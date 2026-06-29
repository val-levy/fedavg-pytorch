import copy
import json
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from models import TwoNN
from util import get_dataset, make_shards
from client import Client


def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def compute_loss(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            total_loss += criterion(model(images), labels).item()
            n_batches += 1
    return total_loss / n_batches


def interpolate_weights(sd0, sd1, theta):
    return {k: (1 - theta) * sd0[k] + theta * sd1[k] for k in sd0}


def train_shard(shard, start_model, device, epochs, lr, batch_size):
    loader = DataLoader(shard, batch_size=batch_size, shuffle=True)
    client = Client(client_id=0, train_loader=loader, device=device, learn_rate=lr)
    return client.train(start_model, epochs=epochs).state_dict


def sweep(sd0, sd1, full_loader, device, n_theta=21):
    eval_model = TwoNN().to(device)
    thetas = torch.linspace(0, 1, n_theta).tolist()
    losses = []
    for theta in thetas:
        eval_model.load_state_dict(interpolate_weights(sd0, sd1, theta))
        losses.append(compute_loss(eval_model, full_loader, device))
    return thetas, losses


def main():
    EPOCHS = 10
    LR = 0.1
    BATCH_SIZE = 10
    N_THETA = 21

    device = get_device()
    print(f"Device: {device}")

    train_dataset = get_dataset(train=True)
    shards = make_shards(train_dataset, 2, iid=True)
    full_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    all_thetas = None
    all_losses = {}

    for mode in ("shared", "independent"):
        print(f"\n--- {mode} init ---")
        if mode == "shared":
            base = TwoNN()
            m0, m1 = base, copy.deepcopy(base)
        else:
            m0, m1 = TwoNN(), TwoNN()

        sd0 = train_shard(shards[0], m0, device, EPOCHS, LR, BATCH_SIZE)
        print(f"  model 0 trained")
        sd1 = train_shard(shards[1], m1, device, EPOCHS, LR, BATCH_SIZE)
        print(f"  model 1 trained")

        thetas, losses = sweep(sd0, sd1, full_loader, device, N_THETA)
        all_thetas = thetas
        all_losses[mode] = losses
        print(f"  loss range: {min(losses):.4f} – {max(losses):.4f}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(all_thetas, all_losses["shared"], "o-", label="Shared init", markersize=4)
    ax.plot(all_thetas, all_losses["independent"], "s-", label="Independent init", markersize=4)
    ax.set_xlabel("Mixing weight θ")
    ax.set_ylabel("Training loss (cross-entropy)")
    ax.set_title("Loss landscape under linear model interpolation (McMahan et al. Fig. 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = f"results/figure1_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot → {plot_path}")

    json_path = f"results/figure1_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"thetas": all_thetas, "losses": all_losses}, f, indent=2)
    print(f"Data → {json_path}")


if __name__ == "__main__":
    main()
