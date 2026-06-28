import argparse
import datetime
import json
import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from client import Client
from evaluate import compute_accuracy
from models import TwoNN, CNN
from server import Server
from util import get_dataset, make_shards

MODELS = {"twonn": TwoNN, "cnn": CNN}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
TOTAL_IMAGES = 60_000
train_dataset = get_dataset(train=True)

DEFAULTS = {"E": 1, "K": 100, "B": 64, "L": 0.01, "C": 0.1, "iid": True}

EVAL_ROUNDS = [*range(0, 21), 30, 50, 100, 200, 300, 400, 500, 1000]
MAX_ROUNDS = max(EVAL_ROUNDS)
EVAL_SET = set(EVAL_ROUNDS)


def parse_trial(s):
    config = dict(DEFAULTS)
    for part in s.split(","):
        key, val = part.strip().split("=")
        key = key.strip().upper()
        config[key] = float(val) if key in ("L", "C") else int(val)
    return config


def trial_label(config):
    dist = "IID" if config["iid"] else "non-IID"
    return f"E={config['E']} K={config['K']} B={config['B']} L={config['L']} C={config['C']} {dist}"


def run_trial(config, target=None, check_every=5, max_rounds=None, model_cls=TwoNN):
    """Train one trial, evaluating at EVAL_ROUNDS (plus every `check_every` rounds).

    Stops early at the first evaluated round where test accuracy >= `target`.
    Returns (checkpoints, target_round) where target_round is the round the
    target was first reached, or None if it was never reached.
    """
    if max_rounds is None:
        max_rounds = MAX_ROUNDS
    shards = make_shards(train_dataset, config["K"], iid=config["iid"])

    clients = [
        Client(
            i,
            DataLoader(shard, batch_size=config["B"], shuffle=True),
            device=DEVICE,
            learn_rate=config["L"],
        )
        for i, shard in enumerate(shards)
    ]

    server = Server(model_cls().to(DEVICE), clients=clients, device=DEVICE)

    # round 0: accuracy before any training
    checkpoints = {0: compute_accuracy(server.global_model)}
    target_round = None

    for r in range(1, max_rounds + 1):
        server.train_round(epochs=config["E"], fraction=config["C"])

        if r in EVAL_SET or r % check_every == 0 or r == max_rounds:
            acc = compute_accuracy(server.global_model)
            checkpoints[r] = acc
            print(f"  Round {r}/{max_rounds}  acc={acc:.4f}", end="\r")
            if target is not None and acc >= target:
                target_round = r
                print(f"\n  Reached target {target:.0%} at round {r} (acc={acc:.4f})")
                break
        else:
            print(f"  Round {r}/{max_rounds}", end="\r")

    print()
    return checkpoints, target_round


def run_lr_sweep(config, lrs, target, check_every, max_rounds, model_cls=TwoNN):
    """Run `config` once per learning rate; return a list of result dicts and the best LR.

    "Best" = fewest rounds to reach target (configs that never reach it rank last,
    broken by highest final accuracy).
    """
    sweep = []
    for lr in lrs:
        cfg = config | {"L": lr}
        print(f"\n  -- L={lr}")
        checkpoints, target_round = run_trial(
            cfg, target=target, check_every=check_every, max_rounds=max_rounds, model_cls=model_cls
        )
        final_acc = checkpoints[max(checkpoints)]
        sweep.append({
            "lr": lr,
            "target_round": target_round,
            "final_acc": final_acc,
            "checkpoints": checkpoints,   # per-round accuracy curve (for Figure-2 style plots)
        })

    # rank: reached-target first (lowest round), then by best final accuracy
    def rank(item):
        reached = item["target_round"] is not None
        return (not reached, item["target_round"] or float("inf"), -item["final_acc"])

    best = min(sweep, key=rank)
    return sweep, best["lr"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "FedAvg experiment runner. trains each config up to 5000 rounds, "
            f"evaluating at rounds {EVAL_ROUNDS}"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trial",
        action="append",
        metavar="KEY=VAL,...",
        help=(
            "Trial config as comma-separated key=value pairs "
            "(E=epochs, K=clients, B=batch_size, L=learn_rate). "
            "Unset keys use defaults. Repeatable for multiple lines on the plot."
        ),
    )
    iid_group = parser.add_mutually_exclusive_group()
    iid_group.add_argument("--iid",     dest="iid", action="store_true",  default=True,  help="IID data distribution (default)")
    iid_group.add_argument("--non-iid", dest="iid", action="store_false",                help="Non-IID: sort by label before sharding")
    parser.add_argument("--target", type=float, default=0.97, help="Stop a trial once test accuracy reaches this (0 to disable)")
    parser.add_argument("--check-every", type=int, default=5, help="Evaluate accuracy every N rounds (for early-stop detection)")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="Cap on communication rounds per trial")
    parser.add_argument("--lr-sweep", metavar="L1,L2,...", help="Comma-separated learning rates to sweep per trial; reports the best L")
    parser.add_argument("--model", choices=sorted(MODELS), default="twonn", help="Model architecture")
    parser.add_argument("--bench", type=int, default=0, metavar="N", help="Time N rounds of the first --trial, print sec/round, and exit")
    parser.add_argument("--output", default="results", help="Directory to save plot and JSON data")
    args = parser.parse_args()

    if not args.trial:
        parser.error("Provide at least one --trial, e.g.  --trial E=1,B=10  --trial E=5,B=64")

    target = args.target if args.target > 0 else None
    model_cls = MODELS[args.model]

    # --- benchmark mode: measure wall-time per round on this machine ---
    if args.bench:
        import time
        config = parse_trial(args.trial[0]) | {"iid": args.iid}
        shards = make_shards(train_dataset, config["K"], iid=config["iid"])
        clients = [Client(i, DataLoader(s, batch_size=config["B"], shuffle=True),
                          device=DEVICE, learn_rate=config["L"]) for i, s in enumerate(shards)]
        server = Server(model_cls().to(DEVICE), clients=clients, device=DEVICE)
        server.train_round(epochs=config["E"], fraction=config["C"])  # warm-up (not timed)
        t0 = time.time()
        for _ in range(args.bench):
            server.train_round(epochs=config["E"], fraction=config["C"])
        dt = (time.time() - t0) / args.bench
        print(f"\n{args.model} {trial_label(config)} on {DEVICE}: {dt:.2f} s/round "
              f"(~{dt/60:.2f} min/100 rounds)")
        return

    # --- learning-rate sweep mode ---
    if args.lr_sweep:
        lrs = [float(x) for x in args.lr_sweep.split(",")]
        best_per_config = {}
        for t in args.trial:
            config = parse_trial(t) | {"iid": args.iid}
            label = trial_label({**config, "L": "?"})
            print(f"\n=== LR sweep: {label} ===")
            sweep, best_lr = run_lr_sweep(config, lrs, target, args.check_every, args.max_rounds, model_cls=model_cls)
            best_per_config[label] = {"best_lr": best_lr, "sweep": sweep}
            print(f"\n  Sweep results for {label}:")
            for item in sweep:
                tr = item["target_round"]
                tr_str = str(tr) if tr is not None else "not reached"
                print(f"    L={item['lr']:<6} rounds-to-target={tr_str:<12} final_acc={item['final_acc']:.4f}")
            print(f"  >> Best L = {best_lr}")

        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_path = os.path.join(args.output, f"lr_sweep_{timestamp}.json")
        with open(sweep_path, "w") as f:
            json.dump({"target": target, "model": args.model, "lrs": lrs, "results": best_per_config}, f, indent=2)
        print(f"\nBest learning rates:")
        for label, info in best_per_config.items():
            print(f"  {label:<45} L={info['best_lr']}")
        print(f"\nSweep data saved to {sweep_path}")
        return
    configs = [parse_trial(t) | {"iid": args.iid} for t in args.trial]
    results = {}
    target_rounds = {}

    for config in configs:
        label = trial_label(config)
        print(f"\nRunning: {label}")
        checkpoints, target_round = run_trial(config, target=target, check_every=args.check_every, max_rounds=args.max_rounds, model_cls=model_cls)
        results[label] = checkpoints
        target_rounds[label] = target_round
        last_round = max(checkpoints)
        print(f"  Accuracy at round {last_round}: {checkpoints[last_round]:.4f}")

    if target is not None:
        print(f"\nRounds to reach {target:.0%} target accuracy:")
        for label in results:
            tr = target_rounds[label]
            print(f"  {label:<45} {tr if tr is not None else 'not reached'}")

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_path = os.path.join(args.output, f"experiment_{timestamp}.json")
    with open(data_path, "w") as f:
        json.dump({"target": target, "target_rounds": target_rounds, "trials": results}, f, indent=2)
    print(f"\nData saved to {data_path}")

    plt.figure(figsize=(10, 5))
    for label, checkpoints in results.items():
        xs = sorted(checkpoints.keys())
        ys = [checkpoints[r] for r in xs]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xscale("log")
    plt.xlabel("Communication Round (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("FedAvg: Accuracy vs. Communication Rounds")
    plt.xticks(EVAL_ROUNDS, labels=EVAL_ROUNDS, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plot_path = os.path.join(args.output, f"experiment_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
