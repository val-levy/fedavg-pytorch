import torch
from torch.utils.data import DataLoader

from util import get_dataset

from models import TwoNN
from pathlib import Path


test_dataset = get_dataset(train=False)
TEST_BATCHSIZE = 1_000

def find_latest_weights(dir="./trained-weights"):
    directory = Path(dir)
    matches = list(directory.rglob("fedavg_*.pt")) #used rglob (not reg. glob) here bc i dont know if ill have multiple datasets in the future (ex: subdirs)
    return sorted(matches)[-1]


def compute_accuracy(model):
    """Evaluate an in-memory model and return test accuracy as a float."""
    device = next(model.parameters()).device
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCHSIZE, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            predicted = torch.argmax(model(images), dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate(model=TwoNN()):
    training_weights_path = find_latest_weights()

    global_model = model
    global_model.load_state_dict(torch.load(training_weights_path, weights_only=True))

    accuracy = compute_accuracy(global_model)
    print(f'Model accuracy: {accuracy}')


        
        


if __name__ == "__main__":
    evaluate()