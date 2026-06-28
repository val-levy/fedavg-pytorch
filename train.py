import torch
from torch.utils.data import DataLoader


from models import TwoNN
from server import Server
from client import Client
from util import get_dataset, make_shards

import argparse
import datetime
import os

#GLOBALS
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
TOTAL_IMAGES = 60_000
NUM_ROUNDS = 10
train_dataset = get_dataset(train=True)


def save_weights(model, dir="./trained-weights"):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), f"{dir}/fedavg_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

def train(epochs=1, num_clients=100, batch_size=64, learn_rate=0.01, fraction=0.1, iid=True):

    shards = make_shards(train_dataset, num_clients, iid=iid)

    clients = []
    for i, shard in enumerate(shards):
        client = Client(
            i,
            DataLoader(
                shard,
                batch_size=batch_size,
                shuffle=True
            ),
            device=DEVICE,
            learn_rate=learn_rate
        )
        clients.append(client)

    server = Server(
        TwoNN(),
        clients=clients,
        device=DEVICE
    )

    #training rounds
    print("Training...\n\n")
    for i in range(NUM_ROUNDS):
        print(f'Training Round: {i}')
        server.train_round(epochs=epochs, fraction=fraction)
        print(f'Round complete.')
    save_weights(server.global_model)
    print("Training Complete!")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FedAvg training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-E", "--epochs",      type=int,   default=1,    help="Local epochs per client per round")
    parser.add_argument("-K", "--clients",     type=int,   default=100,  help="Number of clients")
    parser.add_argument("-B", "--batch-size",  type=int,   default=64,   help="Local minibatch size")
    parser.add_argument("-L", "--learn-rate",  type=float, default=0.01, help="Client learning rate")
    parser.add_argument("-C", "--fraction",    type=float, default=0.1,  help="Fraction of clients selected per round")
    iid_group = parser.add_mutually_exclusive_group()
    iid_group.add_argument("--iid",     dest="iid", action="store_true",  default=True,  help="IID data distribution (default)")
    iid_group.add_argument("--non-iid", dest="iid", action="store_false",                help="Non-IID: sort by label before sharding")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        num_clients=args.clients,
        batch_size=args.batch_size,
        learn_rate=args.learn_rate,
        fraction=args.fraction,
        iid=args.iid,
    )
