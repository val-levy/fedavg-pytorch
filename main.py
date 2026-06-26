import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from models import TwoNN, CNN
from server import Server
from client import Client
from util import get_dataset

#GLOBALS
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

NUM_CLIENTS = 100
TOTAL_IMAGES = 60_000
NUM_ROUNDS = 10
#set batch size
TRAIN_BATCHSIZE = 64
TEST_BATCHSIZE = 1_000
#set datasets
train_dataset = get_dataset(train=True)
test_dataset = get_dataset(train=False)



def main():

    sizes = []
    for _ in range(NUM_CLIENTS):
        sizes.append( TOTAL_IMAGES // NUM_CLIENTS )
    
    shards = random_split(train_dataset, sizes)

    clients = []
    for i, shard in enumerate(shards):
        client = Client(
            i, 
            DataLoader(
                shard,
                batch_size=TRAIN_BATCHSIZE, 
                shuffle=True
            ), 
            device=DEVICE
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
        print(f'Training Round: {i}\n')
        server.train_round()
        print(f'Round complete.\n')


    return


if __name__ == "__main__":
    main()
