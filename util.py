#pull data
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dataset(path='./data', train=False):
    """
    Default is test data. Set optional `train` parameter to True to set to train.

    Default path is './data'.
    """
    return datasets.MNIST(
        root=path,
        train=train,
        download=True,
        transform=transform
        )

def make_shards(dataset, num_clients, iid=True, shards_per_client=2):
    """
    Split dataset into num_clients shards.

    IID: random shuffle then split evenly across clients.

    Non-IID (McMahan et al. "pathological" split): sort by label, cut into
    `shards_per_client * num_clients` equal contiguous shards, then assign
    `shards_per_client` *random* shards to each client. With the default of 2
    and K=100 on MNIST this gives 200 shards of 300 examples, 2 per client, so
    each client holds at most 2 digit classes.
    """
    n = len(dataset)

    if iid:
        shard_size = n // num_clients
        indices = torch.randperm(n).tolist()
        return [
            Subset(dataset, indices[i * shard_size : (i + 1) * shard_size])
            for i in range(num_clients)
        ]

    # --- non-IID pathological split ---
    num_shards = shards_per_client * num_clients
    shard_size = n // num_shards

    labels = torch.tensor([dataset[i][1] for i in range(n)])
    sorted_indices = torch.argsort(labels).tolist()

    # carve the label-sorted indices into num_shards contiguous shards
    shards = [
        sorted_indices[s * shard_size : (s + 1) * shard_size]
        for s in range(num_shards)
    ]

    # randomly assign shards_per_client shards to each client
    shard_order = torch.randperm(num_shards).tolist()
    clients = []
    for c in range(num_clients):
        picked = shard_order[c * shards_per_client : (c + 1) * shards_per_client]
        client_indices = [idx for s in picked for idx in shards[s]]
        clients.append(Subset(dataset, client_indices))
    return clients
