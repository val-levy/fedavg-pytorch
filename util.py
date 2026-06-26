#pull data
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


# Uncomment out the following lines and run this script to download datasets
# get_dataset()
# get_dataset(train=True)