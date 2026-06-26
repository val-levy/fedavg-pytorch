import torch
from torch import nn
from torch.utils.data import DataLoader

from models import TwoNN, CNN
from server import Server
from client import Client

#GLOBALS
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

