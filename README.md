# FedAvg in PyTorch

A from-scratch PyTorch implementation of the Federated Averaging (FedAvg) algorithm introduced in *Communication-Efficient Learning of Deep Networks from Decentralized Data*.

## Features

- Simulated client-server federated learning on MNIST
- Local client training with SGD
- Weighted model aggregation (FedAvg)
- Modular architecture with separate Client, Server, and Model classes
- Timestamped model checkpointing
- Separate training and evaluation scripts

## Project Structure

```
.
├── train.py 
├── evaluate.py
├── client.py
├── server.py
├── models.py
├── util.py
├── trained-weights/  # saved model checkpoints (created on first run)
└── README.md
```

## Usage

**Train** — downloads MNIST automatically, runs FedAvg for N rounds, and saves weights to `trained-weights/`:
```bash
python train.py
```

**Evaluate** — loads the latest saved weights and prints test accuracy:
```bash
python evaluate.py
```

## References

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
