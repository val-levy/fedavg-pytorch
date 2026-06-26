# FedAvg in PyTorch

A from-scratch PyTorch implementation of the Federated Averaging (FedAvg) algorithm introduced in *Communication-Efficient Learning of Deep Networks from Decentralized Data*.

## Features

- Simulated client-server federated learning
- Local client training with SGD
- Weighted model aggregation (FedAvg)
- Modular architecture with separate Client, Server, and Model classes
- Built using PyTorch

## Project Structure

```sh
.
├── client.py
├── server.py
├── models.py
├── datasets.py
├── main.py
└── README.md
```

## References

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
