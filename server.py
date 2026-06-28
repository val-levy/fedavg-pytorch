import torch
from torch import nn

import random


#SERVER CLASS
class Server:

    def __init__(self, global_model, clients, device = None):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def set_device(self, device_update=None):
        if device_update is not None:
            self.device = device_update
            return
        if self.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
        )

    # C - fraction of clients chosen
    def select_clients(self, fraction=None):
        """
        Randomly choose clients for this current round

        Returns a list of sampled clients
        """

        if fraction is None:
            fraction = 0.1

        k = max(1, int(fraction * len(self.clients))) # if the clients selected is less than 1, round up to 1
        return random.sample(self.clients, k)


    def train_round(self, epochs=1, fraction=0.1):
        selected_clients = self.select_clients(fraction=fraction)

        client_updates = []

        for client in selected_clients:
            update = client.train(self.global_model, epochs=epochs)
            client_updates.append(update)
        
        self.aggregate(client_updates)

    def aggregate(self, client_updates):
        """
        Average the client models into one new global model
        """

        total_samples = sum(update.num_samples for update in client_updates)

        new_state = {}

        for key in self.global_model.state_dict().keys():
            new_state[key] = sum(
                update.state_dict[key] * (update.num_samples /total_samples) for update in client_updates
            )

        self.global_model.load_state_dict(new_state)
