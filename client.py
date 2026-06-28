import torch
from torch import nn

import copy


#CLIENTUPDATE CLASS
class ClientUpdate:
    def __init__(self, state_dict, num_samples, loss):
        self.state_dict = state_dict
        self.num_samples = num_samples
        self.loss = loss



#CLIENT CLASS
class Client:

    def __init__(self, client_id, train_loader, device=None, learn_rate=None):
        self.client_id = client_id
        self.train_loader = train_loader #random subset of server dataset
        self.device = device
        self.learn_rate = learn_rate

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

    def train(self, global_model, epochs=1):
        """
        Train the current local dataset on a copy of the global model

        Returns updated weights, length of data trained

        USAGE: state_dict, num_samples = client.train(global_model)
        """

        #first set the learn rate if there was one specified
        if self.learn_rate is not None:
            lr = self.learn_rate
        else:
            lr = 0.01 # set default to 0.01 as a baseline

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train() #set model to train mode

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        total_loss = 0
        num_batches = 0

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                outputs = local_model(images)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

                loss.backward()
                optimizer.step()
        
        return ClientUpdate(
            state_dict=local_model.state_dict(), 
            num_samples=len(self.train_loader.dataset),
            loss=total_loss/num_batches
        )