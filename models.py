from torch import nn

# 2 layer neural network
class TwoNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,10)
        )
    
    def forward(self,x):
        return self.net(x)
    

#convolutional neural network
class CNN(nn.Module):
    super().__init__()
    #todo