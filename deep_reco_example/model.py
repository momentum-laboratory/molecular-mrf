import torch.nn as nn

class Network(nn.Module):

    def __init__(self, sig_n):
        super(Network, self).__init__()
        self.l1 = nn.Linear(sig_n, 300)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(300, 300)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(300, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x