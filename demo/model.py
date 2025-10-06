import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, sched_iter, n_hidden=2, n_neurons=300):
        super(Network, self).__init__()
        self.input = nn.Linear(sched_iter + 2, n_neurons)
        self.output = nn.Linear(n_neurons, 2)
        
        self.hidden = nn.ModuleList([nn.Linear(n_neurons, n_neurons) for i in range(n_hidden)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_neurons) for i in range(n_hidden)])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.input(x))

        for layer, b_norm in zip(self.hidden, self.batch_norms):
            x = torch.relu(layer(x))            
            x = self.dropout(x)
            x = b_norm(x)

        x = self.output(x)
        x = torch.sigmoid(x)
        return x
