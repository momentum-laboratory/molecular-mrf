
import torch
import torch.nn as nn
import numpy as np

## CNN network used for MTC quantification
## The network contains eight convolutional layers, and the depths of feature space 
## after passing the sequential layers are 128, 512, 1024, 256, 64, 16, 4, and 4.
## The last layer has a sigmoid function to normalize the MTC parameters to [0 1]
## Since the ranges of each parameter are very differnet, normalization makes the summed loss countable for all parameters. 

class nnModel(nn.Module):
    def __init__(self,ds_num,device):
        super(nnModel, self).__init__()
        conv1 = nn.Conv2d(in_channels=ds_num, out_channels=128, kernel_size=3, stride=1,padding=1)
        conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1,padding=1)
        conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1,padding=1)
        conv4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1,padding=1)
        conv5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1,padding=1)
        conv6 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1,padding=1)     
        conv7 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1,padding=1) 
        conv8 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1) 
        
        torch.nn.init.xavier_uniform_(conv1.weight)
        torch.nn.init.xavier_uniform_(conv2.weight)
        torch.nn.init.xavier_uniform_(conv3.weight)
        torch.nn.init.xavier_uniform_(conv4.weight)
        torch.nn.init.xavier_uniform_(conv5.weight)
        torch.nn.init.xavier_uniform_(conv6.weight)
        torch.nn.init.xavier_uniform_(conv7.weight)
        torch.nn.init.xavier_uniform_(conv8.weight)

        relu = nn.ReLU()
        sig  = nn.Sigmoid()
        
        self.device=device
        self.fc_module = nn.Sequential(
            conv1,
            relu,
            conv2,
            relu,
            conv3,
            relu,
            conv4,
            relu,
            conv5,
            relu,
            conv6,
            relu,        
            conv7,
            relu,
            conv8,
            sig
        )


    def forward(self,input):
        out = self.fc_module(input)
        
        out = torch.multiply(out,0.9999)
        out = torch.add(out,0.0001)
        ############ When quatificatoin estimates are 0, the it will cause error in the following signal-generation part (in case dividing with 0).
        ############ By setting the minimum 0.0001, no error will occur

        return out
