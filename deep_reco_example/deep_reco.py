# -*- coding: utf-8 -*-
""" Deep CEST/MT-MRF reconstruction
A deep NN is used for mapping CEST parameters from raw CEST-MRF data
Or Perlman 2021 (operlman@mgh.harvard.edu)
Updated on March 2023 to fit py-cest-mrf changes (OP).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import time
import sys

# >>> User input
train_flag = False # (if false, than only inference)
# <<<<

dtype = torch.DoubleTensor

# Use GPU if available (otherwise use CPU)
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU found and will be used")
else:
    device = 'cpu'
    "GPU was not found. Using CPU"

# Schedule iterations
# number of raw images in the CEST-MRF acquisition schedule
sched_iter = 30

# Training properties
learning_rate = 0.0001
batch_size = 256
num_epochs = 100
noise_std = 0.002  # noise level for training


def normalize_range(original_array, original_min, original_max, new_min, new_max):
    """ Normalizing data to a new range (e.g. to [-1, 1] or [1, 1])
    :param original_array:   input array
    :param original_min: current minimum (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: normalized array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (original_array - a) / (b - a) * (d - c) + c


def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    """ Un-normalizing data to its original range (e.g. to [0, 1400])
    :param normalized_array:  normalized array
    :param original_min: minimum value (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: original array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a


# Organizing the training data
class Dataset(Dataset):
    def __init__(self):

        # If the dictionary was created in the matlab-CEST-MRF code
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # training_data = sio.loadmat('dict.mat')['dict']
        # self.fs_list = training_data['fs'][0][0][:, 0]
        # self.ksw_list = training_data['ksw'][0][0][:, 0].astype(np.float)
        # sig = training_data['sig'][0][0]

        # If the dictionary was created in the py-CEST-MRF code
        training_data = sio.loadmat('dict.mat')
        self.fs_list = training_data['fs_0'].transpose()[:, 0]
        self.ksw_list = training_data['ksw_0'].transpose()[:, 0]
        sig = training_data['sig'].transpose()


        # 2-norm normalization of the dictionary signals
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))

        # Training dictionary size
        # self.len = training_data['ksw'][0][0].size         # matlab-cest-mrf version
        self.len = training_data['ksw_0'].transpose().size  # py-cest-mrf version
        print("There are " + str(self.len) + " entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        return fs, ksw, norm_sig


# Defining the NN architecture
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(sched_iter, 300).type(dtype)
        self.relu1 = nn.ReLU().type(dtype)
        self.l2 = nn.Linear(300, 300).type(dtype)
        self.relu2 = nn.ReLU().type(dtype)
        self.l3 = nn.Linear(300, 2).type(dtype)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x
        
        
# Calculating the min and max fs and ksw for the entire dictionary (will be used for normalization later)
# matlab-cest-mrf-version
# temp_data = sio.loadmat('dict.mat')['dict']
# min_fs = np.min(temp_data['fs'][0][0])
# min_ksw = np.min(temp_data['ksw'][0][0].astype(np.float))
# max_fs = np.max(temp_data['fs'][0][0])
# max_ksw = np.max(temp_data['ksw'][0][0].astype(np.float))

# py-cest-mrf version
temp_data = sio.loadmat('dict.mat')
min_fs = np.min(temp_data['fs_0'])
min_ksw = np.min(temp_data['ksw_0'].transpose().astype(np.float))
max_fs = np.max(temp_data['fs_0'])
max_ksw = np.max(temp_data['ksw_0'].transpose().astype(np.float))

min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).type(dtype)
max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).type(dtype)
del temp_data, min_fs, min_ksw, max_fs, max_ksw

# Initializing the reconstruction network
reco_net = Network().to(device)

# Setting optimizer
optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

# Loading the training dataset
dataset = Dataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8)

loss_per_epoch = np.zeros(num_epochs)

# Storing current time
t0 = time.time()

#   Training loop   #
# ################# #
if train_flag:
    # Initializing the reconstruction network
    reco_net = Network().to(device)
    
    # Setting optimizer
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)
    
    # Loading the training dataset
    dataset = Dataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    
    loss_per_epoch = np.zeros(num_epochs)
    
    # Storing current time
    t0 = time.time()
    
    #   Training loop   #
    # ################# #
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    
        # Cumulative loss
        cum_loss = 0
        counter = np.nan
    
        for counter, dict_params in enumerate(train_loader, 0):
            cur_fs, cur_ksw, cur_norm_sig = dict_params
    
            target = torch.stack((cur_fs, cur_ksw), dim=1)
    
            # Normalizing the target
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                     original_max=max_param_tensor, new_min=-1, new_max=1).to(device)
    
            # Adding noise to the input signals (trajectories)
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()).type(dtype) * noise_std
    
            # Forward step
            prediction = reco_net(noised_sig.to(device))
    
            # Batch loss (MSE)
            loss = torch.mean((prediction - target) ** 2)
    
            # Backward step
            optimizer.zero_grad()
            loss.backward()
    
            # Optimization step
            optimizer.step()
    
            # Storing Cumulative loss
            cum_loss += loss.item()
    
        # Average loss for this epoch
        loss_per_epoch[epoch] = cum_loss / (counter + 1)
        print('Loss = {}'.format(loss_per_epoch[epoch]))
        print('=====')
    
    
    # Displaying the runtime:
    RunTime = time.time() - t0
    print("")
    if RunTime < 60:  # if less than a minute
        print('Total Training time: ' + str(RunTime) + ' sec')
    elif RunTime < 3600:  # if less than an hour
        print('Total Training time: ' + str(RunTime / 60.0) + ' min')
    else:  # If took more than an hour
        print('Total Training time: ' + str(RunTime / 3600.0), ' hour')
    
    # Saving optimized model parameters
    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  #
        'loss_per_epoch': loss_per_epoch,
    }, 'checkpoint')
    print("The optimized model, optimizer state, and loss history were saved to the file: 'checkpoint'")

else:  # inference only
    loaded_checkpoint = torch.load('checkpoint')

    reco_net = Network().to(device)
    reco_net.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    loss_per_epoch = loaded_checkpoint['loss_per_epoch']

# Plotting training loss
plt.figure()
plt.plot(np.arange(100) + 1, loss_per_epoch)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('MSE Loss', fontsize=20)
plt.title('Training Loss', fontsize=20)
plt.show()

#    Testing   #
# ############ #

#  >>> Optional - loading a previously saved model, loss, and optimizer state
# checkpoint = torch.load('checkpoint')
# reco_net.load_state_dict(checkpoint['model_state_dict'])
# loss_per_epoch = checkpoint['loss_per_epoch']
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# <<<

# Loading the acquired data
acquired_data = sio.loadmat('acquired_data.mat')['acquired_data'].astype(np.float)
[_, c_acq_data, w_acq_data] = np.shape(acquired_data)

# Reshaping the acquired data to the shape expected by the NN (e.g. 30 x ... )
acquired_data = np.reshape(acquired_data, (sched_iter, c_acq_data * w_acq_data), order='F')

# 2-norm normalization of the dictionary signals
acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))

# Transposing for compatibility with the NN - now each row is a trajectory
acquired_data = acquired_data.T

# Converting to tensor
acquired_data = Variable(torch.from_numpy(acquired_data).type(dtype), requires_grad=False).to(device)

# Storing current time
t0 = time.time()

# Predicting the test-data "labels"
prediction = reco_net(acquired_data)

# Displaying the runtime:
RunTime = time.time() - t0
print("")
if RunTime < 60:  # if less than a minute
    print('Prediction time: ' + str(RunTime) + ' sec')
elif RunTime < 3600:  # if less than an hour
    print('Prediction time: ' + str(RunTime / 60.0) + ' min')
else:  # If took more than an hour
    print('Prediction time: ' + str(RunTime / 3600.0), ' hour')


# Un-normalizing to go back to physical units
prediction = un_normalize_range(prediction, original_min=min_param_tensor.to(device),
                                 original_max=max_param_tensor.to(device), new_min=-1, new_max=1)

# Reshaping back to the image dimension
quant_map_fs = prediction.cpu().detach().numpy()[:, 0]
quant_map_fs = quant_map_fs.T
quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

quant_map_ksw = prediction.cpu().detach().numpy()[:, 1]
quant_map_ksw = quant_map_ksw.T
quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

# Saving output maps
sio.savemat('nn_reco_maps.mat', {'quant_map_fs': quant_map_fs, 'quant_map_ksw': quant_map_ksw})

# >>> Displaying output maps
pdf_handle = PdfPages('deep_reco_results.pdf')

plt.figure()
plt.subplot(121)
plt.imshow(quant_map_fs * 110e3 / 3, cmap='viridis', clim=(0, 120))
plt.title('[L-arg] (mM)', fontsize=20)
cb = plt.colorbar(ticks=np.arange(0.0, 120+20, 20), orientation='horizontal', fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=20)
plt.axis("off")

plt.subplot(122)
plt.imshow(quant_map_ksw, cmap='magma', clim=(0, 500))
cb = plt.colorbar(ticks=np.arange(0.0, 500+100, 100), orientation='horizontal', fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=20)
plt.axis("off")
plt.title('k$_{sw}$ (Hz)', fontsize=20)

# plt.show() # for screen display
pdf_handle.savefig() # storing to pdf instead of display on screen

pdf_handle.close()
