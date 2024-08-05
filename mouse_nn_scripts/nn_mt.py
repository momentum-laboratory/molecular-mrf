import torch
import numpy as np
from torch.utils.data import DataLoader

import time
import os
# import sys

import scipy.io as sio
import matplotlib.pyplot as plt

import tqdm

from utils.colormaps import b_viridis, b_winter
from utils.normalization import normalize_range, un_normalize_range
from utils.image_overlay import image_overlay
from utils.seed import set_seed

# from sequential_nn_example.configs import ConfigMouse
# from sequential_nn_example.sequences import write_sequence
from sequential_nn_example.dataset import MTDataset
from sequential_nn_example.model import Network

from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

from my_funcs.plot_functions import t1_t2_pixel_reader

from my_funcs.cest_functions import bruker_dataset_creator
from my_funcs.cest_functions import dicom_data_arranger

import cv2
import pandas as pd

def main():
    data_f = 'data'
    output_f = 'results'

    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Navigate up one directory level
    mt_dict_fn = os.path.join(parent_dir, 'data', 'exp', 'mouse_new_t_mt', 'MT52', 'dict.pkl')  # dict directory

    txt_file_name = 'labarchive_notes.txt'
    scan_name = '24_04_04_glu_mouse_37deg'
    sub_name = '3_mouse3_two_ears'  # '1_mouse1_left' (t1t2 moves)
    fp_prtcl_name = 'MT52'
    glu_mouse_fn = os.path.join(parent_dir, 'data', 'scans', scan_name, sub_name)
    mask_fn = os.path.join(glu_mouse_fn, 'mask.npy')

    # Schedule iterations
    # number of raw images in the CEST-MRF acquisition schedule
    sched_iter = 30

    # Training properties
    learning_rate = 2e-4
    batch_size = 1024
    num_epochs = 150
    noise_std = 0.01  # noise level for training

    min_delta = 0.05  # minimum absolute change in the loss function
    patience = np.inf

    device = initialize_device()
    print(f"Using device: {device}")

    min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor = define_min_max(mt_dict_fn)

    # Loading the training dataset
    train_dataset = MTDataset(mt_dict_fn)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    reco_net = train_network(train_loader, device, sched_iter, learning_rate, num_epochs, noise_std, patience,
                             min_delta, min_param_tensor, max_param_tensor, min_water_t1t2_tensor,
                             max_water_t1t2_tensor)


# Function to initialize device
def initialize_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Function to train the network
def train_network(train_loader, device, sched_iter, learning_rate, num_epochs, noise_std, patience, min_delta,
                  min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor):
    # Initializing the reconstruction network
    reco_net = Network(sched_iter, n_hidden=2, n_neurons=300).to(device)

    # print amount of parameters
    print('Number of model parameters: ', sum(p.numel() for p in reco_net.parameters() if p.requires_grad))

    # Setting optimizer
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

    # Storing current time
    t0 = time.time()

    loss_per_epoch = []
    loss_per_epoch_test = []

    patience_counter = 0
    min_loss = 100

    reco_net.train()

    pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # Cumulative loss
        cum_loss = 0
        counter = np.nan

        for counter, dict_params in enumerate(train_loader, 0):
            cur_fs, cur_ksw, cur_t1w, cur_t2w, cur_norm_sig = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1)
            input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1)

            # Normalizing the target and input_water_t1t2
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                     original_max=max_param_tensor, new_min=0, new_max=1).to(device)

            input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                               original_max=max_water_t1t2_tensor, new_min=0, new_max=1)

            # Adding noise to the input signals (trajectories) a
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()) * noise_std

            # noised_sig = noised_sig / torch.linalg.norm(noised_sig, dim=1, ord=2, keepdim=True)

            # adding the water_t1t2 input as two additional elements in the noised_sig vector
            noised_sig = torch.hstack((input_water_t1t2, noised_sig))

            # Forward step
            prediction = reco_net(noised_sig.to(device).float())

            # Batch loss (MSE)
            loss = torch.mean((prediction.float() - target.float()) ** 2)

            # Backward step
            optimizer.zero_grad()
            loss.backward()

            # Optimization step
            optimizer.step()
            # Storing Cumulative loss
            cum_loss += loss.item()

        # Average loss for this epoch
        loss_per_epoch.append(cum_loss / (counter + 1))

        pbar.set_description(f'Epoch: {epoch + 1}/{num_epochs}, Loss = {loss_per_epoch[-1]}')
        pbar.update(1)

        if (min_loss - loss_per_epoch[-1]) / min_loss > min_delta:
            min_loss = loss_per_epoch[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print('Early stopping!')
            break
    pbar.close()
    print(f"Training took {time.time() - t0:.2f} seconds")

    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  #
        'loss_per_epoch': loss_per_epoch,
        'loss_per_epoch_test': loss_per_epoch_test
    }, 'checkpoint_m.pt')

    return reco_net

def define_min_max(mt_dict_fn):
    dictionary = pd.read_pickle(mt_dict_fn)

    # min_fs = np.min(dictionary['fs_0'])  # uncomment if non-zero minimum limit is required
    # min_ksw = np.min(dictionary['ksw_0'].transpose().astype(float))  # uncomment if non-zero minimum limit needed
    max_fs = np.max(dictionary['fs_0'])
    max_ksw = np.max(dictionary['ksw_0'].transpose().astype(float))

    min_t1w = np.min(dictionary['t1w'])
    min_t2w = np.min(dictionary['t2w'].transpose().astype(float))
    max_t1w = np.max(dictionary['t1w'])
    max_t2w = np.max(dictionary['t2w'].transpose().astype(float))

    min_param_tensor = torch.tensor(np.hstack((0, 0)), requires_grad=False)  # can be switched to  min_fs, min_ksw
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False)

    min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False)
    max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False)

    return min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    set_seed(2024)

    main()
