import torch
import matplotlib.pyplot as plt
import numpy as np

import time
import matplotlib
from cmcrameri import cm

import os

from utils.colormaps import b_viridis, b_winter

import scipy.io as sio

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define model
    sig_n = 30
    human_net = torch.jit.load('human_net.pt')
    human_net.to(device)
    
    # load checkpoint
    chk_fn = f'checkpoint.pt'
    state_dict = torch.load(chk_fn)

    scaling = state_dict['scaling']
    params = state_dict['recon_tissues']

    # load data
    data_fn = f'data_to_match.npy'
    acquired_data = np.load(data_fn)
    # zoom in
    acquired_data = acquired_data[:,30:220,50:200]
    print("Raw MRF 2D slice shape:" + str(acquired_data.shape))

    _, c_acq_data, w_acq_data = np.shape(acquired_data)

    # Reshaping the acquired data to the shape expected by the NN (e.g. 30 x ... )
    data = np.reshape(acquired_data, (sig_n, c_acq_data * w_acq_data), order='F')
    data = data / (np.linalg.norm(data, ord=2, axis=0) + 1e-8)

    # Transposing for compatibility with the NN - now each row is a trajectory
    data = data.T
    data = torch.from_numpy(data).to(device).float()

    # Switching to evaluation mode
    human_net.eval()

    t0 = time.time()
    prediction = human_net(data)
    print(f"Prediction took {time.time() - t0:.5f} seconds")

    # scaling the prediction
    for i, param in enumerate(params):
        prediction[:, i] = prediction[:, i] * scaling[param]

    # Reshaping back to the image dimension

    quant_maps = {}
    for i, param in enumerate(params):
        quant_maps[param] = prediction[:, i].detach().cpu().numpy()
        quant_maps[param] = quant_maps[param].T
        quant_maps[param] = np.reshape(quant_maps[param], (c_acq_data, w_acq_data), order='F')

    sio.savemat('quant_maps.mat', quant_maps)
    
    # load brain mask (created with SAM model)
    mask_fn = 'human_mask.npy'
    mask = np.load(mask_fn)

    # zoom in
    mask = mask[30:220,50:200]

    # Plotting
    ranges = {
        'T1w': (500, 2500),
        'T2w': (70, 130),
        'M0s': (0.0, 0.5),
        'M0ss': (0, 12),
        'Ksw': (0, 50),
        'Kssw': (0, 60)
    }

    ticks = {
        'T1w': np.arange(ranges['T1w'][0], ranges['T1w'][1] + 500, 500),
        'T2w': np.arange(ranges['T2w'][0], ranges['T2w'][1] + 60, 20),
        'M0s': np.arange(ranges['M0s'][0], ranges['M0s'][1] + 0.1, 0.1),
        'M0ss': np.arange(ranges['M0ss'][0], ranges['M0ss'][1] + 2 , 2),
        'Ksw': np.arange(ranges['Ksw'][0], ranges['Ksw'][1] + 10, 10),
        'Kssw': np.arange(ranges['Kssw'][0], ranges['Kssw'][1] + 10, 20)
    }

    colormaps = {
        'T1w': cm.lipari,
        'T2w': cm.navia,
        'M0s': b_viridis,
        'M0ss':  b_viridis,
        'Ksw': 'magma',
        'Kssw': 'magma'
    }

    positions = {
        'T1w': (0, 0),
        'T2w': (1, 0),
        'M0s': (0, 1),
        'M0ss': (0, 2),
        'Ksw': (1, 1),
        'Kssw': (1, 2)
    }

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom' 
    plt.rcParams['mathtext.rm'] = 'Arial' 
    plt.rcParams['mathtext.it'] = 'Arial' 
    plt.rcParams['mathtext.bf'] = 'Arial:bold'

    plt.figure(figsize=(18,10))

    matplotlib.rc('xtick', labelsize=25) 
    matplotlib.rc('ytick', labelsize=25) 
    plt.rcParams.update({'font.size': 25})

    for i, param in enumerate(params):
        position = positions[param]
        plt.subplot2grid((2, 3), position)
        signal = quant_maps[param]
        title = param
        if param in ['T1w', 'T2w']:
            title = param[:1] + f'$_{{{param[1]}}}$' +' (ms)'

        if param in ['M0s', 'M0ss']:
            signal = signal * 100
            title = 'f' + f'$_{{{param[2:]}}}$' +' (%)'
        if param in ['Ksw', 'Kssw']:
            signal = signal
            title = param.lower()[:1] + f'$_{{{param[1:]}}}$' +' (s$^{-1}$)'
            
        plt.imshow(signal*mask, cmap=colormaps[param])
        plt.title(title)       
        plt.colorbar(orientation='vertical', ticks=ticks[param])
        plt.clim(ranges[param])
        plt.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08, hspace=0.18)  # Adjust spacing between subplots
    # plt.tight_layout()

    plt.savefig('human_results.eps', format='eps')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
