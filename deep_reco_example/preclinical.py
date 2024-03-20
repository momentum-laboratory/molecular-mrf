import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import tqdm

from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

from utils.normalization import normalize_range, un_normalize_range
from utils.colormaps import b_viridis
from utils.seed import set_seed

from deep_reco_example.dataset import DatasetMRF
from deep_reco_example.model import Network
from deep_reco_example.configs import ConfigPreclinical


def main():
    # Schedule iteration (signal dimension)
    # number of raw images in the CEST-MRF acquisition schedule
    sig_n = 30

    # Training properties
    learning_rate = 0.0003
    batch_size = 512
    num_epochs = 100
    noise_std = 0.002  # noise level for training

    patience = 10  # number of epochs to wait before early stopping
    min_delta = 0.01  # minimum absolute change in loss to be considered as an improvement

    device = initialize_device()
    print(f"Using device: {device}")

    data_folder = r'data'
    output_folder = r'results'

    cfg = ConfigPreclinical().get_config()

    write_yaml_dict(cfg)

    dictionary = generate_dict(cfg)
    min_param_tensor, max_param_tensor = define_min_max(dictionary)

    train_loader = prepare_dataloader(dictionary, batch_size=batch_size)
    reco_net = Network(sig_n).to(device)
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)
    reco_net = train_network(train_loader, reco_net, optimizer, device, learning_rate, num_epochs, noise_std,
                             min_param_tensor, max_param_tensor, patience, min_delta)

    data_fn = os.path.join(data_folder, 'acquired_data.mat')
    eval_data, c_acq_data, w_acq_data = load_and_preprocess_data(data_fn, sig_n)
    quant_maps = evaluate_network(reco_net, eval_data, device, min_param_tensor, max_param_tensor,
                                  c_acq_data=c_acq_data, w_acq_data=w_acq_data)

    # load mask from created using dot-product values
    mask = np.load('../dot_prod_example/mask.npy')
    save_and_plot_results(quant_maps, output_folder, mask)


def load_and_preprocess_data(data_fn, sig_n):
    acquired_data = sio.loadmat(data_fn)['acquired_data'].astype(np.float)
    _, c_acq_data, w_acq_data = np.shape(acquired_data)

    # Reshaping the acquired data to the shape expected by the NN (e.g. 30 x ... )
    acquired_data = np.reshape(acquired_data, (sig_n, c_acq_data * w_acq_data), order='F')
    acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))

    # Transposing for compatibility with the NN - now each row is a trajectory
    acquired_data = acquired_data.T

    acquired_data = torch.from_numpy(acquired_data).float()
    acquired_data.requires_grad = False

    return acquired_data, c_acq_data, w_acq_data

def initialize_device():
    """Initialize device (GPU/CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataloader(data, batch_size):
    """Prepare DataLoader for training."""
    dataset = DatasetMRF(data)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)


def train_network(train_loader, reco_net, optimizer, device, learning_rate, num_epochs, noise_std, min_param_tensor,
                  max_param_tensor, patience, min_delta):
    """Train the network."""
    t0 = time.time()
    loss_per_epoch = []
    patience_counter = 0
    min_loss = 100

    pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # Cumulative loss
        cum_loss = 0

        for counter, dict_params in enumerate(train_loader, 0):
            cur_fs, cur_ksw, cur_norm_sig = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1)

            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                     original_max=max_param_tensor, new_min=-1, new_max=1).to(device).float()

            # Adding noise to the input signals (trajectories)
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()) * noise_std
            noised_sig = noised_sig.to(device).float()

            prediction = reco_net(noised_sig)

            # Batch loss (MSE)
            loss = torch.mean((prediction - target) ** 2)

            # Backward step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

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
    },  'checkpoint.pt')

    return reco_net


def evaluate_network(reco_net, data, device, min_param_tensor, max_param_tensor, c_acq_data=30, w_acq_data=126):
    """Evaluate the network on new data."""
    reco_net.eval()
    with torch.no_grad():
        inputs = data.to(device).float()
        outputs = reco_net(inputs)

    outputs = un_normalize_range(outputs, original_min=min_param_tensor.to(device),
                                 original_max=max_param_tensor.to(device), new_min=-1, new_max=1)

    quant_map_fs = outputs.cpu().detach().numpy()[:, 0]
    quant_map_fs = quant_map_fs.T
    quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

    quant_map_ksw = outputs.cpu().detach().numpy()[:, 1]
    quant_map_ksw = quant_map_ksw.T
    quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

    quant_maps = {'fs': quant_map_fs, 'ksw': quant_map_ksw}

    return quant_maps


def save_and_plot_results(quant_maps, output_folder, mask):
    """Save quantitative maps and generate plots."""
    os.makedirs(output_folder, exist_ok=True)
    # Saving output maps
    out_fn = 'nn_reco_maps_preclinical.mat'
    out_fn = os.path.join(output_folder, out_fn)
    sio.savemat(out_fn, quant_maps)

    fig_fn = os.path.join(output_folder, 'deep_reco_preclinical.eps')
    plt.figure(figsize=(10, 5))
    # [L-arg] (mM)
    plt.subplot(121)
    plt.imshow(quant_maps['fs'] * 110e3/3 * mask, cmap=b_viridis, clim=(0, 120))
    plt.colorbar(ticks=np.arange(0, 121, 20), fraction=0.046, pad=0.04)
    plt.title('[L-arg] (mM)')
    plt.axis("off")
    # ksw (Hz)
    plt.subplot(122)
    plt.imshow(quant_maps['ksw'] * mask, cmap='magma', clim=(0, 500))
    plt.colorbar(ticks=np.arange(0, 501, 100), fraction=0.046, pad=0.04)
    plt.title('k$_{sw}$ (s$^{-1}$)')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_fn, format="eps")
    plt.close()


def generate_dict(cfg):
    yaml_fn = cfg['yaml_fn']
    seq_fn = cfg['seq_fn']
    dict_fn = cfg['dict_fn']

    dictionary = generate_mrf_cest_dictionary(seq_fn=seq_fn, param_fn=yaml_fn, dict_fn=dict_fn,
                                              num_workers=cfg['num_workers'],
                                              axes='xy')  # axes can also be 'z' if no readout is simulated
    return preprocess_dict(dictionary)


def preprocess_dict(dictionary):
    """Preprocess the dictionary for dot-matching"""
    dictionary['sig'] = np.array(dictionary['sig'])
    for key in dictionary.keys():
        if key != 'sig':
            dictionary[key] = np.expand_dims(np.squeeze(np.array(dictionary[key])), 0)
    print(dictionary['sig'].shape)

    return dictionary


def define_min_max(dictionary):
    # load the data and define range for normalization
    min_fs = np.min(dictionary['fs_0'])
    min_ksw = np.min(dictionary['ksw_0'].transpose().astype(np.float))
    max_fs = np.max(dictionary['fs_0'])
    max_ksw = np.max(dictionary['ksw_0'].transpose().astype(np.float))

    min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False)
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False)

    return min_param_tensor, max_param_tensor


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    set_seed(2024)

    main()
