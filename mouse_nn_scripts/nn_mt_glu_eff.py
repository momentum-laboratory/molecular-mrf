import multiprocessing
import platform
import torch
import numpy as np
from torch.utils.data import DataLoader

import time
import os
import tqdm

from utils.normalization import normalize_range, un_normalize_range
from utils.seed import set_seed

from sequential_nn.dataset import GluMemDataset
from sequential_nn.model import Network
from sequential_nn.multi_dict import pkl_2_dat, define_min_max

from torch.utils.tensorboard import SummaryWriter

# %load_ext tensorboard
# %tensorboard --logdir=./runs --port 6007

def main():
    torch.multiprocessing.freeze_support()
    # i fucked up 'mouse_aacid_k_regular_nn_noise_005'
    dict_name_category = '1300'  # 1300, dense, aacid_k
    dict_name = 'mouse_20'  # mouse_20, dense_1300,
    fp_prtcl_name = '107a'

    # Schedule iterations
    # number of raw images in the CEST-MRF acquisition schedule
    sched_iter = 30
    add_iter = 4

    # Training properties
    learning_rate = 2e-4
    batch_size = 1024
    num_epochs = 150
    noise_std = 1e-2  # noise level for training, 1e-2

    min_delta = 0.05  # minimum absolute change in the loss function
    patience = np.inf

    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Navigate up one directory level
    glu_dict_folder_fn = os.path.join(parent_dir, 'data', 'exp', 'mt_glu_dicts', dict_name_category, 'glu', dict_name,
                                      fp_prtcl_name)  # dict folder directory
    memmap_fn = os.path.join(glu_dict_folder_fn, 'dict.dat')

    if not os.path.exists(memmap_fn):
        pkl_2_dat(glu_dict_folder_fn, sched_iter, add_iter)

    net_name = f'noise_{noise_std}_min_0'
    nn_fn = os.path.join(current_dir, 'mouse_nns', 'glu_mt_nns', dict_name_category, 'glu', dict_name, fp_prtcl_name,
                         f'{net_name}.pt')  # nn directory

    device = initialize_device()
    print(f"Using device: {device}")

    (min_param_tensor, max_param_tensor,
     min_water_t1t2_tensor, max_water_t1t2_tensor,
     min_mt_param_tensor, max_mt_param_tensor) = define_min_max(memmap_fn, sched_iter, add_iter, device)

    # Convert tensors to numpy arrays
    min_param_array = min_param_tensor.cpu().numpy()
    max_param_array = max_param_tensor.cpu().numpy()
    min_water_t1t2_array = min_water_t1t2_tensor.cpu().numpy()
    max_water_t1t2_array = max_water_t1t2_tensor.cpu().numpy()
    min_mt_param_array = min_mt_param_tensor.cpu().numpy()
    max_mt_param_array = max_mt_param_tensor.cpu().numpy()

    if not os.path.exists(os.path.dirname(nn_fn)):
        os.makedirs(os.path.dirname(nn_fn))
    # Save all arrays to a single .npz file
    np.savez(os.path.join(os.path.dirname(nn_fn), 'min_max_values_min_0.npz'),
             min_param=min_param_array,
             max_param=max_param_array,
             min_water_t1t2=min_water_t1t2_array,
             max_water_t1t2=max_water_t1t2_array,
             min_mt_param=min_mt_param_array,
             max_mt_param=max_mt_param_array)

    # Loading the training dataset
    # train_dataset = GluMemDataset(memmap_fn, sched_iter, add_iter, chunk_size=10000000)
    train_dataset = GluMemDataset(memmap_fn, sched_iter, add_iter)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=3)

    train_network(train_loader, device, sched_iter, add_iter, learning_rate, num_epochs, noise_std, patience,
                  min_delta, min_param_tensor, max_param_tensor, min_water_t1t2_tensor,
                  max_water_t1t2_tensor, min_mt_param_tensor, max_mt_param_tensor, nn_fn)


# Function to initialize device
def initialize_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Function to train the network
def train_network(train_loader, device, sched_iter, add_iter, learning_rate, num_epochs, noise_std, patience, min_delta,
                  min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor,
                  min_mt_param_fs_ksw, max_mt_param_fs_ksw, nn_fn):
    nn_folder = os.path.dirname(nn_fn)  # Navigate up one directory level
    if not os.path.exists(nn_folder):
        os.makedirs(nn_folder)

    # Initializing the reconstruction network
    reco_net = Network(sched_iter, add_iter=add_iter, n_hidden=2, n_neurons=300).to(device)

    # Print amount of parameters
    print('Number of model parameters: ', sum(p.numel() for p in reco_net.parameters() if p.requires_grad))

    # Setting optimizer
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

    # Training loop

    # Storing current time
    t0 = time.time()
    writer = SummaryWriter()

    loss_per_epoch = []
    patience_counter = 0
    min_loss = 100

    reco_net.train()

    pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # Cumulative loss
        cum_loss = 0
        counter = np.nan

        for counter, dict_params in enumerate(train_loader, 0):
            reco_net, cum_loss = train_step(device, noise_std, reco_net, optimizer, cum_loss, dict_params,
                                            min_param_tensor, max_param_tensor,
                                            min_water_t1t2_tensor, max_water_t1t2_tensor,
                                            min_mt_param_fs_ksw, max_mt_param_fs_ksw, writer, epoch)

            del dict_params
            torch.cuda.empty_cache()

        # Average loss for this epoch
        loss_per_epoch.append(cum_loss / (counter + 1))
        # writer.add_scalar("Loss/train", loss_per_epoch, epoch)

        pbar.set_description(f'Epoch: {epoch + 1}/{num_epochs}, Loss = {loss_per_epoch[-1]}')
        pbar.update(1)

        # Early stopping logic
        if (min_loss - loss_per_epoch[-1]) / min_loss > min_delta:
            min_loss = loss_per_epoch[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print('Early stopping!')
            break

        # Save model checkpoint every 25 epochs (excluding epoch 0)
        if epoch % 10 == 0 and epoch != 0:
            print(f"\nSaved epoch {epoch} model")
            torch.save({
                'model_state_dict': reco_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_per_epoch': loss_per_epoch,
                'noise_std': noise_std,
                'epoch': epoch
            }, nn_fn)

            torch.cuda.empty_cache()

    pbar.close()
    print(f"Training took {time.time() - t0:.2f} seconds")

    # Save final model checkpoint
    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_per_epoch': loss_per_epoch,
        'noise_std': noise_std,
    }, nn_fn)

    writer.flush()
    writer.close()

    return reco_net


def train_step(device, noise_std, reco_net, optimizer, cum_loss, dict_params, min_param_tensor, max_param_tensor,
               min_water_t1t2_tensor, max_water_t1t2_tensor, min_mt_param_fs_ksw, max_mt_param_fs_ksw, writer, epoch):
    cur_fs, cur_ksw, cur_t1w, cur_t2w, cur_mt_fs, cur_mt_ksw, cur_norm_sig = dict_params

    target = torch.stack((cur_fs, cur_ksw), dim=1).to(device)
    input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(device)
    input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(device)

    # Normalizing the target and input_water_t1t2
    target = normalize_range(original_array=target, original_min=min_param_tensor,
                             original_max=max_param_tensor, new_min=0, new_max=1).to(device)

    input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                       original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(device)

    input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_fs_ksw,
                                      original_max=max_mt_param_fs_ksw, new_min=0, new_max=1).to(device)

    # Adding noise to the input signals (trajectories)
    noised_sig = cur_norm_sig.to(device) + torch.randn(cur_norm_sig.size()).to(device) * noise_std

    # noised_sig = noised_sig / torch.linalg.norm(noised_sig, dim=1, ord=2, keepdim=True)

    # adding the mt_fs_ksw and t1, t2 as additional nn input
    noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2, noised_sig.to(device))).to(device)
    del input_water_t1t2, input_mt_fs_ksw

    # Forward step
    prediction = reco_net(noised_sig.float())
    del noised_sig

    # Batch loss (MSE)
    loss = torch.mean((prediction.float() - target.float()) ** 2)
    del target

    # Backward step
    optimizer.zero_grad()
    loss.backward()

    # Optimization step
    optimizer.step()

    # Storing Cumulative loss
    cum_loss += loss.item()

    torch.cuda.empty_cache()

    return reco_net, cum_loss


if __name__ == '__main__':
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    set_seed(2024)

    main()
