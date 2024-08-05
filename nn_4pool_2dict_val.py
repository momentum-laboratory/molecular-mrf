import argparse
import multiprocessing
import platform
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import itertools
import pandas as pd

import time
import os
# import tqdm
import tqdm.notebook as tqdm

from utils.normalization import normalize_range, un_normalize_range
from utils.seed import set_seed

from sequential_nn.dataset import GluAmide4pool
from sequential_nn.model import Network
from sequential_nn.multi_dict import pkl_2_dat

from torch.utils.tensorboard import SummaryWriter


def main(args):
    # Freeze support
    torch.multiprocessing.freeze_support()

    # Paths to dict and nn:
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Navigate up one directory level
    glu_dict_folder_fn = os.path.join(
        parent_dir, 'data', 'exp', 'mt_amide_glu_dicts',
        args.dict_name_category, 'glu', args.fp_prtcl_names[0])  # dict folder directory
    glu_memmap_fn = os.path.join(glu_dict_folder_fn, 'M0_dict.dat')

    amide_dict_folder_fn = os.path.join(
        parent_dir, 'data', 'exp', 'mt_amide_glu_dicts',
        args.dict_name_category, 'glu', args.fp_prtcl_names[1])  # dict folder directory
    amide_memmap_fn = os.path.join(amide_dict_folder_fn, 'M0_dict.dat')

    for memmap_fn in [glu_memmap_fn, amide_memmap_fn]:
        if not os.path.exists(memmap_fn):
            pkl_2_dat(glu_dict_folder_fn, args.sched_iter, args.add_iter, memmap_fn)

    net_name = f'{args.dict_name_category}_2dict_noise_{args.noise_std}_lr_{args.learning_rate}_{args.batch_size}'
    nn_fn = os.path.join(current_dir, 'mouse_nns', 'glu_amide_mt_nns',
                         args.dict_name_category, '2dict', f'M0_{net_name}.pt')  # nn directory

    # min max value calc and save:
    min_max_params = define_min_max(memmap_fn, args.sched_iter, args.add_iter, device)

    min_max_saver(min_max_params, nn_fn)

    # Load the shared dataset
    full_dataset = GluAmide4pool(glu_memmap_fn, amide_memmap_fn, args.sched_iter, args.add_iter)
    dataset_size = len(full_dataset)
    print(dataset_size)

    # Split indices for training, validation, and test sets
    train_indices, val_indices, test_indices = split_dataset_indices(dataset_size, val_ratio=0.2, test_ratio=0.1)

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4)

    train_network(args, train_loader, val_loader, test_loader, net_name, nn_fn, min_max_params)


# Function to split dataset indices
def split_dataset_indices(dataset_size, val_ratio=0.2, test_ratio=0.1):
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    test_split = int(test_ratio * dataset_size)
    val_split = int(val_ratio * dataset_size) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]
    return train_indices, val_indices, test_indices

# Function to initialize device
def initialize_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Function to train the network
def train_network(args, train_loader, val_loader, test_loader, net_name, nn_fn, min_max_params):
    nn_folder = os.path.dirname(nn_fn)  # Navigate up one directory level
    if not os.path.exists(nn_folder):
        os.makedirs(nn_folder)

    # Initializing the reconstruction network
    reco_net = Network(args.sched_iter*2, add_iter=4, n_hidden=2, n_neurons=300, output_dim=4).to(args.device)

    # Print amount of parameters
    print('Number of model parameters: ', sum(p.numel() for p in reco_net.parameters() if p.requires_grad))

    # Setting optimizer
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Storing current time
    t0 = time.time()
    # Get today's date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    writer = SummaryWriter(log_dir=f'runs/{net_name}')

    loss_per_epoch = []
    val_loss_per_epoch = []
    patience_counter = 0
    min_loss = 100

    reco_net.train()
    cur_val_loss = float('inf')

    pbar = tqdm.tqdm(total=args.num_epochs)
    for epoch in range(args.num_epochs):
        # Cumulative loss
        cum_loss = 0
        counter = np.nan

        num_steps = len(train_loader)
        inner_pbar = tqdm.tqdm(total=num_steps)
        for counter, dict_params in enumerate(train_loader, 0):
            reco_net, cum_loss = train_step(args, reco_net, optimizer, cum_loss, dict_params,
                                            writer, epoch, counter, min_max_params, num_steps)
            inner_pbar.set_description(f'Step: {counter+1}/{num_steps}')
            inner_pbar.update(1)

            del dict_params
            torch.cuda.empty_cache()
        inner_pbar.close()

        # Average loss for this epoch
        loss_per_epoch.append(cum_loss / (counter + 1))

        # Validate the model
        val_loss = validate(args, reco_net, val_loader, min_max_params)
        val_loss_per_epoch.append(val_loss)

        writer.add_scalar("Loss/train", loss_per_epoch[-1], epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        pbar.set_description(f'Epoch: {epoch + 1}/{args.num_epochs}, '
                             f'Train Loss = {loss_per_epoch[-1]}, '
                             f'Val Loss = {val_loss_per_epoch[-1]}')
        pbar.update(1)

        # Early stopping logic
        if (min_loss - val_loss_per_epoch[-1]) / min_loss > args.min_delta:
            min_loss = val_loss_per_epoch[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > args.patience:
            print('Early stopping!')
            break

        # # Scheduler step
        # scheduler.step()

        # Save model checkpoint when val loss gets better
        if val_loss <= cur_val_loss:
            print(f"\nSaved epoch {epoch} model")
            torch.save({
                'model_state_dict': reco_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_per_epoch': loss_per_epoch,
                'val_loss_per_epoch': val_loss_per_epoch,
                'noise_std': args.noise_std,
                'epoch': epoch
            }, nn_fn)

            torch.cuda.empty_cache()
        cur_val_loss = val_loss

    pbar.close()
    print(f"Training took {time.time() - t0:.2f} seconds")

    writer.flush()
    writer.close()

    # Test the model
    test_loss = test(args, reco_net, test_loader, min_max_params)
    print(f"Test Loss: {test_loss}")

    return reco_net


def train_step(args, reco_net, optimizer, cum_loss, dict_params, writer, epoch, counter, min_max_params, num_steps):
    # min max params
    (min_param_tensor, max_param_tensor,
     min_water_t1t2_tensor, max_water_t1t2_tensor,
     min_mt_param_tensor, max_mt_param_tensor,
     min_amide_param_tensor, max_amide_param_tensor) = min_max_params
    # dict params
    (cur_fs, cur_ksw,
     cur_t1w, cur_t2w,
     cur_mt_fs, cur_mt_ksw,
     cur_amide_fs, cur_amide_ksw,
     cur_glu_norm_sig, cur_amide_norm_sig) = dict_params

    target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
    input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
    input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)
    input_amide_fs_ksw = torch.stack((cur_amide_fs, cur_amide_ksw), dim=1).to(args.device)

    # Normalizing the target and input_water_t1t2
    target_glu_fs_ksw = normalize_range(original_array=target, original_min=min_param_tensor,
                             original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

    target_amide_fs_ksw = normalize_range(original_array=input_amide_fs_ksw, original_min=min_amide_param_tensor,
                                  original_max=max_amide_param_tensor, new_min=0, new_max=1).to(args.device)

    input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                       original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

    input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                      original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

    # Adding noise to the input signals (trajectories)
    glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std
    amide_noised_sig = cur_amide_norm_sig + torch.randn(cur_amide_norm_sig.size()) * args.noise_std

    # adding the mt_fs_ksw and t1, t2 as additional nn input
    target = torch.hstack((target_glu_fs_ksw, target_amide_fs_ksw))
    del target_glu_fs_ksw, target_amide_fs_ksw
    noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                               glu_noised_sig.to(args.device), amide_noised_sig.to(args.device))).to(args.device)
    del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig, amide_noised_sig

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

    writer.add_scalar("Loss/train_step", loss.item(), counter+epoch*num_steps)

    torch.cuda.empty_cache()

    return reco_net, cum_loss


def validate(args, reco_net, val_loader, min_max_params):
    reco_net.eval()
    val_loss = 0
    with torch.no_grad():
        for dict_params in val_loader:
            # min max params
            (min_param_tensor, max_param_tensor,
             min_water_t1t2_tensor, max_water_t1t2_tensor,
             min_mt_param_tensor, max_mt_param_tensor,
             min_amide_param_tensor, max_amide_param_tensor) = min_max_params
            # dict params
            (cur_fs, cur_ksw,
             cur_t1w, cur_t2w,
             cur_mt_fs, cur_mt_ksw,
             cur_amide_fs, cur_amide_ksw,
             cur_glu_norm_sig, cur_amide_norm_sig) = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
            input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
            input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)
            input_amide_fs_ksw = torch.stack((cur_amide_fs, cur_amide_ksw), dim=1).to(args.device)

            # Normalizing the target and input_water_t1t2
            target_glu_fs_ksw = normalize_range(original_array=target, original_min=min_param_tensor,
                                                original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

            target_amide_fs_ksw = normalize_range(original_array=input_amide_fs_ksw,
                                                  original_min=min_amide_param_tensor,
                                                  original_max=max_amide_param_tensor, new_min=0, new_max=1).to(
                args.device)

            input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                               original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

            input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                              original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

            # Adding noise to the input signals (trajectories)
            glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std
            amide_noised_sig = cur_amide_norm_sig + torch.randn(cur_amide_norm_sig.size()) * args.noise_std

            # adding the mt_fs_ksw and t1, t2 as additional nn input
            target = torch.hstack((target_glu_fs_ksw, target_amide_fs_ksw))
            del target_glu_fs_ksw, target_amide_fs_ksw
            noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                                       glu_noised_sig.to(args.device),
                                       amide_noised_sig.to(args.device))).to(args.device)
            del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig, amide_noised_sig

            # Forward step
            prediction = reco_net(noised_sig.float())
            del noised_sig

            loss = torch.mean((prediction.float() - target.float()) ** 2)

            val_loss += loss.item()

    return val_loss / len(val_loader)


def test(args, reco_net, test_loader, min_max_params):
    reco_net.eval()
    test_loss = 0
    with torch.no_grad():
        for dict_params in test_loader:
            # min max params
            (min_param_tensor, max_param_tensor,
             min_water_t1t2_tensor, max_water_t1t2_tensor,
             min_mt_param_tensor, max_mt_param_tensor,
             min_amide_param_tensor, max_amide_param_tensor) = min_max_params
            # dict params
            (cur_fs, cur_ksw,
             cur_t1w, cur_t2w,
             cur_mt_fs, cur_mt_ksw,
             cur_amide_fs, cur_amide_ksw,
             cur_glu_norm_sig, cur_amide_norm_sig) = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
            input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
            input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)
            input_amide_fs_ksw = torch.stack((cur_amide_fs, cur_amide_ksw), dim=1).to(args.device)

            # Normalizing the target and input_water_t1t2
            target_glu_fs_ksw = normalize_range(original_array=target, original_min=min_param_tensor,
                                                original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

            target_amide_fs_ksw = normalize_range(original_array=input_amide_fs_ksw,
                                                  original_min=min_amide_param_tensor,
                                                  original_max=max_amide_param_tensor, new_min=0, new_max=1).to(
                args.device)

            input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                               original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

            input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                              original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

            # Adding noise to the input signals (trajectories)
            glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std
            amide_noised_sig = cur_amide_norm_sig + torch.randn(cur_amide_norm_sig.size()) * args.noise_std

            # adding the mt_fs_ksw and t1, t2 as additional nn input
            target = torch.hstack((target_glu_fs_ksw, target_amide_fs_ksw))
            del target_glu_fs_ksw, target_amide_fs_ksw
            noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                                       glu_noised_sig.to(args.device), amide_noised_sig.to(args.device))).to(
                args.device)
            del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig, amide_noised_sig

            # Forward step
            prediction = reco_net(noised_sig.float())

            loss = torch.mean((prediction.float() - target.float()) ** 2)

            test_loss += loss.item()

    return test_loss / len(test_loader)

def define_min_max(memmap_fn, sched_iter, add_iter, device):
    num_columns = sched_iter + add_iter + 2
    memmap_array = np.memmap(memmap_fn, dtype=np.float64, mode='r')
    num_rows = memmap_array.size // num_columns  # Calculate the number of rows
    memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]

    min_fs = np.min(memmap_array[:, 4])  # uncomment if non-zero minimum limit is required
    min_ksw = np.min(memmap_array[:, 5].transpose().astype(float))  # uncomment if non-zero minimum limit needed
    max_fs = np.max(memmap_array[:, 4])
    max_ksw = np.max(memmap_array[:, 5].transpose().astype(float))

    min_t1w = np.min(memmap_array[:, 2])
    min_t2w = np.min(memmap_array[:, 3].transpose().astype(float))
    max_t1w = np.max(memmap_array[:, 2])
    max_t2w = np.max(memmap_array[:, 3].transpose().astype(float))

    min_mt_fs = np.min(memmap_array[:, 6])
    min_mt_ksw = np.min(memmap_array[:, 7].transpose().astype(float))
    max_mt_fs = np.max(memmap_array[:, 6])
    max_mt_ksw = np.max(memmap_array[:, 7].transpose().astype(float))

    min_amine_fs = np.min(memmap_array[:, 0])
    min_amine_ksw = np.min(memmap_array[:, 1].transpose().astype(float))
    max_amine_fs = np.max(memmap_array[:, 0])
    max_amine_ksw = np.max(memmap_array[:, 1].transpose().astype(float))

    min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).to(device)  # can be switched to  min_fs, min_ksw
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).to(device)

    min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False).to(device)
    max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False).to(device)

    min_mt_param_tensor = torch.tensor(np.hstack((min_mt_fs, min_mt_ksw)), requires_grad=False).to(device)  # can be switched to  min_fs, min_ksw
    max_mt_param_tensor = torch.tensor(np.hstack((max_mt_fs, max_mt_ksw)), requires_grad=False).to(device)  # can be switched to  min_fs, min_ksw

    min_amine_param_tensor = torch.tensor(np.hstack((min_amine_fs, min_amine_ksw)), requires_grad=False).to(device)  # can be switched to  min_fs, min_ksw
    max_amine_param_tensor = torch.tensor(np.hstack((max_amine_fs, max_amine_ksw)),
                                       requires_grad=False).to(device)  # can be switched to  min_fs, min_ksw

    return (min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor,
            min_mt_param_tensor, max_mt_param_tensor, min_amine_param_tensor, max_amine_param_tensor)

def min_max_saver(min_max_params, nn_fn):
    # Convert tensors to numpy arrays
    min_param_array = min_max_params[0].cpu().numpy()
    max_param_array = min_max_params[1].cpu().numpy()
    min_water_t1t2_array = min_max_params[2].cpu().numpy()
    max_water_t1t2_array = min_max_params[3].cpu().numpy()
    min_mt_param_array = min_max_params[4].cpu().numpy()
    max_mt_param_array = min_max_params[5].cpu().numpy()
    min_amide_param_array = min_max_params[6].cpu().numpy()
    max_amide_param_array = min_max_params[7].cpu().numpy()

    if not os.path.exists(os.path.dirname(nn_fn)):
        os.makedirs(os.path.dirname(nn_fn))

    # Save all arrays to a single .npz file
    np.savez(os.path.join(os.path.dirname(nn_fn), 'min_max_values.npz'),
             min_param=min_param_array,
             max_param=max_param_array,
             min_water_t1t2=min_water_t1t2_array,
             max_water_t1t2=max_water_t1t2_array,
             min_mt_param=min_mt_param_array,
             max_mt_param=max_mt_param_array,
             min_amide_param=min_amide_param_array,
             max_amide_param=max_amide_param_array)


if __name__ == '__main__':
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    set_seed(2024)

    # Training properties
    parser = argparse.ArgumentParser()
    # Initialize device:
    device = initialize_device()
    parser.add_argument('--device', default=device)
    parser.add_argument('--dict-name-category', type=str, default='glu_conc_high_500')  # glu_amide_lim
    parser.add_argument('--fp-prtcl-names', default=['107a', '51_Amide'])
    parser.add_argument('--sched-iter', type=int, default=30)
    parser.add_argument('--add-iter', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--noise-std', type=float, default=2e-3)
    parser.add_argument('--min-delta', type=float, default=0.05)  # minimum absolute change in the loss function
    parser.add_argument('--patience', default=np.inf)

    args = parser.parse_args()
    main(args)
