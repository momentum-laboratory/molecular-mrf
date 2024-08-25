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
import tqdm
# import tqdm.notebook as tqdm

from utils.normalization import normalize_range, min_max_yaml
from utils.seed import set_seed

from sequential_nn.dataset import GluAmide3pool
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
        parent_dir, 'data', 'exp', '3pool',
        args.dict_name_category, args.fp_prtcl_names[0])  # dict folder directory
    glu_memmap_fn = os.path.join(glu_dict_folder_fn, 'dict.dat')

    net_name = (f'{args.dict_name_category}_glu_train_{args.norm_type}_{args.sched_iter}_noise_{args.noise_std}_lr_{args.learning_rate}'
                f'_{args.step_size}_{args.gamma}_{args.batch_size}_2')  # _noise_{args.noise_std}
    nn_fn = os.path.join(current_dir, 'mouse_nns', '3pool',
                         args.dict_name_category, 'glu', f'{net_name}.pt')  # nn directory

    # min max value calc and save:
    yaml_file_path = os.path.join(os.path.dirname(glu_memmap_fn), 'scenario.yaml')
    min_max_params = min_max_yaml(yaml_file_path, scenario_type=args.scenario_type, device=args.device)

    min_max_saver(min_max_params, nn_fn)

    # Load the shared dataset
    full_dataset = GluAmide3pool(glu_memmap_fn, args.sched_iter, args.add_iter, args.norm_type)
    dataset_size = len(full_dataset)
    print(dataset_size)

    # Split indices for training, validation, and test sets
    train_indices, val_indices, test_indices = split_dataset_indices(dataset_size, val_ratio=0, test_ratio=0)

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
    reco_net = Network(args.sched_iter, add_iter=4, n_hidden=2, n_neurons=300, output_dim=2).to(args.device)

    # Print amount of parameters
    print('Number of model parameters: ', sum(p.numel() for p in reco_net.parameters() if p.requires_grad))

    # Setting optimizer
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Storing current time
    t0 = time.time()
    # Get today's date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    writer = SummaryWriter(log_dir=f'runs/{net_name}')

    loss_per_epoch = []
    loss_per_step = []
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
            if counter % 200 == 0:
                loss_per_step.append(cum_loss / (counter + 1))
            inner_pbar.set_description(f'Step: {counter+1}/{num_steps}')
            inner_pbar.update(1)

            del dict_params
            torch.cuda.empty_cache()
        inner_pbar.close()

        # Average loss for this epoch
        loss_per_epoch.append(cum_loss / (counter + 1))

        # # Validate the model
        # val_loss = validate(args, reco_net, val_loader, min_max_params)
        # val_loss_per_epoch.append(val_loss)
        #
        # writer.add_scalar("Loss/train", loss_per_epoch[-1], epoch)
        # writer.add_scalar("Loss/val", val_loss, epoch)

        pbar.set_description(f'Epoch: {epoch + 1}/{args.num_epochs}, '
                             f'Train Loss = {loss_per_epoch[-1]}, '
                             # f'Val Loss = {val_loss_per_epoch[-1]}'
                             )
        pbar.update(1)

        # # Early stopping logic
        # if (min_loss - val_loss_per_epoch[-1]) / min_loss > args.min_delta:
        #     min_loss = val_loss_per_epoch[-1]
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #
        # if patience_counter > args.patience:
        #     print('Early stopping!')
        #     break

        # Scheduler step
        scheduler.step()

        # Save model checkpoint when val loss gets better
        # if val_loss <= cur_val_loss:
        if epoch % 5 == 0:
            print(f"\nSaved epoch {epoch} model")
            torch.save({
                'model_state_dict': reco_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_per_epoch': loss_per_epoch,
                'loss_per_step': loss_per_step,
                'val_loss_per_epoch': val_loss_per_epoch,
                'noise_std': args.noise_std,
                'epoch': epoch
            }, nn_fn)

            torch.cuda.empty_cache()
        # cur_val_loss = val_loss

    pbar.close()
    print(f"Training took {time.time() - t0:.2f} seconds")

    writer.flush()
    writer.close()

    # # Test the model
    # test_loss = test(args, reco_net, test_loader, min_max_params)
    # print(f"Test Loss: {test_loss}")

    return reco_net


def train_step(args, reco_net, optimizer, cum_loss, dict_params, writer, epoch, counter, min_max_params, num_steps):
    # min max params
    (min_param_tensor, max_param_tensor,
     min_water_t1t2_tensor, max_water_t1t2_tensor,
     min_mt_param_tensor, max_mt_param_tensor) = min_max_params
    # dict params
    (cur_fs, cur_ksw,
     cur_t1w, cur_t2w,
     cur_mt_fs, cur_mt_ksw,
     cur_glu_norm_sig) = dict_params

    target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
    input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
    input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)

    # Normalizing the target and input_water_t1t2
    target = normalize_range(original_array=target, original_min=min_param_tensor,
                             original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

    input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                       original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

    input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                      original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

    # Adding noise to the input signals (trajectories)
    glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std

    # adding the mt_fs_ksw and t1, t2 as additional nn input
    noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                               glu_noised_sig.to(args.device))).to(args.device)
    del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig

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
             min_mt_param_tensor, max_mt_param_tensor) = min_max_params
            # dict params
            (cur_fs, cur_ksw,
             cur_t1w, cur_t2w,
             cur_mt_fs, cur_mt_ksw,
             cur_glu_norm_sig) = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
            input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
            input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)

            # Normalizing the target and input_water_t1t2
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                                original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

            input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                               original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

            input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                              original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

            # Adding noise to the input signals (trajectories)
            glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std

            # adding the mt_fs_ksw and t1, t2 as additional nn input
            noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                                       glu_noised_sig.to(args.device))).to(args.device)
            del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig

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
             min_mt_param_tensor, max_mt_param_tensor) = min_max_params
            # dict params
            (cur_fs, cur_ksw,
             cur_t1w, cur_t2w,
             cur_mt_fs, cur_mt_ksw,
             cur_glu_norm_sig) = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1).to(args.device)
            input_water_t1t2 = torch.stack((cur_t1w, cur_t2w), dim=1).to(args.device)
            input_mt_fs_ksw = torch.stack((cur_mt_fs, cur_mt_ksw), dim=1).to(args.device)

            # Normalizing the target and input_water_t1t2
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                                original_max=max_param_tensor, new_min=0, new_max=1).to(args.device)

            input_water_t1t2 = normalize_range(original_array=input_water_t1t2, original_min=min_water_t1t2_tensor,
                                               original_max=max_water_t1t2_tensor, new_min=0, new_max=1).to(args.device)

            input_mt_fs_ksw = normalize_range(original_array=input_mt_fs_ksw, original_min=min_mt_param_tensor,
                                              original_max=max_mt_param_tensor, new_min=0, new_max=1).to(args.device)

            # Adding noise to the input signals (trajectories)
            glu_noised_sig = cur_glu_norm_sig + torch.randn(cur_glu_norm_sig.size()) * args.noise_std  # args.noise_std

            # adding the mt_fs_ksw and t1, t2 as additional nn input
            noised_sig = torch.hstack((input_mt_fs_ksw, input_water_t1t2,
                                       glu_noised_sig.to(args.device))).to(args.device)
            del input_mt_fs_ksw, input_water_t1t2, glu_noised_sig

            # Forward step
            prediction = reco_net(noised_sig.float())

            loss = torch.mean((prediction.float() - target.float()) ** 2)

            test_loss += loss.item()

    return test_loss / len(test_loader)


def min_max_saver(min_max_params, nn_fn):
    # Convert tensors to numpy arrays
    min_param_array = min_max_params[0].cpu().numpy()
    max_param_array = min_max_params[1].cpu().numpy()
    min_water_t1t2_array = min_max_params[2].cpu().numpy()
    max_water_t1t2_array = min_max_params[3].cpu().numpy()
    min_mt_param_array = min_max_params[4].cpu().numpy()
    max_mt_param_array = min_max_params[5].cpu().numpy()

    if not os.path.exists(os.path.dirname(nn_fn)):
        os.makedirs(os.path.dirname(nn_fn))

    # Save all arrays to a single .npz file
    np.savez(os.path.join(os.path.dirname(nn_fn), 'min_max_values.npz'),
             min_param=min_param_array,
             max_param=max_param_array,
             min_water_t1t2=min_water_t1t2_array,
             max_water_t1t2=max_water_t1t2_array,
             min_mt_param=min_mt_param_array,
             max_mt_param=max_mt_param_array)


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
    parser.add_argument('--scenario-type', type=str, default='3pool')
    parser.add_argument('--norm-type', type=str, default='2norm')  # glu_amide_lim
    parser.add_argument('--dict-name-category', type=str, default='0_40_7000_9500_seq_30')  # glu_amide_lim
    parser.add_argument('--fp-prtcl-names', default=['107a'])
    parser.add_argument('--sched-iter', type=int, default=30)
    parser.add_argument('--add-iter', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--step-size', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--noise-std', type=float, default=1e-2)
    parser.add_argument('--min-delta', type=float, default=0.05)  # minimum absolute change in the loss function
    parser.add_argument('--patience', default=np.inf)

    args = parser.parse_args()
    main(args)
