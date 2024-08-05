import os
import re
import pandas as pd
import numpy as np
import glob
import time
import torch

def pkl_2_dat(glu_dict_folder_fn, sched_iter, add_iter, memmap_fn):
    """ Converts Pickle files to .dat files"""
    # Currently assumes constant dict size!
    # Deal with pre-existing .dat files (initialize new .dat if rerun, or avoid any process if .dat already exists)
    start_time = time.time()

    pattern = os.path.join(glu_dict_folder_fn, 'dict.pkl')
    glu_dict_fn = glob.glob(pattern)
    pattern = os.path.join(glu_dict_folder_fn, 'dict_*.pkl')
    glu_dict_fns = glob.glob(pattern)

    if len(glu_dict_fns) == 0:
        # if there are no 'dict_{idx}.csv' matching files, refer to the 'dict.csv' file case
        print('Single dict case')
        glu_dict_fns = glu_dict_fn
    else:
        # if there are 'dict_{idx}.csv' matching files, refer to the multi-dict case
        print(f'Multi dict case ({len(glu_dict_fns)})')

    for glu_dict_fn in glu_dict_fns:
        # Loading the training dataset
        training_data = pd.read_pickle(glu_dict_fn)
        n_dp, _ = training_data.shape

        data_cur = np.zeros([n_dp, 2+add_iter+sched_iter]).astype(np.float64)
        sig = np.vstack(training_data['sig'].values)[:, :]  # [#, 31]

        # Fill the memory-mapped array with ['fs_0', 'ksw_0', 't1w', 't2w', 'fs_1', 'ksw_1'] data
        # Fill the memory-mapped array with 'sig' data (excluding M0)
        if add_iter == 2:
            data_cur[:, :4] = training_data[
                ['fs_0', 'ksw_0', 't1w', 't2w']].values
            data_cur[:, 4:] = sig  # [#, 31]
        if add_iter == 4:
            data_cur[:, :6] = training_data[
                ['fs_0', 'ksw_0', 't1w', 't2w', 'fs_1', 'ksw_1']].values
            data_cur[:, 6:] = sig  # [#, 31]
        elif add_iter == 6:
            data_cur[:, :8] = training_data[
                ['fs_0', 'ksw_0', 't1w', 't2w', 'fs_1', 'ksw_1', 'fs_2', 'ksw_2']].values
            data_cur[:, 8:] = sig  # [#, 31]

        del sig, training_data

        # Check if the memmap file exists
        if not os.path.exists(memmap_fn):
            # Create an initial memmap array if it doesn't exist
            np.memmap(memmap_fn, dtype=np.float64, mode='w+', shape=(n_dp*len(glu_dict_fns), 2+add_iter+sched_iter))

        memmap_array = np.memmap(memmap_fn, dtype=np.float64, mode='r+',
                                 shape=(n_dp * len(glu_dict_fns), 2 + add_iter + sched_iter))
        memmap_array[-data_cur.shape[0]:] = data_cur

    end_time = time.time()

    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f'Total time taken for memmap generation: {total_time//60:.0f} minutes {total_time%60:.0f} seconds')

    memmap_shape = (n_dp * len(glu_dict_fns), 2 + add_iter + sched_iter)
    return memmap_shape

def define_min_max(memmap_fn, sched_iter, add_iter, device):
    # 0- fs0, 1- ksw0, 2- t1w, 3- t2w, 4- fs1, 5- ksw1
    # currently I used non-zero minima

    num_columns = sched_iter + add_iter +2
    memmap_array = np.memmap(memmap_fn, dtype=np.float64, mode='r+')
    num_rows = memmap_array.size // num_columns  # Calculate the number of rows
    memmap_array.shape = (num_rows, num_columns)

    min_fs = np.min(memmap_array[:, 0])
    min_ksw = np.min(memmap_array[:, 1])
    max_fs = np.max(memmap_array[:, 0])
    max_ksw = np.max(memmap_array[:, 1])

    min_t1w = np.min(memmap_array[:, 2])
    min_t2w = np.min(memmap_array[:, 3])
    max_t1w = np.max(memmap_array[:, 2])
    max_t2w = np.max(memmap_array[:, 3])

    min_mt_fs = np.min(memmap_array[:, 4])
    min_mt_ksw = np.min(memmap_array[:, 5])
    max_mt_fs = np.max(memmap_array[:, 4])
    max_mt_ksw = np.max(memmap_array[:, 5])

    del memmap_array

    min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).to(device)
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).to(device)

    min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False).to(device)
    max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False).to(device)

    min_mt_param_tensor = torch.tensor(np.hstack((min_mt_fs, min_mt_ksw)), requires_grad=False).to(device)
    max_mt_param_tensor = torch.tensor(np.hstack((max_mt_fs, max_mt_ksw)), requires_grad=False).to(device)

    return (min_param_tensor, max_param_tensor,
            min_water_t1t2_tensor, max_water_t1t2_tensor,
            min_mt_param_tensor, max_mt_param_tensor)