import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import glob
import random

class NoShuffleMultiDataset(Dataset):
    def __init__(self, glu_dict_folder_fn, add_iter, M0_flag=False):
        """ Initialize with paths to Pickle files """
        self.glu_dict_folder_fn = glu_dict_folder_fn
        self.add_iter = add_iter
        self.M0_flag = M0_flag

        pattern = os.path.join(glu_dict_folder_fn, 'dict.pkl')
        glu_dict_fn = glob.glob(pattern)
        pattern = os.path.join(glu_dict_folder_fn, 'dict_*.pkl')
        self.glu_dict_fns = glob.glob(pattern)

        if len(self.glu_dict_fns) == 0:
            print('Single dict case')
            self.glu_dict_fns = glu_dict_fn
        else:
            print(f'Multi dict case ({len(self.glu_dict_fns)})')

        self.dataset_order = list(range(len(self.glu_dict_fns)))
        random.shuffle(self.dataset_order)
        self.current_dataset_idx = 0
        self.load_next_dataset()

    def load_next_dataset(self):
        if self.current_dataset_idx < len(self.dataset_order):
            dict_i = self.dataset_order[self.current_dataset_idx]
            glu_dict_fn = self.glu_dict_fns[dict_i]
            self.current_data = pd.read_pickle(glu_dict_fn)
            n_dp, _ = self.current_data.shape
            if self.add_iter==2:
                self.fs_list = self.current_data['fs_0'].values
                self.ksw_list = self.current_data['ksw_0'].values
                self.t1w_list = self.current_data['t1w'].values
                self.t2w_list = self.current_data['t2w'].values
            elif self.add_iter==4:
                self.fs_list = self.current_data['fs_0'].values
                self.ksw_list = self.current_data['ksw_0'].values
                self.t1w_list = self.current_data['t1w'].values
                self.t2w_list = self.current_data['t2w'].values
                self.mt_fs_list = self.current_data['fs_1'].values
                self.mt_ksw_list = self.current_data['ksw_1'].values
            elif self.add_iter==6:
                self.fs_list = self.current_data['fs_1'].values
                self.ksw_list = self.current_data['ksw_1'].values
                self.t1w_list = self.current_data['t1w'].values
                self.t2w_list = self.current_data['t2w'].values
                self.mt_fs_list = self.current_data['fs_2'].values
                self.mt_ksw_list = self.current_data['ksw_2'].values
                self.amide_fs_list = self.current_data['fs_0'].values
                self.amide_ksw_list = self.current_data['ksw_0'].values
            else:
                print('add_iter Error')

            sig = np.vstack(self.current_data['sig'].values).T[1:, :]
            if not self.M0_flag:
                self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
            else:
                M0 = np.vstack(self.current_data['sig'].values).T[0, :]  # M0
                self.norm_sig_list = sig / M0

            self.len = len(self.current_data)
            self.current_dataset_idx += 1
        else:
            print("All datasets have been loaded for this epoch")

    def __len__(self):
        return self.len

    def get_current_data(self):
        """ Return the current dataset """
        return self.current_data

    def __getitem__(self, idx):
        if idx >= self.len:
            self.load_next_dataset()
            idx = 0  # Reset index after loading new dataset

        if self.add_iter == 2:
            fs = self.fs_list[idx]
            ksw = self.ksw_list[idx]
            t1w = self.t1w_list[idx]
            t2w = self.t2w_list[idx]
            mt_fs = 'Nan'
            mt_ksw = 'Nan'
            amide_fs = 'Nan'
            amide_ksw = 'Nan'
            norm_sig = self.norm_sig_list[:, idx]
        elif self.add_iter == 4:
            fs = self.fs_list[idx]
            ksw = self.ksw_list[idx]
            t1w = self.t1w_list[idx]
            t2w = self.t2w_list[idx]
            mt_fs = self.mt_fs_list[idx]
            mt_ksw = self.mt_ksw_list[idx]
            amide_fs = 'Nan'
            amide_ksw = 'Nan'
            norm_sig = self.norm_sig_list[:, idx]
        elif self.add_iter == 6:
            fs = self.fs_list[idx]
            ksw = self.ksw_list[idx]
            t1w = self.t1w_list[idx]
            t2w = self.t2w_list[idx]
            mt_fs = self.mt_fs_list[idx]
            mt_ksw = self.mt_ksw_list[idx]
            amide_fs = self.amide_fs_list[idx]
            amide_ksw = self.amide_ksw_list[idx]
            norm_sig = self.norm_sig_list[:, idx]
        else:
            print('add_iter Error')

        return (
            torch.tensor(fs),
            torch.tensor(ksw),
            torch.tensor(t1w),
            torch.tensor(t2w),
            torch.tensor(mt_fs),
            torch.tensor(mt_ksw),
            torch.tensor(amide_fs),
            torch.tensor(amide_ksw),
            torch.tensor(norm_sig)
        )

def define_min_max(dataset, device):
    min_fs, min_ksw = float('inf'), float('inf')
    max_fs, max_ksw = float('-inf'), float('-inf')

    min_t1w, min_t2w = float('inf'), float('inf')
    max_t1w, max_t2w = float('-inf'), float('-inf')

    min_mt_fs, min_mt_ksw = float('inf'), float('inf')
    max_mt_fs, max_mt_ksw = float('-inf'), float('-inf')

    min_amine_fs, min_amine_ksw = float('inf'), float('inf')
    max_amine_fs, max_amine_ksw = float('-inf'), float('-inf')

    # Iterate over each dataset to compute min and max
    for _ in range(len(dataset.glu_dict_fns)):
        current_data = dataset.get_current_data()

        min_fs = min(min_fs, np.min(current_data['fs_1']))
        min_ksw = min(min_ksw, np.min(current_data['ksw_1'].transpose().astype(float)))
        max_fs = max(max_fs, np.max(current_data['fs_1']))
        max_ksw = max(max_ksw, np.max(current_data['ksw_1'].transpose().astype(float)))

        min_t1w = min(min_t1w, np.min(current_data['t1w']))
        min_t2w = min(min_t2w, np.min(current_data['t2w'].transpose().astype(float)))
        max_t1w = max(max_t1w, np.max(current_data['t1w']))
        max_t2w = max(max_t2w, np.max(current_data['t2w'].transpose().astype(float)))

        min_mt_fs = min(min_mt_fs, np.min(current_data['fs_2']))
        min_mt_ksw = min(min_mt_ksw, np.min(current_data['ksw_2'].transpose().astype(float)))
        max_mt_fs = max(max_mt_fs, np.max(current_data['fs_2']))
        max_mt_ksw = max(max_mt_ksw, np.max(current_data['ksw_2'].transpose().astype(float)))

        min_amine_fs = min(min_amine_fs, np.min(current_data['fs_0']))
        min_amine_ksw = min(min_amine_ksw, np.min(current_data['ksw_0'].transpose().astype(float)))
        max_amine_fs = max(max_amine_fs, np.max(current_data['fs_0']))
        max_amine_ksw = max(max_amine_ksw, np.max(current_data['ksw_0'].transpose().astype(float)))

        dataset.load_next_dataset()

    min_param_tensor = torch.tensor([min_fs, min_ksw], requires_grad=False).to(device)
    max_param_tensor = torch.tensor([max_fs, max_ksw], requires_grad=False).to(device)

    min_water_t1t2_tensor = torch.tensor([min_t1w, min_t2w], requires_grad=False).to(device)
    max_water_t1t2_tensor = torch.tensor([max_t1w, max_t2w], requires_grad=False).to(device)

    min_mt_param_tensor = torch.tensor([min_mt_fs, min_mt_ksw], requires_grad=False).to(device)
    max_mt_param_tensor = torch.tensor([max_mt_fs, max_mt_ksw], requires_grad=False).to(device)

    min_amine_param_tensor = torch.tensor([min_amine_fs, min_amine_ksw], requires_grad=False).to(device)
    max_amine_param_tensor = torch.tensor([max_amine_fs, max_amine_ksw], requires_grad=False).to(device)

    return (min_param_tensor, max_param_tensor, min_water_t1t2_tensor, max_water_t1t2_tensor,
            min_mt_param_tensor, max_mt_param_tensor, min_amine_param_tensor, max_amine_param_tensor)

