import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class DatasetMRF(Dataset):
    def __init__(self, data):
        self.fs_list = data['fs_0'].transpose()[:, 0]
        self.ksw_list = data['ksw_0'].transpose()[:, 0]
        sig = data['sig'].transpose()

        # l2-norm normalization of the dictionary signals
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))

        # Training dictionary size
        self.len = data['ksw_0'].transpose().size 
        print("There are " + str(self.len) + " entries in the dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        return fs, ksw, norm_sig

