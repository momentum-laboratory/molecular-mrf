import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


# Organizing the training data
class SequentialDataset(Dataset):
    def __init__(self, dict_fn):
        training_data = sio.loadmat(dict_fn)
        self.fs_list = training_data['fs_0'].transpose()[:, 0]
        self.ksw_list = training_data['ksw_0'].transpose()[:, 0]
        self.t1w_list = training_data['t1w'].transpose()[:, 0]
        self.t2w_list = training_data['t2w'].transpose()[:, 0]
        sig = training_data['sig'].transpose()

        # 2-norm normalization of the dictionary signals
        self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
        # self.norm_sig_list = sig
        
        # Training dictionary size
        self.len = training_data['ksw_0'].transpose().size  # py-cest-mrf version
        print("There are " + str(self.len) + " entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        t1w = self.t1w_list[index]
        t2w = self.t2w_list[index]
        norm_sig = self.norm_sig_list[:, index]
        return fs, ksw, t1w, t2w, norm_sig
