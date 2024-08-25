import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import pandas as pd
import os


# Organizing the training data
class MTDataset(Dataset):
    def __init__(self, mt_dict_fn, norm_type, sched_iter):
        training_data = pd.read_pickle(mt_dict_fn)
        self.fs_list = training_data['fs_0'].values
        self.ksw_list = training_data['ksw_0'].values
        self.t1w_list = training_data['t1w'].values
        self.t2w_list = training_data['t2w'].values
        sig = np.vstack(training_data['sig'].values).T  # [31, dict len]

        # Normalization of the dictionary signals
        if sig.shape[0] == 30:
            self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
        else:
            if norm_type == '2norm':
                self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
                if sched_iter == 30:
                    sig = sig[1:, :]
                    self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
            elif norm_type == 'M0':
                self.norm_sig_list = sig / sig[0:1, :]
                if sched_iter == 30:
                    self.norm_sig_list = self.norm_sig_list[1:, :]
            # elif norm_type=='mean_std':
            #     self.norm_sig_list = sig / sig[0:1, :]
            else:
                print(f'There is no normalization "{norm_type}" case')

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


# class GluMemDataset(Dataset):
#     # CPU dataset GPU output
#     def __init__(self, memmap_fn, sched_iter, add_iter):
#         num_columns = sched_iter + add_iter + 2
#         memmap_array = np.memmap(memmap_fn, dtype=np.float64, mode='r')
#         num_rows = memmap_array.size // num_columns  # Calculate the number of rows
#         memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]
#
#         self.memmap_array = memmap_array
#         del memmap_array
#         self.len = num_rows
#
#         print("There are " + str(self.len) + " entries in the training dictionary")
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         fs = self.memmap_array[index, 0]
#         ksw = self.memmap_array[index, 1]
#         t1w = self.memmap_array[index, 2]
#         t2w = self.memmap_array[index, 3]
#         mt_fs = self.memmap_array[index, 4]
#         mt_ksw = self.memmap_array[index, 5]
#
#         norm_sig = (self.memmap_array[index, 6:]).T
#
#         return (
#             torch.tensor(fs),
#             torch.tensor(ksw),
#             torch.tensor(t1w),
#             torch.tensor(t2w),
#             torch.tensor(mt_fs),
#             torch.tensor(mt_ksw),
#             torch.tensor(norm_sig)
#         )


# class Dataset_4pool(Dataset):
#     def __init__(self, glu_dict_fn, norm_type, sched_iter):
#         training_data = pd.read_pickle(glu_dict_fn)
#         # fix that strange order (alphabetical!)?
#         self.fs_list = training_data['fs_1'].values
#         self.ksw_list = training_data['ksw_1'].values
#         self.t1w_list = training_data['t1w'].values
#         self.t2w_list = training_data['t2w'].values
#         self.mt_fs_list = training_data['fs_2'].values
#         self.mt_ksw_list = training_data['ksw_2'].values
#         self.amine_fs_list = training_data['fs_0'].values
#         self.amine_ksw_list = training_data['ksw_0'].values
#         sig = np.vstack(training_data['sig'].values).T[:, :]
#
#         # 2-norm normalization of the dictionary signals
#         self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
#         # self.norm_sig_list = sig
#
#         # Training dictionary size
#         self.len = training_data['ksw_0'].transpose().size  # py-cest-mrf version
#         print("There are " + str(self.len) + " entries in the training dictionary")
#         del training_data
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         fs = self.fs_list[index]
#         ksw = self.ksw_list[index]
#         t1w = self.t1w_list[index]
#         t2w = self.t2w_list[index]
#         mt_fs = self.mt_fs_list[index]
#         mt_ksw = self.mt_ksw_list[index]
#         amine_fs = self.amine_fs_list[index]
#         amine_ksw = self.amine_ksw_list[index]
#         norm_sig = self.norm_sig_list[:, index]
#
#         return (
#             torch.tensor(fs),
#             torch.tensor(ksw),
#             torch.tensor(t1w),
#             torch.tensor(t2w),
#             torch.tensor(mt_fs),
#             torch.tensor(mt_ksw),
#             torch.tensor(amine_fs),
#             torch.tensor(amine_ksw),
#             torch.tensor(norm_sig)
#         )

# class GluMemDataset_4pool(Dataset):
#     # CPU dataset GPU output
#     def __init__(self, memmap_fn, sched_iter, add_iter, norm_type):
#         num_columns = sched_iter + add_iter + 2
#         memmap_array = np.memmap(memmap_fn, dtype=np.float64, mode='r')
#         num_rows = memmap_array.size // num_columns  # Calculate the number of rows
#         memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]
#
#         # fix that strange order (alphabetical!)?
#         self.fs_list = memmap_array[:, 4]
#         self.ksw_list = memmap_array[:, 5]
#         self.t1w_list = memmap_array[:, 2]
#         self.t2w_list = memmap_array[:, 3]
#         self.amide_fs_list = memmap_array[:, 0]
#         self.amide_ksw_list = memmap_array[:, 1]
#         self.mt_fs_list = memmap_array[:, 6]
#         self.mt_ksw_list = memmap_array[:, 7]
#         sig = memmap_array[:, 8:].T  # [#, 30]
#         del memmap_array
#
#         # Normalization of the dictionary signals
#         if norm_type=='2norm':
#             self.norm_sig_list = sig / np.linalg.norm(sig, axis=0, ord=2)
#             if sched_iter == 30:
#                 self.norm_sig_list = self.norm_sig_list[1:, :]
#         elif norm_type=='M0':
#             self.norm_sig_list = sig / sig[0:1, :]
#             if sched_iter == 30:
#                 self.norm_sig_list = self.norm_sig_list[1:, :]
#         # elif norm_type=='mean_std':
#         #     self.norm_sig_list = sig / sig[0:1, :]
#         else:
#             print(f'There is no normalization "{norm_type}" case')
#
#         print("There are " + str(self.len) + " entries in the training dictionary")
#
#         self.len = num_rows
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         fs = self.fs_list[index]
#         ksw = self.ksw_list[index]
#         t1w = self.t1w_list[index]
#         t2w = self.t2w_list[index]
#         mt_fs = self.mt_fs_list[index]
#         mt_ksw = self.mt_ksw_list[index]
#         amide_fs = self.amide_fs_list[index]
#         amide_ksw = self.amide_ksw_list[index]
#         norm_sig = self.norm_sig_list[:, index]
#
#         return (
#             torch.tensor(fs),
#             torch.tensor(ksw),
#             torch.tensor(t1w),
#             torch.tensor(t2w),
#             torch.tensor(mt_fs),
#             torch.tensor(mt_ksw),
#             torch.tensor(amide_fs),
#             torch.tensor(amide_ksw),
#             torch.tensor(norm_sig)
#         )
class GluAmide3pool(Dataset):
    # CPU dataset GPU output
    def __init__(self, glu_memmap_fn, sched_iter, add_iter, norm_type):
        num_columns = 30 + add_iter + 2
        glu_memmap_array = np.memmap(glu_memmap_fn, dtype=np.float64, mode='r')
        num_rows = glu_memmap_array.size // num_columns  # Calculate the number of rows
        glu_memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]
        self.fs_list = glu_memmap_array[:, 0]
        self.ksw_list = glu_memmap_array[:, 1]
        self.t1w_list = glu_memmap_array[:, 2]
        self.t2w_list = glu_memmap_array[:, 3]
        self.mt_fs_list = glu_memmap_array[:, 4]
        self.mt_ksw_list = glu_memmap_array[:, 5]
        sig_glu = glu_memmap_array[:, 6:].T  # [30/31, :]

        # Normalization of the dictionary signals
        if sig_glu.shape[0] == 30:
            self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
        else:
            if norm_type=='2norm':
                self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
                if sched_iter == 30:
                    sig_glu = sig_glu[1:, :]
                    self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
            elif norm_type=='M0':
                self.glu_norm_sig_list = sig_glu / sig_glu[0:1, :]
                if sched_iter == 30:
                    self.glu_norm_sig_list = self.glu_norm_sig_list[1:, :]
            # elif norm_type=='mean_std':
            #     self.norm_sig_list = sig / sig[0:1, :]
            else:
                print(f'There is no normalization "{norm_type}" case')

        del glu_memmap_array
        self.len = num_rows

        print("There are " + str(self.len) + " entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        t1w = self.t1w_list[index]
        t2w = self.t2w_list[index]
        mt_fs = self.mt_fs_list[index]
        mt_ksw = self.mt_ksw_list[index]
        glu_norm_sig = self.glu_norm_sig_list[:, index]
        return (
            torch.tensor(fs),
            torch.tensor(ksw),
            torch.tensor(t1w),
            torch.tensor(t2w),
            torch.tensor(mt_fs),
            torch.tensor(mt_ksw),
            torch.tensor(glu_norm_sig),
        )


class GluAmide4pool(Dataset):
    # CPU dataset GPU output
    def __init__(self, glu_memmap_fn, amide_memmap_fn, sched_iter, add_iter, norm_type):
        num_columns = 31 + add_iter + 2
        glu_memmap_array = np.memmap(glu_memmap_fn, dtype=np.float64, mode='r')
        amide_memmap_array = np.memmap(amide_memmap_fn, dtype=np.float64, mode='r')
        num_rows = glu_memmap_array.size // num_columns  # Calculate the number of rows
        glu_memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]
        amide_memmap_array.shape = (num_rows, num_columns)  # [#, 30+6]

        # fix that strange order (alphabetical!)?
        self.fs_list = glu_memmap_array[:, 4]
        self.ksw_list = glu_memmap_array[:, 5]
        self.t1w_list = glu_memmap_array[:, 2]
        self.t2w_list = glu_memmap_array[:, 3]
        self.amide_fs_list = glu_memmap_array[:, 0]
        self.amide_ksw_list = glu_memmap_array[:, 1]
        self.mt_fs_list = glu_memmap_array[:, 6]
        self.mt_ksw_list = glu_memmap_array[:, 7]
        sig_glu = glu_memmap_array[:, 8:].T  # [30/31, :]
        sig_amide = amide_memmap_array[:, 8:].T  # [30/31, ;]

        # Normalization of the dictionary signals
        if sig_glu.shape[0] == 30:
            self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
            self.amide_norm_sig_list = sig_amide / np.linalg.norm(sig_amide, axis=0, ord=2)
        else:
            if norm_type == '2norm':
                self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
                self.amide_norm_sig_list = sig_amide / np.linalg.norm(sig_amide, axis=0, ord=2)
                if sched_iter == 30:
                    sig_glu = sig_glu[1:, :]
                    sig_amide = sig_amide[1:, :]
                    self.glu_norm_sig_list = sig_glu / np.linalg.norm(sig_glu, axis=0, ord=2)
                    self.amide_norm_sig_list = sig_amide / np.linalg.norm(sig_amide, axis=0, ord=2)
            elif norm_type == 'M0':
                self.glu_norm_sig_list = sig_glu / sig_glu[0:1, :]
                self.amide_norm_sig_list = sig_amide / sig_amide[0:1, :]
                if sched_iter == 30:
                    self.glu_norm_sig_list = self.glu_norm_sig_list[1:, :]
                    self.amide_norm_sig_list = self.amide_norm_sig_list[1:, :]
            # elif norm_type=='mean_std':
            #     self.norm_sig_list = sig / sig[0:1, :]
            else:
                print(f'There is no normalization "{norm_type}" case')

        del glu_memmap_array, amide_memmap_array
        self.len = num_rows

        print("There are " + str(self.len) + " entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        t1w = self.t1w_list[index]
        t2w = self.t2w_list[index]
        mt_fs = self.mt_fs_list[index]
        mt_ksw = self.mt_ksw_list[index]
        amide_fs = self.amide_fs_list[index]
        amide_ksw = self.amide_ksw_list[index]
        glu_norm_sig = self.glu_norm_sig_list[:, index]
        amide_norm_sig = self.amide_norm_sig_list[:, index]

        return (
            torch.tensor(fs),
            torch.tensor(ksw),
            torch.tensor(t1w),
            torch.tensor(t2w),
            torch.tensor(mt_fs),
            torch.tensor(mt_ksw),
            torch.tensor(amide_fs),
            torch.tensor(amide_ksw),
            torch.tensor(glu_norm_sig),
            torch.tensor(amide_norm_sig)
        )
