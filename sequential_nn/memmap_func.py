import os
import re
import pandas as pd
import numpy as np



# def memmap_single_dict_processing(glu_dict_fn, glu_memmap_fn, shape):
#     # Loading the training dataset
#     training_data = pd.read_pickle(glu_dict_fn)
#     num_samples = len(training_data)
#
#     # Memory-map the concatenated data
#     training_memmap_data = np.memmap(glu_memmap_fn, dtype=np.float32, mode='w+',
#                                   shape=shape)  # Adjust shape as per your data
#
#     # Fill the memory-mapped array with data
#     training_memmap_data[:, :6] = training_data[
#         ['fs_0', 'ksw_0', 't1w', 't2w', 'fs_1', 'ksw_1']].values
#     training_memmap_data[:, 6:] = np.vstack(training_data['sig'].values)[:, 1:]  # [#, 30]
#
# def memmap_creator(glu_dict_fn, glu_memmap_fn, shape):
#     if os.path.basename(glu_dict_fn) == 'dict.pkl':
#         # I need to norm this case!!!!!!!
#         print(f'Single dict found, memmap processing begins')
#
#         memmap_single_dict_processing(glu_dict_fn, glu_memmap_fn, shape)
#         memmap_fns = [glu_dict_fn]
#
#     else:
#         glu_dict_folder_fn = os.path.dirname(glu_dict_fn)
#         memmap_folder_fn = os.path.dirname(glu_memmap_fn)
#         memmap_fns = [os.path.join(glu_dict_folder_fn, name) for name in os.listdir(glu_dict_folder_fn) if
#                       re.match(r'^dict_\d+\.pkl$', name)]
#         n_dicts = len(memmap_fns) # Count the files that match the pattern dict_{i}
#         print(f'{n_dicts} dicts found, memmap processing begins')
#
#         # for dict_idx in range(1, n_dicts+1):
#         #     cur_dict_fn = os.path.join(glu_dict_folder_fn, f'dict_{dict_idx}.pkl')
#         #     cur_memmap_fn = os.path.join(memmap_folder_fn, f'cur_memmap_{dict_idx}.dat')
#         #     memmap_single_dict_processing(cur_dict_fn, cur_memmap_fn, sched_iter, add_iter)
#         #
#         # sig_norm = memmap_sig_norm(glu_dict_fn, glu_memmap_fn, shape=shape, dtype='float32')
#         # np.save(os.path.join('sequential_nn', 'sig_norms.npy'), sig_norm)
#     return memmap_fns
#
#
# def memmap_sig_norm(glu_dict_fn, glu_memmap_fn, shape, dtype='float32'):
#     glu_dict_folder_fn = os.path.dirname(glu_dict_fn)
#     memmap_folder_fn = os.path.dirname(glu_memmap_fn)
#     memmap_fns = [os.path.join(glu_dict_folder_fn, name) for name in os.listdir(glu_dict_folder_fn) if
#                   re.match(r'^dict_\d+\.pkl$', name)]
#     n_dicts = len(memmap_fns)  # Count the files that match the pattern dict_{i}
#
#     sigs = []
#     for dict_idx in range(1, n_dicts+1):
#         cur_memmap_fn = os.path.join(memmap_folder_fn, f'cur_memmap_{dict_idx}.dat')
#         training_data = np.memmap(cur_memmap_fn, dtype=dtype, mode='r', shape=shape)
#         sigs.append(np.vstack(training_data[:, 6:]).T)  # [[30, #]... ]
#         print(f'Done with dict {dict_idx} signal')
#
#     stacked_sigs = np.hstack(sigs)  # [30, 7*#]
#     # del sigs
#     sig_norm = np.linalg.norm(stacked_sigs, axis=0, ord=2)  # [7*#]
#     return sig_norm
#
# def delete_memmat_files(glu_memmap_fn):
#     memmap_folder_fn = os.path.dirname(glu_memmap_fn)
#
#     # List all files in the directory
#     files = os.listdir(memmap_folder_fn)
#
#     # Iterate over each file and delete if it matches the pattern 'memmat_*.dat'
#     for file in files:
#         if file.startswith('cur_memmat_') and file.endswith('.dat') or file == 'cur_memat.dat':
#             file_path = os.path.join(memmap_folder_fn, file)
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except Exception as e:
#                 print(f"Error deleting {file_path}: {e}")