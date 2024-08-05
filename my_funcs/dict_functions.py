import os
import glob
import numpy as np
import scipy.io as sio

def dict_conc_func(mrf_files_fn):
    """
    Rescale t1/t2 maps
    :param mrf_files_fn: path to mrf_files containing 'dict_*.mat' files
    :return full_dict: full dict, should I sort it?
    """
    dict_file_l = glob.glob(os.path.join(mrf_files_fn, 'dict_*.mat'))  # works up to 9

    full_dict = sio.loadmat(dict_file_l[0])  # first dict
    for cur_dict_fn in dict_file_l[1:]:
        cur_dict = sio.loadmat(cur_dict_fn)
        # concat dict
        full_dict['t1w'] = np.hstack([full_dict['t1w'], cur_dict['t1w']])
        full_dict['t2w'] = np.hstack([full_dict['t2w'], cur_dict['t2w']])
        full_dict['f'] = np.hstack([full_dict['f'], cur_dict['f']])
        full_dict['t1s_0'] = np.hstack([full_dict['t1s_0'], cur_dict['t1s_0']])
        full_dict['t2s_0'] = np.hstack([full_dict['t2s_0'], cur_dict['t2s_0']])
        full_dict['fs_0'] = np.hstack([full_dict['fs_0'], cur_dict['fs_0']])
        full_dict['ksw_0'] = np.hstack([full_dict['ksw_0'], cur_dict['ksw_0']])
        full_dict['t1s_1'] = np.hstack([full_dict['t1s_1'], cur_dict['t1s_1']])
        full_dict['t2s_1'] = np.hstack([full_dict['t2s_1'], cur_dict['t2s_1']])
        full_dict['fs_1'] = np.hstack([full_dict['fs_1'], cur_dict['fs_1']])
        full_dict['ksw_1'] = np.hstack([full_dict['ksw_1'], cur_dict['ksw_1']])
        full_dict['sig'] = np.vstack([full_dict['sig'], cur_dict['sig']])  # final: # 21120000 X 30

    sig = full_dict['sig'].shape
    print(f'final sig shape: {sig}')

    return full_dict