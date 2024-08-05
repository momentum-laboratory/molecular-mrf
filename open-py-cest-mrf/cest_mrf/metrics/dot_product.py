import scipy.io as sio
import numpy as np
from numpy import linalg as la

from typing import List, Dict, Optional

def dot_prod_indexes(synt_sig:List, acquired_data:List, batch_size:int = 256, restrict:Dict = None):
    """
    Perform dot product between synthetic signals and acquired data in batches.

    :param synt_sig: numpy array representing synthetic signals.
    :param acquired_data: numpy array representing acquired data.
    :param batch_size: size of each batch for processing.
    :param restrict: dictionary containing constraints for each synthetic signal. 
        e.g. {'t1w': {'dict': np.array, 'map': np.array, 'step': 0.1}},
        where dict is the required value in the dict, e.g. dictionary['b0_inhom'].T,
        where map is the some quantitative map to restrict to, e.g. b0_map[...,n_slice] 
        where step is the maximum difference allowed between the dictionary and the map.
    :return: dict containing dot products and indexes reshaped to original dimensions.
    """
    # NV, Apr 1, 2024
    
    n_iter, r_raw_data, c_raw_data = acquired_data.shape
    data = acquired_data.reshape((n_iter, r_raw_data * c_raw_data), order='F')

    if restrict is not None:
        constraint_masks = {}
        for k, v in restrict.items():
            restrict[k]['map'] = v['map'].reshape((r_raw_data * c_raw_data), order='F')


    dp = np.zeros((1, r_raw_data * c_raw_data))
    dp_indexes = np.zeros((1, r_raw_data * c_raw_data))

    norm_dict = synt_sig / (la.norm(synt_sig, axis=0) + 1e-5)
    norm_data = data / (la.norm(data, axis=0) + 1e-5)

    assert norm_data.shape[1] % batch_size == 0, "The number of image pixels must be divisible by batch_size"

    for batch_start in range(0, norm_data.shape[1], batch_size):
        batch_end = batch_start + batch_size
        # print(norm_data[:, batch_start:batch_end].T.shape, norm_dict.shape)
        current_score = norm_data[:, batch_start:batch_end].T @ norm_dict

        if restrict is not None:
            for k, v in restrict.items():
                constraint = np.abs(v['dict']-v['map'][batch_start:batch_end]) < v['step']
                current_score *= constraint.T

        dp[0, batch_start:batch_end] = np.max(current_score, axis=1)
        dp_indexes[0, batch_start:batch_end] = np.argmax(current_score, axis=1)

    
    ret = {
        'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
        'dp_indexes': dp_indexes.reshape((r_raw_data, c_raw_data), order='F'),
    }

    return ret

def dot_prod_matching(dictionary = None, acquired_data = None, dict_fn = None, acquired_data_fn = None, batch_size = 256):
    """
    :param dict_fn: path to dictionary (.mat) with filename
    :param acquired_data_fn: path to acquired data (.mat) with filename
    :param dictionary: dictionary with fields: t1w, t2w, t1s, t2s, fs, ksw, sig 
    :param acquired_data: acquired data with dimensions: n_iter x r_raw_data x c_raw_data
    :param batch_size: batch size for dot product matching
    :return: quant_maps - quantitative maps dictionary with the fields: dp, t1w, t2w, fs, ksw
    """
    #  OP, Mar 2, 2023

    if acquired_data_fn is not None:
        acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']
    elif acquired_data is None:
        raise Exception("Either acquired_data or acquired_data_fn must be specified")
    
    if dict_fn is not None:
        synt_dict = sio.loadmat(dict_fn)
    elif dictionary is not None:
        synt_dict = dictionary
    elif dict_fn is None and dictionary is None:
        raise Exception("Either dictionary or dict_fn must be specified")
    
    if len(synt_dict.keys()) < 4:
        for k in synt_dict.keys():
            if k[0] != '_':
                key = k
        synt_dict = synt_dict[key][0]
        dict_t1w = synt_dict['t1w'][0].transpose()
        dict_t2w = synt_dict['t2w'][0].transpose()
        dict_t1s = synt_dict['t1s'][0].transpose()
        dict_t2s = synt_dict['t2s'][0].transpose()
        dict_fs = synt_dict['fs'][0].transpose()
        dict_ksw = synt_dict['ksw'][0].transpose()
        synt_sig = synt_dict['sig'][0]
    else:
        dict_t1w = synt_dict['t1w']
        dict_t2w = synt_dict['t2w']
        dict_t1s = synt_dict['t1s_0']
        dict_t2s = synt_dict['t2s_0']
        dict_fs = synt_dict['fs_0']
        dict_ksw = synt_dict['ksw_0']

        synt_sig = np.transpose(synt_dict['sig'])  # e.g. 30 x 665,873

    # % Number of schedule iterations and raw data dimensions
    n_iter = np.shape(acquired_data)[0]
    r_raw_data = np.shape(acquired_data)[1]
    c_raw_data = np.shape(acquired_data)[2]

    #  Reshaping image data to voxel - associated columns
    data = acquired_data.reshape((n_iter, r_raw_data * c_raw_data), order='F')

    # Output quantitative maps, initially as zero - vectors
    dp = np.zeros((1, r_raw_data * c_raw_data))
    t1w = np.copy(dp)
    t2w = np.copy(dp)
    t1s = np.copy(dp)
    t2s = np.copy(dp)
    fs = np.copy(dp)
    ksw = np.copy(dp)

    # 2 - norm normalization
    # equivalent to normc in matlab
    norm_dict = synt_sig / (la.norm(synt_sig, axis=0) + 1e-10)
    norm_data = data / (la.norm(data, axis=0) + 1e-10)

    # Matching in batches due to memory considerations
    # The number of image pixel needs to be dividable by batch_size
    assert np.shape(data)[1] / batch_size == np.round(np.shape(data)[1] / batch_size)

    batch_indices = range(0, data.shape[1], batch_size)
    for ind in range(np.size(batch_indices)):
        #  Dot - product for the current batch
        current_score = np.dot(np.transpose(norm_data[:, batch_indices[ind]: batch_indices[ind] + batch_size]),
                               norm_dict)
        # current_score = norm_data[:, batch_indices[ind]: batch_indices[ind] + batch_size].T @ norm_dict

        # Finding maximum dot - product and storing the corresponding parameters
        dp[0, batch_indices[ind]: batch_indices[ind] + batch_size] = np.max(current_score, axis=1)
        dp_ind = np.argmax(current_score, axis=1)


        t1w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1w[0, dp_ind]
        t2w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2w[0, dp_ind]
        t1s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1s[0, dp_ind]
        t2s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2s[0, dp_ind]
        fs[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fs[0, dp_ind]
        ksw[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_ksw[0, dp_ind]

    # Reshaping the output to the original image dimensions
    quant_maps = {'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
                  't1w': t1w.reshape((r_raw_data, c_raw_data), order='F'),
                  't2w': t2w.reshape((r_raw_data, c_raw_data), order='F'),
                  'fs': fs.reshape((r_raw_data, c_raw_data), order='F'),
                  'ksw': ksw.reshape((r_raw_data, c_raw_data), order='F'),
                  't1s': t1s.reshape((r_raw_data, c_raw_data), order='F'),
                  't2s': t2s.reshape((r_raw_data, c_raw_data), order='F')
                }

    return quant_maps