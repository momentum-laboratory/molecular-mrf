import numpy as np
import scipy.io as sio
from numpy import linalg as la
import pandas as pd

def dot_prod_matching(dict_fn, acquired_data_fn):
    """
    :param dict_fn: path to dictionary (.mat) with filename
    :param acquired_data_fn: path to acquired data (.mat) with filename
    :return: quant_maps - quantitative maps dictionary with the fields: dp, t1w, t2w, fs, ksw
    """
    #  OP, Mar 2, 2023

    acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']  # 31 channels

    synt_df = pd.read_pickle(dict_fn)

    synt_sig = np.vstack(synt_df['sig'].values).T
    dict_t1w = synt_df['t1w'].values[np.newaxis, :]
    dict_t2w = synt_df['t2w'].values[np.newaxis, :]
    dict_t1s = synt_df['t1s_0'].values[np.newaxis, :]
    dict_t2s = synt_df['t2s_0'].values[np.newaxis, :]
    dict_fs = synt_df['fs_0'].values[np.newaxis, :]
    dict_ksw = synt_df['ksw_0'].values[np.newaxis, :]
    dict_m_id = np.array(range(0, synt_sig.shape[-1]))

    if synt_sig.shape[0] == 31:  # if M0 was simulated
        acquired_data = acquired_data[1:, :]
        synt_sig = synt_sig[1:, :]  # e.g. 30 x dict len
    elif synt_sig.shape[0] == 30:  # if M0 was *not* simulated
        acquired_data = acquired_data[1:, :]
    
    # If more than one pool were simulated (could be MT)
    if 'fs_1' in synt_df.columns.values:
        dict_fss = synt_df['fs_1'].values[np.newaxis, :]
        dict_kssw = synt_df['ksw_1'].values[np.newaxis, :]

    # Number of schedule iterations and raw data dimensions
    n_iter = np.shape(acquired_data)[0]
    r_raw_data = np.shape(acquired_data)[1]
    c_raw_data = np.shape(acquired_data)[2]

    # Reshaping image data to voxel - associated columns
    data = acquired_data.reshape((n_iter, r_raw_data * c_raw_data), order='F')

    # Output quantitative maps, initially as zero - vectors
    dp = np.zeros((1, r_raw_data * c_raw_data))
    t1w = np.copy(dp)
    t2w = np.copy(dp)
    t1s = np.copy(dp)
    t2s = np.copy(dp)
    fs = np.copy(dp)
    ksw = np.copy(dp)
    m_id = np.copy(dp)

    # If more than one pool were simulated (could be MT)
    if 'fs_1' in synt_df.columns.values:
        fss = np.copy(dp)
        kssw = np.copy(dp)

    # 2-norm normalization
    norm_dict = synt_sig / la.norm(synt_sig, axis=0)
    norm_data = data / la.norm(data, axis=0)

    # # M0 normalization
    # norm_dict = synt_sig[1:, :] / synt_sig[1:2, :]
    # norm_data = data[1:, :] / data[1:2, :]

    # Matching in batches due to memory considerations
    # (can be modified according to available RAM)
    batch_size = 64

    # The number of image pixel needs to be dividable by batch_size
    assert np.shape(data)[1] / batch_size == np.round(np.shape(data)[1] / batch_size)

    batch_indices = range(0, data.shape[1], batch_size)
    for ind in range(np.size(batch_indices)):
        #  Dot - product for the current batch
        current_score = np.dot(np.transpose(norm_data[:, batch_indices[ind]: batch_indices[ind] + batch_size]),
                               norm_dict)
        # Finding maximum dot - product and storing the corresponding parameters
        dp[0, batch_indices[ind]: batch_indices[ind] + batch_size] = np.max(current_score, axis=1)
        dp_ind = np.argmax(current_score, axis=1)

        t1w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1w[0, dp_ind]
        t2w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2w[0, dp_ind]
        t1s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1s[0, dp_ind]
        t2s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2s[0, dp_ind]
        fs[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fs[0, dp_ind]
        ksw[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_ksw[0, dp_ind]
        m_id[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_m_id[dp_ind]
        # If more than one pool were simulated (could be MT)
        if 'fs_1' in synt_df.columns.values:
            fss[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fss[0, dp_ind]
            kssw[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_kssw[0, dp_ind]


    # Reshaping the output to the original image dimensions
	# If more than one pool were simulated (could be MT)
    if 'fs_1' in synt_df.columns.values:
        quant_maps = {'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
		          't1w': t1w.reshape((r_raw_data, c_raw_data), order='F'),
		          't2w': t2w.reshape((r_raw_data, c_raw_data), order='F'),
		          'fs': fs.reshape((r_raw_data, c_raw_data), order='F'),
		          'ksw': ksw.reshape((r_raw_data, c_raw_data), order='F'),
                  'match_id': m_id.reshape((r_raw_data, c_raw_data), order='F'),
		          'fss': fss.reshape((r_raw_data, c_raw_data), order='F'),
		          'kssw': kssw.reshape((r_raw_data, c_raw_data), order='F')}
    else:
        quant_maps = {'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
                  't1w': t1w.reshape((r_raw_data, c_raw_data), order='F'),
                  't2w': t2w.reshape((r_raw_data, c_raw_data), order='F'),
                  'fs': fs.reshape((r_raw_data, c_raw_data), order='F'),
                  'ksw': ksw.reshape((r_raw_data, c_raw_data), order='F'),
                  'match_id': m_id.reshape((r_raw_data, c_raw_data), order='F')}

    return quant_maps
