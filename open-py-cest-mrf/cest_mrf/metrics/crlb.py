import numpy as np

from copy import copy
from itertools import product

from numpy.linalg import inv, LinAlgError

def crb_calc(dictionary: dict = None,
             signals: list = None,
             sigma:float = 0.008,
             norm:bool = True,
             reg:float = 1e-3,
             verbose:bool = True):
    '''
    Function to calculate Cramer-Rao Lower Bound (CRB).
    It helps to estimate of quantification accuracy for a given MRF dictionary
    :param dictionary: MRF dictionary without signal key
    :param signals: signal array corresponding to dictionary
    :param sigma: standard deviation of a noise
    :param norm: to normalize CRB or not (sqrt(CRB(theta)) / theta) <- shows relative error
    :param reg: regularization coefficient to avoid singular matrix
    :param verbose: print CRB matrix and sensitivity
    :return: CRB matrix size of (p,p) where p is number of differentiable parameters in dictionary
    '''
    assert dict is not None and list is not None, 'dictionary and signals must be defined'

    f_dict = {}
    keys = list(dictionary.keys())
    vals = list(dictionary.values())

    vals = np.squeeze(vals)

    for i in range(len(vals[0])):
        value = []
        for j, k in enumerate(keys):
            value.append(vals[j][i])
        value = tuple(value)
        f_dict[value] = signals[i]


    d_idx = {k: i for i, k in enumerate(keys)}

    # find unique values for every variable
    unique_vals = {}
    idxs = {}
    for k, v in dictionary.items():
        unique_vals[k] = np.unique(v)
        idxs[k] = 0

    # find variables which are changing and thus differentiable
    diff_vars = []
    num_points = 1
    for k, v in unique_vals.items():
        if len(v) > 1:
            diff_vars.append(k)
            num_points *= len(v)

    # Calculate Jacobian
    num_all = len(unique_vals.keys())
    num_dvars = len(diff_vars)
    num_vars = num_all - num_dvars
    sig_len = len(signals[0])
    J = np.ones((num_dvars, num_points, sig_len)) * 1e6
    for_norm = np.ones((num_dvars, num_points))
    for dvar_i, dvar in enumerate(diff_vars):
        d = copy(unique_vals)
        d.pop(dvar, None)
        point = 0

        for v in product(*d.values()):
            values0 = list(v)
            values1 = list(v)
            n_dvar = len(unique_vals[dvar])
            for i_dvar in range(n_dvar):
                if i_dvar == 0:
                    x0 = unique_vals[dvar][i_dvar]
                    x1 = unique_vals[dvar][i_dvar + 1]
                elif i_dvar == n_dvar - 1:
                    x0 = unique_vals[dvar][i_dvar - 1]
                    x1 = unique_vals[dvar][i_dvar]
                else:
                    x0 = unique_vals[dvar][i_dvar - 1]
                    x1 = unique_vals[dvar][i_dvar + 1]

                if len(values0) < num_all:
                    values0.insert(d_idx[dvar], x0)
                    values1.insert(d_idx[dvar], x1)
                else:
                    values0[d_idx[dvar]] = x0
                    values1[d_idx[dvar]] = x1


                y0 = f_dict[tuple(values0)]
                y1 = f_dict[tuple(values1)]

                J[dvar_i, point, :] = (y1 - y0) / (x1 - x0)
                for_norm[dvar_i, point] = unique_vals[dvar][i_dvar]
                point += 1

    CRB = np.zeros((num_points, num_dvars, num_dvars))
    for i in range(num_points):
        I_ij = np.matmul(J[:, i, :], J[:, i, :].T) / sigma ** 2
        try:
            # V_ij = 1 / (I_ij)
            V_ij = inv(I_ij)
        except LinAlgError:
            print('WARNING: I_ij is a singular matrix.')
            print(I_ij)
            print(f'WARNING: use regularization with {reg} coefficient')
            V_ij = inv(I_ij + reg * np.eye(*I_ij.shape))

        CRB[i] = V_ij
        if norm:
            CRB[i] = CRB[i]  / for_norm[:, i] ** 2

    mCRB = np.mean(CRB, axis=0)
    meCRB = np.median(CRB, axis=0)

    if verbose:
        print('Mean CRB matrix: \n', mCRB)
        print('Median CRB matrix: \n', meCRB)

        if norm:
            print('Normalized CRLB for diagonal elements (sensitivity), %: ')
            print(dict(zip(diff_vars, [mCRB[i, i] * 100 for i in range(num_dvars)])))
        else:
            print('Cramer Rao Lower Bound for diagonal elements: ')
            print(dict(zip(diff_vars, [mCRB[i, i] for i in range(num_dvars)])))

    return CRB, diff_vars

