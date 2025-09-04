import os
from itertools import product
import time

import concurrent.futures

from .load import read_mrf_simulation_params
from ..simulation.simulate import simulate_mrf

import math
from scipy.io import savemat


import tqdm

# new func to rename keys in the final dictionary
def key_map(key):
    """
    Maps a given key to a new key based on predefined mapping.
    """
    mapping = {'tw1': 't1w', 'tw2': 't2w', 'fww': 'fw',
               'ts1': 't1s', 'ts2': 't2s', 'fss': 'fs', 'ksw': 'ksw',
               'tm1': 't1m', 'tm2': 't2m', 'fmm': 'fm', 'lmm': 'lineshape',
               'dsw': 'dsw', 'dmw': 'dmw', 'kmw': 'kss'}
    for old_key, new_key in mapping.items():
        if key.startswith(old_key):
            return key.replace(old_key, new_key)
    return None  # Return the original key if no mapping is found

def inverse_key_map(key):
    """
    Maps a given key to its original key based on predefined inverse mapping.
    """
    inverse_mapping = {v: k for k, v in {'tw1': 't1w', 'tw2': 't2w', 'fww': 'fw',
                                         'ts1': 't1s', 'ts2': 't2s', 'fss': 'fs', 'ksw': 'ksw',
                                         'tm1': 't1m', 'tm2': 't2m', 'fmm': 'fm', 'lmm': 'lineshape',
                                         'dsw': 'dsw', 'dmw': 'dmw', 'kmw': 'kss'}.items()}
    for new_key, old_key in inverse_mapping.items():
        if key.startswith(new_key):
            return key.replace(new_key, old_key)
    return None  # Return the original key if no inverse mapping is found

def check_dict(dict_):
    # Even single value must be an array to get the next code working
    for k, v in dict_['variables'].items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            v = [v]
            dict_['variables'][k] = v
    return dict_


def prepare_dictionary(dict_, equals=None):
    # generate unique combination
    """
    Input:  dict_:    dictionary variable
            equals:   list of keys [(key1,key2,factor)], which should be equal in all combinations (second equal to first multiplied by factor)

    return: dict_:    dictionary variable, filled with all combinations
            num_comb: total number of all combinations
    """
    
    dict_ = check_dict(dict_)

    var_names0 = dict_['variables'].keys()
    var_names0 = list(var_names0)
    var_names = var_names0.copy()

    if equals is not None:
        assert isinstance(equals, list), "equals must be a list of tuples"

        for pair_factor in equals:
            pair_factor = list(pair_factor)
            assert len(pair_factor) == 3, "equals must be a list of tuples with three elements (x, x*factor, factor) (second eqauls to first multiplied by factor)"

            pair_factor = [inverse_key_map(p) for p in pair_factor[:2]]
            if pair_factor[0] not in var_names0 or pair_factor[1] not in var_names0:
                raise ValueError(f"Key {key_map(pair_factor[0])} or {key_map(pair_factor[1])} not in dictionary variables")
            if pair_factor[0] == pair_factor[1]:
                raise ValueError(f"Key {key_map(pair_factor[0])} is equal to itself")
            if pair_factor[1] in var_names:
                var_names.remove(pair_factor[1])

    combinations = list(product(*[dict_['variables'][name] for name in var_names]))
    num_comb = len(combinations)
    print(f"Found {num_comb} different parameter combinations.")

    # put all combinations in dict
    for i, name in enumerate(var_names):
        dict_[name] = [x[i] for x in combinations]

    if equals is not None:
        for pair_factor in equals:
            pair_factor = list(pair_factor)
            factor = pair_factor[2]
            pair_factor = [inverse_key_map(p) for p in pair_factor[:2]]

            dict_[pair_factor[1]] = [x * factor for x in dict_[pair_factor[0]]]

    return dict_, num_comb


def generate_mrf_cest_dictionary(seq_fn=None,
                                 param_fn=None,
                                 dict_fn=None,
                                 num_workers=None,
                                 shuffle=True,
                                 axes='xy',
                                 equals=None):
    if seq_fn is None and param_fn is None:
        raise Exception(".seq and .yaml files must be specified")

    if dict_fn is None:
        head, tail = os.path.split(param_fn)
        fn, _ = os.path.splitext(tail)
        dict_fn = os.path.join(head, fn + '.mat')

    config, dictionary, options = read_mrf_simulation_params(param_fn)

    if 'variables' not in dictionary:
        raise ValueError('No parameter variation in yaml file...')
    
    if shuffle:
        dictionary, num_comb = prepare_dictionary(dictionary, equals=equals)
        dictionary.pop('variables', None)

    else:
        num_comb = len(dictionary['variables'][list(dictionary['variables'].keys())[0]])
        for k, v in dictionary['variables'].items():
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                v = [v]*num_comb
                dictionary['variables'][k] = v
            elif len(v) == 1:
                v = v*num_comb
                dictionary['variables'][k] = v
        dictionary = dictionary['variables']
        print(f"Found {num_comb} different parameter combinations.")

    print('Dictionary generation started. Please wait...')

    if num_workers > 1 and num_comb > num_workers:
        pbar = tqdm.tqdm(total=num_comb)
        # Divide dict to dicts for multithreading
        dicts = []
        # Iterate over the keys in the dictionary
        for key in dictionary.keys():
            values = dictionary[key]
            chunk_size = math.ceil(len(values) / num_workers)
            for i in range(num_workers):
                # Create a new dictionary if it does not exist
                if i == len(dicts):
                    dicts.append({})
                # Get the start and end indices of the current chunk
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(values))
                dicts[i][key] = values[start:end]

        start = time.perf_counter()
        signals = {}
        from copy import deepcopy
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_simulate = \
                {executor.submit(simulate_mrf, deepcopy(dicts[i]), deepcopy(options), seq_file=seq_fn, id_num = i, axes=axes): i for i in range(num_workers)}
            for future in concurrent.futures.as_completed(future_to_simulate):
                try:
                    sim_params, signal, id_num = future.result()
                except Exception as exc:
                    print(f' generated an exception: {exc}')
                    raise exc
                else:
                    # print(f'Future {id_num} is finished')
                    pbar.update(len(signal))
                    pbar.set_description(f'CPU thread #{id_num} is finished')
                    signals[id_num] = signal

        end = time.perf_counter()
        pbar.close()
        s = (end-start)
        print(f"Dictionary simulation took {s:.03f} s.")

        combined_signals = []
        for i in sorted(list(signals.keys())):
            combined_signals.extend(signals[i])
    else:
        start = time.perf_counter()
        sim_params, combined_signals, _ = simulate_mrf(dictionary, options, seq_file=seq_fn, axes=axes)
        end = time.perf_counter()
        s = (end-start)
        print(f"Dictionary simulation took {s:.03f} s.")

    # rename the keys to more readable
    new_dict = {}
    for key, value in dictionary.items():
        # if key.startswith('dsw') or key.startswith('dmw'):
        #     continue
        new_key = key_map(key)
        new_dict[new_key] = value

    dictionary = new_dict
    dictionary['sig'] = combined_signals

    savemat(dict_fn, dictionary)

    return dictionary
