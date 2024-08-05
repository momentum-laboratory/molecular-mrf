import os
from itertools import product
import time

import concurrent.futures

from .load import read_mrf_simulation_params
from ..simulation.simulate import simulate_mrf
from sequential_nn.multi_dict import pkl_2_dat

import math
import pandas as pd

import tqdm

def check_dict(dict_):
    # Even single value must be an array to get the next code working
    for k, v in dict_['variables'].items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            v = [v]
            dict_['variables'][k] = v
    return dict_


# def prepare_dictionary(dict_, add_iter, equals=None):
#     # generate unique combination
#     """
#     Input:  dict_:    dictionary variable
#             equals:   list of keys (pairs), which should be equal in all combinations (first equal to second)
#
#     return: dict_:    dictionary variable, filled with all combinations
#             num_comb: total number of all combinations
#     """
#
#     dict_ = check_dict(dict_)
#
#     var_names = dict_['variables'].keys()
#     var_names = list(var_names)
#
#     if equals is not None:
#         for pair in equals:
#             if pair[0] not in var_names or pair[1] not in var_names:
#                 raise ValueError(f"Key {pair[0]} or {pair[1]} not in dictionary variables")
#             if pair[0] == pair[1]:
#                 raise ValueError(f"Key {pair[0]} is equal to itself")
#             var_names.remove(pair[1])
#
#     combinations = list(product(*[dict_['variables'][name] for name in var_names]))
#     num_comb = len(combinations)
#     print(f"Found {num_comb} different parameter combinations.")
#
#     # put all combinations in dict
#     for i, name in enumerate(var_names):
#         dict_[name] = [x[i] for x in combinations]
#         if (add_iter == 2) & (name == 'ts1_1'):  # 2pool
#             dict_['ts1_1'] = dict_['tw1']
#             a=3
#         if (add_iter == 4) & (name == 'ts1_0'):  # 3pool
#             dict_['ts1_0'] = dict_['tw1']
#         if (add_iter == 6) & (name == 'ts1_1'):  # 4pool
#             # dict_['ts1_0'] = dict_['tw1']  # 0: Amide
#             dict_['ts1_1'] = dict_['tw1']  # MT
#
#     if equals is not None:
#         for pair in equals:
#             dict_[pair[1]] = dict_[pair[0]]
#
#     return dict_, num_comb
#
#
# def generate_mrf_cest_dictionary(seq_fn=None,
#                                  param_fn=None,
#                                  dict_fn=None,
#                                  num_workers=None,
#                                  shuffle=True,
#                                  axes='xy',
#                                  equals=None,
#                                  add_iter=2):
#     if seq_fn is None and param_fn is None:
#         raise Exception(".seq and .yaml files must be specified")
#
#     if dict_fn is None:
#         head, tail = os.path.split(param_fn)
#         fn, _ = os.path.splitext(tail)
#         dict_fn = os.path.join(head, fn + '.mat')
#
#     config, dictionary, options = read_mrf_simulation_params(param_fn)
#
#     if 'variables' not in dictionary:
#         raise ValueError('No parameter variation in yaml file...')
#
#     if shuffle:
#         dictionary, num_comb = prepare_dictionary(dictionary, add_iter, equals=equals)
#         dictionary.pop('variables', None)
#
#     else:
#         num_comb = len(dictionary['variables'][list(dictionary['variables'].keys())[0]])
#         for k, v in dictionary['variables'].items():
#             if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
#                 v = [v]*num_comb
#                 dictionary['variables'][k] = v
#             elif len(v) == 1:
#                 v = v*num_comb
#                 dictionary['variables'][k] = v
#         dictionary = dictionary['variables']
#         print(f"Found {num_comb} different parameter combinations.")
#
#     print('Dictionary generation started. Please wait...')
#
#     if num_workers > 1 and num_comb > num_workers:
#         pbar = tqdm.tqdm(total=num_comb)
#         # Divide dict to dicts for multithreading
#         dicts = []
#         # Iterate over the keys in the dictionary
#         for key in dictionary.keys():
#             values = dictionary[key]
#             chunk_size = math.ceil(len(values) / num_workers)
#             for i in range(num_workers):
#                 # Create a new dictionary if it does not exist
#                 if i == len(dicts):
#                     dicts.append({})
#                 # Get the start and end indices of the current chunk
#                 start = i * chunk_size
#                 end = min((i + 1) * chunk_size, len(values))
#                 dicts[i][key] = values[start:end]
#
#         start = time.perf_counter()
#         signals = {}
#         from copy import deepcopy
#         with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#             future_to_simulate = \
#                 {executor.submit(simulate_mrf, deepcopy(dicts[i]), deepcopy(options), seq_file=seq_fn, id_num = i, axes=axes): i for i in range(num_workers)}
#             for future in concurrent.futures.as_completed(future_to_simulate):
#                 try:
#                     sim_params, signal, id_num = future.result()
#                 except Exception as exc:
#                     print(f' generated an exception: {exc}')
#                     raise exc
#                 else:
#                     # print(f'Future {id_num} is finished')
#                     pbar.update(len(signal))
#                     pbar.set_description(f'CPU thread #{id_num} is finished')
#                     signals[id_num] = signal
#
#         end = time.perf_counter()
#         pbar.close()
#         s = (end-start)
#         print(f"Dictionary simulation took {s:.03f} s.")
#
#         combined_signals = []
#         for i in sorted(list(signals.keys())):
#             combined_signals.extend(signals[i])
#     else:
#         start = time.perf_counter()
#         sim_params, combined_signals, _ = simulate_mrf(dictionary, options, seq_file=seq_fn, axes=axes)
#         end = time.perf_counter()
#         s = (end-start)
#         print(f"Dictionary simulation took {s:.03f} s.")
#
#     # rename the keys to more readable
#     mapping = {'tw1':'t1w','tw2':'t2w','fww':'f',
#                'ts1':'t1s','ts2':'t2s','fss':'fs', 'ksw': 'ksw',
#                'tm1':'t1m','tm2':'t2m','fmm':'fm','lmm':'lineshape'}
#     new_dict = {}
#     for key, value in dictionary.items():
#         for old_key, new_key in mapping.items():
#             if key.startswith(old_key):
#                 # replace the old_key at the start of key with new_key
#                 new_key_name = key.replace(old_key, new_key)
#                 new_dict[new_key_name] = value
#                 break
#     dictionary = new_dict
#     dictionary['sig'] = combined_signals
#
#     # Convert the dictionary to a pandas DataFrame
#     dictionary_df = pd.DataFrame(dictionary)
#     dictionary_df.to_pickle(dict_fn)  # To save as pickle
#     glu_dict_folder_fn = os.path.dirname(dict_fn)
#     memmap_fn = os.path.join(glu_dict_folder_fn, 'dict.dat')
#     pkl_2_dat(glu_dict_folder_fn, 31, add_iter, memmap_fn)
#
#     return dictionary


def prepare_dictionary_4pool(dict_, add_iter, equals=None):
    # generate unique combination
    """
    Input:  dict_:    dictionary variable
            equals:   list of keys (pairs), which should be equal in all combinations (first equal to second)

    return: dict_:    dictionary variable, filled with all combinations
            num_comb: total number of all combinations
    """

    dict_ = check_dict(dict_)

    var_names = dict_['variables'].keys()
    var_names = list(var_names)

    if equals is not None:
        for pair in equals:
            if pair[0] not in var_names or pair[1] not in var_names:
                raise ValueError(f"Key {pair[0]} or {pair[1]} not in dictionary variables")
            if pair[0] == pair[1]:
                raise ValueError(f"Key {pair[0]} is equal to itself")
            var_names.remove(pair[1])

    combinations = list(product(*[dict_['variables'][name] for name in var_names]))
    num_comb = len(combinations)
    print(f"Found {num_comb} different parameter combinations.")

    # put all combinations in dict
    for i, name in enumerate(var_names):
        dict_[name] = [x[i] for x in combinations]
        if add_iter==2:
            # print('2pool case')
            if name == 'ts1_0':
                dict_['ts1_0'] = dict_['tw1']
        elif add_iter==4:
            # print('3pool case')
            if name == 'ts1_0':
                dict_['ts1_0'] = dict_['tw1']
        elif add_iter==6:
            # print('4pool case')
            if name == 'ts1_0':
                dict_['ts1_0'] = dict_['tw1']
            elif name == 'ts1_1':
                dict_['ts1_1'] = dict_['tw1']
            elif name == 'ts1_2':
                dict_['ts1_2'] = dict_['tw1']

    if equals is not None:
        for pair in equals:
            dict_[pair[1]] = dict_[pair[0]]

    return dict_, num_comb

def generate_mrf_cest_dictionary_4pool(seq_fn=None,
                                       param_fn=None,
                                       dict_fn=None,
                                       num_workers=None,
                                       shuffle=True,
                                       axes='xy',
                                       equals=None,
                                       add_iter=6):
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
        dictionary, num_comb = prepare_dictionary_4pool(dictionary, add_iter, equals=equals)
        dictionary.pop('variables', None)

    else:
        num_comb = len(dictionary['variables'][list(dictionary['variables'].keys())[0]])
        for k, v in dictionary['variables'].items():
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                v = [v] * num_comb
                dictionary['variables'][k] = v
            elif len(v) == 1:
                v = v * num_comb
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
                {executor.submit(simulate_mrf, deepcopy(dicts[i]), deepcopy(options), seq_file=seq_fn, id_num=i,
                                 axes=axes): i for i in range(num_workers)}
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
        s = (end - start)
        print(f"Dictionary simulation took {s:.03f} s.")

        combined_signals = []
        for i in sorted(list(signals.keys())):
            combined_signals.extend(signals[i])
    else:
        start = time.perf_counter()
        sim_params, combined_signals, _ = simulate_mrf(dictionary, options, seq_file=seq_fn, axes=axes)
        end = time.perf_counter()
        s = (end - start)
        print(f"Dictionary simulation took {s:.03f} s.")

    # rename the keys to more readable
    mapping = {'tw1': 't1w', 'tw2': 't2w', 'fww': 'f',
               'ts1': 't1s', 'ts2': 't2s', 'fss': 'fs', 'ksw': 'ksw',
               'tm1': 't1m', 'tm2': 't2m', 'fmm': 'fm', 'lmm': 'lineshape'}
    new_dict = {}
    for key, value in dictionary.items():
        for old_key, new_key in mapping.items():
            if key.startswith(old_key):
                # replace the old_key at the start of key with new_key
                new_key_name = key.replace(old_key, new_key)
                new_dict[new_key_name] = value
                break
    dictionary = new_dict
    dictionary['sig'] = combined_signals

    # Convert the dictionary to a pandas DataFrame
    dictionary_df = pd.DataFrame(dictionary)
    dictionary_df.to_pickle(dict_fn)  # To save as pickle
    glu_dict_folder_fn = os.path.dirname(dict_fn)
    memmap_fn = os.path.join(glu_dict_folder_fn, 'dict.dat')
    pkl_2_dat(glu_dict_folder_fn, 31, add_iter, memmap_fn)

    return dictionary