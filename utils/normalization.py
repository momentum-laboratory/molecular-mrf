import yaml
import numpy as np
import torch

def normalize_range(original_array, original_min, original_max, new_min, new_max):
    """ Normalizing data to a new range (e.g. to [-1, 1] or [1, 1])
    :param original_array:   input array
    :param original_min: current minimum (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: normalized array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (original_array - a) / (b - a) * (d - c) + c


def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    """ Un-normalizing data to its original range (e.g. to [0, 1400])
    :param normalized_array:  normalized array
    :param original_min: minimum value (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: original array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a


def min_max_yaml(yaml_file_path, scenario_type, device):
    # Open and read the YAML file
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    if scenario_type == '2pool':
        min_fs = data['cest_pool']['Amine']['f'][0]  # uncomment if non-zero minimum limit is required
        max_fs = data['cest_pool']['Amine']['f'][-1]
        min_ksw = float(data['cest_pool']['Amine']['k'][0])  # uncomment if non-zero minimum limit needed
        max_ksw = float(data['cest_pool']['Amine']['k'][-1])

        min_t1w = data['water_pool']['t1'][0]
        max_t1w = data['water_pool']['t1'][-1]
        min_t2w = data['water_pool']['t2'][0]
        max_t2w = data['water_pool']['t2'][-1]

        min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).to(device)

        min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False).to(device)
        max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False).to(device)

        min_max_params = (min_param_tensor, max_param_tensor,
                          min_water_t1t2_tensor, max_water_t1t2_tensor)

    elif scenario_type == '3pool':
        min_fs = data['cest_pool']['Amine']['f'][0]  # uncomment if non-zero minimum limit is required
        max_fs = data['cest_pool']['Amine']['f'][-1]
        min_ksw = float(data['cest_pool']['Amine']['k'][0])  # uncomment if non-zero minimum limit needed
        max_ksw = float(data['cest_pool']['Amine']['k'][-1])

        min_t1w = data['water_pool']['t1'][0]
        max_t1w = data['water_pool']['t1'][-1]
        min_t2w = data['water_pool']['t2'][0]
        max_t2w = data['water_pool']['t2'][-1]

        min_mt_fs = data['cest_pool']['mt_pool']['f'][0]  # uncomment if non-zero minimum limit is required
        max_mt_fs = data['cest_pool']['mt_pool']['f'][-1]
        min_mt_ksw = float(data['cest_pool']['mt_pool']['k'][0])  # uncomment if non-zero minimum limit needed
        max_mt_ksw = float(data['cest_pool']['mt_pool']['k'][-1])

        min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).to(device)

        min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False).to(device)
        max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False).to(device)

        min_mt_param_tensor = torch.tensor(np.hstack((min_mt_fs, min_mt_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_mt_param_tensor = torch.tensor(np.hstack((max_mt_fs, max_mt_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw

        min_max_params = (min_param_tensor, max_param_tensor,
                          min_water_t1t2_tensor, max_water_t1t2_tensor,
                          min_mt_param_tensor, max_mt_param_tensor)

    elif scenario_type == '4pool':
        min_fs = data['cest_pool']['Glu']['f'][0]  # uncomment if non-zero minimum limit is required
        max_fs = data['cest_pool']['Glu']['f'][-1]
        min_ksw = float(data['cest_pool']['Glu']['k'][0])  # uncomment if non-zero minimum limit needed
        max_ksw = float(data['cest_pool']['Glu']['k'][-1])

        min_t1w = data['water_pool']['t1'][0]
        max_t1w = data['water_pool']['t1'][-1]
        min_t2w = data['water_pool']['t2'][0]
        max_t2w = data['water_pool']['t2'][-1]

        min_mt_fs = data['cest_pool']['mt_pool']['f'][0]  # uncomment if non-zero minimum limit is required
        max_mt_fs = data['cest_pool']['mt_pool']['f'][-1]
        min_mt_ksw = float(data['cest_pool']['mt_pool']['k'][0])  # uncomment if non-zero minimum limit needed
        max_mt_ksw = float(data['cest_pool']['mt_pool']['k'][-1])

        min_amide_fs = data['cest_pool']['Amide']['f'][0]  # uncomment if non-zero minimum limit is required
        max_amide_fs = data['cest_pool']['Amide']['f'][-1]
        min_amide_ksw = float(data['cest_pool']['Amide']['k'][0])  # uncomment if non-zero minimum limit needed
        max_amide_ksw = float(data['cest_pool']['Amide']['k'][-1])

        min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).to(device)

        min_water_t1t2_tensor = torch.tensor(np.hstack((min_t1w, min_t2w)), requires_grad=False).to(device)
        max_water_t1t2_tensor = torch.tensor(np.hstack((max_t1w, max_t2w)), requires_grad=False).to(device)

        min_mt_param_tensor = torch.tensor(np.hstack((min_mt_fs, min_mt_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_mt_param_tensor = torch.tensor(np.hstack((max_mt_fs, max_mt_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw

        min_amide_param_tensor = torch.tensor(np.hstack((min_amide_fs, min_amide_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw
        max_amide_param_tensor = torch.tensor(np.hstack((max_amide_fs, max_amide_ksw)), requires_grad=False).to(
            device)  # can be switched to  min_fs, min_ksw

        min_max_params = (min_param_tensor, max_param_tensor,
                          min_water_t1t2_tensor, max_water_t1t2_tensor,
                          min_mt_param_tensor, max_mt_param_tensor,
                          min_amide_param_tensor, max_amide_param_tensor)

    return min_max_params