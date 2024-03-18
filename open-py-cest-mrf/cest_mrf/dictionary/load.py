import yaml
import os

def read_mrf_simulation_params(yaml_fn):
    """
    Read .yaml file which contains the parameters of MRF (or not) simulation.
    :param yaml_fn: .yaml file name
    :return: params - raw yaml data, dict_ - dictionary with data in a required structure, options - scanner definitions
    """

    # init output
    options = {}
    dict_ = {}

    # check for file
    if not os.path.isfile(yaml_fn):
        raise ValueError('yaml parameter file does not exist!')

    # read struct
    with open(yaml_fn, 'r') as f:
        params = yaml.safe_load(f)

    # get water pool
    if 'water_pool' not in params:
        raise ValueError('Water pool must be defined in "water_pool"')
    wp = params['water_pool']
    not_params = ['f', 't1', 't2']
    not_contains = any(item not in wp.keys() for item in not_params)
    if not_contains:
        raise ValueError('"water_pool" must contain "f", "t1" and "t2"')


    # firt fill possible dict_ variables
    dict_['variables'] = {}
    dict_['variables']['tw1'] = wp['t1']
    dict_['variables']['tw2'] = wp['t2']
    dict_['variables']['fww'] = wp['f']

    # CEST pools
    num_pools = 0
    if 'cest_pool' in params:
        cp = params['cest_pool']
        pool_names = list(cp.keys())
        num_pools = len(pool_names)
        # PMEX['cest_pool'] = {}
        for i, p in enumerate(pool_names):
            cpool = cp[p]
            not_params = ['f', 't1', 't2', 'k', 'dw']
            not_contains = any(item not in cpool.keys() for item in not_params)
            if not_contains:
                raise ValueError(f'{p} must contain "f/c", "t1" , "t2", "k" and "dw"')

            # fill the dictionary struct
            dict_['variables']['ts1' + '_' + str(i)] = cpool['t1']
            dict_['variables']['ts2' + '_' + str(i)] = cpool['t2']
            dict_['variables']['fss' + '_' + str(i)] = cpool['f']
            dict_['variables']['ksw' + '_' + str(i)] = cpool['k']
            dict_['variables']['dsw' + '_' + str(i)] = cpool['dw']


    else:
        print('No CEST pools found in param files! specify with "cest_pool" if needed')

    # MT pool
    if 'mt_pool' in params:
        mt = params['mt_pool']
        not_params = ['f', 't1', 't2', 'k', 'dw', 'lineshape']
        not_contains = any(item not in mt.keys() for item in not_params)
        if not_contains:
            raise ValueError('"mt_pool" must contain "f", "t1", "t2", "k", "dw" and "lineshape"')
        if mt['lineshape'] not in  ['SuperLorentzian', 'Lorentzian', 'Gaussian']:
            raise ValueError('lineshape must be "SuperLorentzian", "Lorentzian" or "Gaussian"')

        # fill the dictionary struct
        dict_['variables']['tm1'] = mt['t1']
        dict_['variables']['tm2'] = mt['t2']
        dict_['variables']['fmm'] = mt['f']
        dict_['variables']['kmw'] = mt['k']
        dict_['variables']['dmw'] = mt['dw']
        dict_['variables']['lmm'] = mt['lineshape']

    else:
        print('No MT pools found in param files! specify with "mt_pool" if needed')

    # scanner parameters
    if 'b0' not in params or 'gamma' not in params:
        raise ValueError('Parameter file must contain "b0" and "gamma"')
    options['scanner'] = {}
    options['scanner']['b0'] = params['b0']  # field strength [T]
    options['scanner']['gamma'] = params['gamma']  # gyromagnetic ratio [rad/uT]
    if 'b0_inhom' in params:
        options['scanner']['b0_inhom'] = params['b0_inhom']

    if 'rel_b1' in params:
        options['scanner']['rel_b1'] = params['rel_b1']

    # more optional parameters
    if 'verbose' in params:
        options['verbose'] = bool(params['verbose'])
    if 'reset_init_mag' in params:
        options['reset_init_mag'] = bool(params['reset_init_mag'])
    if 'max_pulse_samples' in params:
        options['max_pulse_samples'] = params['max_pulse_samples']
    if 'scale' in params:
        options['scale'] = params['scale']

    return params, dict_, options

