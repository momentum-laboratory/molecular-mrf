from bmctool.params import Params
import numpy as np

class ParamsMRF(Params):
    """
    Class to store simulation parameters for MRF.
    It acts as a proxy to Params class.
    """
    def __init__(self, set_defaults: bool = False):
        self.params_dict = {}
        self.num_comb = 0
        self.n_cest_pools = 0

        super().__init__(set_defaults)

    def _transform_dict(self, dictionary):
        """
        The function transforms the dictionary to the required form
        It was a hot-fix, so probably will be changed
        TODO: rewrite
        """
        # var_names = {
        #     'cest_pool':  ['ts1', 'ts1', 'fss', 'ksw'],
        #     'mt_pool':    ['tm1', 'tm1', 'fmm', 'kmw'],
        #     'water_pool': ['tw1', 'tw1', 'fww']
        # }
        new_dict = {}
        pools = {'s':'cest_pool',
                'm': 'mt_pool',
                'w': 'water_pool'}
        for k, v in dictionary.items():
            if k == 'variables':
                continue
            pool = pools[k[1]]
            if pool not in new_dict.keys():
                new_dict[pool] = {}
                if pool == 'cest_pool':
                    pool_id = k[4:]
                    if pool_id not in new_dict[pool].keys():
                        new_dict[pool][pool_id] = {}
                        self.n_cest_pools += 1
                else:
                    new_dict[pool] = {}
            else:
                if pool == 'cest_pool':
                    pool_id = k[4:]
                    if pool_id not in new_dict[pool].keys():
                        new_dict[pool][pool_id] = {}
                        self.n_cest_pools += 1

            par = k[0]
            if 't' == par:
                n_k = 'r' + k[2]
                n_v = 1 / np.array(v)
                if pool == 'cest_pool':
                    pool_id = k[4:]
                    new_dict[pool][pool_id][n_k] = n_v
                else:
                    new_dict[pool][n_k] = n_v
            else:
                n_k = par
                if n_k == 'd':
                    n_k = 'dw'
                elif n_k == 'l':
                    n_k = 'lineshape'
                n_v = v
                if pool == 'cest_pool':
                    pool_id = k[4:]
                    new_dict[pool][pool_id][n_k] = n_v
                else:
                    new_dict[pool][n_k] = n_v
        return new_dict

    def set_params_dict(self,
                        dictionary: dict = None,
                        options: dict = None):
        """
        Set MRF dictionary with parameters.
        Parameters checking should be performed before passing.
        """
        if dictionary is None or options is None:
            raise ValueError("dictionary and options should be defined")

        self.params_dict = self._transform_dict(dictionary)
        self.num_comb = len(dictionary['tw1'])

        # Initialize with first values
        # set required parameters
        self.set_scanner(**options['scanner'])
        self.set_options(**{k: v for k, v in options.items() if k in ['reset_init_mag', 'max_pulse_samples', 'scale',
                                                                   'par_calc', 'verbose']})

        self.set_water_pool(**{k: v[0] for k, v in self.params_dict['water_pool'].items()})

        # set optional parameters
        if 'cest_pool' in self.params_dict:
            for _, pool in self.params_dict['cest_pool'].items():
                self.set_cest_pool(**{k: v[0] for k, v in pool.items()})
        if 'mt_pool' in self.params_dict:
            self.set_mt_pool(**{k: v[0] for k, v in self.params_dict['mt_pool'].items()})

        self.set_m_vec()

        if self.options['verbose']:
            self.print_settings()

    def __setitem__(self, item, data):
        pass
    def __getitem__(self, item : int):
        if item >= self.num_comb:
            raise ValueError("requested item ID is > num_comb")

        # Update values
        self.update_water_pool(**{k: v[item] for k, v in self.params_dict['water_pool'].items()})

        # set optional parameters
        if 'cest_pool' in self.params_dict:
            for pool_id, pool in self.params_dict['cest_pool'].items():
                self.update_cest_pool(pool_idx = int(pool_id), **{k: v[item] for k, v in pool.items()})
        if 'mt_pool' in self.params_dict:
            self.update_mt_pool(**{k: v[item] for k, v in self.params_dict['mt_pool'].items()})

        self.set_m_vec()

        if self.options['verbose']:
            self.print_settings()

        return self

    def set_m_vec(self) -> np.array:
        """
        Sets the initial magnetization vector (fully relaxed) from the defined pools
        e. g. for 2 CEST pools: [MxA, MxB, MxD, MyA, MyB, MyD, MzA, MzB, MzD, MzC]
        with A: water pool, B: 1st CEST pool, D: 2nd CEST pool, C: MT pool

        :return: array of the initial magnetizations
        """
        if not self.water_pool:
            raise Exception('No water pool defined before assignment of magnetization vector.')

        if self.cest_pools:
            n_total_pools = len(self.cest_pools) + 1
        else:
            n_total_pools = 1

        m_vec = np.zeros(n_total_pools * 3)
        m_vec[n_total_pools * 2] = self.water_pool['f']
        if self.cest_pools:
            for ii in range(1, n_total_pools):
                m_vec[n_total_pools * 2 + ii] = self.cest_pools[ii - 1]['f']
        if self.mt_pool:
            m_vec = np.append(m_vec, self.mt_pool['f'])
        if type(self.options['scale']) == int or type(self.options['scale']) == float:
            m_vec = m_vec * self.options['scale']

        self.m_vec = m_vec
        return m_vec

