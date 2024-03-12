import numpy as np

class Config:
    def get_config(self):
        return self.cfg

class ConfigClinical(Config):
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'scenario3T.yaml'
        config['seq_fn'] = 'acq_protocol3T.seq'
        config['dict_fn'] = 'dict3T.mat'

        # Water_pool
        config['water_pool'] = {}
        # config['water_pool']['t1'] = np.arange(2700, 3200, 100) / 1000
        # config['water_pool']['t1'] = config['water_pool']['t1'].tolist()  # vary t1
        config['water_pool']['t1'] = [2.8]
        config['water_pool']['t2'] = np.arange(400, 1350, 50) / 1000
        config['water_pool']['t2'] = config['water_pool']['t2'].tolist()  # vary t2
        config['water_pool']['f'] = 1

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = [2800 / 1000]
        config['cest_pool']['Amine']['t2'] = [40 / 1000]
        config['cest_pool']['Amine']['k'] = np.arange(100, 510, 10).tolist()
        config['cest_pool']['Amine']['dw'] = 3
        config['cest_pool']['Amine']['f'] = np.arange(10, 120, 5) * 3 / 110000
        config['cest_pool']['Amine']['f'] = config['cest_pool']['Amine']['f'].tolist()

        # Fill initial magnetization info
        # this is important now for the mrf simulation! For the regular pulseq-cest
        # simulation, we usually assume athat the magnetization reached a steady
        # state after the readout, which means we can set the magnetization vector
        # to a specific scale, e.g. 0.5. This is because we do not simulate the
        # readout there. For mrf we include the readout in the simulation, which
        # means we need to carry the same magnetization vector through the entire
        # sequence. To avoid that the magnetization vector gets set to the initial
        # value after each readout, we need to set reset_init_mag to false
        config['scale'] = 1
        config['reset_init_mag'] = 0

        # Fill scanner info
        config['b0'] = 2.89
        config['gamma'] = 267.5153
        config['freq'] = 127.7153
        config['b0_inhom'] = 0
        config['rel_b1'] = 1

        # Fill additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

        self.cfg = config


class ConfigPreclinical(Config):
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'scenario.yaml'
        config['seq_fn'] = 'acq_protocol.seq'
        config['dict_fn'] = 'dict.mat'

        # Water_pool
        config['water_pool'] = {}
        # config['water_pool']['t1'] = np.arange(2500, 3350, 50) / 1000
        # config['water_pool']['t1'] = config['water_pool']['t1'].tolist()  # vary t1
        config['water_pool']['t1'] = [2.8]
        config['water_pool']['t2'] = np.arange(600, 1250, 50) / 1000
        config['water_pool']['t2'] = config['water_pool']['t2'].tolist()  # vary t2
        config['water_pool']['f'] = 1

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = [2800 / 1000]
        config['cest_pool']['Amine']['t2'] = [40 / 1000]
        config['cest_pool']['Amine']['k'] = np.arange(100, 510, 10).tolist()
        config['cest_pool']['Amine']['dw'] = 3
        config['cest_pool']['Amine']['f'] = np.arange(10, 125, 5) * 3 / 110000
        config['cest_pool']['Amine']['f'] = config['cest_pool']['Amine']['f'].tolist()

        # Fill initial magnetization info
        # this is important now for the mrf simulation! For the regular pulseq-cest
        # simulation, we usually assume athat the magnetization reached a steady
        # state after the readout, which means we can set the magnetization vector
        # to a specific scale, e.g. 0.5. This is because we do not simulate the
        # readout there. For mrf we include the readout in the simulation, which
        # means we need to carry the same magnetization vector through the entire
        # sequence. To avoid that the magnetization vector gets set to the initial
        # value after each readout, we need to set reset_init_mag to false
        config['scale'] = 1
        config['reset_init_mag'] = 0

        # Fill scanner info
        config['b0'] = 9.4
        config['gamma'] = 267.5153
        config['b0_inhom'] = 0
        config['rel_b1'] = 1

        # Fill additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

        self.cfg = config
