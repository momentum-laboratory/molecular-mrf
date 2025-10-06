import numpy as np

class Config:
    def get_config(self):
        return self.cfg


class ConfigPreclinical(Config):
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'scenario.yaml'
        config['seq_fn'] = 'acq_protocol.seq'
        config['dict_fn'] = 'dict.mat'

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = np.arange(2500, 3350, 50) / 1000
        config['water_pool']['t1'] = config['water_pool']['t1'].tolist()  # vary t1
        config['water_pool']['t2'] = np.arange(600, 1250, 50) / 1000
        config['water_pool']['t2'] = config['water_pool']['t2'].tolist()  # vary t2
        config['water_pool']['f'] = 1

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = [2800 / 1000]
        config['cest_pool']['Amine']['t2'] = [40 / 1000]
        config['cest_pool']['Amine']['k'] = np.arange(100, 1410, 10).tolist()
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
        config['num_workers'] = 28

        self.cfg = config


class ConfigMouse(Config):
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'scenario_mouse.yaml'
        config['seq_fn'] = 'acq_protocol_mouse.seq'
        config['dict_fn'] = 'dict_mouse.mat'

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = np.arange(1300, 2300+100,100) / 1000
        config['water_pool']['t1'] = config['water_pool']['t1'].tolist()
        config['water_pool']['t2'] = np.arange(40, 110+10, 10) / 1000
        config['water_pool']['t2'] = config['water_pool']['t2'].tolist()
        config['water_pool']['f'] = 1

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['MT'] = {}
        config['cest_pool']['MT']['t1'] = [2200 / 1000]
        config['cest_pool']['MT']['t2'] = [10e-6]
        config['cest_pool']['MT']['k'] = np.arange(5, 100+5, 5).tolist()
        config['cest_pool']['MT']['dw'] = -2.5
        config['cest_pool']['MT']['f'] = np.arange(2e3, 30e3+2e3, 2e3) * 1 / 110e3
        config['cest_pool']['MT']['f'] = config['cest_pool']['MT']['f'].tolist()

        # Fill initial magnetization info
        config['scale'] = 1
        config['reset_init_mag'] = 0

        # Fill scanner info
        config['b0'] = 7
        config['gamma'] = 267.5153
        config['b0_inhom'] = 0
        config['rel_b1'] = 1
        
        # Fill additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 16

        self.cfg = config
