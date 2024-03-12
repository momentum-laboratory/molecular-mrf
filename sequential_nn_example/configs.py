import numpy as np

class Config:
    def get_config(self):
        return self.cfg

class ConfigIohexol(Config):
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'scenario_io.yaml'
        config['seq_fn'] = 'acq_protocol_io.seq'
        config['dict_fn'] = 'dict_io.mat'

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = np.arange(2000, 3600+100,100) / 1000
        config['water_pool']['t1'] = config['water_pool']['t1'].tolist()
        config['water_pool']['t2'] = np.arange(200, 900+50, 50) / 1000
        config['water_pool']['t2'] = config['water_pool']['t2'].tolist()
        config['water_pool']['f'] = 1

        # Solute pool  
        config['cest_pool'] = {}
        config['cest_pool']['Iohexol'] = {}
        config['cest_pool']['Iohexol']['protons'] = 2
        config['cest_pool']['Iohexol']['t1'] = [2200 / 1000]
        config['cest_pool']['Iohexol']['t2'] = [40 / 1000]
        config['cest_pool']['Iohexol']['k'] = np.arange(50, 1800+10, 10).tolist()        
        config['cest_pool']['Iohexol']['f'] = np.arange(10, 100+5, 5) * config['cest_pool']['Iohexol']['protons'] / 110e3
        config['cest_pool']['Iohexol']['f'] = config['cest_pool']['Iohexol']['f'].tolist()
        config['cest_pool']['Iohexol']['dw'] = 4.3

        # Fill initial magnetization info
        config['scale'] = 1
        config['reset_init_mag'] = 0

        # Fill scanner info
        config['b0'] = 4.7
        config['gamma'] = 267.5153
        config['b0_inhom'] = 0
        config['rel_b1'] = 1

        # Fill additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

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
