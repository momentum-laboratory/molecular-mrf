import yaml
# Write the yaml-file
# We vary T1w, T2w, Ksw and M0s in one file

def write_yaml_dict(cfg: dict, yaml_fn=None):
    if yaml_fn is None:
        if 'yaml_fn' in cfg.keys():
            yaml_fn = cfg['yaml_fn']
        else:
            raise ValueError("yaml_fn must be defined")
    
    with open(yaml_fn, 'w') as file:
        yaml.safe_dump(cfg, file, default_flow_style=None)

def write_yaml_2pool(cfg, yaml_fn=None):
    if yaml_fn is None:
        raise ValueError("yaml_fn must be defined")

    yaml_struct = {}

    # Small description
    yaml_struct['description'] = 'A parameter file for L-arginine phantom CEST-MRF experiment'

    # Water_pool
    yaml_struct['water_pool'] = {}
    yaml_struct['water_pool']['t1'] = cfg.water_t1
    yaml_struct['water_pool']['t2'] = cfg.water_t2
    yaml_struct['water_pool']['f'] = cfg.water_f

    # # Solute pool
    yaml_struct['cest_pool'] = {}
    yaml_struct['cest_pool']['Amine'] = {}

    yaml_struct['cest_pool']['Amine']['t1'] = cfg.cest_amine_t1
    yaml_struct['cest_pool']['Amine']['t2'] = cfg.cest_amine_t2
    yaml_struct['cest_pool']['Amine']['k'] = cfg.cest_amine_k
    yaml_struct['cest_pool']['Amine']['f'] = cfg.cest_amine_f
    yaml_struct['cest_pool']['Amine']['dw'] = cfg.cest_amine_dw

    yaml_struct['scale'] = cfg.magnetization_scale
    yaml_struct['reset_init_mag'] = cfg.magnetization_reset

    # Fill scanner info
    yaml_struct['b0']       = cfg.b0
    yaml_struct['gamma']    = cfg.gamma
    yaml_struct['b0_inhom'] = cfg.b0_inhom
    yaml_struct['rel_b1']   = cfg.rel_b1

    # Fill additional info
    yaml_struct['verbose'] = cfg.verbose
    yaml_struct['max_pulse_samples'] = cfg.max_pulse_samples

    # Write the file
    with open(yaml_fn, 'w') as file:
        yaml.safe_dump(yaml_struct, file, default_flow_style=None)
        
        
        
def write_yaml_3pool(cfg, yaml_fn=None):
    if yaml_fn is None:
        raise ValueError("yaml_fn must be defined")

    yaml_struct = {}

    # Small description
    yaml_struct['description'] = 'A parameter file for L-arginine phantom CEST-MRF experiment'

    # Water_pool
    yaml_struct['water_pool'] = {}
    yaml_struct['water_pool']['t1'] = cfg.water_t1
    yaml_struct['water_pool']['t2'] = cfg.water_t2
    yaml_struct['water_pool']['f'] = cfg.water_f

    # # Solute pool
    yaml_struct['cest_pool'] = {}
    yaml_struct['cest_pool']['Amine'] = {}

    yaml_struct['cest_pool']['Amine']['t1'] = cfg.cest_amine_t1
    yaml_struct['cest_pool']['Amine']['t2'] = cfg.cest_amine_t2
    yaml_struct['cest_pool']['Amine']['k'] = cfg.cest_amine_k
    yaml_struct['cest_pool']['Amine']['f'] = cfg.cest_amine_f
    yaml_struct['cest_pool']['Amine']['dw'] = cfg.cest_amine_dw

    # MT pull simulated as another CEST pool (full x,y,z components played out)
    yaml_struct['cest_pool']['mt_pool'] = {}

    yaml_struct['cest_pool']['mt_pool']['t1'] = cfg.cest_mt_t1
    yaml_struct['cest_pool']['mt_pool']['t2'] = cfg.cest_mt_t2
    yaml_struct['cest_pool']['mt_pool']['k'] = cfg.cest_mt_k
    yaml_struct['cest_pool']['mt_pool']['f'] = cfg.cest_mt_f
    yaml_struct['cest_pool']['mt_pool']['dw'] = cfg.cest_mt_dw
    

    yaml_struct['scale'] = cfg.magnetization_scale
    yaml_struct['reset_init_mag'] = cfg.magnetization_reset

    # Fill scanner info
    yaml_struct['b0']       = cfg.b0
    yaml_struct['gamma']    = cfg.gamma
    yaml_struct['b0_inhom'] = cfg.b0_inhom
    yaml_struct['rel_b1']   = cfg.rel_b1

    # Fill additional info
    yaml_struct['verbose'] = cfg.verbose
    yaml_struct['max_pulse_samples'] = cfg.max_pulse_samples

    # Write the file
    with open(yaml_fn, 'w') as file:
        yaml.safe_dump(yaml_struct, file, default_flow_style=None)

write_yaml = write_yaml_2pool # compatibility
        
