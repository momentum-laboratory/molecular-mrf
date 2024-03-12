import numpy as np

class config_params():
    yaml_fn = 'scenario.yaml'
    seq_fn = 'acq_protocol.seq'
    dict_fn = 'dict.mat'
    # Water_pool
    water_t1 = np.arange(2500, 3350, 50) / 1000

    water_t1 = water_t1.tolist()  # vary t1

    water_t2 = np.arange(600, 1250, 50) / 1000

    water_t2 = water_t2.tolist()  # vary t2
    water_f = 1

    # Solute pool
    cest_amine_t1 = [2800 / 1000]  # fix solute t1
    cest_amine_t2 = [40 / 1000]  # fix solute t2
    cest_amine_k = np.arange(100, 1410, 10).tolist()  # vary solute exchange rate

    cest_amine_dw = 3  # fixed solute exchange rate at 3 ppm

    # solute concentration * protons / water concentration
    cest_amine_sol_conc = np.arange(10, 125, 5)  # solute concentration

    cest_amine_protons = 3
    cest_amine_water_conc = 110000
    cest_amine_f = cest_amine_sol_conc * cest_amine_protons / cest_amine_water_conc
    cest_amine_f = cest_amine_f.tolist()
    # cest_amine_f = [2.7273E-04]
    # Fill initial magnetization info
    # this is important now for the mrf simulation! For the regular pulseq-cest
    # simulation, we usually assume athat the magnetization reached a steady
    # state after the readout, which means we can set the magnetization vector
    # to a specific scale, e.g. 0.5. This is because we do not simulate the
    # readout there. For mrf we include the readout in the simulation, which
    # means we need to carry the same magnetization vector through the entire
    # sequence. To avoid that the magnetization vector gets set to the initial
    # value after each readout, we need to set reset_init_mag to false
    magnetization_scale = 1
    magnetization_reset = 0

    # Fill scanner info
    b0 = 9.4 # [T]
    gamma = 267.5153  # [rad / uT]
    b0_inhom = 0
    rel_b1 = 1

    # Fill additional info
    verbose = 0  # no unneccessary console log wished
    max_pulse_samples = 100  # block pulses are only 1 sample anyways

    num_workers = 16


