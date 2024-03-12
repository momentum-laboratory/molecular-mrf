from bmctool.params import Params

from BMCSimulator import BMCSimulator
from BMCSimulator import SimulationParameters, WaterPool, MTPool, CESTPool
from BMCSimulator import Lorentzian, SuperLorentzian, NoLineshape

from .SimulationParametersMRF import ParamsMRF

import numpy as np

# Function based on pypulseq-cest function from
# https://github.com/KerstinKaspar/pypulseq-cest/blob/main/pypulseq_cest/parser.py
# Function updated to be compatible with new version of pypulse-cest
# InitMagnetizationVectors -> SetInitialMagnetizationVector
# And it can be used for updating of existing SimulationParameters object
def parse_params(sp: Params,
                 sp_sim: SimulationParameters = None) -> SimulationParameters:
    """
    parsing python parameters into the according C++ functions
    :param sp: simulation parameter object
    :param seq_file: location of the seq-file to simulate
    :return: SWIG object for C++ object handling
    """
    lineshape_map = {'Lorentzian': Lorentzian, 'SuperLorentzian': SuperLorentzian, None: NoLineshape}
    # Make it updatable
    if sp_sim == None:
        sp_sim = SimulationParameters()

    # init magnetization vector
    # sp_sim.InitMagnetizationVectors(sp.m_vec, get_num_adc_events(seq_file=seq_file))
    sp_sim.SetInitialMagnetizationVector(sp.m_vec)

    # construct water pool
    water_pool = WaterPool(sp.water_pool['r1'], sp.water_pool['r2'], sp.water_pool['f'])
    sp_sim.SetWaterPool(water_pool)
    if sp.mt_pool:
        try:
            lineshape = lineshape_map[sp.mt_pool['lineshape']]
        except KeyError:
            print(sp.mt_pool['lineshape'] + ' is not a valid lineshape for MT Pool. Use NoLineshape')
            lineshape = NoLineshape
        mt_pool = MTPool(sp.mt_pool['r1'], sp.mt_pool['r2'], sp.mt_pool['f'], sp.mt_pool['dw'], sp.mt_pool['k'], lineshape)
        sp_sim.SetMTPool(mt_pool)
    # sp_sim.InitCESTPoolMemory(len(sp.cest_pools))
    sp_sim.SetNumberOfCESTPools(len(sp.cest_pools))
    for i in range(len(sp.cest_pools)):
        cest_pool = CESTPool(sp.cest_pools[i]['r1'], sp.cest_pools[i]['r2'], sp.cest_pools[i]['f'], sp.cest_pools[i]['dw'], sp.cest_pools[i]['k'])
        sp_sim.SetCESTPool(cest_pool, i)
    sp_sim.InitScanner(sp.scanner['b0'], sp.scanner['rel_b1'], sp.scanner['b0_inhomogeneity'], sp.scanner['gamma'])
    if 'verbose' in sp.options.keys():
        sp_sim.SetVerbose(sp.options['verbose'])
    if 'reset_init_mag' in sp.options.keys():
        sp_sim.SetUseInitMagnetization(sp.options['reset_init_mag'])
    if 'max_pulse_samples' in sp.options.keys():
        sp_sim.SetMaxNumberOfPulseSamples(sp.options['max_pulse_samples'])
    return sp_sim


def simulate_mrf(dictionary: dict,
                 options: dict,
                 seq_file: str = None,
                 id_num: int = 0,
                 axes: str = None
                 ) -> (ParamsMRF, list, int):
    sim_params = ParamsMRF()
    sim_params.set_params_dict(dictionary, options)
    n_cest = sim_params.n_cest_pools

    idx = range(sim_params.num_comb)

    sp = parse_params(sim_params[idx[0]]) # create pointer to SimulationParameters object
    sf = BMCSimulator(sp, seq_file) # link the SP object with BMCSim

    # start = time.perf_counter()
    signal = []
    for i in idx:
        # update_params(sim_params[i], sp) # update data in the original object
        parse_params(sim_params[i], sp) # update data in the original object

        # start = time.perf_counter()

        # Output e.g. for three pools is the following vector:
        # [MxA, MxB, MxD, MyA, MyB, MyD, MzA, MzB, MzD, MzC]
        # with A: water pool, B: 1st CEST pool, D: 2nd CEST pool, C: MT pool
        m_out = sf.RunSimulation()

        # print(f"One simulation took {time.perf_counter() - start:.06f} s.")
        if axes.lower() == 'xy':
            mx_water = m_out[0,:]
            my_water = m_out[n_cest+1,:]
            s = np.sqrt(mx_water**2 + my_water**2)
        elif axes.lower() == 'z':
            s = m_out[(n_cest+1)*2]
        else:
            raise AttributeError(f'{axes} unknown axes parameter')
        signal.append(s)

        # if (i+1)%10000 == 0:
        #     # TODO: implement checkpoint saving in a pickle file to have possibility of interrupting
        #     print(f'Worker {id_num}: {i+1} combinations are calculated, {time.perf_counter() - start:.03f} s elapsed')
        #     start = time.perf_counter()
    return sim_params, signal, id_num
