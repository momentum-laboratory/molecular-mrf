import os

import time
import numpy as np
import pypulseq

from cest_mrf.write_scenario import write_yaml_2pool
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary_4pool  # generate_mrf_cest_dictionary,

from my_funcs.cest_functions import bruker_dataset_creator

""" Data to fill by user #1: """
class ConfigParams:
    # Scanner stats (maybe could be pulled from subject):
    b0 = 7  # [T]
    gyro_ratio_hz = 42.5764  # gamma for H [Hz/uT]
    gamma = 267.5153  # gamma (gyro_ratio_rad) for H [rad / uT], maybe change write_scenario later
    b0_inhom = 0
    rel_b1 = 1

    # Water_pool
    water_t1 = np.arange(1300, 2300 + 100, 100) / 1000
    water_t1 = water_t1.tolist()  # vary t1

    water_t2 = np.arange(40, 110 + 10, 10) / 1000
    water_t2 = water_t2.tolist()  # vary t2

    water_f = 1

    # Solute pool stats:
    cest_amine_t1 = [0 / 1000]  # fix solute t1 (copies water t1s)
    cest_amine_t2 = [0.04 / 1000]  # from ksw paper
    cest_amine_k = np.arange(5, 100 + 5, 5).tolist()
    cest_amine_dw = -2.5

    cest_amine_sol_conc = np.arange(2e3, 30e3 + 2e3, 2e3)
    cest_amine_protons = 1
    cest_amine_water_conc = 110000
    cest_amine_f = (cest_amine_sol_conc * cest_amine_protons
                    / cest_amine_water_conc)  # solute concentration * protons / # water concentration
    cest_amine_f = cest_amine_f.tolist()

    # Fill initial magnetization info
    # this is important now for the mrf simulation! For the regular pulseq-cest
    # simulation, we usually assume that the magnetization reached a steady
    # state after the readout, which means we can set the magnetization vector
    # to a specific scale, e.g. 0.5. This is because we do not simulate the
    # readout there. For mrf we include the readout in the simulation, which
    # means we need to carry the same magnetization vector through the entire
    # sequence. To avoid that the magnetization vector gets set to the initial
    # value after each readout, we need to set reset_init_mag to false
    magnetization_scale = 1
    magnetization_reset = 0

    # Additional (run) stats:
    verbose = 0  # no unneccessary console log wished
    max_pulse_samples = 100  # block pulses are only 1 sample anyways
    num_workers = 16  # 16


    def __init__(self, fp_prtcl_name, general_fn):
        """
        Create ConfigParams paths
        :param fp_prtcl_name: the protocol
        :return: yaml_fn: the yaml file path, root->exp->month->yaml
        :return: seq_fn: the seq file path,  root->exp->month->seq
        :return: dict_fn: the dict file path,  root->exp->month->dict
        """
        # Initialize attributes using input values
        self.fp_prtcl_name = fp_prtcl_name

        # Automatically Initialize (Path saver stats)
        dict_name_category = 'glu_50'
        dict_folder = os.path.join(general_fn, 'exp', '4pool', dict_name_category, 'mt', fp_prtcl_name)
        if not os.path.exists(dict_folder):
            os.makedirs(dict_folder)
        self.yaml_fn = os.path.join(dict_folder, f'scenario.yaml')
        self.seq_fn = os.path.join(dict_folder, f'acq_protocol.seq')
        self.dict_fn = os.path.join(dict_folder, f'dict.pkl')  # pickle format!

""" Dictionary creation: """
def write_sequence(seq_defs: dict = None,
                   seq_fn: str = 'protocol.seq',
                   remove_first_image_flag: bool = False):
    """
    Create sequence for CEST
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :param remove_first_image_flag: if 1, remove
    :return: sequence object
    """

    # >>> Gradients and scanner limits - see pulseq doc for more info
    # Mostly relevant for clinical scanners
    # lims =  mr.opts('MaxGrad',30,'GradUnit','mT/m',...
    #     'MaxSlew',100,'SlewUnit','T/m/s', ...
    #     'rfRingdownTime', 50e-6, 'rfDeadTime', 200e-6, 'rfRasterTime',1e-6)
    # <<<

    # gamma
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]

    # This is the info for the 2d readout sequence. As gradients etc ar
    # simulated as delay, we can just add a delay after the imaging pulse for
    # simulation which has the same duration as the actual sequence
    # the flip angle of the readout sequence:
    imaging_pulse = pypulseq.make_block_pulse(seq_defs['FA'] * np.pi / 180, duration=2.1e-3)  # duration 2.1 or 3???
    # =='system', lims== should be added for clinical scanners

    # the duration of the readout sequence:
    te = 20e-3
    imaging_delay = pypulseq.make_delay(te)

    # init sequence
    # seq = SequenceSBB()
    seq = pypulseq.Sequence()
    # seq = SequenceSBB(lims) for clinical scanners

    if not remove_first_image_flag:
        # M0 pulse if required (when first image is #not# removed!)
        seq.add_block(pypulseq.make_delay(seq_defs['Trec_M0']))  # M0 delay
        seq.add_block(imaging_pulse)  # add imaging block
        # seq.add_block(imaging_delay)
        pseudo_adc = pypulseq.make_adc(1, duration=1e-3)
        seq.add_block(pseudo_adc)
        # add relaxation block after M0 measurement
        seq.add_block(pypulseq.make_delay(seq_defs['Trec_M0'] - te))  # net recovery time

    # Loop b1s
    for idx, B1 in enumerate(seq_defs['B1pa']):

        if idx > 0:  # add relaxation block after first measurement
            seq.add_block(pypulseq.make_delay(seq_defs['Trec'] - te))  # net recovery time

        # saturation pulse
        current_offset_ppm = seq_defs['offsets_ppm'][idx]
        current_offset_hz = current_offset_ppm * seq_defs['B0'] * gyro_ratio_hz
        fa_sat = B1 * gyro_ratio_rad * seq_defs['tp']  # flip angle of sat pulse

        # add pulses
        for n_p in range(seq_defs['n_pulses']):
            # If B1 is 0 simulate delay instead of a saturation pulse
            if B1 == 0:
                seq.add_block(pypulseq.make_delay(seq_defs['tp']))  # net recovery time
            else:
                sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
                # =='system', lims== should be added for clinical scanners
                seq.add_block(sat_pulse)
            # delay between pulses
            if n_p < seq_defs['n_pulses'] - 1:
                seq.add_block(pypulseq.make_delay(seq_defs['td']))
        # Spoiling
        # additional feature of the SequenceSBB class
        # seq.add_spoiler_gradients(lims)

        # Imaging pulse
        seq.add_block(imaging_pulse)
        seq.add_block(imaging_delay)
        # seq.add_pseudo_ADC_block()  # additional feature of the SequenceSBB class
        pseudo_adc = pypulseq.make_adc(1, duration=1e-3)
        seq.add_block(pseudo_adc)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.version_revision = 1  # compatibility with Matlab CEST-MRF code
    seq.write(seq_fn)


""" Whole process and run as main: """
def main():
    """ Data to fill by user #2: """
    # Root stats:
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Navigate up two directory level
    general_fn = os.path.join(parent_dir, 'data')

    # Subject stats:
    glu_phantom_fns = [os.path.join(general_fn, 'scans', '24_04_04_glu_tumor_mouse_37deg',
                                    '1_mouse1_left')]  # Mouse

    # Protocol stats:
    fp_prtcl_names = ['MT52']

    remove_first_image_flag = False  # False=not removed M0 (31) / True=removed M0 (30)

    """ Data to fill by user #2 end: """

    for phantom_i in range(len(glu_phantom_fns)):
        glu_phantom_fn = glu_phantom_fns[phantom_i]
        txt_file_name = 'labarchive_notes.txt'

        print(f'################################# start of phantom {phantom_i+1} #################################')
        for fp_prtcl_name in fp_prtcl_names:
            print(f'------------------------------ start of protocol {fp_prtcl_name} ------------------------------')
            """ Acquired scan arrangement: """
            glu_phantom_dicom_fn, glu_phantom_mrf_files_fn, bruker_dataset = \
                bruker_dataset_creator(glu_phantom_fn, txt_file_name, fp_prtcl_name)

            """ Dictionary creation: """
            cfg = ConfigParams(fp_prtcl_name, general_fn)

            # Define output filenames
            yaml_fn = cfg.yaml_fn
            seq_fn = cfg.seq_fn
            dict_fn = cfg.dict_fn

            # Define scanner stats:
            gyro_ratio_hz = cfg.gyro_ratio_hz
            b0 = cfg.b0

            # Write the .yaml according to the config.py file
            write_yaml_2pool(cfg, yaml_fn)

            # Write the seq file for a 2d experiment
            # for more info about the seq file, check out the pulseq-cest repository
            seq_defs = {}
            seq_defs['n_pulses'] = 1                                             # number of pulses (1 in preclinical)
            seq_defs['tp'] = (bruker_dataset['PVM_MagTransPulse1'].value[0]
                              / 1000)
            seq_defs['td'] = 0                                                   # interpulse delay [s] (0 in preclinical)
            seq_defs['Trec'] = (bruker_dataset['Fp_TRs'].value[-1] -
                                bruker_dataset['Fp_SatDur'].value[-1]) / 1000    # delay before readout [s] (TR-Tsat)
            seq_defs['Trec_M0'] = (bruker_dataset['Fp_TRs'].value[0] -
                                   bruker_dataset['Fp_SatDur'].value[0]) / 1000  # delay before m0 readout [s]
            seq_defs['M0_offset'] = (bruker_dataset['Fp_SatOffset'].value[0] /
                                     (gyro_ratio_hz * b0))                       # dummy m0 offset [ppm]
            seq_defs['DCsat'] = (seq_defs['tp'] /
                                 (seq_defs['tp'] + seq_defs['td']))              # duty cycle (1)
            seq_defs['offsets_ppm'] = (bruker_dataset['Fp_SatOffset'].value[1:] /
                                       (gyro_ratio_hz * b0))                     # offset vector [ppm]

            # seq_defs['offsets_ppm'] = np.round((bruker_dataset['Fp_SatOffset'].value[1:] /
            #                                     (gyro_ratio_hz * b0)), decimals=1) # here!!!

            seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])                  # number of repetition
            seq_defs['Tsat'] = (seq_defs['n_pulses'] *
                                (seq_defs['tp'] + seq_defs['td']) -
                                seq_defs['td'])                                  # (3 [s])
            seq_defs['B0'] = b0                                                  # B0 [T]
            seq_defs['FA'] = int(bruker_dataset['Fp_FlipAngle'].value[0])             # FA

            seqid = os.path.splitext(seq_fn)[1][1:]
            seq_defs['seq_id_string'] = seqid  # unique seq id

            # B1 variation
            seq_defs['B1pa'] = bruker_dataset['Fp_SatPows'].value[1:]

            # Create .seq file
            write_sequence(seq_defs=seq_defs, seq_fn=seq_fn, remove_first_image_flag=remove_first_image_flag)

            start = time.perf_counter()
            generate_mrf_cest_dictionary_4pool(seq_fn=seq_fn, param_fn=yaml_fn, dict_fn=dict_fn, num_workers=cfg.num_workers,
                                         axes='xy', add_iter=2)  # axes can also be 'z' if no readout is simulated
            # equals=[['tw1', 'ts1_0']]
            end = time.perf_counter()
            s = (end - start)
            print(f"Dictionary simulation and preparation took {s:.03f} s.")

            print(f'------------------------------ end of protocol {fp_prtcl_name} ------------------------------')

        print(f'################################# end of phantom {phantom_i + 1} #################################')


if __name__ == "__main__":
    main()
