import os
import re

import time
import numpy as np
import pypulseq
import scipy.io as sio
from numpy import linalg as la
import pydicom as dcm
from brukerapi.dataset import Dataset

from cest_mrf.write_scenario import write_yaml_2pool
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching

from my_funcs.cest_functions import bruker_dataset_creator
from my_funcs.cest_functions import dicom_data_arranger
from my_funcs.path_functions import make_folder

from my_funcs.cest_functions import z_spec_rearranger
from my_funcs.cest_functions import offset_rearranger

from my_funcs.plot_functions import mask_roi_finder

from my_funcs.b0_mapping_functions import wassr_b0_mapping
from my_funcs.b0_mapping_functions import b0_correction


""" Data to fill by user #1: """
class ConfigParams:
    # Scanner stats (maybe could be pulled from subject):
    b0 = 7  # [T]
    gyro_ratio_hz = 42.5764  # gamma for H [Hz/uT]
    gamma = 267.5153  # gamma (gyro_ratio_rad) for H [rad / uT], maybe change write_scenario later
    b0_inhom = 0
    rel_b1 = 1

    # Water_pool stats:
    water_t1 = np.arange(3500, 4700+100, 50) / 1000  # from t1 maps (pbs) 3500-5000
    water_t1 = water_t1.tolist()  # vary t2

    water_t2 = np.arange(400, 1900+100, 50) / 1000  # from t2 maps (pbs) 1100-1800
    water_t2 = water_t2.tolist()  # vary t2

    water_f = 1

    # Solute pool stats:
    cest_amine_t1 = [1000 / 1000]  # from ksw paper
    cest_amine_t2 = [40 / 1000]  # from ksw paper
    cest_amine_k = np.arange(3000, 12000+100, 100).tolist()
    cest_amine_dw = 3

    cest_amine_sol_conc = np.arange(12, 20 + 1, 1)
    cest_amine_protons = 3  # 3
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
    num_workers = 24  # 16


    def __init__(self, fp_prtcl_name):
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
        self.yaml_fn = os.path.join('exp', 'z_sim', fp_prtcl_name, 'scenario.yaml')
        self.seq_fn = os.path.join('exp', 'z_sim', fp_prtcl_name, 'acq_protocol.seq')
        self.dict_fn = os.path.join('exp', 'z_sim', fp_prtcl_name, 'dict.csv')
        # self.dict_fn = os.path.join('exp', 'feb', fp_prtcl_name, 'dict.mat')


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
    seq = pypulseq.Sequence()
    # the duration of the readout sequence:
    te = 20e-3
    imaging_delay = pypulseq.make_delay(te)
    imaging_pulse = pypulseq.make_block_pulse(seq_defs['FA'] * np.pi / 180, duration=2.1e-3)  # duration 2.1 or 3???
    pseudo_adc = pypulseq.make_adc(1, duration=1e-3)

    # saturation pulse
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]
    fa_sat = seq_defs['B1pa'] * gyro_ratio_rad * seq_defs['tp']  # flip angle of sat pulse

    # M0
    seq.add_block(pypulseq.make_delay(seq_defs['Trec_M0'] - te))  # - te?
    current_offset_hz = seq_defs['M0_offset'] * seq_defs['B0'] * gyro_ratio_hz
    sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
    seq.add_block(sat_pulse)
    seq.add_block(imaging_pulse)
    seq.add_block(imaging_delay)
    seq.add_block(pseudo_adc)

    for current_offset_ppm in seq_defs['offsets_ppm']:
        current_offset_hz = current_offset_ppm * seq_defs['B0'] * gyro_ratio_hz
        seq.add_block(sat_pulse)
        seq.add_block(imaging_delay)

        # Non-M0
        seq.add_block(pypulseq.make_delay(seq_defs['Trec'] - te))  # - te
        sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
        seq.add_block(sat_pulse)
        seq.add_block(imaging_pulse)
        seq.add_block(imaging_delay)
        seq.add_block(pseudo_adc)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.version_revision = 1  # compatibility with Matlab CEST-MRF code
    seq.write(seq_fn)


    # # >>> Gradients and scanner limits - see pulseq doc for more info
    # # Mostly relevant for clinical scanners
    # # lims =  mr.opts('MaxGrad',30,'GradUnit','mT/m',...
    # #     'MaxSlew',100,'SlewUnit','T/m/s', ...
    # #     'rfRingdownTime', 50e-6, 'rfDeadTime', 200e-6, 'rfRasterTime',1e-6)
    # # <<<
    #
    # # gamma
    # gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    # gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]
    #
    # # This is the info for the 2d readout sequence. As gradients etc ar
    # # simulated as delay, we can just add a delay after the imaging pulse for
    # # simulation which has the same duration as the actual sequence
    # # the flip angle of the readout sequence:
    # imaging_pulse = pypulseq.make_block_pulse(seq_defs['FA'] * np.pi / 180, duration=2.1e-3)  # duration 2.1 or 3???
    # # =='system', lims== should be added for clinical scanners
    #
    # # the duration of the readout sequence:
    # te = 20e-3
    # imaging_delay = pypulseq.make_delay(te)
    #
    # # init sequence
    # # seq = SequenceSBB()
    # seq = pypulseq.Sequence()
    # # seq = SequenceSBB(lims) for clinical scanners
    #
    # if not remove_first_image_flag:
    #     # saturation pulse
    #     current_offset_ppm = seq_defs['M0_offset']
    #     B1 = seq_defs['B1pa'][0]
    #     current_offset_hz = current_offset_ppm * seq_defs['B0'] * gyro_ratio_hz
    #     fa_sat = B1 * gyro_ratio_rad * seq_defs['tp']  # flip angle of sat pulse
    #
    #     sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
    #     seq.add_block(sat_pulse)
    #     # Imaging pulse
    #     seq.add_block(imaging_pulse)
    #     seq.add_block(imaging_delay)  # ???
    #     # seq.add_pseudo_ADC_block()  # additional feature of the SequenceSBB class
    #     pseudo_adc = pypulseq.make_adc(1, duration=1e-3)
    #     seq.add_block(pseudo_adc)
    #     seq.add_block(pypulseq.make_delay(seq_defs['Trec_M0'] - te))  # net recovery time
    #
    # # Loop b1s
    # for idx, B1 in enumerate(seq_defs['B1pa']):
    #
    #     if idx > 0:  # add relaxation block after first measurement
    #         seq.add_block(pypulseq.make_delay(seq_defs['Trec'] - te))  # net recovery time
    #
    #     # saturation pulse
    #     current_offset_ppm = seq_defs['offsets_ppm'][idx]
    #     current_offset_hz = current_offset_ppm * seq_defs['B0'] * gyro_ratio_hz
    #     fa_sat = B1 * gyro_ratio_rad * seq_defs['tp']  # flip angle of sat pulse
    #
    #     # add pulses
    #     for n_p in range(seq_defs['n_pulses']):
    #         # If B1 is 0 simulate delay instead of a saturation pulse
    #         if B1 == 0:
    #             seq.add_block(pypulseq.make_delay(seq_defs['tp']))  # net recovery time
    #         else:
    #             sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
    #             # =='system', lims== should be added for clinical scanners
    #             seq.add_block(sat_pulse)
    #         # delay between pulses
    #         if n_p < seq_defs['n_pulses'] - 1:
    #             seq.add_block(pypulseq.make_delay(seq_defs['td']))
    #     # Spoiling
    #     # additional feature of the SequenceSBB class
    #     # seq.add_spoiler_gradients(lims)
    #
    #     # Imaging pulse
    #     seq.add_block(imaging_pulse)
    #     seq.add_block(imaging_delay)
    #     # seq.add_pseudo_ADC_block()  # additional feature of the SequenceSBB class
    #     pseudo_adc = pypulseq.make_adc(1, duration=1e-3)
    #     seq.add_block(pseudo_adc)
    #
    # def_fields = seq_defs.keys()
    # for field in def_fields:
    #     seq.set_definition(field, seq_defs[field])
    #
    # seq.version_revision = 1  # compatibility with Matlab CEST-MRF code
    # seq.write(seq_fn)


""" Whole process and run as main: """
def main():
    """ Data to fill by user #2: """
    # Root stats:
    general_fn = os.path.abspath(os.curdir)

    glu_phantom_fns = [os.path.join(general_fn, 'scans', '23_12_08_glu_phantom_37deg',
                                    '1_glu_phantom_12_16_20mM_ph7_37deg')]  # December 23

    # Protocol stats:
    fp_prtcl_names = ['EPI_0p7uT', 'EPI_2uT', 'EPI_4uT', 'EPI_6uT']
    fp_prtcl_names = ['EPI_4uT', 'EPI_6uT']

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
            cfg = ConfigParams(fp_prtcl_name)

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
            seq_defs['Trec'] = 5    # delay before readout [s] (TR-Tsat)
            seq_defs['Trec_M0'] = 5  # delay before m0 readout [s]
            seq_defs['M0_offset'] = (bruker_dataset['SatFreqList'].value[0] /
                                     (gyro_ratio_hz * b0))                       # dummy m0 offset [ppm]
            seq_defs['DCsat'] = (seq_defs['tp'] /
                                 (seq_defs['tp'] + seq_defs['td']))              # duty cycle (1)
            seq_defs['B1pa'] = bruker_dataset['PVM_MagTransPower'].value  # B1
            seq_defs['offsets_ppm'] = (bruker_dataset['SatFreqList'].value[1:] /
                                       (gyro_ratio_hz * b0))                     # offset vector [ppm]

            # seq_defs['offsets_ppm'] = np.round((bruker_dataset['SatFreqList'].value[1:] /
            #                            (gyro_ratio_hz * b0)) , decimals=1) # here!!!
            # seq_defs['offsets_ppm'] = np.array([7., -7., 6.75, -6.75, 6.5, -6.5, 6.25, -6.25, 6.,
            #        -6., 5.75, -5.75, 5.5, -5.5, 5.25, -5.25, 5., -5.,
            #        4.75, -4.75, 4.5, -4.5, 4.25, -4.25, 4., -4., 3.75,
            #        -3.75, 3.5, -3.5, 3.25, -3.25, 3., -3., 2.75, -2.75,
            #        2.5, -2.5, 2.25, -2.25, 2., -2., 1.75, -1.75, 1.5,
            #        -1.5, 1.25, -1.25, 1., -1., 0.75, -0.75, 0.5, -0.5,
            #        0.25, -0.25, 0.])
            seq_defs['offsets_ppm'] = np.arange(7, -7.25, -0.25)

            seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])                  # number of repetition
            seq_defs['Tsat'] = (seq_defs['n_pulses'] *
                                (seq_defs['tp'] + seq_defs['td']) -
                                seq_defs['td'])                                  # (3 [s])
            seq_defs['B0'] = b0                                                  # B0 [T]
            seq_defs['FA'] = 90  # FA, I don't actually know

            seqid = os.path.splitext(seq_fn)[1][1:]
            seq_defs['seq_id_string'] = seqid  # unique seq id


            # Create .seq file
            write_sequence(seq_defs=seq_defs, seq_fn=seq_fn, remove_first_image_flag=remove_first_image_flag)

            start = time.perf_counter()
            generate_mrf_cest_dictionary(seq_fn=seq_fn, param_fn=yaml_fn, dict_fn=dict_fn, num_workers=cfg.num_workers,
                                         axes='xy')  # axes can also be 'z' if no readout is simulated
            end = time.perf_counter()
            s = (end - start)
            print(f"Dictionary simulation and preparation took {s:.03f} s.")

            # start = time.perf_counter()
            # quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acquired_data_fn)  # I changed it!!!
            # end = time.perf_counter()
            # s = (end - start)
            # print(f"Dot product matching took {s:.03f} s.")
            #
            # # save acquired data to: root->scans->date->subject->E->mrf_files->quant_maps.mat
            # quant_maps_fn = os.path.join(glu_phantom_mrf_files_fn, 'quant_maps.mat')
            # sio.savemat(quant_maps_fn, quant_maps)
            # print('quant_maps.mat saved')

            print(f'------------------------------ end of protocol {fp_prtcl_name} ------------------------------')

        print(f'################################# end of phantom {phantom_i + 1} #################################')


if __name__ == "__main__":
    main()
