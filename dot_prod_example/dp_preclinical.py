import os
import time
import numpy as np
import pypulseq
import scipy.io as sio
from numpy import linalg as la

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from cest_mrf.write_scenario import write_yaml_2pool, write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching


import os
import sys
# Add the parent directory to sys.path to find the utils package
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.colormaps import b_viridis, b_winter  

from configs import ConfigPreclinical
from sequences import write_sequence_preclinical

def main():
    data_f = 'data'
    output_f = 'results'
    
    cfg = ConfigPreclinical().get_config()

    # Define output filenames
    yaml_fn = cfg['yaml_fn']
    seq_fn = cfg['seq_fn']
    dict_fn = cfg['dict_fn']

    # # Write the .yaml according to the config.py file (inside cest_mrf folder)
    # write_yaml_2pool(cfg, yaml_fn)
    write_yaml_dict(cfg, yaml_fn)

    # Write the seq file for a 2d experiment
    # for more info about the seq file, check out the pulseq-cest repository
    seq_defs = {}
    seq_defs['n_pulses'] = 1  # number of pulses
    seq_defs['tp'] = 3  # pulse duration [s]
    seq_defs['td'] = 0  # interpulse delay [s]
    seq_defs['Trec'] = 1  # delay before readout [s]
    seq_defs['Trec_M0'] = 'NaN'  # delay before m0 readout [s]
    seq_defs['M0_offset'] = 'NaN'  # dummy m0 offset [ppm]
    seq_defs['DCsat'] = seq_defs['tp'] / (seq_defs['tp'] + seq_defs['td'])  # duty cycle
    seq_defs['offsets_ppm'] = [3.0] * 30  # offset vector [ppm]
    seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])  # number of repetition
    seq_defs['Tsat'] = seq_defs['n_pulses'] * (seq_defs['tp'] + seq_defs['td']) - seq_defs['td']
    seq_defs['B0'] = cfg['b0']  # B0 [T]

    seqid = os.path.splitext(seq_fn)[1][1:]
    seq_defs['seq_id_string'] = seqid  # unique seq id

    # we vary B1 for the dictionary generation
    seq_defs['B1pa'] = [5, 5, 3, 3.75, 2.5, 1.75, 5.5, 6, 3.75,
                        5.75, 0.25, 3, 6, 4.5, 3.75, 3.5, 3.5, 0, 3.75, 6, 3.75, 4.75, 4.5,
                        4.25, 3.25, 5.25, 5.25, 0.25, 4.5, 5.25]

    # Create .seq file
    write_sequence_preclinical(seq_defs=seq_defs, seq_fn=seq_fn)

