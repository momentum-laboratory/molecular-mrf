from sequences import create_seq_defs_loas, write_sequence_clinical
import os
import numpy as np


def main():
    lens = [10, 40]
    seq_fn = 'loas_mtc'
    fa = 90 
    type_s = 'scanner'

    for l in lens:
        seq_defs, lims = create_seq_defs_loas(l, clinical=True)
        fn = f'{seq_fn}_{l}.seq'
        TR = seq_defs['tsat'] + seq_defs['trec']
        seq_defs['seq_id_string'] = f'{seq_fn}_{l}'
        write_sequence_clinical(seq_defs, fn, lims, type=type_s)


if __name__ == '__main__':
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)

    main()