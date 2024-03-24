from sequences import create_seq_defs_unsupervised
import os
import numpy as np


def main():
    seq_fn = 'mt_unsuper'
    fa = 90 

    seq_defs = create_seq_defs_unsupervised()
    fn = f'{seq_fn}.txt'
    TR = seq_defs['tsat'] + seq_defs['trec']
    
    with open(fn, 'w') as file:
        file.write(f'{len(seq_defs["b1"])}\n')
    
    for i, off in enumerate(seq_defs["offsets_ppm"]):

        line = f"{TR[i]*1000:.0f} {seq_defs['b1'][i] :.2f} {off} {fa} {seq_defs['tsat'][i]*1000:.0f}\n"

        with open(fn, 'a') as file:
            file.write(line)


if __name__ == '__main__':
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)

    main()