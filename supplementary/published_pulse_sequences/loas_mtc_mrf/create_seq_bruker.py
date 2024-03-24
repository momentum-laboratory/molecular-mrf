from sequences import create_seq_defs_loas
import os
import numpy as np

def main():
    lens = [10, 40]
    seq_fn = 'loas_mtc'
    fa = 90 

    for l in lens:
        seq_defs = create_seq_defs_loas(l)
        fn = f'{seq_fn}_{l}.txt'
        TR = np.array(seq_defs['tsat']) + np.array(seq_defs['trec'])

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