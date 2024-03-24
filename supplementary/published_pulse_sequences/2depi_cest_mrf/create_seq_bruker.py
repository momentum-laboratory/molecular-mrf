import numpy as np
import pypulseq as pp
from write_sequence import write_clinical_sequence_gaussian
import os 


def main():
    B1 = [2, 2, 1.7, 1.5, 1.2, 1.2, 3, 0.5, 3, 1, 2.2, 3.2, 1.5, 0.7, 1.5, 2.2, 2.5, 1.2, 3, 0.2, 1.5, 2.5, 0.7, 4,
            3.2, 3.5, 1.5, 2.7, 0.7, 0.5]
    N = len(B1)
    ppm = [3.5] * N
    TR = [3.5] * N
    Tsat = [2.560] * N

    ppm = np.array(ppm)
    TR = np.array(TR)
    Tsat = np.array(Tsat)
    B1 = np.array(B1)

    seq_fn = '2D_EPI_CEST.txt'

    seq_defs = {}

    seq_defs["num_meas"] = N  # number of repetition

    seq_defs["tsat"] = Tsat  # saturation time [s]
    seq_defs["trec"] = TR - seq_defs["tsat"]  # net recovery time [s]

    seqid = os.path.splitext(seq_fn)[1][1:]
    seq_defs['seq_id_string'] = seqid  # unique seq id

    seq_defs['b1pa'] = B1
    seq_defs["b1rms"] = seq_defs["b1pa"]

    seq_defs["offsets_ppm"] = ppm  # offset list [ppm]

    
    fa = 90
    with open(seq_fn, 'w') as file:
        file.write(f'{len(B1)}\n')
    for i, off in enumerate(seq_defs["offsets_ppm"]):
        line = f"{TR[i] * 1000:.0f} {seq_defs['b1pa'][i] :.2f} {off} {fa} {seq_defs['tsat'][i] * 1000:.0f}\n"

        with open(seq_fn, 'a') as file:
            file.write(line)
    



if __name__ == "__main__":
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)
    
    main()