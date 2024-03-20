import numpy as np
import os


def main():
    # sequence definitions from https://github.com/kherz/pulseq-cest-library/blob/master/seq-library/APTw_3T_000_2uT_1block_2s_braintumor/APTw_3T_000_2uT_1block_2s_braintumor.py
    defs: dict = {}
    defs["b1pa"] = 2.0  # B1 peak amplitude [µT] (b1rms calculated below)
    defs["b0"] = 3  # B0 [T]
    defs["n_pulses"] = 1  # number of pulses  #
    defs["tp"] = 2  # pulse duration [s]
    defs["td"] = 0  # interpulse delay [s]
    defs["trec"] = 3.5  # recovery time [s]
    defs["trec_m0"] = 3.5  # recovery time before M0 [s]
    defs["m0_offset"] = -300  # m0 offset [ppm]
    defs["offsets_ppm"] = np.append(defs["m0_offset"], np.linspace(-4, 4, 33))  # offset vector [ppm]

    defs["dcsat"] = (defs["tp"]) / (defs["tp"] + defs["td"])  # duty cycle
    defs["num_meas"] = defs["offsets_ppm"].size  # number of repetition
    defs["tsat"] = defs["n_pulses"] * (defs["tp"] + defs["td"]) - defs["td"]  # saturation time [s]


    TR = defs['tp'] + defs['trec']
    TR = TR * 1e3

    Tsat = defs["tp"]*1000

    fa = 90

    f_name = 'cest-weighted.txt'

    with open(f_name, 'w') as file:
        file.write(f'{len(defs["offsets_ppm"] )}\n')

    for i, off in enumerate(defs["offsets_ppm"]):

        line = f'{TR:.0f} {defs["b1pa"] :.2f} {off} {fa} {Tsat:.0f}\n'

        with open(f_name, 'a') as file:
            file.write(line)

if __name__ == "__main__":
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)
    
    main()
