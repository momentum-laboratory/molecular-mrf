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

    gamma = 267.5153  # [rad / uT]
    seq_fn = '16msGaussianDRONE.seq'
    type_s = 'scanner' # type of sequence can be simulator or scanner

    seq_defs = {}

    seq_defs["num_meas"] = N  # number of repetition

    seq_defs["tp"] = 16e-3  # pulse duration [s]
    seq_defs["td"] = 0  # interpulse delay [s]
    seq_defs["dcsat"] = (seq_defs["tp"]) / (seq_defs["tp"] + seq_defs["td"])  # duty cycle

    seq_defs["n_pulses"] = np.ceil(Tsat / (seq_defs["tp"] + seq_defs["td"])).astype(int)  # number of pulses

    seq_defs["tsat"] = Tsat  # saturation time [s]
    seq_defs["trec"] = TR - seq_defs["tsat"]  # net recovery time [s]

    seq_defs["spoiling"] = True

    seqid = os.path.splitext(seq_fn)[1][1:]
    seq_defs['seq_id_string'] = seqid  # unique seq id

    seq_defs['b1pa'] = B1
    seq_defs["b1rms"] = seq_defs["b1pa"]

    seq_defs["offsets_ppm"] = ppm  # offset list [ppm]

    lims = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        gamma=gamma/2/np.pi*1e6,
    )

    seq_defs["gamma_hz"] = lims.gamma * 1e-6
    seq_defs["freq"] = 127.7292
    seq_defs['b0'] = seq_defs['freq'] / seq_defs["gamma_hz"]

    write_clinical_sequence_gaussian(seq_defs, seq_fn, lims, type = type_s)

if __name__ == "__main__":
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)
    
    main()