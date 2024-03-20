from sequences import write_sequence_clinical, setup_sequence_definitions_clinical
import os


def main():
    seq_fn = 'cest_mrf_clinical.seq'
    freq = 127.7292
    gamma = 267.5153
    b1 = [5, 5, 3, 3.75, 2.5, 1.75, 5.5, 6, 3.75,
                5.75, 0.25, 3, 6, 4.5, 3.75, 3.5, 3.5, 0, 3.75, 6, 3.75, 4.75, 4.5,
                4.25, 3.25, 5.25, 5.25, 0.25, 4.5, 5.25]
    type_s = 'scanner' # type of sequence can be simulator or scanner

    seq_defs, lims = setup_sequence_definitions_clinical(b1, seq_fn, gamma, freq)
    write_sequence_clinical(seq_defs, seq_fn, lims, type=type_s)


if __name__ == '__main__':
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)

    main()