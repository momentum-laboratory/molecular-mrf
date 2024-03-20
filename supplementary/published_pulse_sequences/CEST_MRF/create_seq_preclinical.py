from sequences import write_sequence_preclinical, setup_sequence_definitions_preclinical
import os


def main():
    seq_fn = 'cest_mrf_preclinical.seq'
    b0 = 9.4
    b1 = [5, 5, 3, 3.75, 2.5, 1.75, 5.5, 6, 3.75,
                5.75, 0.25, 3, 6, 4.5, 3.75, 3.5, 3.5, 0, 3.75, 6, 3.75, 4.75, 4.5,
                4.25, 3.25, 5.25, 5.25, 0.25, 4.5, 5.25]
                
    seq_defs = setup_sequence_definitions_preclinical(b0, b1, seq_fn)
    write_sequence_preclinical(seq_defs, seq_fn)

if __name__ == '__main__':
    # change folder to script directory if needed
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if current_directory != script_directory:
        os.chdir(script_directory)

    main()