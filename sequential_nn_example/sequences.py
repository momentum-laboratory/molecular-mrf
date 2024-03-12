import pypulseq as pp
import numpy as np

def write_sequence(seq_defs: dict = None,
                   seq_fn: str = 'protocol.seq'):
    """
    Create sequence for CEST
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :return: sequence object
    """

    # gamma
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]

    imaging_pulse = pp.make_block_pulse(60 * np.pi / 180, duration=2.1e-3)

    # the duration of the readout sequence:
    te = 20e-3
    imaging_delay = pp.make_delay(te)

    pseudo_adc = pp.make_adc(1, duration=1e-3)

    # init sequence
    seq = pp.Sequence()

    # Loop b1s
    for idx, B1 in enumerate(seq_defs['B1pa']):

        if idx > 0:  # add relaxtion block after first measurement
            seq.add_block(pp.make_delay(seq_defs['Trec'][idx] - te))  # net recovery time

        # saturation pulse
        # current_offset_ppm = seq_defs['offsets_ppm'][idx]
        current_offset_hz = seq_defs['offsets_ppm'][idx] * seq_defs['B0'] * gyro_ratio_hz
        fa_sat = B1 * gyro_ratio_rad * seq_defs['tp']  # flip angle of sat pulse

        # add pulses
        for n_p in range(seq_defs['n_pulses']):
            # If B1 is 0 simulate delay instead of a saturation pulse
            if B1 == 0:
                seq.add_block(pp.make_delay(seq_defs['tp']))  # net recovery time
            else:
                sat_pulse = pp.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
                # =='system', lims== should be added for clinical scanners
                seq.add_block(sat_pulse)
            # delay between pulses
            if n_p < seq_defs['n_pulses'] - 1:
                seq.add_block(pp.make_delay(seq_defs['td']))
            
        # Imaging pulse
        seq.add_block(imaging_pulse)
        seq.add_block(imaging_delay)

        seq.add_block(pseudo_adc)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.version_revision = 1  # compatibility with Matlab CEST-MRF code
    seq.write(seq_fn)