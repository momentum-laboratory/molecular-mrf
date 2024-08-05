import numpy as np
import pypulseq

def write_sequence(seq_defs: dict = None,
                   seq_fn: str = 'protocol.seq'):
    '''
    Create sequence for CEST
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :return: sequence object
    '''

    # >>> Gradients and scanner limits - see pulseq doc for more info
    # Mostly relevant for clinical scanners
    # lims =  mr.opts('MaxGrad',30,'GradUnit','mT/m',...
    #     'MaxSlew',100,'SlewUnit','T/m/s', ...
    #     'rfRingdownTime', 50e-6, 'rfDeadTime', 200e-6, 'rfRasterTime',1e-6)
    # <<<

    # gamma
    gyroRatio_hz  = 42.5764           # for H [Hz/uT]
    gyroRatio_rad = gyroRatio_hz * 2 *np.pi  # [rad/uT]

    # This is the info for the 2d readout sequence. As gradients etc ar
    # simulated as delay, we can just add a delay afetr the imaging pulse for
    # simulation which has the same duration as the actual sequence
    # the flip angle of the readout sequence:
    imagingPulse = pypulseq.make_block_pulse(60 * np.pi / 180, duration=2.1e-3)
    # =='system', lims== should be added for clinical scanners

    # the duration of the readout sequence:
    te = 20e-3
    imagingDelay = pypulseq.make_delay(te)

    # init sequence
    # seq = SequenceSBB()
    seq = pypulseq.Sequence()
    # seq = SequenceSBB(lims) for clinical scanners

    # M0 pulse if required
    # seq.add_block(pypulseq.make_delay(seq_defs.Trec_M0)) # add a delay for m0
    # seq.add_block(imagingPulse)# add imaging block
    # seq.add_pseudo_ADC_block()    # additional feature of the SequenceSBB class
    # seq.add_block(imagingDelay)

    # Loop b1s
    for idx, B1 in enumerate(seq_defs['B1pa']):

        if idx > 0:  # add relaxtion block after first measurement
            seq.add_block(pypulseq.make_delay(seq_defs['Trec'] - te))  # net recovery time

        # saturation pulse
        current_offset_ppm = seq_defs['offsets_ppm'][idx]
        current_offset_hz = seq_defs['offsets_ppm'][idx] * seq_defs['B0'] * gyroRatio_hz
        fa_sat = B1 * gyroRatio_rad * seq_defs['tp']  # flip angle of sat pulse

        # add pulses
        for n_p in range(seq_defs['n_pulses']):
            # If B1 is 0 simulate delay instead of a saturation pulse
            if B1 == 0:
                seq.add_block(pypulseq.make_delay(seq_defs['tp']))  # net recovery time
            else:
                sat_pulse = pypulseq.make_block_pulse(fa_sat, duration=seq_defs['tp'], freq_offset=current_offset_hz)
                # =='system', lims== should be added for clinical scanners
                seq.add_block(sat_pulse)
            # delay between pulses
            if n_p < seq_defs['n_pulses'] - 1:
                seq.add_block(pypulseq.make_delay(seq_defs['td']))
        # Spoiling
        # additional feature of the SequenceSBB class
        # seq.add_spoiler_gradients(lims)

        # Imaging pulse
        seq.add_block(imagingPulse)
        seq.add_block(imagingDelay)
        # seq.add_pseudo_ADC_block()  # additional feature of the SequenceSBB class
        pseudo_ADC = pypulseq.make_adc(1, duration=1e-3)
        seq.add_block(pseudo_ADC)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.version_revision = 1 #compatibility with Matlab CEST-MRF code
    seq.write(seq_fn)

    return seq

