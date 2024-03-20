import numpy as np
import pypulseq as pp
import os


def setup_sequence_definitions_clinical(b1, seq_fn, gamma = 267.5153, freq = 127.7153):
    """Set up the sequence definitions based on B1 values and configuration."""

    num_meas = len(b1)
    seq_defs = {
        "n_pulses": 13,
        "num_meas": num_meas,
        "tp": 100e-3,
        "td": 100e-3,
        "offsets_ppm": np.ones(num_meas) * 3.0,
        "dcsat": 100e-3 / (100e-3 + 100e-3),
        "tsat": np.ones(num_meas) * 2.5,
        "trec": np.ones(num_meas) * 3.5 - np.ones(num_meas) * 2.5,
        "spoiling": True,
        "b1": b1,
        "seq_id_string": os.path.splitext(seq_fn)[1][1:],
        "freq": freq,
    }

    lims = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        gamma=gamma / 2 / np.pi * 1e6,
    )

    seq_defs["gamma_hz"] = lims.gamma * 1e-6
    seq_defs["freq"] = freq
    seq_defs['b0'] = seq_defs['freq'] / seq_defs["gamma_hz"]

    return seq_defs, lims


def setup_sequence_definitions_preclinical(b0, b1, seq_fn):
    """Set up the sequence definitions based on configuration."""

    seq_defs = {
        'n_pulses': 1,  # number of pulses
        'tp': 3,  # pulse duration [s]
        'td': 0,  # interpulse delay [s]
        'Trec': 1,  # delay before readout [s]
        'Trec_M0': 'NaN',  # delay before m0 readout [s]
        'M0_offset': 'NaN',  # dummy m0 offset [ppm]
        'offsets_ppm': [3.0] * 30,  # offset vector [ppm]
        'B0': b0,  # B0 [T]
        'B1pa': b1
    }
    seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])  # number of measurements
    seq_defs['DCsat'] = seq_defs['tp'] / (seq_defs['tp'] + seq_defs['td'])  # duty cycle
    seq_defs['Tsat'] = seq_defs['n_pulses'] * (seq_defs['tp'] + seq_defs['td']) - seq_defs['td']
    seq_defs['seq_id_string'] = os.path.splitext(seq_fn)[1][1:]  # unique seq id

    return seq_defs


def write_sequence_preclinical(seq_defs: dict = None,
                   seq_fn: str = 'protocol.seq'):
    """
    Create preclinical continous-wave sequence for CEST with simple readout
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :return: sequence object
    """

    # >>> Gradients and scanner limits - see pulseq doc for more info
    # Mostly relevant for clinical scanners
    # lims =  mr.opts('MaxGrad',30,'GradUnit','mT/m',...
    #     'MaxSlew',100,'SlewUnit','T/m/s', ...
    #     'rfRingdownTime', 50e-6, 'rfDeadTime', 200e-6, 'rfRasterTime',1e-6)
    # <<<

    # gamma
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]

    # This is the info for the 2d readout sequence. As gradients etc ar
    # simulated as delay, we can just add a delay afetr the imaging pulse for
    # simulation which has the same duration as the actual sequence
    # the flip angle of the readout sequence:
    imaging_pulse = pp.make_block_pulse(60 * np.pi / 180, duration=2.1e-3)

    # the duration of the readout sequence:
    te = 20e-3
    imaging_delay = pp.make_delay(te)

    # init sequence
    seq = pp.Sequence()

    # Loop b1s
    for idx, B1 in enumerate(seq_defs['B1pa']):

        if idx > 0:  # add relaxtion block after first measurement
            seq.add_block(pp.make_delay(seq_defs['Trec'] - te))  # net recovery time

        # saturation pulse
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
        pseudo_adc = pp.make_adc(1, duration=1e-3)
        seq.add_block(pseudo_adc)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.write(seq_fn)
    return seq


def write_sequence_clinical(seq_defs: dict, seq_fn:str, lims = None, type = 'scanner'):
    """
    Create clinical pulsed-wave sequence for CEST with complex readout
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :param lims: scanner limits
    :param type: type of sequence (scanner or simulation)
    :return: sequence object
    """

    GAMMA_HZ = seq_defs["gamma_hz"]

    tp_sl = 1e-3  # duration of tipping pulse for sl
    td_sl = lims.rf_dead_time + lims.rf_ringdown_time  # delay between tip and sat pulse
    sl_time_per_sat = 2 * (tp_sl + td_sl)  # additional time of sl pulses for 1 sat pulse
    assert (seq_defs["trec"] >= sl_time_per_sat).all(), "DC too high for SL preparation pulses!"

    sl_pause_time = 250e-6

    # spoiler
    spoil_amp = 0.8 * lims.max_grad  # Hz/m
    rise_time = 1.0e-3  # spoiler rise time in seconds
    spoil_dur = 4500e-6 + rise_time  # complete spoiler duration in seconds

    gx_spoil, gy_spoil, gz_spoil = [
        pp.make_trapezoid(channel=c, system=lims, amplitude=spoil_amp, duration=spoil_dur, rise_time=rise_time)
        for c in ["x", "y", "z"]
    ]
    if type == 'scanner':
        sl_pause_time = sl_pause_time - lims.rf_dead_time - lims.rf_ringdown_time

    min_fa = 1

    pseudo_adc = pp.make_adc(num_samples=1, duration=1e-3)
    offsets_hz = seq_defs["offsets_ppm"] * seq_defs["freq"]  # convert from ppm to Hz

    phase_cycling = 50 / 180 * np.pi
    seq = pp.Sequence()

    for m, b1 in enumerate(seq_defs["b1"]):
        # reset accumulated phase
        accum_phase = 0

        # prep and set rf pulse
        flip_angle_sat = b1 * GAMMA_HZ * 2 * np.pi * seq_defs["tp"]
        sat_pulse = pp.make_block_pulse(flip_angle=flip_angle_sat, duration=seq_defs["tp"], freq_offset=offsets_hz[m],
                                        system=lims)
        accum_phase = np.mod(offsets_hz[m] * 2 * np.pi * seq_defs["tp"], 2 * np.pi)

        # prep spin lock pulses
        flip_angle_tip = np.arctan(b1 / (seq_defs["offsets_ppm"][m] * seq_defs["b0"] + 1e-8))
        pre_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl, phase_offset=-(np.pi / 2),
                                           system=lims)
        post_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl,
                                            phase_offset=accum_phase + (np.pi / 2), system=lims)

        sat_pulse.freq_offset = offsets_hz[m]
        for n in range(seq_defs["n_pulses"]):
            pre_sl_pulse.phase_offset = pre_sl_pulse.phase_offset + phase_cycling
            sat_pulse.phase_offset = sat_pulse.phase_offset + phase_cycling
            post_sl_pulse.phase_offset = post_sl_pulse.phase_offset + phase_cycling

            if b1 == 0:
                seq.add_block(pp.make_delay(seq_defs["tp"]))
                seq.add_block(pp.make_delay(sl_time_per_sat))
            else:
                if flip_angle_tip > min_fa/180*np.pi:
                    seq.add_block(pre_sl_pulse)
                    seq.add_block(pp.make_delay(sl_pause_time))
                else:
                    seq.add_block(pp.make_delay(pp.calc_duration(pre_sl_pulse)+sl_pause_time))

                seq.add_block(sat_pulse)

                if flip_angle_tip > min_fa/180*np.pi:
                    seq.add_block(pp.make_delay(sl_pause_time))
                    seq.add_block(post_sl_pulse)
                else:
                    seq.add_block(pp.make_delay(pp.calc_duration(post_sl_pulse)+sl_pause_time))

                if n < seq_defs["n_pulses"] - 1:
                    seq.add_block(pp.make_delay(seq_defs["td"] - sl_time_per_sat))

        if type == 'scanner':
            seq.add_block(pp.make_delay(100e-6)) # hardware related delay

        if seq_defs["spoiling"]:
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            if type == 'scanner':
                seq.add_block(pp.make_delay(100e-6)) # hardware related delay

        # Readout sequence
        if type == 'scanner':
            seq.add_block(pseudo_adc)
            readout_time = pp.calc_duration(pseudo_adc)
        else:
            n_shots = 45
            tr1 = 28.370e-3
            tp1 = 2e-3
            tpause = tr1 - tp1
            fa1 = 15 * np.pi / 180
            flip_pulse = pp.make_block_pulse(flip_angle=fa1, duration=tp1)
            relax_time_readout = pp.make_delay(tpause)

            for ii in range(n_shots):
                seq.add_block(flip_pulse)
                seq.add_block(relax_time_readout)

                if ii == 0:
                    seq.add_block(pseudo_adc)

            # readout_time = n_shots * (tp1 + tpause) + tp1
            readout_time = 1e-3

        # add delay
        if m < len(seq_defs["b1"]) - 1:
            seq.add_block(pp.make_delay(seq_defs["trec"][m]-readout_time))

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.write(seq_fn)
    return seq