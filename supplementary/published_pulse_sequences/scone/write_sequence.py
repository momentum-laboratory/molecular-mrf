import pypulseq as pp
from simplified_gaussian import make_gauss_pulse
import numpy as np

def write_clinical_sequence_gaussian(seq_defs, seq_fn, lims, type = 'scanner'):
    GAMMA_HZ = seq_defs["gamma_hz"]

    tp_sl = 1e-3  # duration of tipping pulse for sl
    td_sl = lims.rf_dead_time + lims.rf_ringdown_time  # delay between tip and sat pulse
    sl_time_per_sat = 2 * (tp_sl + td_sl)  # additional time of sl pulses for 1 sat pulse
    assert (seq_defs["trec"] >= sl_time_per_sat).all(), "DC too high for SL preparation pulses!"
    # print(td_sl,sl_time_per_sat)
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

    for m, b1 in enumerate(seq_defs["b1rms"]):
        # reset accumulated phase
        accum_phase = 0

        # prep and set rf pulse
        flip_angle_sat = b1 * GAMMA_HZ * 2 * np.pi * seq_defs["tp"]
        sat_pulse = make_gauss_pulse(b1=b1, duration=seq_defs["tp"], freq_offset=offsets_hz[m],
                                        system=lims, time_bw_product = 2.26)
        accum_phase = np.mod(offsets_hz[m] * 2 * np.pi * seq_defs["tp"], 2 * np.pi)

        # prep spin lock pulses
        flip_angle_tip = np.arctan(b1 / (seq_defs["offsets_ppm"][m] * seq_defs["b0"] + 1e-8))

        # make_block_pulse
        pre_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl, phase_offset=-(np.pi / 2),
                                           system=lims)
        post_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl,
                                            phase_offset=accum_phase + (np.pi / 2), system=lims)


        sat_pulse.freq_offset = offsets_hz[m]

        n_pulses = seq_defs["n_pulses"]
        if not isinstance(n_pulses, (int, float, np.int64)):
            n_pulses = seq_defs["n_pulses"][m]
        
        for n in range(n_pulses):
            pre_sl_pulse.phase_offset = pre_sl_pulse.phase_offset + phase_cycling
            sat_pulse.phase_offset = sat_pulse.phase_offset + phase_cycling
            post_sl_pulse.phase_offset = post_sl_pulse.phase_offset + phase_cycling
            # print(flip_angle_tip)

            if b1 == 0:
                seq.add_block(pp.make_delay(seq_defs["tp"]))
                seq.add_block(pp.make_delay(sl_time_per_sat))
            else:
                if flip_angle_tip > min_fa/180*np.pi:
                    seq.add_block(pre_sl_pulse)
                    seq.add_block(pp.make_delay(sl_pause_time))
                else:
                    # print('here')
                    # print(flip_angle_tip)
                    seq.add_block(pp.make_delay(pp.calc_duration(pre_sl_pulse)+sl_pause_time))

                seq.add_block(sat_pulse)

                if flip_angle_tip > min_fa/180*np.pi:
                    seq.add_block(pp.make_delay(sl_pause_time))
                    seq.add_block(post_sl_pulse)
                else:
                    seq.add_block(pp.make_delay(pp.calc_duration(post_sl_pulse)+sl_pause_time))

                if n < n_pulses - 1:
                    seq.add_block(pp.make_delay(seq_defs["td"]))

        if type == 'scanner':
            seq.add_block(pp.make_delay(100e-6)) # hardware related delay

        if seq_defs["spoiling"]:
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            if type == 'scanner':
                seq.add_block(pp.make_delay(100e-6)) # hardware related delay

        # readout sequence
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
        if m < len(seq_defs["b1rms"]) - 1:
            seq.add_block(pp.make_delay(seq_defs["trec"][m]-readout_time))

    def_fields = seq_defs.keys()
    for field in def_fields:
        print(f"Setting {field} to {seq_defs[field]}")
        if isinstance(seq_defs[field] , (np.ndarray, np.generic) ):
            seq_defs[field] = seq_defs[field].tolist()
        seq.set_definition(field, seq_defs[field])

    seq.write(seq_fn)
