# This code creates a CEST MRF dictionary for a pulsed clinical sequence
# Author: N. Vladimirov, 2023

import time

from cest_mrf.write_scenario import write_yaml
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

import pypulseq as pp
import numpy as np
import os

class config():
    yaml_fn = 'scenario_clinical.yaml'
    seq_fn = 'clinical_seq.seq'
    dict_fn = 'dict_seq.mat'

    # Water_pool
    water_t1 = [3.0]

    water_t2 = np.arange(400, 1550, 50) / 1000
    water_t2 = water_t2.tolist()  # vary t2
    water_f = 1

    # Solute pool
    cest_amine_t1 = [3000 / 1000]  # fix solute t1
    cest_amine_t2 = [40 / 1000]  # fix solute t2
    cest_amine_k = np.arange(100, 1410, 10).tolist()  # vary solute exchange rate
    cest_amine_dw = 3  # fixed solute exchange rate at 3 ppm

    # solute concentration * protons / water concentration
    cest_amine_sol_conc = np.arange(10, 125, 5)  # solute concentration
    cest_amine_protons = 3
    cest_amine_water_conc = 110000
    cest_amine_f = cest_amine_sol_conc * cest_amine_protons / cest_amine_water_conc
    cest_amine_f = cest_amine_f.tolist()

    magnetization_scale = 1
    magnetization_reset = 0

    # Fill scanner info
    b0 = 3 # [T]
    gamma = 267.5153  # [rad / uT]
    b0_inhom = 0
    rel_b1 = 1

    # Fill additional info
    verbose = 0
    max_pulse_samples = 100  # block pulses are only 1 sample

    num_workers = 18

def write_clinical_sequence(seq_defs, seq_fn, lims, type = 'scanner'):
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

    for m, b1 in enumerate(seq_defs["b1rms"]):
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
        seq.set_definition(field, seq_defs[field])
    # seq.version_revision = 1 #compatibility with Matlab CEST-MRF code

    seq.write(seq_fn)


def generate_dict(cfg, x):
    # Define output filenames
    yaml_fn = cfg.yaml_fn
    seq_fn = cfg.seq_fn
    dict_fn = cfg.dict_fn

    # Write the yaml-file
    write_yaml(cfg, yaml_fn)

    # Write the seq file for a 2d experiment
    # for more info about the seq file, check out the pulseq-cest repository
    seq_defs = {}
    seq_defs["n_pulses"] = 13  # number of pulses
    seq_defs["num_meas"] = 31  # number of repetition
    seq_defs["tp"] = 100e-3  # pulse duration [s]
    seq_defs["td"] = 100e-3  # interpulse delay [s]
    seq_defs["offsets_ppm"] = np.ones(seq_defs["num_meas"]) * 3.0

    TR = np.ones(seq_defs["num_meas"]) * 3.5
    TR[0] = 15
    Tsat = np.ones(seq_defs["num_meas"]) * 2.5
    seq_defs["offsets_ppm"][0] = 0

    seq_defs["dcsat"] = (seq_defs["tp"]) / (seq_defs["tp"] + seq_defs["td"])  # duty cycle
    seq_defs["tsat"] = Tsat  # saturation time [s]
    seq_defs["trec"] = TR - seq_defs["tsat"]  # net recovery time [s]
    seq_defs["spoiling"] = True

    seqid = os.path.splitext(seq_fn)[1][1:]
    seq_defs['seq_id_string'] = seqid  # unique seq id

    # we vary B1 for the dictionary generation
    seq_defs['b1pa'] = x
    seq_defs["b1rms"] = seq_defs["b1pa"]

    lims = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        gamma=cfg.gamma/2/np.pi*1e6,
    )

    seq_defs["gamma_hz"] = lims.gamma * 1e-6
    seq_defs["freq"] = 127.7292
    seq_defs['b0'] = seq_defs['freq'] / seq_defs["gamma_hz"]

    write_clinical_sequence(seq_defs, seq_fn, lims, type = 'simulation')

    start = time.perf_counter()
    dictionary = generate_mrf_cest_dictionary(seq_fn=seq_fn, param_fn=yaml_fn, dict_fn=dict_fn, num_workers=cfg.num_workers, axes='xy')
    end = time.perf_counter()
    s = (end-start)
    print(f"Dictionary simulation and preparation took {s:.03f} s.")
    return dictionary

def main():
    b1 = [0, 2, 2, 1.7, 1.5, 1.2, 1.2, 3, 0.5, 3, 1, 2.2, 3.2, 1.5, 0.7, 1.5, 2.2, 2.5, 1.2, 3, 0.2, 1.5, 2.5, 0.7, 4,
         3.2, 3.5, 1.5, 2.7, 0.7, 0.5]

    cfg = config()
    dictionary = generate_dict(cfg,b1)

if __name__ == '__main__':
    main()
