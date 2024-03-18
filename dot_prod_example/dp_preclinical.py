import os
import sys
import time
import numpy as np
import scipy.io as sio

from matplotlib import pyplot as plt

from utils.colormaps import b_viridis

from dot_prod_example.configs import ConfigPreclinical
from dot_prod_example.sequences import write_sequence_preclinical

from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching

FOLDER = os.path.dirname(os.path.realpath(__file__))

def setup_sequence_definitions(cfg):
    """ Setup the sequence definitions based on configuration."""
    seq_defs = {
        'n_pulses': 1,  # number of pulses
        'tp': 3,  # pulse duration [s]
        'td': 0,  # interpulse delay [s]
        'Trec': 1,  # delay before readout [s]
        'Trec_M0': 'NaN',  # delay before m0 readout [s]
        'M0_offset': 'NaN',  # dummy m0 offset [ppm]
        'offsets_ppm': [3.0] * 30,  # offset vector [ppm]
        'B0': cfg['b0'],  # B0 [T]
        'B1pa': [5, 5, 3, 3.75, 2.5, 1.75, 5.5, 6, 3.75,
                 5.75, 0.25, 3, 6, 4.5, 3.75, 3.5, 3.5, 0, 3.75, 6, 3.75, 4.75, 4.5,
                 4.25, 3.25, 5.25, 5.25, 0.25, 4.5, 5.25]
    }
    seq_defs['num_meas'] = len(seq_defs['offsets_ppm'])  # number of measurements
    seq_defs['DCsat'] = seq_defs['tp'] / (seq_defs['tp'] + seq_defs['td'])  # duty cycle
    seq_defs['Tsat'] = seq_defs['n_pulses'] * (seq_defs['tp'] + seq_defs['td']) - seq_defs['td']
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:]  # unique seq id

    return seq_defs


def generate_quant_maps(data_f, dict_fn):
    """Run dot product matching and save quant maps."""
    acq_fn = os.path.join(data_f, 'acquired_data.mat')
    quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acq_fn)
    return quant_maps


def visualize_and_save_results(quant_maps, output_f):
    """Visualize quant maps and save them as PDF."""
    os.makedirs(output_f, exist_ok=True)

    mat_fn = os.path.join(output_f, 'quant_maps.mat')
    sio.savemat(mat_fn, quant_maps)
    print('quant_maps.mat saved')

    mask = quant_maps['dp'] > 0.99974
    mask_fn = os.path.join(FOLDER, 'mask.npy')
    np.save(mask_fn, mask)

    fig_fn = os.path.join(output_f, 'dot_product_results.eps')
    fig, axes = plt.subplots(1, 3, figsize=(30, 25))
    color_maps = [b_viridis, 'magma', 'magma']
    data_keys = ['fs', 'ksw', 'dp']
    titles = ['[L-arg] (mM)', 'k$_{sw}$ (s$^{-1}$)', 'Dot product']
    clim_list = [(0, 120), (0, 500), (0.999, 1)]
    tick_list = [np.arange(0, 140, 20), np.arange(0, 600, 100), np.arange(0.999, 1.0005, 0.0005)]

    for ax, color_map, key, title, clim, ticks in zip(axes.flat, color_maps, data_keys, titles, clim_list, tick_list):
        vals = quant_maps[key] * (key == 'fs' and 110e3 / 3 or 1) * mask
        plot = ax.imshow(vals, cmap=color_map)
        plot.set_clim(*clim)
        ax.set_title(title, fontsize=25)
        cb = plt.colorbar(plot, ax=ax, ticks=ticks, orientation='vertical', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=25)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_fn, format="eps")
    plt.close()
    print("Resulting plots saved as EPS")


def main():
    data_f = os.path.join(FOLDER, 'data')
    output_f = os.path.join(FOLDER, 'results')

    cfg = ConfigPreclinical().get_config()

    seq_fn = os.path.join(FOLDER, cfg['seq_fn'])
    yaml_fn = os.path.join(FOLDER, cfg['yaml_fn'])
    dict_fn = os.path.join(FOLDER, cfg['dict_fn'])

    # Write configuration and sequence files
    write_yaml_dict(cfg, yaml_fn)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_preclinical(seq_defs=seq_defs, seq_fn=seq_fn)

    # Dictionary generation
    dictionary = generate_mrf_cest_dictionary(seq_fn=seq_fn, param_fn=yaml_fn, dict_fn=dict_fn,
                                 num_workers=cfg['num_workers'], axes='xy')

    # Dot product matching and quant map generation
    start_time = time.perf_counter()
    quant_maps = generate_quant_maps(data_f, dict_fn)
    print(f"Dot product matching took {time.perf_counter() - start_time:.03f} s.")

    # Visualization and saving results
    visualize_and_save_results(quant_maps, output_f)


if __name__ == '__main__':
    main()
