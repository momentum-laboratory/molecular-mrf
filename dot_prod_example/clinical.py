import os
import time
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pypulseq as pp

from utils.colormaps import b_viridis

from dot_prod_example.configs import ConfigClinical
from dot_prod_example.sequences import write_sequence_clinical

from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching


def setup_sequence_definitions(cfg, b1):
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
        "seq_id_string": os.path.splitext(cfg['seq_fn'])[1][1:],
        "freq": cfg['freq'],
    }

    lims = create_scanner_limits(cfg)

    seq_defs["gamma_hz"] = lims.gamma * 1e-6
    seq_defs["freq"] = cfg['freq']
    seq_defs['b0'] = seq_defs['freq'] / seq_defs["gamma_hz"]

    return seq_defs, lims


def create_scanner_limits(cfg):
    """Create scanner limits object."""
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        gamma=cfg['gamma'] / 2 / np.pi * 1e6,
    )


def preprocess_image(data_f, file_name):
    """Load and preprocess the image data."""
    data_fn = os.path.join(data_f, file_name)
    img = sio.loadmat(data_fn)['dataToMatch']
    return np.nan_to_num(img)[:, 19:-19, :]


def create_masks(quant_maps):
    """Create different masks based on quant_map criteria and save them."""
    mask_dp = quant_maps['dp'] > 0.9995
    mask_ksw = quant_maps['ksw'] > 100
    mask_fs = quant_maps['fs'] * 110e3 / 3 > 10
    mask = mask_ksw * mask_dp * mask_fs

    np.save('mask_dp.npy', mask_dp)
    np.save('mask_ksw_3T.npy', mask_ksw)
    np.save('mask_fs_3T.npy', mask_fs)
    np.save('mask_3T.npy', mask)

    return mask


def visualize_and_save_results(quant_maps, output_f, mask):
    """Visualize quant maps, apply mask, and save as eps."""
    fig_fn = os.path.join(output_f, 'dot_product_results_3T.eps')
    os.makedirs(output_f, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(30, 25))
    color_maps = [b_viridis, 'magma', 'magma']
    data_keys = ['fs', 'ksw', 'dp']
    titles = ['[L-arg] (mM)', 'k$_{sw}$ (s$^{-1}$)', 'Dot product']
    clim_list = [(0, 120), (0, 1400), (0.999, 1)]
    tick_list = [np.arange(0, 140, 20), np.arange(0, 1500, 200), np.arange(0.999, 1.0005, 0.0005)]

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


def preprocess_dict(dictionary):
    """Preprocess the dictionary for dot-matching"""
    dictionary['sig'] = np.array(dictionary['sig'])
    for key in dictionary.keys():
        if key != 'sig':
            dictionary[key] = np.expand_dims(np.squeeze(np.array(dictionary[key])), 0)
    print(dictionary['sig'].shape)

    return dictionary


def main():
    data_f = 'data'
    output_f = 'results'

    cfg = ConfigClinical().get_config()

    write_yaml_dict(cfg)

    # Write sequence file
    b1_values = [2, 2, 1.7, 1.5, 1.2, 1.2, 3, 0.5, 3, 1, 2.2, 3.2, 1.5, 0.7, 1.5, 2.2, 2.5, 1.2, 3, 0.2, 1.5, 2.5, 0.7,
                 4, 3.2, 3.5, 1.5, 2.7, 0.7, 0.5]
    
    seq_defs, lims = setup_sequence_definitions(cfg, b1_values)
    write_sequence_clinical(seq_defs, cfg['seq_fn'], lims, type='simulation')

    # Dictionary generation
    dictionary = generate_mrf_cest_dictionary(seq_fn=cfg['seq_fn'], param_fn=cfg['yaml_fn'], dict_fn=cfg['dict_fn'],
                                              num_workers=cfg['num_workers'], axes='xy')
    dictionary = preprocess_dict(dictionary)

    # Load and preprocess image data
    img = preprocess_image(data_f, 'dataToMatch_30_126_88_slice75.mat')
    start = time.perf_counter()
    quant_maps = dot_prod_matching(dictionary=dictionary, acquired_data=img, batch_size=img.shape[1] * 2)
    print(f"Dot product matching took {time.perf_counter() - start:.03f} s.")

    out_fn = 'quant_maps_3T.mat'
    sio.savemat(os.path.join(output_f, out_fn), quant_maps)

    mask = create_masks(quant_maps)

    # Visualize and save results
    visualize_and_save_results(quant_maps, output_f, mask)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
