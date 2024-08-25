import os
import numpy as np
from my_funcs.cest_functions import z_spec_rearranger, offset_rearranger, wassr_b0_mapping, b0_correction
from my_funcs.cest_functions import bruker_dataset_creator, dicom_data_arranger
from scipy.interpolate import splev, splrep
from my_funcs.mask_functions import mask_processor


# z-spec for roi
def correct_b0(subject_dict, glu_mouse_fn, txt_file_name, mask):
    # given:
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    b0 = 7

    b1_names = subject_dict['z_b1s_names']

    # WASSR image
    wassr_dicom_fn, wassr_mrf_files_fn, wassr_bruker_dataset = bruker_dataset_creator(glu_mouse_fn, txt_file_name,
                                                                                      'WASSR')
    wassr_data = dicom_data_arranger(wassr_bruker_dataset, wassr_dicom_fn)
    M0_wassr, arr_wassr_spec = z_spec_rearranger(wassr_data)  # (21, 64, 64)

    wassr_norm = np.divide(arr_wassr_spec, np.where(M0_wassr == 0, 1e-8, M0_wassr))  # (22, 64, 64) full_mask
    offset_hz = offset_rearranger(wassr_bruker_dataset['SatFreqList'].value)
    offset_ppm = offset_hz / (gyro_ratio_hz * b0)
    b0_map = wassr_b0_mapping(wassr_norm, mask, w_x=offset_ppm, MainFieldMHz=gyro_ratio_hz * b0)

    z_dict = {}
    for b1_i, b1_name in enumerate(b1_names):  # 0.7, 1.5, 2, 4, 6
        b1_dict = {}
        # z-spec
        glu_phantom_dicom_fn, glu_phantom_mrf_files_fn, bruker_dataset = bruker_dataset_creator(glu_mouse_fn,
                                                                                                txt_file_name,
                                                                                                b1_name)  # (72/58/52, 64, 64) [M0,7,-7,6.75,-6.75,...,0.25,-0.25,0]
        cest_data = dicom_data_arranger(bruker_dataset, glu_phantom_dicom_fn)
        M0_cest, arr_z_spec = z_spec_rearranger(cest_data)
        z_spec_norm = np.divide(arr_z_spec, np.where(M0_cest == 0, 1e-8, M0_cest))  # (51/57, 64, 64) full_mask

        # offset vector
        offset_hz = offset_rearranger(bruker_dataset['SatFreqList'].value)
        offset_ppm = offset_hz / (gyro_ratio_hz * b0)

        b0_cor_zspec = b0_correction(b0_map, z_spec_norm, offset_hz)  # have not checked!

        fr_s = int(bruker_dataset['SatFreqStart'].value)
        fr_e = int(bruker_dataset['SatFreqEnd'].value)
        fr_i = round(bruker_dataset['SatFreqInc'].value, 2)

        ppm = np.arange(fr_s, fr_e - fr_i, -fr_i)

        b1_dict['ppm'] = ppm
        b1_dict['M0'] = M0_cest
        b1_dict[f'z_b1cor'] = b0_cor_zspec
        b1_dict[f'z_nob1cor'] = z_spec_norm
        z_dict[f'{b1_name}'] = b1_dict

    return z_dict


def aacid_roi(subject_dict, z_dict, roi_mask):
    # Iterate over each channel
    for b1_i, b1_name in enumerate(subject_dict['z_b1s_names']):
        # Iterate over each image
        ppm = z_dict[b1_name]['ppm']
        M0 = z_dict[b1_name]['M0']
        z_b1cor = z_dict[b1_name]['z_b1cor']
        z_nob1cor = z_dict[b1_name]['z_nob1cor']

        cha_n, r_n, c_n = z_b1cor.shape

        if b1_name == '1p5uT':
            MainFieldMHz = 298
            B1_uT = 1.5
            gyro_ratio_hz = 42.5764
            w1_Hz = round(B1_uT * gyro_ratio_hz)  # satpwr for scan (Hz)
            w_Hz = ppm * MainFieldMHz  # Scanned frequency offsets (ppm * MHz = Hz)

            # Calculate mean value for each channel in the masked regions (excluding zeros)
            roi_mask = roi_mask.astype(bool)
            expanded_roi_mask = np.repeat(roi_mask[np.newaxis, :, :], z_b1cor.shape[0], axis=0)

            masked_z_b1cor = np.ma.masked_where(expanded_roi_mask == 0, z_b1cor)
            roi_mean_z = np.mean(masked_z_b1cor, axis=(1, 2))
            Interp_w_Hz = np.arange(min(w_Hz), max(w_Hz) + 1, 1)  # Spline interpolation of scanned offsets every 1 Hz
            tck = splrep(w_Hz[::-1], roi_mean_z[::-1], s=0)  # Spline representation of the data (I needed to reverse)
            M_2p75 = splev(2.75 * MainFieldMHz, tck)
            M_3p5 = splev(3.5 * MainFieldMHz, tck)

            M_6 = roi_mean_z[np.argmin(abs(ppm - 6))]
            # print(f'3.5: {M_3p5}, 2.5: {M_2p75}, 6: {M_6}')
            aacid_roi = (M_3p5 * (M_6 - M_2p75)) / (M_2p75 * (M_6 - M_3p5))

            ph_roi = (-4 * aacid_roi + 12.8)  # in-vivo calibration
            # ph_roi = (aacid_roi - 6.5) / (-0.64)  # in-vitro calibration
            # ph_roi = (aacid_roi - 1.79) / (-0.236)  # in-vitro egg calibration physiological (not accurate!!!)
            # ph_roi = (aacid_roi - 1.95) / (-0.176)  # in-vitro egg calibration all (not accurate!!!)

    return aacid_roi, ph_roi, roi_mean_z


def aacid_per_pixel(subject_dict, parent_dir, txt_file_name):
    highres_img_idx = subject_dict['highres_img_idx']
    resratio = subject_dict['resratio']
    shift_up, shift_right = subject_dict['t_shift']
    glu_mouse_fn = os.path.join(parent_dir, 'data', 'scans', subject_dict['scan_name'], subject_dict['sub_name'])

    mask = mask_processor(glu_mouse_fn, subject_dict['resratio'], subject_dict['mask_slice'])
    mask = np.roll(mask, shift=(shift_up, shift_right), axis=(0, 1))
    z_dict = correct_b0(subject_dict, glu_mouse_fn, txt_file_name, mask)

    # Iterate over each channel
    for b1_i, b1_name in enumerate(subject_dict['z_b1s_names']):
        # Iterate over each image
        ppm = z_dict[b1_name]['ppm']
        z_b1cor = z_dict[b1_name]['z_b1cor']
        z_nob1cor = z_dict[b1_name]['z_nob1cor']

        cha_n, r_n, c_n = z_b1cor.shape

        aacid_map = np.zeros([r_n, c_n])
        if b1_name == '1p5uT':
            MainFieldMHz = 298
            B1_uT = 1.5
            gyro_ratio_hz = 42.5764
            w1_Hz = round(B1_uT * gyro_ratio_hz)  # satpwr for scan (Hz)

            w_Hz = ppm * MainFieldMHz  # Scanned frequency offsets (ppm * MHz = Hz)

            Interp_w_Hz = np.arange(min(w_Hz), max(w_Hz) + 1, 1)  # Spline interpolation of scanned offsets every 1 Hz
            for r_i in range(r_n):
                for c_i in range(c_n):
                    if mask[r_i, c_i] != 0:
                        cur_z = z_b1cor[:, r_i, c_i]
                        tck = splrep(w_Hz[::-1], z_b1cor[::-1, r_i, c_i],
                                     s=0)  # Spline representation of the data (I needed to reverse)
                        M_2p75 = splev(2.75 * MainFieldMHz, tck)
                        M_3p5 = splev(3.5 * MainFieldMHz, tck)
                        # M_2 = z_b1cor[np.argmin(abs(ppm-2)), r_i, c_i]
                        M_6 = z_b1cor[np.argmin(abs(ppm - 6)), r_i, c_i]

                        aacid_map[r_i, c_i] = (M_3p5 * (M_6 - M_2p75)) / (M_2p75 * (M_6 - M_3p5))
                        # aacid_map[r_i, c_i] = (M_3p5 * (M_6 - M_2)) / (M_2 * (M_6 - M_3p5))

                        # ph = (AACID - 6.5) / (-0.64)

                        # # Add scatter plot z-spectrum element
                        # z_fig.add_trace(go.Scatter(
                        #     x=ppm,
                        #     y=cur_z,
                        #     mode='lines',
                        #     line=dict(color='blue', width=2),
                        #     opacity=0.5
                        # ), row=row_i, col=1)
            # ph_map = (aacid_map - 6.1357681)/-0.6265661  # my bsa calibration
            # ph_map = (aacid_map - 6.5) / (-0.64)  # their bsa calibration
            # AACID = -0.6265661 * ph + 6.1357681

            ph_map = (-4 * aacid_map + 12.8)  # in vivo calibration
            # ph_map = (aacid_map - 2.8357681) / -0.2365661  # projection trick
            ph_map[ph_map < 0] = 0
            ph_map[ph_map > 10] = 11
            ph_map = ph_map * mask
            f_aacid_map = aacid_map

    return f_aacid_map, ph_map, z_dict, ppm