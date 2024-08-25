import os
import cv2
import numpy as np
import pydicom as dcm
import pandas as pd
from brukerapi.dataset import Dataset
from my_funcs.path_functions import find_n

from my_funcs.b0_mapping_functions import wassr_b0_mapping
from my_funcs.b0_mapping_functions import b0_correction


def bruker_dataset_creator(subject_fn, txt_file_name, fp_prtcl_name):
    """
    Create bruker dataset
    :param subject_fn: the subject's file path, root->scans->date->subject
    :param txt_file_name: the descriptory txt file name
    :param fp_prtcl_name: the protocol name
    :return: dicom_fn: the subject's dicom file path, root->scans->date->subject->E->pdata->1->dicom
    :return: mrf_files_fn: the subject's mrf_files path, root->scans->date->subject->E->mrf_files
    :return: bruker_dataset: the required bruker dataset
    """
    # Find E number where protocol was saved:
    glu_phantom_txt_fn = os.path.join(subject_fn, txt_file_name)  # root->scans->date->subject
    exp_id = find_n(glu_phantom_txt_fn, fp_prtcl_name)  # root->scans->date->subject->txt file
    # Paths to relevant files:
    fid_fn = os.path.join(subject_fn, f'{exp_id}', 'pdata', '1', '2dseq')  # fid file path
    dicom_fn = os.path.join(subject_fn, f'{exp_id}', 'pdata', '1', 'dicom')  # dicom folder path
    mrf_files_fn = os.path.join(subject_fn, f'{exp_id}', 'mrf_files')  # mrf_files folder path

    bruker_dataset = Dataset(fid_fn)
    bruker_dataset.add_parameter_file('method')  # load bruker dataset (with method only!)

    return dicom_fn, mrf_files_fn, bruker_dataset

def bruker_dataset_creator_csv(subject_fn, fp_prtcl_name):
    """
    Create bruker dataset
    """
    # Find E number where protocol was saved:
    doc_df = pd.read_csv(os.path.join(subject_fn, 'scan_doc.csv'))
    exp_id = doc_df[doc_df['scan name'] == fp_prtcl_name]['export_idx'].values[0]
    # Paths to relevant files:
    fid_fn = os.path.join(subject_fn, f'{exp_id}', 'pdata', '1', '2dseq')  # fid file path
    dicom_fn = os.path.join(subject_fn, f'{exp_id}', 'pdata', '1', 'dicom')  # dicom folder path
    mrf_files_fn = os.path.join(subject_fn, f'{exp_id}', 'mrf_files')  # mrf_files folder path

    bruker_dataset = Dataset(fid_fn)
    bruker_dataset.add_parameter_file('method')  # load bruker dataset (with method only!)

    return dicom_fn, mrf_files_fn, bruker_dataset


# def dicom_data_arranger(brkr_dataset, phantom_dicom_fn):
#     """
#     Create sequence for CEST
#     :param brkr_dataset: the bruker dataset for shape
#     :param phantom_dicom_fn: the dicom folder path
#     :return: dicom_data: the arranged dicom data (31/40/52/58, 64, 64)
#     """
#     channel_n = brkr_dataset.shape[-1]  # 31 channels (first one is M0)
#     im_shape = brkr_dataset.shape[:-2]  # (64, 64)
#     dicom_data = np.zeros((channel_n, im_shape[0], im_shape[1]))  # (31, 64, 64) #-1
#
#     for img_i in range(channel_n):
#         formatted_i = "{:02d}".format(img_i+1)  # index as '01' digits
#         dicom_data[img_i, :, :] = dcm.dcmread(os.path.join(phantom_dicom_fn,
#                                                               f'MRIm{formatted_i}.dcm')).pixel_array
#
#     return dicom_data

def dicom_data_arranger(brkr_dataset, phantom_dicom_fn):
    """
    Create sequence for CEST
    :param brkr_dataset: the bruker dataset for shape
    :param phantom_dicom_fn: the dicom folder path
    :return: dicom_data: the arranged dicom data (31/40/52/58, 64, 64)
    """
    channel_n = brkr_dataset.shape[-1]  # 31 channels (first one is M0)
    im_shape = brkr_dataset.shape[:-2]  # (64, 64/40)
    dicom_data = np.zeros((channel_n, im_shape[1], im_shape[0]))  # (31, 64/40, 64) #-1

    for img_i in range(channel_n):
        formatted_i = "{:02d}".format(img_i+1)  # index as '01' digits
        dicom_data[img_i, :, :] = dcm.dcmread(os.path.join(phantom_dicom_fn,
                                                              f'MRIm{formatted_i}.dcm')).pixel_array

    return dicom_data

def m0_remover(dicom_data):
    """
    Removes first image (M0 image), assuming n channels are first in shape
    :param dicom_data: the arranged dicom data
    :return: n0_M0_dicom_data: the arranged dicom data (30, 64, 64)
    """
    no_m0_dicom_data = np.copy(dicom_data[1:, :, :])

    return no_m0_dicom_data


def m0_normalizer(arr_dicom_data):
    """
    Normalizes channels by M0 (M0 image)
    :param arr_dicom_data: the arranged dicom data (31/52/58, 64, 64)
    :return: m0_norm_cest: the m0 arranged dicom data (30/51/57, 64, 64)
    """
    m0_cest = arr_dicom_data[0:1, :, :]
    complete_cest = arr_dicom_data[1:, :, :]

    m0_norm_cest = np.divide(complete_cest, np.where(m0_cest == 0, 1e-8, m0_cest))  # (30/51/57, 64, 64)

    return m0_norm_cest

def m0_normalizer_new(arr_dicom_data):
    """
    Normalizes channels by M0 (M0 image)
    :param arr_dicom_data: the arranged dicom data (31/52/58, 64, 64)
    :return: m0_norm_cest: the m0 arranged dicom data (31/52/58, 64, 64)
    """
    m0_cest = arr_dicom_data[0:1, :, :]

    m0_norm_cest = np.divide(arr_dicom_data, np.where(m0_cest == 0, 1e-8, m0_cest))  # (30/51/57, 64, 64)

    return m0_norm_cest

def sig_m0_normalizer(sig):
    """
    Normalizes signal by M0
    :param sig: the unnormalized signal (31)
    :return: M0_norm_sig: the M0 normalized signal (30)
    """
    if sig.shape[0] == 30:
        raise('Signal array does not contain M0, its length is 30')

    M0_sig = sig[0]
    M0_norm_sig = sig[1:] / M0_sig

    return M0_norm_sig

def z_spec_rearranger(z_spec_data):
    """
    Rearranges cest dicom images to be ordered []
    :param z_spec_data: the pre-arranged data (52/58, 64, 64) [M0,7,-7,6.75,-6.75,...,0.25,-0.25,0]
    :return: m0_cest: the M0 data (1, 64, 64) [M0]
    :return: z_spec_rearr: the arranged z-spectrum data (51/57, 64, 64) [M0,7,6.75,...,-6.75,-7]
    """
    cha_n, row_n, col_n = z_spec_data.shape  # (52/58, 64, 64)

    m0_cest = z_spec_data[0:1, :, :]  # (1, 64, 64)

    # rearrange cest images (without M0)
    mid_i = int(cha_n/2 - 1)
    last_i = int(cha_n-1)
    z_spec_rearr = np.zeros((cha_n-1, row_n, col_n))  # (64, 64, 51/57) [7,6.75,...,0.25,0,-0.25,...,-6.75,-7]
    z_spec_rearr[0:mid_i, :, :] = z_spec_data[1:-1:2, :, :]          # positives (by order) [7,6.75,...,0.25], (25/28, 64, 64)
    z_spec_rearr[mid_i:(mid_i+1), :, :] = z_spec_data[-1:, :, :]     # zero [0], (1, 64, 64)
    z_spec_rearr[last_i:mid_i:-1, :, :] = z_spec_data[2:-1:2, :, :]  # negatives (by order) [-0.25, ...,-6.75,-7], (25/28, 64, 64)

    return m0_cest, z_spec_rearr

def offset_rearranger(offset):
    """
    Rearranges offset data
    :param offset: the pre-arranged data (22/52/58) [M0,7,-7,6.75,-6.75,...,0.25,-0.25,0]
    :return: offset_rearr: the arranged (21/51/57) [7,-7,6.75,-6.75,...,0.25,-0.25,0]
    """
    cha_n = len(offset)  # 22/52/58

    # rearrange cest images (without M0)
    mid_i = int(cha_n/2 - 1)
    last_i = int(cha_n-1)
    offset_rearr = np.zeros((cha_n-1))  # (21/51/57) [7,6.75,...,0.25,0,-0.25,...,-6.75,-7]
    offset_rearr[0:mid_i] = offset[1:-1:2]      # positives (by order) [7,6.75,...,0.25]
    offset_rearr[mid_i:(mid_i+1)] = offset[-1:]        # zero [0]
    offset_rearr[last_i:mid_i:-1] = offset[2:-1:2]  # negatives (by order)

    return offset_rearr

def z_unarranger(arranged_z):
    """
    Rearranges offset data
    :param arranged_z: the arranged z_spec (51/57) [7,-7,6.75,-6.75,...,0.25,-0.25,0]
    :return: unarranged_z: the normalized un-arranged data (51/57) [7,-7,6.75,-6.75,...,0.25,-0.25,0]
    """
    cha_n = len(arranged_z)  # 51/57

    # rearrange cest images (without M0)
    mid_i = int(cha_n/2 - 1)
    last_i = int(cha_n-1)
    z_unarr = np.zeros((cha_n))  # (51/57)
    z_unarr[0:-1:2] = arranged_z[0:(mid_i+1)]
    z_unarr[-1:] = arranged_z[(mid_i+1): (mid_i+2)]
    z_unarr[1:-1:2] = arranged_z[last_i:(mid_i+1):-1]

    return z_unarr

def sigs_rearranger(signals):
    """
    Rearranges offset data
    :param offset: the pre-arranged data (22/52/58, dict_len) [M0,7,-7,6.75,-6.75,...,0.25,-0.25,0]
    :return: z_spectra: the arranged (21/51/57, dict_len) [7,-7,6.75,-6.75,...,0.25,-0.25,0]
    """
    cha_n, dict_len = signals.shape  # 22/52/58

    # rearrange cest images (without M0)
    mid_i = int(cha_n/2 - 1)
    last_i = int(cha_n-1)
    z_spectra = np.zeros((cha_n-1, dict_len))  # (21/51/57) [7,6.75,...,0.25,0,-0.25,...,-6.75,-7]
    signal = signals / signals[0, :]
    z_spectra[0:mid_i, :] = signal[1:-1:2, :]      # positives (by order) [7,6.75,...,0.25]
    z_spectra[mid_i:(mid_i+1), :] = signal[-1:]        # zero [0]
    z_spectra[last_i:mid_i:-1, :] = signal[2:-1:2]  # negatives (by order)

    return z_spectra

def z_spectra_creator(general_fn, txt_file_name, subject_dict):
    """
    :param: general_fn:
    :param: txt_file_name:
    :param: subject_dict:
    """
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    b0 = 7

    cest_prtcl_names = subject_dict['z_b1s_names']
    glu_phantom_fn = os.path.join(general_fn, 'scans', subject_dict['scan_name'],
                                  subject_dict['sub_name'])
    # mask
    vial_rois = subject_dict['vial_rois']
    full_mask = subject_dict['full_mask']

    # WASSR image
    wassr_dicom_fn, wassr_mrf_files_fn, wassr_bruker_dataset = bruker_dataset_creator(glu_phantom_fn, txt_file_name,
                                                                                      'WASSR')
    wassr_data = dicom_data_arranger(wassr_bruker_dataset, wassr_dicom_fn)
    M0_wassr, arr_wassr_spec = z_spec_rearranger(wassr_data)  # (21, 64, 64)

    wassr_norm = np.divide(arr_wassr_spec, np.where(M0_wassr == 0, 1e-8, M0_wassr))  # (22, 64, 64) full_mask
    offset_hz = offset_rearranger(wassr_bruker_dataset['SatFreqList'].value)
    offset_ppm = offset_hz / (gyro_ratio_hz * b0)
    b0_map = wassr_b0_mapping(wassr_norm, full_mask, w_x=offset_ppm, MainFieldMHz=gyro_ratio_hz * b0)

    mean_z_spectrum = np.zeros([3, len(cest_prtcl_names), 57])
    mean_z_spectrum_not_cor = np.zeros([3, len(cest_prtcl_names), 57])
    for cest_prtcl_i, cest_prtcl_name in enumerate(cest_prtcl_names):  # 0.7, 2, 4, 6
        # z-spec
        glu_phantom_dicom_fn, glu_phantom_mrf_files_fn, bruker_dataset = bruker_dataset_creator(glu_phantom_fn,
                                                                                                txt_file_name,
                                                                                                cest_prtcl_name)  # (58/52, 64, 64) [M0,7,-7,6.75,-6.75,...,0.25,-0.25,0]
        cest_data = dicom_data_arranger(bruker_dataset, glu_phantom_dicom_fn)
        M0_cest, arr_z_spec = z_spec_rearranger(cest_data)
        z_spec_norm = np.divide(arr_z_spec, np.where(M0_cest == 0, 1e-8, M0_cest))  # (51/57, 64, 64) full_mask

        # offset vector
        offset_hz = offset_rearranger(bruker_dataset['SatFreqList'].value)
        offset_ppm = offset_hz / (gyro_ratio_hz * b0)

        b0_cor_zspec = b0_correction(b0_map, z_spec_norm, offset_hz)  # have not checked!

        for vial_i, vial_roi in enumerate(vial_rois):
            roi_mask = np.zeros_like(full_mask)
            rr, cc = vial_roi.coords[:, 0], vial_roi.coords[:, 1]
            roi_mask[rr, cc] = 1
            # roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
            roi_mask = roi_mask.astype(np.uint8)

            for c_i in range(z_spec_norm.shape[0]):
                [[mean_vial]], _ = cv2.meanStdDev(z_spec_norm[c_i, :, :], mask=roi_mask)  # before b0 correction
                mean_z_spectrum_not_cor[vial_i, cest_prtcl_i, c_i] = mean_vial
                [[mean_vial]], _ = cv2.meanStdDev(b0_cor_zspec[c_i, :, :], mask=roi_mask)
                mean_z_spectrum[vial_i, cest_prtcl_i, c_i] = mean_vial

    return mean_z_spectrum_not_cor, mean_z_spectrum