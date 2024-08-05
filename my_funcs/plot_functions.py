import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dcm
import scipy.io as sio
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from numpy import linalg as la
import matplotlib.pyplot as plt

import cv2
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import label, regionprops

from my_funcs.cest_functions import z_spectra_creator

# Create mask
def mask_roi_finder(bruker_dataset, erosion_r = 0):
    """
    Create 3 subplots showcasing vial roi's
    :param bruker_dataset: the bruker dataset based on 107a
    :param erosion_r:
    :return: circle_regions: regionprops circle regions (should be 3)
    :return: final_mask: the complete mask
    :return: bg_mask_f: the background mask
    """
    fp_filename = bruker_dataset['Fp_FileName'].value
    if '107a' not in fp_filename:
        raise ValueError("Error: the bruker dataset is not '107a'")

    """cest loading:"""
    cest_data = bruker_dataset.data
    count = np.shape(cest_data)[-1]  # # of images

    """highest contrast image finding:"""
    # Compute RMS contrast for each image
    rms_contrast = np.zeros(count)
    for i in range(count):
        rms_contrast[i] = np.std(cest_data[:, :, :, i])

    # Find index of image with highest contrast
    index = np.argmax(rms_contrast)

    # Load the image with highest contrast (always 1 out of 2 image options)
    contrasted_image = np.transpose(np.uint8(cest_data[:, :, 0, index]))

    """circle locating:"""
    # Binarize the image using thresholding
    _, image_binary = cv2.threshold(contrasted_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform connected component labeling
    label_image = label(~image_binary)

    # Specify the minimum number of pixels for separating regions
    min_pixels = 10

    # Create circular structuring element
    radius = int(min_pixels / 2)
    circular_footprint = np.zeros((min_pixels, min_pixels), dtype=bool)
    y, x = np.ogrid[:min_pixels, :min_pixels]
    mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2
    circular_footprint[mask] = 1

    # Perform erosion and dilation using circular structuring element
    binary_separated_1 = binary_erosion(label_image != 0, footprint=circular_footprint)
    binary_separated_2 = binary_dilation(binary_separated_1, footprint=circular_footprint)

    d_l_lim = 15
    a_l_lim = 300
    if erosion_r != 0:  # if not 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_r+1, 2*erosion_r+1))
        binary_separated_2 = binary_erosion(binary_separated_2, footprint=kernel)
        # plt.imshow(binary_separated_2)
        # plt.show()
        d_l_lim = 8
        a_l_lim = 140

    label_image = label(binary_separated_2)


    # Find the regions that correspond to the circles
    circle_regions = []
    for region in regionprops(label_image):
        if d_l_lim < region.equivalent_diameter < 22:
            if a_l_lim < region.area < 500:
                if region.axis_minor_length / region.axis_major_length > 0.5:
                    circle_regions.append(region)

    # Create a mask for each circle and set the corresponding pixels to one
    mask = np.zeros_like(contrasted_image)
    for region in circle_regions:
        rr, cc = region.coords[:, 0], region.coords[:, 1]
        mask[rr, cc] = 1

    # Set the pixels outside the circles to zero
    final_mask = mask

    if len(regionprops(label_image)) < 3:
        raise f'Expected 3 circles, but found {len(regionprops(label_image))} circle areas'

    """background locating:"""
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        image_binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=25,  # Lowering this value allows more circles to be detected
        param2=15,  # Lowering this value allows weaker circles to be detected
        minRadius=20,
        maxRadius=30
    )

    # Create a mask for a single circle
    bg_mask = np.zeros_like(image_binary)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        single_circle = circles[0, 0]
        cv2.circle(bg_mask, (single_circle[0], single_circle[1]), single_circle[2], 1, thickness=cv2.FILLED)

    bg_mask_logical = bg_mask != 0

    kernel_size = 10  # Adjust the kernel size as needed
    final_mask_logical = cv2.dilate(final_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    final_mask_logical = final_mask_logical != 0

    # Create a mask where big circle does not contain the small circle
    bg_mask_f = np.logical_and(bg_mask_logical, ~final_mask_logical)
    bg_mask_f = bg_mask_f.astype(int)

    return circle_regions, final_mask, bg_mask_f


def mask_check_plot(subject_dict):
    """
    Create 3 subplots showcasing vial roi's
    :param subject_dict: subject_dict
    """
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    vial_rois = subject_dict['vial_rois']
    full_mask = subject_dict['full_mask']
    bg_mask = subject_dict['bg_mask']

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.01,
                        subplot_titles=[f'idx 0: {conc_l[0]}mM, pH {ph_l[0]}',
                                        f'idx 1: {conc_l[1]}mM, pH {ph_l[1]}',
                                        f'idx 2: {conc_l[2]}mM, pH {ph_l[2]}',
                                        'Background'])

    for vial_i in range(len(vial_rois)):
        roi_mask = np.zeros_like(full_mask)
        vial_roi = vial_rois[vial_i]
        rr, cc = vial_roi.coords[:, 0], vial_roi.coords[:, 1]
        roi_mask[rr, cc] = 1

        # Add black mask using Heatmap
        fig.add_trace(go.Heatmap(z=~roi_mask, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'black']], showscale=False), row=1,
                      col=vial_i + 1)
        fig.update_xaxes(row=1, col=vial_i + 1, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, row=1, col=vial_i + 1, showticklabels=False,
                         autorange='reversed')  # Reverse the y-axis

    # Add black mask using Heatmap
    fig.add_trace(go.Heatmap(z=~bg_mask, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'black']], showscale=False), row=1,
                  col=4)
    fig.update_xaxes(row=1, col=4, showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, row=1, col=4, showticklabels=False,
                     autorange='reversed')  # Reverse the y-axis

    fig.update_layout(
        template='plotly_white',  # Set the theme to plotly white
        # plot_bgcolor='white',
        # paper_bgcolor='white',
        title_text=f'107a based mask',
        title=dict(x=0.01, y=0.97),  # Adjust the title position
        # title_font = dict(size=16, color='black'),  # Set title color to black
        # font = dict(color='black'),  # Set text color to black
    )

    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0),
                      height=300,
                      width=1000, )
    fig.show()


def vial_locator(full_mask, vial_rois):
    """
    Create vial location for tag adding
    :param full_mask: whole mask
    :param: vial_rois: vial rois
    :return roi_masks: roi individual masks
    :return vial_loc: vial locations
    """
    vial_loc = np.zeros([3, 2])  # [x,y]

    roi_masks = [np.zeros_like(full_mask), np.zeros_like(full_mask), np.zeros_like(full_mask)]

    for vial_i, vial_roi in enumerate(vial_rois):
        rr, cc = vial_roi.coords[:, 0], vial_roi.coords[:, 1]
        roi_masks[vial_i][rr, cc] = 1

        vial_loc[vial_i] = np.mean(vial_roi.coords[:, 0]), np.mean(vial_roi.coords[:, 1])

    return roi_masks, vial_loc


# Compare txt and bruker dataset
def compare_txt_method(prtcl_txt_fn, bruker_dataset, fp_prtcl_name):
    """
    Print protocol name, Trec, Tsat comparisons & B1, offsets comparison figure
    :param prtcl_txt_fn: location of protocol txt file root->protocol_text_files->glutamate->txt file
    :param bruker_dataset: the bruker dataset of scan
    :param fp_prtcl_name: the protocol name
    """
    # compare seq file to method file to protocol txt file
    # data from txt file:
    prtcl_txt_df = pd.read_csv(prtcl_txt_fn, sep='\s+', skiprows=1, header=None)

    offsets_ppm_txt = prtcl_txt_df.iloc[:, 2].to_numpy()
    B1pa_txt = prtcl_txt_df.iloc[:, 1].to_numpy()

    Trec_M0_txt = int((prtcl_txt_df.iloc[0, 0] - prtcl_txt_df.iloc[0, -1]) / 1000)
    Trec_txt = int((prtcl_txt_df.iloc[1, 0] - prtcl_txt_df.iloc[1, -1]) / 1000)
    Tsat_txt = int(prtcl_txt_df.iloc[1, -1] / 1000)
    FA_txt = int(prtcl_txt_df.iloc[1, -2])

    # data from bruker data:
    fp_filename_scan = bruker_dataset['Fp_FileName'].value

    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    b0 = 7
    offsets_ppm_scan = (bruker_dataset['Fp_SatOffset'].value / (gyro_ratio_hz * b0))  # offset vector [ppm]
    B1pa_scan = bruker_dataset['Fp_SatPows'].value  # B1 vector [uT]

    Trec_M0_scan = int((bruker_dataset['Fp_TRs'].value[0] -
                    bruker_dataset['Fp_SatDur'].value[0]) / 1000)  # delay before m0 readout [s]
    Trec_scan = int((bruker_dataset['Fp_TRs'].value[-1] -
                 bruker_dataset['Fp_SatDur'].value[-1]) / 1000)  # delay before readout [s] (TR-Tsat = 1)
    Tsat_scan = int(bruker_dataset['PVM_MagTransPulse1'].value[0] / 1000)
    FA_scan = int(bruker_dataset['Fp_FlipAngle'].value[0])

    # compare:
    if fp_prtcl_name.lower() in fp_filename_scan.lower():
        print(f"The protocol '{fp_prtcl_name}' matches between scan data and txt file")
    elif fp_prtcl_name.lower() in f'{fp_filename_scan.lower()[1:4]}{fp_filename_scan.lower()[11]}':
        print(f"The protocol '{fp_prtcl_name}' matches between scan data and txt file")
    else:
        raise ValueError(f"Error: The protocol '{fp_prtcl_name}' doesn't match scan data's protocol {fp_filename_scan}")

    if Trec_M0_txt != Trec_M0_scan:
        raise ValueError(f"Error: The Trec M0 '{Trec_M0_txt}' doesn't match scan data's Trec M0 {Trec_scan}")

    if Trec_txt != Trec_scan:
        raise ValueError(f"Error: The Trec '{Trec_txt}' doesn't match scan data's Trec {Trec_scan}")

    if Tsat_txt != Tsat_scan:
        raise ValueError(f"Error: The Tsat '{Tsat_txt}' doesn't match scan data's Tsat {Tsat_scan}")

    if FA_txt != FA_scan:
        raise ValueError(f"Error: The FA '{FA_txt}' doesn't match scan data's FA {FA_scan}")

    # Create a DataFrame
    parameters = ['Trec_M0', 'Trec', 'Tsat', 'FA']
    txt_results = [Trec_M0_txt, Trec_txt, Tsat_txt, FA_txt]
    method_results = [Trec_M0_scan, Trec_scan, Tsat_scan, FA_scan]
    df = pd.DataFrame({
        'Parameter': parameters * 2,
        'File Type': ['txt'] * len(parameters) + ['method'] * len(parameters),
        'Result': txt_results + method_results
    })

    # compare B1 and offsets:
    fig = make_subplots(rows=3, cols=1, subplot_titles=['T comparison', 'B1 comparison', 'offsets ppm comparison'],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.01)

    # Colors for the scatter plots
    color_txt = '#1f77b4'
    color_method = '#ff7f0e'

    # First plot (Trec, Trec comparison)
    fig.add_trace(go.Bar(
        x=df[df['File Type'] == 'txt']['Parameter'],
        y=df[df['File Type'] == 'txt']['Result'],
        marker_color=color_txt,
        text=df[df['File Type'] == 'txt']['Result']),
        row=1, col=1)

    # Add bars for 'method' files
    fig.add_trace(go.Bar(
    x = df[df['File Type'] == 'method']['Parameter'],
    y = df[df['File Type'] == 'method']['Result'],
    marker_color=color_method,
    text=df[df['File Type'] == 'method']['Result']),
    row = 1, col = 1)
    fig.update_yaxes(row=1, col=1, title_text=f'Ts [s]', tick0=0, dtick=5, range=[0, 15])

    x_axis = np.arange(1, 31)

    scatter_B1_txt = go.Scatter(x=x_axis, y=B1pa_txt, mode='lines', name='protocol txt file',
                                legendgroup='protocol txt file', line=dict(color=color_txt))
    scatter_B1_scan = go.Scatter(x=x_axis, y=B1pa_scan, mode='lines', name='scan method file',
                                 legendgroup='scan method file', line=dict(color=color_method, dash='dash'))
    # Add traces to the first subplot
    fig.add_trace(scatter_B1_txt, row=2, col=1)
    fig.add_trace(scatter_B1_scan, row=2, col=1)
    fig.update_yaxes(row=2, col=1, title_text=f'B1 [uT]', tick0=0, dtick=1, range=[0, 6])

    scatter_offsets_txt = go.Scatter(x=x_axis, y=offsets_ppm_txt, mode='lines', name='protocol txt file',
                                     legendgroup='protocol txt file', showlegend=False,
                                     line=dict(color=color_txt))
    scatter_offsets_scan = go.Scatter(x=x_axis, y=offsets_ppm_scan, mode='lines', name='scan method file',
                                      legendgroup='scan method file', showlegend=False,
                                      line=dict(color=color_method, dash='dash'))
    # # Add traces to the first subplot
    fig.add_trace(scatter_offsets_txt, row=3, col=1)
    fig.add_trace(scatter_offsets_scan, row=3, col=1)
    fig.update_yaxes(row=3, col=1, title_text=f'offsets [ppm]', tick0=0, dtick=1, range=[0, 6])

    # Set layout for better visualization
    fig.update_layout(
        template='plotly_white',  # Set the theme to plotly white,
        title_text=f'{fp_prtcl_name}, parameter comparison',
        showlegend=True,  # Hide legend
        height=500,
        width=700  # Set a width based on your preference
    )

    # Adjust margin to reduce whitespace
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))

    # only show 2 last traces in legend!
    for trace_i, trace in enumerate(fig['data']):
        if trace_i < 2:
            trace['showlegend'] = False

    # Show the plot
    fig.show()

# Normalize vial mrf signals (mean signal per vial)
def norm_pixels(vial_i, full_mask, vial_rois, quant_data_fn, glu_phantom_mrf_files_fn, expected_f, expected_k, expected_t1, expected_t2):
    """
    Create normalized mrf signals
    :param vial_i:
    :param: roi_masks:
    :param: quant_data_fn:
    :param: glu_phantom_mrf_files_fn:
    :param: expected_f:
    :param: expected_k:
    :param: expected_t1:
    :param: expected_t2:
    :return norm_vial_real_sig: real signal from acquit
    :return norm_vial_match_sig: dict match signal
    :return norm_vial_exp_sig: dict expected signal
    """
    roi_masks, vial_loc = vial_locator(full_mask, vial_rois)

    quant_maps = sio.loadmat(quant_data_fn)

    acquired_data_fn = os.path.join(glu_phantom_mrf_files_fn, 'acquired_data.mat')
    acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']

    dict_fn = os.path.join(glu_phantom_mrf_files_fn, 'dict.mat')
    synt_dict = sio.loadmat(dict_fn)
    synt_sig = np.transpose(synt_dict['sig'])   # e.g. 30 x 369,000
    synt_fs = synt_dict['fs_0']
    synt_ksw = synt_dict['ksw_0']
    synt_t1w = synt_dict['t1w']
    synt_t2w = synt_dict['t2w']

    acquired_data = acquired_data[1:, :]
    if synt_sig.shape[0] == 31:
        synt_sig = synt_sig[1:, :]

    # # temp!!!
    # acquired_data = acquired_data[3:, :]
    # synt_sig = synt_sig[3:, :]

    x_locs, y_locs = np.where(roi_masks[vial_i])
    # print(f'{np.mean(x_locs)}, {np.mean(y_locs)}')

    m_ids_pixel = quant_maps['match_id'][x_locs, y_locs].astype('int')
    exp_id_pixel = np.where((synt_fs == expected_f[vial_i]) & (synt_ksw == expected_k[vial_i]) & (synt_t1w == expected_t1[vial_i]) & (synt_t2w == expected_t2[vial_i]))[1][0]
    vial_real_sig = acquired_data[:, x_locs, y_locs]
    vial_real_sig_mean = np.mean(vial_real_sig, axis=1)
    vial_real_sig_std = np.std(vial_real_sig, axis=1)

    vial_match_sig = synt_sig[:, m_ids_pixel]
    vial_match_sig_mean = np.mean(vial_match_sig, axis=1)
    vial_match_sig_std = np.std(vial_match_sig, axis=1)
    vial_exp_sig = synt_sig[:, exp_id_pixel]

    # vial_real_sig_mean = sig_m0_normalizer(vial_real_sig_mean)
    # vial_match_sig_mean = sig_m0_normalizer(vial_match_sig_mean)
    # vial_exp_sig = sig_m0_normalizer(vial_exp_sig)

    # equivalent to normc in matlab
    norm_vial_real_sig = vial_real_sig_mean / la.norm(vial_real_sig_mean, axis=0)
    norm_vial_match_sig = vial_match_sig_mean / la.norm(vial_match_sig_mean, axis=0)
    norm_vial_exp_sig = vial_exp_sig / la.norm(vial_exp_sig, axis=0)

    return norm_vial_real_sig, norm_vial_match_sig, norm_vial_exp_sig

def norm_roi(roi_mask, quant_data_fn, dict_fn, mrf_files_fn, subject_dict, roi_i, real_ts_flag=False):
    """
    Create normalized mrf signals
    :param: roi_mask:
    :param: quant_data_fn:
    :param: mrf_files_fn:
    :param: subject_dict:
    :param: roi_i:
    :param: real_ts_flag:
    :return norm_vial_real_sig: real signal from acquit
    :return norm_vial_match_sig: dict match signal
    """
    if real_ts_flag == True:
        quant_maps = subject_dict['quant_maps'][roi_i]
    else:
        quant_maps = sio.loadmat(quant_data_fn)

    acquired_data_fn = os.path.join(mrf_files_fn, 'acquired_data.mat')
    acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']

    synt_df = pd.read_csv(dict_fn, header=0)

    synt_sig = np.transpose(np.array([np.fromstring(cell.strip('[]').replace('\n', '').replace('  ', ' '), sep=' ') for cell in synt_df['sig']]))  # e.g. 30/31 x ???

    acquired_data = acquired_data[1:, :]
    if synt_sig.shape[0] == 31:
        synt_sig = synt_sig[1:, :]

    r_locs, c_locs = np.where(roi_mask)

    m_ids_pixel = quant_maps['match_id'][r_locs, c_locs].astype('int')
    real_sig = acquired_data[:, r_locs, c_locs]
    n_real_sig = real_sig / la.norm(real_sig, axis=0)
    # n_real_sig = real_sig[1:, :] / real_sig[1:2, :]
    n_real_sig_mean = np.mean(n_real_sig, axis=1)
    n_real_sig_std = np.std(n_real_sig, axis=1)

    match_sig = synt_sig[:, m_ids_pixel]
    n_match_sig = match_sig / la.norm(match_sig, axis=0)
    # n_match_sig = match_sig[1:, :] / match_sig[1:2, :]
    n_match_sig_mean = np.mean(n_match_sig, axis=1)
    n_match_sig_std = np.std(n_match_sig, axis=1)

    return (n_real_sig_mean, n_match_sig_mean), (n_real_sig_std, n_match_sig_std), (n_real_sig, n_match_sig)

# Find real t1 and t2
def find_n(filename, target_str):
    # find export file num
    with open(filename, 'r') as file:
        for line in file:
            match = re.search(rf'{target_str}.*\(E(\d+)\)', line)
            if match:
                n = int(match.group(1))
                return n
    print(f"Error: '{target_str}' not found in the file.")
    return None


def t1_t2_pixel_reader(glu_phantom_fn, txt_file_name, image_idx, t_type, image_file=2):
    """
    Rescale t1/t2 maps
    :param glu_phantom_fn: path root->scans->date->subject
    :param: txt_file_name: name of export specifying test file
    :param image_idx: relevant image index
    :param t_type: specify 't1' or 't2'
    :return: t1/t2 pixel data
    """
    if t_type == 't1':
        prtcl_name = 'T1map_RARE'
    elif t_type == 't2':
        prtcl_name = 'T2map_MSME'
    else:
        print(f"Error: 't_type = {t_type}' not found in the file.")

    glu_phantom_txt_fn = os.path.join(glu_phantom_fn, txt_file_name)
    exp_id = find_n(glu_phantom_txt_fn, prtcl_name)
    dcm_fn = os.path.join(glu_phantom_fn, f'{exp_id}', 'pdata', f'{image_file}', 'dicom', f'MRIm{image_idx}.dcm')

    dcm_data = dcm.dcmread(dcm_fn)  # Read DICOM file

    # Access the rescale slope
    pixel_val = dcm_data.pixel_array
    rescale_slope = dcm_data.RescaleSlope
    rescale_intercept = dcm_data.RescaleIntercept

    if rescale_intercept != 0:
        print(f'rescale_intercept is {rescale_intercept}, should be 0!')

    # Rescale
    dcm_pixel_data = pixel_val * rescale_slope + rescale_intercept

    return dcm_pixel_data


# Create figures
def real_t1_t2(t1_pixels, t2_pixels, phantom_choice, subject_dict):
    """
    Create vial location for tag adding
    :param t1_pixels:
    :param: t2_pixels:
    :param: phantom_choice:
    :param: subject_dict:
    """
    full_mask = subject_dict['full_mask']
    vial_rois = subject_dict['vial_rois']
    tag = subject_dict['tags']
    tag_x_loc = subject_dict['tag_x_locs']
    tag_y_loc = subject_dict['tag_y_locs']
    date = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    # Set layout for better visualization
    tag_id = [tag.index('a'), tag.index('b'), tag.index('c')]

    roi_masks, vial_loc = vial_locator(full_mask, vial_rois)

    # Create custom Viridis colormap with black for 0 values
    custom_viridis = np.array(plotly.colors.sequential.Viridis)
    custom_viridis[0] = '#000000'  # Set black for 0 values

    # Create custom Plasma colormap with black for 0 values
    custom_plasma = np.array(plotly.colors.sequential.Plasma)
    custom_plasma[0] = '#000000'  # Set black for 0 values

    # Create subplots with 1 row and 3 columns, increased horizontal spacing
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, subplot_titles=['Real t1 [ms]', 'Real t2 [ms]'])

    # Add heatmaps for the three arrays
    heatmap_t1 = go.Heatmap(z=t1_pixels*full_mask, colorscale=custom_viridis, coloraxis='coloraxis1')
    heatmap_t2 = go.Heatmap(z=t2_pixels*full_mask, colorscale=custom_plasma, coloraxis='coloraxis2')

    fig.add_trace(heatmap_t1, row=1, col=1)
    fig.add_trace(heatmap_t2, row=1, col=2)

    df = pd.DataFrame()

    # Loop over masks
    for j, mask in enumerate([roi_masks[0], roi_masks[1], roi_masks[2]]):
        # Calculate placement
        x_loc = int(np.round(vial_loc[j, 1]))
        y_loc = int(np.round(vial_loc[j, 0]))

        # Calculate mean and std
        [[t1_mean_val]], [[t1_std_val]] = cv2.meanStdDev(t1_pixels, mask=mask)
        [[t2_mean_val]], [[t2_std_val]] = cv2.meanStdDev(t2_pixels, mask=mask)

        roi_row = pd.DataFrame({
            f't1': [f'{round(t1_mean_val)} ± {round(t1_std_val)}'],
            f't2': [f'{round(t2_mean_val)} ± {round(t2_std_val)}'],
            f't1_mean': [t1_mean_val],
            f't2_mean': [t2_mean_val],
            f't1_std': [t1_std_val],
            f't2_std': [t2_std_val]
        })

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, roi_row], ignore_index=True)

        # Annotate subplot with mean ± std
        t1_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(t1_mean_val)} ± {round(t1_std_val)}<b>'
        t2_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(t2_mean_val)} ± {round(t2_std_val)}<b>'
        fig.add_annotation(text=t1_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x1', yref='y1', row=1, col=1)
        fig.add_annotation(text=t2_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x2', yref='y2', row=1, col=2)

    fig.update_layout(
        template='plotly_dark',  # Set the theme to plotly dark
        title_text=f'Phantom {date} {temp}°C: real Glutamate t1 & t2',
        showlegend=False,  # Hide legend
        height=345,
        width=680,  # Set a width based on your preference
        margin=dict(l=10, r=0, t=60, b=20),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    # Add individual titles and separate colorbars
    for i, title in enumerate(['Real t1 [ms]', 'Real t2 [ms]'], start=1):
        fig.update_xaxes(row=1, col=i, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, row=1, col=i, showticklabels=False, autorange='reversed')  # Reverse the y-axis

    # Manually add separate colorbars
    colorbar_t1 = {'colorscale': custom_viridis, 'cmin': 0, 'cmax': 5000}
    colorbar_t2 = {'colorscale': custom_plasma, 'cmin': 0, 'cmax': 1000}

    fig.update_layout(
        coloraxis1=colorbar_t1,
        coloraxis2=colorbar_t2,
        coloraxis_colorbar=dict(x=0.43, y=0.5),
        coloraxis2_colorbar=dict(x=0.995, y=0.5),
    )

    # Show the plot
    fig.show()

    pio.write_image(fig, f'images/{save_name}/subject_{phantom_choice}/t.jpeg')

    t1_t2_df = pd.DataFrame(df.values[tag_id, :], index=['a', 'b', 'c'], columns=df.columns)
    return t1_t2_df

def real_t1_t2_bg(t1_pixels, t2_pixels, phantom_choice, subject_dict):
    """
    Create vial location for tag adding
    :param t1_pixels:
    :param: t2_pixels:
    :param: phantom_choice:
    :param: subject_dict:
    """
    bg_mask = subject_dict['bg_mask']
    date = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']

    # Create custom Viridis colormap with black for 0 values
    custom_viridis = np.array(plotly.colors.sequential.Viridis)
    custom_viridis[0] = '#000000'  # Set black for 0 values

    # Create custom Plasma colormap with black for 0 values
    custom_plasma = np.array(plotly.colors.sequential.Plasma)
    custom_plasma[0] = '#000000'  # Set black for 0 values

    # Create subplots with 1 row and 3 columns, increased horizontal spacing
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, subplot_titles=['Real t1 [ms]', 'Real t2 [ms]'])

    # Add heatmaps for the three arrays
    heatmap_t1 = go.Heatmap(z=t1_pixels*bg_mask, colorscale=custom_viridis, coloraxis='coloraxis1')
    heatmap_t2 = go.Heatmap(z=t2_pixels*bg_mask, colorscale=custom_plasma, coloraxis='coloraxis2')

    bg_mask = (bg_mask * 255).astype(np.uint8)

    fig.add_trace(heatmap_t1, row=1, col=1)
    fig.add_trace(heatmap_t2, row=1, col=2)

    # Calculate placement
    x_loc = 15
    y_loc = 5

    # Calculate mean and std
    [[t1_mean_val]], [[t1_std_val]] = cv2.meanStdDev(t1_pixels, mask=bg_mask)
    [[t2_mean_val]], [[t2_std_val]] = cv2.meanStdDev(t2_pixels, mask=bg_mask)

    # Annotate subplot with mean ± std
    t1_annotation = f'<b>{round(t1_mean_val)} ± {round(t1_std_val)}<b>'
    t2_annotation = f'<b>{round(t2_mean_val)} ± {round(t2_std_val)}<b>'
    fig.add_annotation(text=t1_annotation, x=x_loc, y=y_loc,
                       showarrow=False, xref='x1', yref='y1', row=1, col=1)
    fig.add_annotation(text=t2_annotation, x=x_loc, y=y_loc,
                       showarrow=False, xref='x2', yref='y2', row=1, col=2)

    # Set layout for better visualization
    fig.update_layout(
        template='plotly_dark',  # Set the theme to plotly dark
        title_text=f'Phantom {date} {temp}°C: PBS real t1 & t2',
        showlegend=False,  # Hide legend
        height=345,
        width=680,  # Set a width based on your preference
        margin=dict(l=10, r=0, t=60, b=20),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    # Add individual titles and separate colorbars
    for i, title in enumerate(['Real t1 [ms]', 'Real t2 [ms]'], start=1):
        fig.update_xaxes(row=1, col=i, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, row=1, col=i, showticklabels=False, autorange='reversed')  # Reverse the y-axis

    # Manually add separate colorbars
    colorbar_t1 = {'colorscale': custom_viridis, 'cmin': 0, 'cmax': 5000}
    colorbar_t2 = {'colorscale': custom_plasma, 'cmin': 0, 'cmax': 2000}

    fig.update_layout(
        coloraxis1=colorbar_t1,
        coloraxis2=colorbar_t2,
        coloraxis_colorbar=dict(x=0.43, y=0.5),
        coloraxis2_colorbar=dict(x=0.995, y=0.5),
    )

    # Show the plot
    fig.show()

    roi_row = pd.DataFrame({
        f't1': [f'{round(t1_mean_val)} ± {round(t1_std_val)}'],
        f't2': [f'{round(t2_mean_val)} ± {round(t2_std_val)}'],
        f't1_mean': [t1_mean_val],
        f't2_mean': [t2_mean_val],
        f't1_std': [t1_std_val],
        f't2_std': [t2_std_val]
    })
    t1_t2_bg_df = pd.DataFrame(roi_row.values, index=['bg'], columns=roi_row.columns)

    pio.write_image(fig, f'images/{save_name}/subject_{phantom_choice}/t_bg.jpeg')

    return t1_t2_bg_df

def dict_t1_t2(t1w_array, t2w_array, phantom_choice, fp_prtcl_name, subject_dict):
    """
    Create vial location for tag adding
    :param t1_pixels:
    :param: t2_pixels:
    :param: phantom_choice:
    :param: fp_prtcl_name:
    :param: subject_dict:
    """
    full_mask = subject_dict['full_mask']
    vial_rois = subject_dict['vial_rois']
    tag = subject_dict['tags']
    tag_x_loc = subject_dict['tag_x_locs']
    tag_y_loc = subject_dict['tag_y_locs']
    date = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']

    roi_masks, vial_loc = vial_locator(full_mask, vial_rois)

    # Create custom Viridis colormap with black for 0 values
    custom_viridis = np.array(plotly.colors.sequential.Viridis)
    custom_viridis[0] = '#000000'  # Set black for 0 values

    # Create custom Plasma colormap with black for 0 values
    custom_plasma = np.array(plotly.colors.sequential.Plasma)
    custom_plasma[0] = '#000000'  # Set black for 0 values

    # Create subplots with 1 row and 3 columns, increased horizontal spacing
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, subplot_titles=['Dict t1 [ms]', 'Dict t2 [ms]'])

    # Add heatmaps for the three arrays
    heatmap_t1 = go.Heatmap(z=t1w_array*full_mask, colorscale=custom_viridis, coloraxis='coloraxis1')
    heatmap_t2 = go.Heatmap(z=t2w_array*full_mask, colorscale=custom_viridis, coloraxis='coloraxis2')

    fig.add_trace(heatmap_t1, row=1, col=1)
    fig.add_trace(heatmap_t2, row=1, col=2)

    # Loop over masks
    for j, mask in enumerate([roi_masks[0], roi_masks[1], roi_masks[2]]):
        # Calculate placement
        x_loc = int(np.round(vial_loc[j, 1]))
        y_loc = int(np.round(vial_loc[j, 0]))

        # Calculate mean and std
        [[t1_mean_val]], [[t1_std_val]] = cv2.meanStdDev(t1w_array, mask=mask)
        [[t2_mean_val]], [[t2_std_val]] = cv2.meanStdDev(t2w_array, mask=mask)

        # Annotate subplot with mean ± std
        t1_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(t1_mean_val)} ± {round(t1_std_val)}<b>'
        t2_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(t2_mean_val)} ± {round(t2_std_val)}<b>'
        fig.add_annotation(text=t1_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x1', yref='y1', row=1, col=1)
        fig.add_annotation(text=t2_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x2', yref='y2', row=1, col=2)

    # Set layout for better visualization

    fig.update_layout(
        template='plotly_dark',  # Set the theme to plotly dark
        title_text=f'Phantom {date} {temp}°C: matched Glutamate t1 & t2',
        showlegend=False,  # Hide legend
        height=345,
        width=680,  # Set a width based on your preference
        margin=dict(l=10, r=0, t=60, b=20),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    # Add individual titles and separate colorbars
    for i, title in enumerate(['Real t1 [ms]', 'Real t2 [ms]'], start=1):
        fig.update_xaxes(row=1, col=i, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, row=1, col=i, showticklabels=False, autorange='reversed')  # Reverse the y-axis

    # Manually add separate colorbars
    colorbar_t1 = {'colorscale': custom_viridis, 'cmin': 0, 'cmax': 5000}
    colorbar_t2 = {'colorscale': custom_plasma, 'cmin': 0, 'cmax': 2000}

    fig.update_layout(
        coloraxis1=colorbar_t1,
        coloraxis2=colorbar_t2,
        coloraxis_colorbar=dict(x=0.43, y=0.5),
        coloraxis2_colorbar=dict(x=0.995, y=0.5),
    )

    # Show the plot
    fig.show()

    pio.write_image(fig, f'images/{save_name}/subject_{phantom_choice}/{fp_prtcl_name}_t.jpeg')


def dict_fs_ksw(fs_array, ksw_array, match_array, phantom_choice, fp_prtcl_name, subject_dict):
    """
    Create vial location for tag adding
    :param t1_pixels:
    :param: t2_pixels:
    :param: phantom_choice:
    :param: fp_prtcl_name:
    :param: subject_dict:
    """
    full_mask = subject_dict['full_mask']
    vial_rois = subject_dict['vial_rois']
    tag = subject_dict['tags']
    tag_x_loc = subject_dict['tag_x_locs']
    tag_y_loc = subject_dict['tag_y_locs']
    date = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    f_lims = subject_dict['dict_ranges']['fs_0']
    k_lims = subject_dict['dict_ranges']['ksw_0']
    tag_id = [tag.index('a'), tag.index('b'), tag.index('c')]

    roi_masks, vial_loc = vial_locator(full_mask, vial_rois)

    # Create custom Viridis colormap with black for 0 values
    custom_viridis = np.array(plotly.colors.sequential.Viridis)
    custom_viridis[0] = '#000000'  # Set black for 0 values

    # Create custom Plasma colormap with black for 0 values
    custom_plasma = np.array(plotly.colors.sequential.Plasma)
    custom_plasma[0] = '#000000'  # Set black for 0 values

    # Create custom Plasma colormap with black for 0 values
    custom_plotly3 = np.array(plotly.colors.sequential.Plotly3)
    custom_plotly3[0] = '#000000'  # Set black for 0 values

    # Create subplots with 1 row and 3 columns, increased horizontal spacing
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=['Glu [mM]', 'ksw [Hz]', 'Dot Product'])

    # Add heatmaps for the three arrays
    heatmap_fs = go.Heatmap(z=fs_array*full_mask, colorscale=custom_viridis, coloraxis='coloraxis1')
    heatmap_ksw = go.Heatmap(z=ksw_array*full_mask, colorscale=custom_plasma, coloraxis='coloraxis2')
    heatmap_match = go.Heatmap(z=match_array*full_mask, colorscale=custom_plotly3, coloraxis='coloraxis3')

    fig.add_trace(heatmap_fs, row=1, col=1)
    fig.add_trace(heatmap_ksw, row=1, col=2)
    fig.add_trace(heatmap_match, row=1, col=3)

    df = pd.DataFrame()

    # Loop over masks
    for j, mask in enumerate([roi_masks[0], roi_masks[1], roi_masks[2]]):
        # Calculate placement
        x_loc = int(np.round(vial_loc[j, 1]))
        y_loc = int(np.round(vial_loc[j, 0]))

        # Calculate mean and std
        [[fs_mean_val]], [[fs_std_val]] = cv2.meanStdDev(fs_array, mask=mask)
        [[ksw_mean_val]], [[ksw_std_val]] = cv2.meanStdDev(ksw_array, mask=mask)

        # Annotate subplot with mean ± std
        t1_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(fs_mean_val, 2)} ± {round(fs_std_val, 2)}<b>'
        t2_annotation = f'<b>{conc_l[j]} mM, pH {ph_l[j]}<br>{round(ksw_mean_val)} ± {round(ksw_std_val)}<b>'
        fig.add_annotation(text=t1_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x1', yref='y1', row=1, col=1)
        fig.add_annotation(text=t2_annotation, x=x_loc + tag_x_loc[j], y=y_loc + tag_y_loc[j],
                           showarrow=False, xref='x2', yref='y2', row=1, col=2)

        roi_row = pd.DataFrame({
            f'pH': ph_l[j],
            f'fs_mean': [fs_mean_val],
            f'fs_std': [fs_std_val],
            f'ksw_mean': [ksw_mean_val],
            f'ksw_std': [ksw_std_val]
        })

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, roi_row], ignore_index=True)

    fig.update_layout(
        template='plotly_dark',  # Set the theme to plotly dark
        title_text=f"Phantom {date} {temp}°C: '{fp_prtcl_name}' Glutamate MRF results",
        showlegend=False,  # Hide legend
        height=345,
        width=1020,  # Set a width based on your preference
        margin=dict(l=10, r=0, t=60, b=20),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    # Add individual titles and separate colorbars
    for i, title in enumerate(['Glu [mM]', 'ksw [Hz]', 'Dot Product'], start=1):
        fig.update_xaxes(row=1, col=i, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, row=1, col=i, showticklabels=False, autorange='reversed')  # Reverse the y-axis

    # Manually add separate colorbars
    f_const = 3 / 110000
    colorbar_fs = {'colorscale': custom_viridis, 'cmin': 0, 'cmax': f_lims[1]/f_const}
    colorbar_ksw = {'colorscale': custom_plasma, 'cmin': k_lims[0], 'cmax': k_lims[1]}
    colorbar_match = {'colorscale': custom_plotly3}

    fig.update_layout(
        coloraxis1=colorbar_fs,
        coloraxis2=colorbar_ksw,
        coloraxis3=colorbar_match,
        coloraxis_colorbar=dict(x=0.28, y=0.5),
        coloraxis2_colorbar=dict(x=0.64, y=0.5),
        coloraxis3_colorbar=dict(x=0.995, y=0.5),
    )

    # Show the plot
    fig.show()

    pio.write_image(fig, f'images/{save_name}/subject_{phantom_choice}/{fp_prtcl_name}.jpeg')

    mrf_df = pd.DataFrame(df.values[tag_id, :], index=['a', 'b', 'c'], columns=df.columns)

    return mrf_df

def plot_norm_sig(fp_prtcl_name, phantom_choice, subject_dict, real_ts_flag=False):
    """
    Create fig of normalized mrf signals
    :param: fp_prtcl_name:
    :param: phantom_choice:
    :param: subject_dict:
    :param: plot_norm_sig:
    """
    full_mask = subject_dict['full_mask']
    vial_rois = subject_dict['vial_rois']
    tag = subject_dict['tags']
    date = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    quant_data_fn = subject_dict['quant_data_fn']
    glu_phantom_mrf_files_fn = subject_dict['glu_phantom_mrf_files_fn']

    # # dict_fn = os.path.join('exp', date, fp_prtcl_name, 'dict.csv')
    # dict_fn = os.path.join('exp', fp_prtcl_name, 'dict.csv')
    # synt_df = pd.read_csv(dict_fn, header=0)
    #
    # # filter dict as needed
    # # dict_fn = os.path.join('exp', date, fp_prtcl_name, 'f_dict.csv')
    # dict_fn = os.path.join('exp', fp_prtcl_name, 'f_dict.csv')
    # df_masks = [synt_df[column].between(min_val, max_val) for column, (min_val, max_val) in
    #             subject_dict['dict_ranges'].items()]
    # filtered_df = synt_df[np.all(df_masks, axis=0)]
    # filtered_df.to_csv(path_or_buf=dict_fn, index=False)  # To save as CSV without row indices

    dict_fn = os.path.join('exp', save_name, fp_prtcl_name, 'f_dict.csv')

    abc_id = [tag.index('a'), tag.index('b'), tag.index('c')]  # [2, 0, 1]
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.07, horizontal_spacing=0.01)

    # Colors for the scatter plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    roi_masks, vial_loc = vial_locator(full_mask, vial_rois)

    for roi_i in abc_id:
        roi_mask = roi_masks[roi_i]
        mean_sigs, std_sigs, _ = norm_roi(roi_mask, quant_data_fn, dict_fn,
                                          glu_phantom_mrf_files_fn, subject_dict, roi_i, real_ts_flag)  # real, matched, expected

        x_axis = np.arange(0, len(mean_sigs[0]))
        scatter_real = go.Scatter(x=x_axis, y=mean_sigs[0], mode='lines',
                                  name=f'real {conc_l[roi_i]} mM, pH {ph_l[roi_i]}', line=dict(color=colors[roi_i]))
        scatter_match = go.Scatter(x=x_axis, y=mean_sigs[1], mode='lines',
                                   name=f'matched {conc_l[roi_i]} mM, pH {ph_l[roi_i]}',
                                   line=dict(color=colors[roi_i], dash='dash'), opacity=0.5)

        # Add traces to the first subplot
        fig.add_trace(scatter_real, row=1, col=1)
        fig.add_trace(scatter_match, row=1, col=1)

        fig.update_yaxes(title_text=f'Normalized signal', row=roi_i + 1, col=1)

    # Set layout for better visualization
    fig.update_layout(
        template='plotly_white',  # Set the theme to plotly white
        title_text=f"Signal Close Look: '{fp_prtcl_name}' Phantom {date} {temp}°C",
        showlegend=True,  # Hide legend
        legend=dict(x=1, y=1.05),
        height=300,
        width=900  # Set a width based on your preference
    )

    labels_to_show_in_legend = ['Real Signal', 'Matched Signal']

    # Adjust margin to reduce whitespace
    fig.update_layout(margin=dict(l=0, r=5, t=60, b=0))
    # Set y-axis ticks to values in the range [0, 1] with a step of 0.2
    # fig.update_yaxes(tickmode='linear', tick0=0, dtick=0.1, range=[0.14, 0.23])
    fig.update_yaxes(tickmode='linear', tick0=0, dtick=0.1)

    # Show the plot
    fig.show()
    pio.write_image(fig, f'images/{save_name}/subject_{phantom_choice}/{fp_prtcl_name}_sigs.jpeg')


from my_funcs.fitting_functions import multi_b1_fit
def multi_b_fit_plot(general_fn, txt_file_name, phantom_i, subject_dict, b1_i_lim, real_ts_flag = False):
    """
    :param: general_fn:
    :param: txt_file_name:
    :param: phantom_i:
    :param: subject_dict:
    :param: b1_i_lim:
    :param: real_ts_flag:
    """
    b1s = subject_dict['z_b1s']

    mean_z_spectrum_not_cor, mean_z_spectrum = z_spectra_creator(general_fn, txt_file_name, subject_dict)
    # sio.savemat(f'mar_phantom_2.mat', {'z_spectra': mean_z_spectrum})
    # mean_z_spectrum = mean_z_spectrum / mean_z_spectrum[0:1, 0:1, -1:]  # normalize by -7 ppm
    # mean_z_spectrum_not_cor = mean_z_spectrum_not_cor / mean_z_spectrum_not_cor[0:1, 0:1, -1:]  # normalize by -7 ppm

    b1_v = b1s[b1_i_lim[0]:b1_i_lim[1]]
    zi_v = mean_z_spectrum[0, b1_i_lim[0]:b1_i_lim[1], -1]  # initial magnetization before saturation block (M0=1)
    # zi_v = np.ones((len(b1_v)))
    w_df = np.arange(7, -7.25, -0.25)

    tag = subject_dict['tags']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    month = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    df = pd.DataFrame()

    # Iterate over each image
    for tag_i, cur_tag in enumerate(['a', 'b', 'c']):
        vial_i = tag.index(cur_tag)
        subject_dict['params']['fb'] = conc_l[vial_i] * 3 / 110000
        if real_ts_flag == True:
            t1_l = subject_dict['t1_l']
            t2_l = subject_dict['t2_l']
            # subject_dict['params']['t1w'] = t1_l[vial_i]
            # subject_dict['params']['t2w'] = t2_l[vial_i]
            subject_dict['params']['t1w'] = t1_l
            subject_dict['params']['t2w'] = t2_l
        fig = make_subplots(rows=1, cols=1)

        z_spec_mat = mean_z_spectrum[vial_i, b1_i_lim[0]:b1_i_lim[1], :]
        fit_result, fitted_z = multi_b1_fit(w_df, b1_v, zi_v, z_spec_mat, subject_dict['params'])
        fitted_z_rs = fitted_z.reshape((len(b1_v), 57))

        for b1_i, b1 in enumerate(b1_v):
            cur_z = z_spec_mat[b1_i, :]
            cur_z_fit = fitted_z_rs[b1_i, :]
            fig.add_trace(go.Scatter(x=w_df, y=cur_z, mode='markers', line=dict(color=colors[b1_i]),
                                     name=f'real z, {b1}uT'), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=w_df, y=cur_z_fit, line=dict(color=colors[b1_i]), opacity=0.5,
                           name=f'fit z, {b1}uT'), row=1, col=1)

        fig.update_yaxes(title_text='$M_{sat}/M_0$', row=1, col=1, title_standoff=2)

        # Update layout
        fig.update_layout(template='plotly_white',  # Set the theme to plotly white
                          title_text=f'Phantom {month} - {temp}°C multi-B fitting - {conc_l[vial_i]}mM, pH {ph_l[vial_i]}',
                          height=300, width=650,
                          title=dict(x=0.02, y=0.97),
                          margin=dict(l=45, r=0, t=30, b=0)
                          )  # Adjust the title position

        # Set axes for z-spectrum
        fig.update_xaxes(autorange='reversed', tickmode='linear', tick0=0, dtick=1)
        fig.update_yaxes(tickmode='linear', tick0=0, dtick=0.2, range=[-0.1, 1.15])

        fig.show()

        # After fitting the model
        # print(fit_result.fit_report())  # Print a summary of the fit results
        # print("Parameter Values:")
        # for name, param in fit_result.params.items():
        #     if name == 'fb':
        #         print(f"{name}: {param.value * 110000 / 3:.2f} +/- {param.stderr * 110000 / 3:.2f}")
        #     else:
        #         print(f"{name}: {param.value:.2f} +/- {param.stderr:.2f}")

        fb_val = fit_result.params['fb'].value * 110000 / 3
        fb_std = fit_result.params['fb'].stderr * 110000 / 3
        kb_val = fit_result.params['kb'].value
        kb_std = fit_result.params['kb'].stderr

        roi_row = pd.DataFrame({
            f'fb_mean': [fb_val],
            f'fb_std': [fb_std],
            f'ksw_mean': [kb_val],
            f'ksw_std': [kb_std]
        })

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, roi_row], ignore_index=True)

        pio.write_image(fig, f'images/{save_name}/subject_{phantom_i}/z_spec/multi_B_fit_{conc_l[vial_i]}mM_pH_{ph_l[vial_i]}.jpeg')

    B1fit_df = pd.DataFrame(df.values, index=['a', 'b', 'c'], columns=df.columns)

    return B1fit_df

def z_mtr_plot(general_fn, txt_file_name, phantom_i, subject_dict):
    """
    :param: general_fn:
    :param: txt_file_name:
    :param: phantom_i:
    :param: subject_dict:
    """
    mean_z_spectrum_not_cor, mean_z_spectrum = z_spectra_creator(general_fn, txt_file_name, subject_dict)

    cest_prtcl_names = subject_dict['z_b1s_names']
    tag = subject_dict['tags']
    conc_l = subject_dict['concs']
    ph_l = subject_dict['phs']
    month = subject_dict['month']
    save_name = subject_dict['save_name']
    temp = subject_dict['temp']
    b1s = subject_dict['z_b1s']
    b1s = [f'{x}uT' for x in b1s]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    abc_id = [tag.index('a'), tag.index('b'), tag.index('c')]

    fig = make_subplots(rows=1, cols=len(cest_prtcl_names),
                        subplot_titles=b1s, vertical_spacing=0.07, horizontal_spacing=0.01)
    fig_2 = make_subplots(rows=1, cols=len(cest_prtcl_names),
                          subplot_titles=b1s, vertical_spacing=0.07, horizontal_spacing=0.05)

    # Iterate over each channel
    for B_idx in range(len(cest_prtcl_names)):
        # Iterate over each image
        for tag_i, cur_tag in enumerate(['a', 'b', 'c']):
            vial_i = tag.index(cur_tag)
            z_vial_B = mean_z_spectrum[vial_i, B_idx, :]
            z_vial_B_not_cor = mean_z_spectrum_not_cor[vial_i, B_idx, :]

            ppm_lim = 7

            # MTR asym calculation
            cha_n = len(z_vial_B)
            mid_i = int(cha_n / 2 - 1)
            last_i = int(cha_n - 1)
            positives = z_vial_B[0:(mid_i + 2)]  # positives to 0 (including)
            negatives = z_vial_B[last_i:mid_i:-1]  # negatives to 0 (including)

            MTR_asym = negatives - positives
            ppm_asym = np.linspace(ppm_lim, 0, num=(mid_i + 2))

            ppm = np.linspace(ppm_lim, -ppm_lim, num=cha_n)

            col_idx = B_idx + 1
            fig.add_trace(go.Scatter(x=ppm, y=z_vial_B, line=dict(color=colors[tag_i]),
                                     name=f'{conc_l[vial_i]} mM, pH {ph_l[vial_i]}',
                                     legendgroup=f'{1}_{col_idx}_1'), row=1, col=col_idx)
            # fig.add_trace(go.Scatter(x=ppm, y=z_vial_B_not_cor, line=dict(color='gray'), name=f'{cur_tag} {conc_l[vial_i]}, {ph_l[vial_i]}', legendgroup=f'{1}_{col_idx}_1'), row=1, col=col_idx)  # later delete!

            fig.add_trace(go.Scatter(x=ppm_asym, y=MTR_asym, line=dict(color=colors[tag_i]),
                                     name=f'{conc_l[vial_i]}mM, pH {ph_l[vial_i]}', legendgroup=f'{1}_{col_idx}_2'), row=1,
                          col=col_idx)
            fig.update_xaxes(title_text='ppm', row=1, col=col_idx)

            fig_2.add_trace(go.Scatter(x=ppm_asym, y=MTR_asym, line=dict(color=colors[tag_i]),
                                       name=f'{conc_l[vial_i]}mM, pH {ph_l[vial_i]}', legendgroup=f'{1}_{col_idx}_2'), row=1,
                            col=col_idx)
            fig_2.update_xaxes(title_text='ppm', row=1, col=col_idx)

    fig.update_yaxes(title_text='$M_{sat}/M_0$', row=1, col=1, title_standoff=2)
    fig_2.update_yaxes(title_text='MTR-asym', row=1, col=1)

    # Update layout
    fig.update_layout(template='plotly_white',  # Set the theme to plotly white
                      title_text=f'Phantom {month} - {temp}°C Z-spectrum',
                      height=250, width=250*len(cest_prtcl_names)+150,
                      title=dict(x=0.02, y=0.97),
                      margin=dict(l=45, r=0, t=50, b=0)
                      )  # Adjust the title position

    fig_2.update_layout(template='plotly_white',  # Set the theme to plotly white
                        title_text=f'Phantom {month} - {temp}°C MTR-asym',
                        height=200, width=220*len(cest_prtcl_names)+150,
                        title=dict(x=0.02, y=0.97),
                        margin=dict(l=0, r=0, t=60, b=0)
                        )  # Adjust the title position

    # Set axes for z-spectrum
    fig.update_xaxes(autorange='reversed', tickmode='linear', tick0=0, dtick=1)
    fig.update_yaxes(tickmode='linear', tick0=0, dtick=0.2, range=[-0.1, 1])

    # # Set axes for MTR-asym
    fig_2.update_xaxes(autorange='reversed', dtick=1)
    # fig_2.update_xaxes(tickvals=list(range(5)), ticktext=list(map(str, range(4, -1, -1))))
    fig_2.update_yaxes(tickmode='linear', tick0=0, dtick=0.05, range=[0, 0.25], tickformat='.0%')

    # Show only specific legend groups
    for trace in fig.data:
        trace.showlegend = (trace.legendgroup == '1_1_1')

    # Show only specific legend groups
    for trace in fig_2.data:
        trace.showlegend = (trace.legendgroup == '1_1_2')

    fig.show()
    fig_2.show()

    pio.write_image(fig, f'images/{save_name}/subject_{phantom_i}/z_spec/zspec.jpeg')
    pio.write_image(fig_2, f'images/{save_name}/subject_{phantom_i}/z_spec/mtr.jpeg')


