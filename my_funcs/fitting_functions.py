import time
import numpy as np
from numpy import linalg as la
import os

import cv2
import lmfit
from lmfit import Model

from my_funcs.cest_functions import bruker_dataset_creator
from my_funcs.cest_functions import dicom_data_arranger
from my_funcs.plot_functions import t1_t2_pixel_reader
from my_funcs.plot_functions import mask_roi_finder
from my_funcs.plot_functions import vial_locator

def z_cw_creator(w_df,  # ppm
                 b1_v,
                 zi_v,
                 kb, t1w, t2w,
                 tp, fb, t1s, t2s,
                 gyro_ratio_hz=42.5764,  # gamma for H [Hz/uT]
                 b0=7):
    gamma = 267.5153  # [MHz]

    # Define the function as a lambda expression
    rex_lorentz = lambda da, w1, db, fb, kb, r2b, zi: (
            (
                    (kb * fb) * w1 ** 2 / (da ** 2 + w1 ** 2) *
                    ((da - db) ** 2 + (da ** 2 + w1 ** 2) * r2b / kb + r2b * (kb + r2b))
            ) / (
                    ((2 * np.sqrt((kb + r2b) / kb * w1 ** 2 + (kb + r2b) ** 2)) / 2) ** 2 + db ** 2
            )
    )
    r1w = 1 / t1w
    r2w = 1 / t2w
    r1s = 1 / t1s
    r2s = 1 / t2s
    z_cw = np.array([])  # (4, 57)
    for b1_i, b1 in enumerate(b1_v):
        zi = zi_v[b1_i]
        w1 = b1 * gamma
        w_ref = b0 * gyro_ratio_hz * 2 * np.pi
        da = (w_df - 0) * w_ref  # 0 ppm water
        db = (w_df - 3) * w_ref  # 3 ppm solute

        w1 = np.full(len(w_df), w1)
        da = np.where(da == 0, 1e-8, da)
        theta = np.arctan(w1 / da)

        rex = rex_lorentz(da, w1, db, fb, kb, r2s, zi)  # exchange dependant relaxation (rotating frame)
        reff = r1w * np.cos(theta) ** 2 + r2w * np.sin(theta) ** 2  # R1rho of water
        r1rho = reff + rex
        pz = np.cos(theta)
        pzeff = np.cos(theta)
        r1rho = np.where(r1rho == 0, 1e-8, r1rho)
        zss = np.cos(theta) * pz * r1w / r1rho
        cur_z_cw = (pz * pzeff * zi - zss) * np.exp(-(r1rho * tp)) + zss
        z_cw = np.concatenate((z_cw, cur_z_cw))

    return z_cw


def multi_b1_fit_per_pixel(
        w_df,
        b1_v,
        zi_v,
        z_spec_mat,  # (vial i, b1 i, 7 to -7)
        params
):
    # I will try varying kb, r1w, r2w (maybe r1s, r2s next?)
    piecewisez_cw = lambda w_df, kb, t1w, t2w, tp, fb, t1s, t2s: (
        z_cw_creator(w_df, b1_v, zi_v, kb, t1w, t2w, tp, fb, t1s, t2s))

    ft = Model(piecewisez_cw)
    # Add bounds to parameters
    opts = ft.make_params(method='leastsq')  # Method for optimization

    for param in params.values():
        if param['type'] == 'vary':
            # varying
            opts.add(param['name'], value=param['val'], min=param['min'], max=param['max'])

        elif param['type'] == 'const':
            # constant
            opts.add(param['name'], value=param['val'], vary=False)

    zs = np.array([])
    for b1_i, b1 in enumerate(b1_v):
        zs = np.concatenate((zs, z_spec_mat[b1_i, :]))  # 7 to -7

    # Perform the curve fitting
    result = ft.fit(zs, w_df=w_df,
                    kb=opts['kb'], t1w=opts['t1w'], t2w=opts['t2w'],
                    tp=opts['tp'], fb=opts['fb'], t1s=opts['t1s'], t2s=opts['t2s'],
                    method='least_squares')

    # # Extract fit results and goodness-of-fit
    # fitresult = result.best_values
    # gof = result.fit_report(result.params)
    # ci = lmfit.conf_interval(result, result, sigmas=[2])  # not exact same 2 sigma instead of 95%
    # kb_ci = ci['kb'][2][1] - np.mean([ci['kb'][2][1], ci['kb'][0][1]])  # 95%
    # fb_ci = ci['fb'][2][1] - np.mean([ci['fb'][2][1], ci['fb'][0][1]])  # 95%
    fitted_z = result.best_fit

    return result, fitted_z

def multi_b1_fit(w_df,
                 b1_v,
                 zi_v,
                 z_spec_mat,  # (vial i, b1 i, 7 to -7)
                 params):
    # z_cw = z_cw_creator(params, b1_v, w_df)

    # I will try varying kb, r1w, r2w (maybe r1s, r2s next?)
    piecewisez_cw = lambda w_df, kb, t1w, t2w, tp, fb, t1s, t2s: (
        z_cw_creator(w_df, b1_v, zi_v, kb, t1w, t2w, tp, fb, t1s, t2s))

    # u = piecewisez_cw(w_df, 7000, 1/4, 1/1.8, 3, 0.01, 1/1, 1/0.04)

    ft = Model(piecewisez_cw)
    # Add bounds to parameters
    opts = ft.make_params(method='leastsq')  # Method for optimization

    # varying
    opts.add('kb', value=7500, min=5100, max=12000)  # min=5100, max=12000
    # opts.add('t1w', value=params['t1w'], min=3.35, max=4.2)
    # opts.add('t2w', value=params['t2w'], min=1.5, max=1.9)
    # opts.add('t1s', value=params['t1s'], min=1, max=4.5)
    # opts.add('t2s', value=params['t2s'], min=0.007, max=0.3)

    # constant
    opts.add('tp', value=params['tp'], vary=False)
    opts.add('fb', value=params['fb'], vary=False)
    opts.add('t1w', value=params['t1w'], vary=False)
    opts.add('t2w', value=params['t2w'], vary=False)
    opts.add('t1s', value=params['t1s'], vary=False)
    opts.add('t2s', value=params['t2s'], vary=False)

    # b1s = np.array([])
    zs = np.array([])
    for b1_i, b1 in enumerate(b1_v):
        # b1s = np.concatenate((b1s, np.full(len(w_df), b1)))
        zs = np.concatenate((zs, z_spec_mat[b1_i, :]))  # 7 to -7

    # Perform the curve fitting
    result = ft.fit(zs, w_df=w_df,
                    kb=opts['kb'], t1w=opts['t1w'], t2w=opts['t2w'],
                    tp=opts['tp'], fb=opts['fb'], t1s=opts['t1s'], t2s=opts['t2s'],
                    method='least_squares')

    # # Extract fit results and goodness-of-fit
    # fitresult = result.best_values
    # gof = result.fit_report(result.params)
    # ci = lmfit.conf_interval(result, result, sigmas=[2])  # not exact same 2 sigma instead of 95%
    # kb_ci = ci['kb'][2][1] - np.mean([ci['kb'][2][1], ci['kb'][0][1]])  # 95%
    # fb_ci = ci['fb'][2][1] - np.mean([ci['fb'][2][1], ci['fb'][0][1]])  # 95%
    fitted_z = result.best_fit

    return result, fitted_z

def quesp_calculator(glu_phantom_fn,
                     txt_file_name,
                     f,
                     t1_mean = None,
                     gyro_ratio_hz = 42.5764,  # gamma for H [Hz/uT]
                     b0 = 7,
                     ):
    """
    Calculates quesp match per pixel
    :param glu_phantom_fn: root->scans->date->subject
    :param txt_file_name: the descriptory txt file name
    :param t1_mean: calculated from t1 map, can be changed manually [s]
    :param gyro_ratio_hz: 42.5764 [Hz/uT]
    :param b0: 7 [T]
    :return quesp: dict of quesp resuls
    :return quesp: dict of inverted quesp resuls
    """
    glu_phantom_dicom_fn, glu_phantom_mrf_files_fn, bruker_dataset = bruker_dataset_creator(glu_phantom_fn, txt_file_name, 'quesp')
    _, _, bruker_dataset_mask = bruker_dataset_creator(glu_phantom_fn, txt_file_name, '107a')  # always takes mask from 107a
    vial_rois, mask, bg_mask = mask_roi_finder(bruker_dataset_mask)

    if t1_mean is None:
        t1_pixels = t1_t2_pixel_reader(glu_phantom_fn=glu_phantom_fn, txt_file_name=txt_file_name, image_idx=3,
                                       t_type='t1')
        [[t1_mean]], [[t1_std]] = cv2.meanStdDev(t1_pixels, mask=(bg_mask * 255).astype(np.uint8))
        t1_mean = t1_mean / 1000  # t1 [s]

    tp = (bruker_dataset['PVM_MagTransPulse1'].value[0] / 1000)
    single_offset = (bruker_dataset['Fp_SatOffset'].value / (gyro_ratio_hz * b0))  # 3 ppm
    m = int(len(single_offset)/2)
    b1_v = bruker_dataset['Fp_SatPows'].value[(m-1)::-1]  # [0 1 2 3 4 5 6 0 1 2 3 4 5 6]

    quesp_data = dicom_data_arranger(bruker_dataset, glu_phantom_dicom_fn)

    roi_masks, vial_loc = vial_locator(mask, vial_rois)
    roi_mask = roi_masks[2]
    r_locs, c_locs = np.where(roi_mask)

    sig = quesp_data[:, r_locs, c_locs]
    # n_sig = sig / la.norm(sig, axis=0)

    sig_mean = np.mean(sig, axis=1)
    # n_sig_std = np.std(n_sig, axis=1)

    Zlab = sig_mean[(m-1)::-1]  # positive frequencies in [6 5 4 3 2 1 0] ppm
    Zref = sig_mean[m:][::-1]  # negative frequencies in [6 5 4 3 2 1 0] ppm
    Zlab = np.divide(Zlab, np.where(Zref[-1] == 0, 1e-8, Zref[-1]))  # (7, 64, 64), normalized by last image
    Zref = np.divide(Zref, np.where(Zref[-1] == 0, 1e-8, Zref[-1]))  # (7, 64, 64), normalized by last image

    # # demo data:
    # mask = np.array([1])
    # mask = mask[:, np.newaxis, np.newaxis]  # Reshape the array to (6, 1, 1)
    #
    # Zlab = np.array([0.557904227609697,	0.569624060150376,	0.585461099429258,	0.607014028056112,	0.641097536551172,	0.708425055033020])
    # Zref = np.array([0.990082147866159,	0.998496240601504,	0.995494142385101,	1.00020040080160,	0.999399158822351,	0.999299579747849])
    # Zlab = Zlab[:, np.newaxis, np.newaxis]  # Reshape the array to (6, 1, 1)
    # Zref = Zref[:, np.newaxis, np.newaxis]  # Reshape the array to (6, 1, 1)
    # # x_data_full = np.broadcast_to(x_data_3d, (7, 64, 64))  # Broadcast the array to (7, 64, 64)
    #
    # b1_v = np.array([35, 30, 25, 20, 15, 10])
    # Zlab = np.divide(Zlab, np.where(Zref[-1] == 0, 1e-8, Zref[-1]))  # (7, 64, 64), normalized by last image
    # Zref = np.divide(Zref, np.where(Zref[-1] == 0, 1e-8, Zref[-1]))  # (7, 64, 64), normalized by last image

    t1_mean = 4

    P = {'tp': tp,
         'R1': 1/t1_mean,
         'Zi': 1,
         'pulsed': 0,
         'B1': b1_v,
         'fB': f * 3 / 110000}

    quesp = {
        'kBA': [np.zeros([1]), np.zeros([1])],
        'fB': [np.zeros([1]), np.zeros([1])],
        'MTR': [],
        'w_x': [],
        'rsquare': np.zeros([1])  #result.rsquared
        # I didn't save fit
    }

    quesp_inv = {
        'kBA': [np.zeros([1]), np.zeros([1])],
        'fB': [np.zeros([1]), np.zeros([1])],
        'MTR': [],
        'w_x': [],
        'rsquare': np.zeros([1])  #result.rsquared
        # I didn't save fit
    }
    for ii in range(2):
        if ii==0:
            # quesp
            MTR = Zref-Zlab
            quesp['MTR'] = MTR
            modelstr = lambda w1, fb, kb, R1, x: (
                    fb * kb * w1 ** 2 / (w1 ** 2 + kb ** 2) / (R1 + fb * kb * w1 ** 2 / (w1 ** 2 + kb ** 2)) -
                    (P['Zi'] - R1 / (R1 + fb * kb * w1 ** 2 / (w1 ** 2 + kb ** 2))) *
                    np.exp(-(R1 + fb * kb * w1 ** 2 / (w1 ** 2 + kb ** 2)) * x) + (P['Zi'] - 1) * np.exp(-R1 * x))

        elif ii==1:
            # inverse quesp
            Zref_i = np.divide(1, np.where(Zref == 0, 1e-8, Zref))
            Zlab_i = np.divide(1, np.where(Zlab == 0, 1e-8, Zlab))
            MTR = Zlab_i - Zref_i
            quesp_inv['MTR'] = MTR
            modelstr = lambda w1, fb, kb, R1, x: fb * kb * w1 ** 2 / (w1 ** 2 + kb ** 2) / R1 * x / x

            # I didn't copy pulsed part

        x_data = np.transpose(P['B1']) * gyro_ratio_hz * 2 * np.pi # (7,)
        # x_data_3d = x_data[:, np.newaxis, np.newaxis]  # Reshape the array to (7, 1, 1)
        # x_data_full = np.broadcast_to(x_data_3d, (7, 64, 64))  # Broadcast the array to (7, 64, 64)

        # Create a Model object with the defined function
        ft = Model(modelstr)

        # Add bounds to parameters
        opts = ft.make_params(method='leastsq')  # Method for optimization
        opts.add('fb', value=P['fB'], vary=False)
        # opts.add('fb', value=0.000135, min=0.0000135, max=0.0008181818)  # 0-30
        opts.add('kb', value=7500, min=2000, max=12000)
        opts.add('R1', value=P['R1'], vary=False)
        opts.add('x', value=P['tp'], vary=False)

        # print(r_ind, c_ind)
        # Perform the curve fitting
        y_data = MTR  # (7,) example

        result = ft.fit(y_data, w1=x_data, fb=opts['fb'], kb=opts['kb'], R1=opts['R1'], x=opts['x'],
                        method='least_squares')  # I reversed the vectors (b1 from high to low)

        # Extract fit results and goodness-of-fit
        fitresult = result.best_values
        gof = result.fit_report(result.params)
        # ci = lmfit.conf_interval(result, result, sigmas=[2])  # not exact same 2 sigma instead of 95%
        # kb_ci = ci['kb'][2][1] - np.mean([ci['kb'][2][1], ci['kb'][0][1]])  # 95%
        # fb_ci = ci['fb'][2][1] - np.mean([ci['fb'][2][1], ci['fb'][0][1]])  # 95%

        if ii == 0:
            quesp['kBA'][0] = fitresult['kb']
            # quesp['kBA'][1][r_ind, c_ind] = kb_ci
            quesp['fB'][0] = fitresult['fb'] * 110000 / 3
            # quesp['fB'][1][r_ind, c_ind] = fb_ci
            quesp['w_x'] = x_data
            quesp['rsquare'] = result.rsquared

        elif ii == 1:
            quesp_inv['kBA'][0] = fitresult['kb']
            # quesp['kBA'][1][r_ind, c_ind] = kb_ci
            quesp_inv['fB'][0] = fitresult['fb'] * 110000 / 3
            # quesp['fB'][1][r_ind, c_ind] = fb_ci
            quesp_inv['w_x'] = x_data
            quesp_inv['rsquare'] = result.rsquared

    return quesp, quesp_inv

# general_fn = os.path.dirname(os.path.abspath(os.curdir))
# glu_phantom_fn = os.path.join(general_fn, 'scans', '24_02_12_glu_phantom_vardeg',
#                                 '2_glu_phantom_20_16_12mM_ph7_dec_37deg')  # February
# txt_file_name = 'labarchive_notes.txt'
# quesp, quesp_inv = quesp_calculator(glu_phantom_fn, txt_file_name, 20)
# a = 5