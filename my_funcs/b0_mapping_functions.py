import time
import numpy as np

from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate


# Define the Lorentzian function
def lorentz_iN(delta, cen, amp, sig, offset):  # delta must be first?
    """
    Creates lorentzian fit from params and dw vector?
    :param delta: w_Hz
    :param par: [center, amp, sigma, offset]
    :return b0_map:
    """
    denum = 1 + (sig / (delta - cen)) ** 2
    # y_fit = np.sqrt(par[3] ^ 2 + (par[1]] / denum + par[3]) ** 2)
    y_fit = offset + amp / denum
    # y_fit = sum(sum(abs(np.sqrt(par[3]] + (par[1] / denum) ** 2) - y)))
    return y_fit


def wassr_b0_mapping(Z_Ims_3DMat,
                     Brain_Mask,
                     w_x: np.array = np.arange(1, -1.1, -0.1),
                     B1_uT: float = 0.3,
                     MainFieldMHz: float = 298,
                     ):
    """
    Creates B0 WASSR mapping
    :param Z_Ims_3DMat: CEST z-spectrum images [21,64,64]
    :param w_x: [1, 0.9, ..., 0, -0.1, -0.2, ... -1]
    :param B1_uT: B1 pulse used [uT], 0.3 in WASSR
    :param MainFieldMHz:
    :param Brain_Mask:
    :return b0_map:
    """
    # B1 frequency (Hz)
    gyro_ratio_hz = 42.5764
    w1_Hz = round(B1_uT*gyro_ratio_hz)  # satpwr for WASSER (Hz)

    # Scanned frequency offsets (Hz)
    w_Hz = w_x * MainFieldMHz # (ppm * MHz = Hz)

    # Masking the z images to include only brain regions
    Z_Ims_3DMat = Z_Ims_3DMat.transpose(1, 2, 0)  # (21, 64, 64) -> (64, 64, 21)
    broadcasted_mask = np.broadcast_to(Brain_Mask[:, :, np.newaxis], Z_Ims_3DMat.shape)  # Broadcast the mask to all channels
    Masked_Z_Ims_3DMat = Z_Ims_3DMat * broadcasted_mask

    # Initializing B0map
    B0Map = np.zeros_like(Brain_Mask, dtype=float)

    # Spline interpolation of scanned offsets every 1 Hz
    Interp_w_Hz = np.arange(min(w_Hz), max(w_Hz)+1, 1)

    start = time.perf_counter()
    for r_ind in range(B0Map.shape[0]):
        for c_ind in range(B0Map.shape[1]):
            # Mapping only if the mask is nonzero in this pixel
            if Brain_Mask[r_ind, c_ind]:
                # Currant pixel Z
                CurZ = Masked_Z_Ims_3DMat[r_ind,c_ind,:]

                #Interpolating the pixel Z-spectrum every 1Hz
                tck = splrep(w_Hz[::-1], CurZ[::-1], s=0)  # Spline representation of the data (I needed to reverse)

                yy0 = np.argmin(splev(Interp_w_Hz, tck, der=0)) # Interpolate at specific Interp_w_Hz
                x0 = Interp_w_Hz[yy0]  # corresponding x values
                """up to here all matches up!"""

                x_data = w_Hz
                y_data = CurZ / np.max(CurZ)

                # Initial guess and bounds
                initial_guess = [x0, 50, w1_Hz * 2, 0.05]  # [center, amp, sigma, offset]
                bounds_lower = [x0 - 200, 1e-3, 1, 0]  # [center, amp, sigma, offset]
                bounds_upper = [x0 + 200, 1000, 500, 1]  # [center, amp, sigma, offset]

                # Perform curve fitting using scipy.optimize.curve_fit
                par, _ = curve_fit(lorentz_iN, x_data, y_data, p0=initial_guess, bounds=(bounds_lower, bounds_upper))

                B0Map[r_ind, c_ind] = par[0]  # center [Hz]

    end = time.perf_counter()
    print(f'WASSR B0 mapping took {end-start:.03f} seconds')
    return B0Map


def b0_correction(b0_map, original_images, w_hz):  # haven't checked it
    """
    :param b0_map: (Hz), optimally from a WASSR or WASABI scan
    :param original_images (3D array, row x col x num_images)
    :param w_hz: saturation frequency offsets (Hz)
    :return: b0_corrected_images (3D array, row x col x num_images)
    """

    # Initialization
    # original_images = original_images.transpose(1, 2, 0)  # (57/51/30, 64, 64) -> (64, 64, 57/51/30)
    b0_corrected_images = np.zeros(np.shape(original_images))

    for r_ind in range(original_images.shape[1]):
        for c_ind in range(original_images.shape[2]):

            # Current pixel original z-spectrum
            cur_pixel_orig_z_vec = original_images[:, r_ind, c_ind].reshape(-1, 1)

            # Current pixel B0 shift (Hz)
            cur_pixel_b0 = b0_map[r_ind, c_ind]

            # Correcting current pixel if B0 shift is not zero
            if cur_pixel_b0 != 0:
                # Cubic spline interpolation (initially flipping for splrep compatibility)
                tck = interpolate.splrep(np.flipud(w_hz - cur_pixel_b0), np.flipud(cur_pixel_orig_z_vec), s=0)
                b0_corrected_images[:, r_ind, c_ind] = np.squeeze(interpolate.splev(w_hz, tck, der=0))

    # b0_corrected_images = b0_corrected_images.transpose(2, 0, 1)  # (64, 64, 57/51/30) -> (57/51/30, 64, 64)

    return b0_corrected_images

