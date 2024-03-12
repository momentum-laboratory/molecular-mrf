import numpy as np
import matplotlib.pyplot as plt
import cv2

def image_overlay(background, overlay, mask, cmap_bg, cmap_ov, clim_ov, clim_bg=None, alphabg=1, alphafg=1):
    # Convert color maps from strings if necessary
    cmap_bg = plt.get_cmap(cmap_bg) if isinstance(cmap_bg, str) else cmap_bg
    cmap_ov = plt.get_cmap(cmap_ov) if isinstance(cmap_ov, str) else cmap_ov

    # Resize mask to match background dimensions
    mask_hr = cv2.resize(mask.astype(float), (background.shape[1], background.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Ensure overlay is resized to match background dimensions as well
    overlay_resized = cv2.resize(overlay, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply normalization and color maps
    if clim_bg:
        norm_bg = plt.Normalize(*clim_bg)
    else:
        norm_bg = plt.Normalize(background.min(), background.max())
    norm_ov = plt.Normalize(*clim_ov)

    # background *= ~mask_hr
    overlay_resized *= mask_hr

    background_norm = norm_bg(background)
    overlay_norm = norm_ov(overlay_resized)

    image_bg = cmap_bg(background_norm)
    image_ov = np.zeros_like(image_bg)
    
    # Apply the mask to the overlay image
    overlay_mapped = cmap_ov(overlay_norm)
    image_ov[mask_hr] = overlay_mapped[mask_hr]

    # set alpha channels based on the mask
    image_bg[mask_hr] *= (1-alphafg)
    image_ov[~mask_hr] *= alphabg

    blendarray = alphabg * image_bg + alphafg * image_ov
    return blendarray