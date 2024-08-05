import os
import nrrd
import numpy as np

from skimage.measure import regionprops
from skimage.draw import line
from skimage import measure
from skimage.morphology import remove_small_objects, binary_opening, disk

from scipy.ndimage import binary_fill_holes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import cv2

def mask_mirrorer(mask, center_line):
    (x0, y0), (x1, y1) = center_line
    height, width = mask.shape
    mirrored_mask = np.zeros_like(mask)

    # Find the indices where the mask is 1
    indices = np.argwhere(mask == 1)

    # Calculate the vector from (x0, y0) to (x1, y1)
    dx = x1 - x0
    dy = y1 - y0

    # Calculate the length of the vector
    length = np.sqrt(dx ** 2 + dy ** 2)

    # Normalize the vector
    dx /= length
    dy /= length

    # Calculate the normal vector (perpendicular to the centerline)
    normal_x = -dy
    normal_y = dx

    # Iterate through each point in the original mask
    for y, x in indices:
        # Calculate the distance from the point (x, y) to the centerline
        distance = (x - x0) * normal_x + (y - y0) * normal_y

        # Calculate the mirrored coordinates
        x_mirror = int(x - 2 * distance * normal_x)
        y_mirror = int(y - 2 * distance * normal_y)

        # Check if the mirrored coordinates are within the mask boundaries
        if 0 <= x_mirror < width and 0 <= y_mirror < height:
            # Mark the mirrored pixel in the mirrored mask as 1
            mirrored_mask[y_mirror, x_mirror] = 1
    mirrored_mask = binary_fill_holes(mirrored_mask)
    return binary_fill_holes(mirrored_mask)

def center_line_creator(mask):
    centroid = regionprops(mask)[0].centroid  # 20
    orientation = regionprops(mask)[0].orientation + 0.01  # again manual

    # regionprops(label_image)[0].
    # Plot the orientation line
    x0, y0 = centroid[1], centroid[0]
    d = 25
    x1 = x0 - np.cos(orientation) * d
    y1 = y0 + np.sin(orientation) * d
    x2 = x0 + np.cos(orientation) * d
    y2 = y0 - np.sin(orientation) * d

    center_line = (x1, y1), (x2, y2)
    return center_line


def contour_finder(cur_mask):
    contour = measure.find_contours(image=cur_mask, level=0.5)[0]
    shift = 0.5
    contour[:, 1] += shift
    contour[:, 0] += shift

    # Find and add additional points to improve the contour plot
    points = []
    for idx0 in range(len(contour)):
        pos0 = contour[idx0]
        idx1 = idx0 + 1
        if len(contour) > idx1:
            pos1 = contour[idx1]
            y1, x1 = tuple(pos1)
            y0, x0 = tuple(pos0)
            if y1 - y0 and x1 - x0:
                point = np.array([y0, x1])
                if y0 % 1:
                    point = np.array([y1, x0])
                points.append((idx0, point))
    points.reverse()
    for idx0, point in points:
        contour = np.insert(contour, idx0 + 1, point, axis=0)

    # Deshift the contour before returning
    contour[:, 1] -= shift
    contour[:, 0] -= shift

    return contour

def downsampler(mask, resratio):
    # Resize the binary array to lower resolution (times 4)
    downsampled_mask = cv2.resize(mask.astype(np.uint8) * 255, None, fx=1/resratio, fy=1/resratio, interpolation=cv2.INTER_NEAREST)
    # Threshold the downsampled array to convert it back to a binary array
    downsampled_mask = np.where(downsampled_mask > 127, 1, 0)

    return downsampled_mask

def mask_processor(glu_mouse_fn, resratio, idx=0):
    mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'mask.seg.nrrd')
    mask = nrrd.read(mask_fn, index_order='C')[0][idx, : ,:]

    mask = downsampler(mask, resratio)

    return mask


def tumor_masks_processor(glu_mouse_fn, resratio, idx=0):
    mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'mask.seg.nrrd')
    tumor_mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'tumor.seg.nrrd')
    mask = nrrd.read(mask_fn, index_order='C')[0][idx, : ,:]
    tumor_mask = nrrd.read(tumor_mask_fn, index_order='C')[0][idx, : ,:]

    center_line = center_line_creator(mask)
    mirror_mask = mask_mirrorer(tumor_mask, center_line)

    # Find the overlapping regions (where both masks are 1)
    overlap = np.logical_and(tumor_mask == 1, mirror_mask == 1)

    # Set the overlapping regions to 0 in both masks
    tumor_mask[overlap] = 0
    mirror_mask[overlap] = 0

    selem = disk(3)
    tumor_mask = binary_opening(tumor_mask, selem)
    mirror_mask = binary_opening(mirror_mask, selem)

    ipsi_mask = downsampler(tumor_mask, resratio)
    contra_mask = downsampler(mirror_mask, resratio)

    return ipsi_mask, contra_mask

def tumor_contra_masks_processor(glu_mouse_fn, resratio, idx=0):
    mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'mask.seg.nrrd')
    tumor_mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'tumor.seg.nrrd')
    contra_mask_fn = os.path.join(glu_mouse_fn, 'slicer_masks', 'contra.seg.nrrd')
    mask = nrrd.read(mask_fn, index_order='C')[0][idx, : ,:]
    tumor_mask = nrrd.read(tumor_mask_fn, index_order='C')[0][idx, : ,:]
    contra_mask = nrrd.read(contra_mask_fn, index_order='C')[0][idx, :, :]

    ipsi_mask = downsampler(tumor_mask, resratio)
    contra_mask = downsampler(contra_mask, resratio)

    return ipsi_mask, contra_mask