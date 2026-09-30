"""
Shared utility functions for MRI-adapted ventriculomegaly feature computation.

Adapted from Sharada Samadani's CT-based pipeline for T1-weighted MRI.
Key change: No HU windowing — Otsu thresholding operates directly on raw MRI intensities.

All 7 feature notebooks import from this module instead of duplicating utility code.
"""

import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
from skimage import measure, filters
import statsmodels.api as sm


# =============================================================================
# Image orientation / coordinate transforms
# =============================================================================

def rotate_image(image, dimension=3):
    """Resamples a NIfTI volume to identity direction cosines for correct axial display.

    SimpleITK reads NIfTI in LPS; OASIS/ABCD data is RAS. This function sets the
    direction to [1,0,0, 0,1,0, 0,0,1] so axial slices appear upright.
    """
    transform = sitk.AffineTransform(dimension)
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) // 2.0))
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.SetMatrix(matrix.ravel())

    extreme_points = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), image.GetDepth())),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), image.GetDepth())),
    ]

    inv_transform = transform.GetInverse()
    extreme_points_transformed = [inv_transform.TransformPoint(pnt) for pnt in extreme_points]
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]

    output_spacing = image.GetSpacing()
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output_origin = [min_x, min_y, min_z]
    output_size = image.GetSize()

    return sitk.Resample(image, output_size, transform, sitk.sitkLinear,
                         output_origin, output_spacing, output_direction)


def rotate_image_coronal(image, dimension=3):
    """Like rotate_image but flips the Z-axis for correct coronal display.

    Coronal volumes (from CT pipeline) used direction cosine with Z=-1.
    For MRI single-volume coronal extraction, this handles the equivalent flip.
    """
    transform = sitk.AffineTransform(dimension)
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) // 2.0))
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    transform.SetMatrix(matrix.ravel())

    extreme_points = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), image.GetDepth())),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), image.GetDepth())),
    ]

    inv_transform = transform.GetInverse()
    extreme_points_transformed = [inv_transform.TransformPoint(pnt) for pnt in extreme_points]
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]

    output_spacing = image.GetSpacing()
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output_origin = [min_x, min_y, min_z]
    output_size = image.GetSize()

    return sitk.Resample(image, output_size, transform, sitk.sitkLinear,
                         output_origin, output_spacing, output_direction)


def translate_image(image, dimension, ac):
    """Translates the image so the AC world coordinates map to (0,0,0)."""
    transform = sitk.AffineTransform(3)
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) // 2.0))
    transform.SetTranslation([x for x in ac])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.SetMatrix(matrix.ravel())

    extreme_points = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), image.GetDepth())),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), image.GetDepth())),
    ]

    inv_transform = transform.GetInverse()
    extreme_points_transformed = [inv_transform.TransformPoint(pnt) for pnt in extreme_points]
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]

    output_spacing = image.GetSpacing()
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output_origin = [min_x, min_y, min_z]
    output_size = image.GetSize()

    return sitk.Resample(image, output_size, transform, sitk.sitkLinear,
                         output_origin, output_spacing, output_direction)


# =============================================================================
# View extraction (for ABCD single-volume T1w support)
# =============================================================================

def extract_coronal_view(img):
    """Extract coronal-oriented volume from a 3D axial volume.

    For ABCD single-volume data: permutes axes so the coronal dimension becomes
    the slice dimension. For OASIS atlas-registered data, direct numpy indexing
    (arr[:, y, :]) works since volumes are isotropic.

    Returns a SimpleITK image with coronal slices as the 'depth' axis.
    """
    return sitk.PermuteAxes(img, [0, 2, 1])


def extract_sagittal_view(img):
    """Extract sagittal-oriented volume from a 3D axial volume.

    Permutes axes so the sagittal dimension becomes the slice dimension.

    Returns a SimpleITK image with sagittal slices as the 'depth' axis.
    """
    return sitk.PermuteAxes(img, [1, 2, 0])


# =============================================================================
# Segmentation primitives
# =============================================================================

def getLargestCC(blobs_labels):
    """Returns the largest connected component from a labeled image."""
    if blobs_labels.max() == 0:
        raise ValueError('Blank segmentation, inspect processing up to here.')
    largestCC = blobs_labels == np.argmax(np.bincount(blobs_labels.flat)[1:]) + 1
    return largestCC


def ventricle_fullbrain_seg(ax_img_arr, visualize=False):
    """Segments ventricles and full brain mask from a 3D volume using Otsu + floodfill.

    For T1w MRI: brain tissue is bright, CSF/ventricles are dark.
    Otsu finds the threshold adaptively — no HU windowing needed.

    Returns
    -------
    ventricles : np.ndarray — binary mask of ventricles
    full_brain : np.ndarray — binary mask of full brain
    """
    ventricles = np.zeros(ax_img_arr.shape)
    full_brain = np.zeros(ax_img_arr.shape)

    for pl in range(len(ax_img_arr)):
        ax_plane = ax_img_arr[pl, :, :]
        if ax_plane.max() == 0:
            continue
        try:
            thresh = filters.threshold_otsu(ax_plane)
        except ValueError:
            continue

        ax_bin = ax_plane > thresh

        im_th = ax_bin.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_floodfill = im_th.copy()
        res = cv.floodFill(np.uint8(im_floodfill), mask, (0, 0), 255)
        full_brain_mask = (1 - res[2]).copy()
        full_brain_mask = full_brain_mask[1:h + 1, 1:w + 1]

        ventricles[pl, :, :] = full_brain_mask - ax_bin
        full_brain[pl, :, :] = full_brain_mask

    if visualize:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.set_title("Ventricles (IS projection)")
        ax1.imshow(np.sum(ventricles, axis=0), cmap="Greens")
        ax2.set_title("Ventricles (AP projection)")
        ax2.imshow(np.sum(ventricles, axis=1)[::-1, :], cmap="gray")
        ax3.set_title("Ventricles (LR projection)")
        ax3.imshow(np.sum(ventricles, axis=2)[::-1, :], cmap="gray")
        plt.tight_layout()
        plt.show()

    return ventricles, full_brain


# =============================================================================
# Mask primitives
# =============================================================================

def ellipse_mask(cen, a, b, shape):
    """Generates a 2D binary elliptical mask."""
    mask = np.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            if a > 0 and b > 0 and (row - int(cen[0]))**2 / a**2 + (col - int(cen[1]))**2 / b**2 <= 1:
                mask[row, col] = 1
    return mask


def circular_mask(cen, rad, shape):
    """Generates a 2D binary circular mask (vectorized)."""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center_sq = (Y - int(cen[0]))**2 + (X - int(cen[1]))**2
    return (dist_from_center_sq < rad**2).astype(np.uint8)


def elliptical_mask(cen, a, b, shape, angle):
    """Generates a 2D binary mask of a rotated ellipse (vectorized)."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    r_shifted = rows - int(cen[0])
    c_shifted = cols - int(cen[1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    term1 = (r_shifted * cos_a + c_shifted * sin_a) ** 2 / a ** 2
    term2 = (r_shifted * sin_a - c_shifted * cos_a) ** 2 / b ** 2
    return (term1 + term2 <= 1).astype(np.float64)


# =============================================================================
# Signal processing
# =============================================================================

def normalize_to_uint8(img_slice):
    """Normalizes a 2D image slice to uint8 (0-255) range.

    MRI intensities are arbitrary (e.g., 0-3126 for OASIS T1w), unlike CT
    which has standardized HU. This replaces the CT window_stack_sitk() step
    by linearly mapping the full intensity range to 0-255, which is required
    for cv2.adaptiveThreshold and consistent thresholding behavior.
    """
    arr = img_slice.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.uint8(255.0 * (arr - mn) / (mx - mn))


def moving_average(x, w):
    """1D moving average with zero-padding to preserve length."""
    x = np.uint8(x)
    pad_left = w // 2
    pad_right = (w - 1) // 2
    zero_pad_x = np.pad(x, (pad_left, pad_right), mode='constant', constant_values=(0, 0))
    return np.convolve(zero_pad_x, np.ones(w), 'valid') / w


# =============================================================================
# Contour tracing (for CallosalAngle and MaxEccLV)
# =============================================================================

def moore_contour(blob):
    """Moore neighborhood contour tracing — returns boundary pixels of a binary blob."""
    if blob.shape == (1, 1):
        raise ValueError('Single pixel in segmentation, inspect processing up to here.')

    blob_padded = np.pad(blob, (1, 1), mode='constant', constant_values=(0, 0))
    boundary_pts = []
    [row, col] = blob_padded.nonzero()
    start_row, start_col = min(row[col == min(col)]), min(col)
    boundary_pts.append((start_row, start_col))

    prev_row, prev_col = start_row, start_col - 1
    d_row, d_col = prev_row - start_row, prev_col - start_col

    moore_neighborhood = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

    count = 1
    b_row, b_col = start_row, start_col

    while count < 2:
        for step in range(len(moore_neighborhood)):
            if (d_row, d_col) == moore_neighborhood[step]:
                step_num = step
                break

        while True:
            cur_row, cur_col = b_row + moore_neighborhood[step_num][0], b_col + moore_neighborhood[step_num][1]
            if blob_padded[cur_row, cur_col] == 1:
                if (cur_row, cur_col) in boundary_pts:
                    count += 1
                    if count == 2:
                        break
                boundary_pts.append((cur_row, cur_col))
                d_row, d_col = prev_row - cur_row, prev_col - cur_col
                b_row, b_col = cur_row, cur_col
                break
            prev_row, prev_col = cur_row, cur_col
            step_num += 1
            if step_num == 8:
                step_num = 0

    moore_boundary_pts = [(x - 1, y - 1) for (x, y) in boundary_pts]
    contour = np.zeros(blob.shape)
    for pts in moore_boundary_pts:
        contour[pts[0], pts[1]] = 1
    return contour


def trace_left_contour(row, col, bot_con_row, ven_contour):
    """Traces left ventricle wall moving SE/S/E direction."""
    left_pts = []
    left_con_pts = []
    flat_area_count = 0
    height, width = ven_contour.shape

    if row > bot_con_row:
        row = bot_con_row

    while row <= bot_con_row and flat_area_count <= 10:
        if len(left_pts) > 1:
            if left_pts[-1][1] == left_pts[-2][1]:
                flat_area_count += 1
            else:
                flat_area_count = 0

        can_move_south = (row + 1 < height)
        can_move_east = (col + 1 < width)

        if can_move_south and can_move_east and (ven_contour[row + 1, col + 1] > 0):
            left_pts.append((col + 1, row + 1))
            left_con_pts.append((col + 1, row + 1))
            row, col = row + 1, col + 1
        elif can_move_east and (ven_contour[row, col + 1] > 0):
            left_pts.append((col + 1, row))
            left_con_pts.append((col + 1, row))
            row, col = row, col + 1
        elif can_move_south and (ven_contour[row + 1, col] > 0):
            left_pts.append((col, row + 1))
            left_con_pts.append((col, row + 1))
            row, col = row + 1, col
        elif can_move_east:
            left_pts.append((col + 1, row))
            row, col = row, col + 1
        else:
            break

    return left_pts, left_con_pts


def trace_right_contour(row, col, bot_con_row, ven_contour):
    """Traces right ventricle wall moving SW/S/W direction."""
    right_pts = []
    right_con_pts = []
    flat_area_count = 0
    height, width = ven_contour.shape

    if row > bot_con_row:
        row = bot_con_row

    while row <= bot_con_row and flat_area_count <= 10:
        if len(right_pts) > 1:
            if right_pts[-1][1] == right_pts[-2][1]:
                flat_area_count += 1
            else:
                flat_area_count = 0

        can_move_south = (row + 1 < height)
        can_move_west = (col - 1 >= 0)

        if can_move_south and can_move_west and (ven_contour[row + 1, col - 1] > 0):
            right_pts.append((col - 1, row + 1))
            right_con_pts.append((col - 1, row + 1))
            row, col = row + 1, col - 1
        elif can_move_west and (ven_contour[row, col - 1] > 0):
            right_pts.append((col - 1, row))
            right_con_pts.append((col - 1, row))
            row, col = row, col - 1
        elif can_move_south and (ven_contour[row + 1, col] > 0):
            right_pts.append((col, row + 1))
            right_con_pts.append((col, row + 1))
            row, col = row + 1, col
        elif can_move_west:
            right_pts.append((col - 1, row))
            row, col = row, col - 1
        else:
            break

    return right_pts, right_con_pts


def fit_lines(left_con_cand_pts, right_con_cand_pts):
    """Fits OLS lines to left and right contour points, returns interior angle in degrees."""
    x_l = [x for (x, y) in left_con_cand_pts]
    y_l = [y for (x, y) in left_con_cand_pts]
    x_l = sm.add_constant(x_l, has_constant='add')
    result_l = sm.OLS(y_l, x_l).fit()
    left_ven_slope = result_l.params[1]

    x_r = [x for (x, y) in right_con_cand_pts]
    y_r = [y for (x, y) in right_con_cand_pts]
    x_r = sm.add_constant(x_r, has_constant='add')
    result_r = sm.OLS(y_r, x_r).fit()
    right_ven_slope = result_r.params[1]

    return 180 - abs(np.degrees(np.arctan(right_ven_slope))) - abs(np.degrees(np.arctan(left_ven_slope)))


# =============================================================================
# AC/PC coordinate loading
# =============================================================================

def parse_acpc_coords(coord_str):
    """Parses '[x, y, z]' string to list of floats."""
    [x, y, z] = coord_str.split(",")
    return [float(x.strip("[")), float(y), float(z.strip("]"))]


def load_acpc_from_txt(txt_path):
    """Loads AC/PC coordinates from a .txt file (Charlie's ART script output).

    Expected format: two lines, each with 'x,y,z' or '[x,y,z]'.
    First line = AC, second line = PC.

    Returns (ac, pc) as lists of floats.
    """
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    ac = parse_acpc_coords(lines[0])
    pc = parse_acpc_coords(lines[1])
    return ac, pc


# =============================================================================
# Dataset configuration helpers
# =============================================================================

# OASIS T88 atlas voxel-space AC/PC (hardcoded for disc1 atlas-registered volumes)
OASIS_AC_VOXEL = (85, 110, 73)
OASIS_PC_VOXEL = (85, 87, 73)
