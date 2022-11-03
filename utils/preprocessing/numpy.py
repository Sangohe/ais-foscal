"""Numpy functions to normalize, resize, compute weights and check for 
the integrity of the medical images and masks of ischemic stroke patients."""

import cv2
import numpy as np

from scipy import ndimage
from typing import Tuple, Callable

# Mask weights.
# ----------------------------------------------------------------


def compute_sample_weights_for_mask(
    mask: np.ndarray, class_weights: np.ndarray
) -> np.ndarray:
    """Create a `sample_weights` array with the same dimensions of
    `mask`. The values of the `sample_weights` array will be determined
    by the `class_weights` array.

    Args:
        mask (np.ndarray): reference mask
        class_weights (np.ndarray): importance of each class

    Returns:
        np.ndarray: sample weights
    """
    assert_mask_integrity(mask)
    assert class_weights.sum() == 1.0
    sample_weights = np.take(class_weights, mask.astype(np.int64))
    return sample_weights


def label_uncertainty(mask: np.ndarray) -> np.ndarray:
    """Computes the label uncertainty weights proposed in:
    https://arxiv.org/abs/2102.04566

    Args:
        mask (np.ndarray): reference mask

    Returns:
        np.ndarray: uncertainty weights
    """
    mask = mask.astype(np.float32)
    std = mask.std()
    exp_num = ndimage.distance_transform_edt(mask) ** 2
    exp_den = 2 * (std**2)
    exp = np.exp(-(exp_num / exp_den))

    return 1 - exp


# Data normalization, resizing and mask binarization.
# ----------------------------------------------------------------


def z_normalization(data: np.ndarray, min_divisor: float = 1e-3) -> np.ndarray:
    """Returns a Z normalized data array

    Args:
        data (np.ndarray): array to be normalized
        min_divisor (float, optional). defaults to 1e-3.

    Returns:
        np.ndarray: Normalized data
    """
    mean = data.mean()
    std = data.std()
    if std < min_divisor:
        std = min_divisor
    return (data - mean) / std


def min_max_normalization(data: np.ndarray) -> np.ndarray:
    """Returns a min-max normalized data array. The values for the
    normalized array will lie between 0 and 1.

    Args:
        data (np.ndarray): array to be normalized

    Returns:
        np.ndarray: Normalized data
    """
    min = data.min()
    max = data.max()
    return (data - min) / (max - min)


def resize_data(data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Function to resize medical images except the manually delineated
    masks. This function applies a bilinear interpolation to all the
    channels/slices in `data` to obtain a resized version.

    Args:
        data (np.ndarray): medical images to resize.
        target_size (Tuple[int, int]): target dimensionality

    Returns:
        np.ndarray: resized data
    """
    resized_data = cv2.resize(data, target_size)
    return resized_data


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Function to resize the manually delineated masks. This function
    applies a nearest neighbor interpolation to all the channels/slices
    in `mask` to obtain a resized version.

    Args:
        mask (np.ndarray): mask to resize.
        target_size (Tuple[int, int]): target dimensionality

    Returns:
        np.ndarray: resized mask
    """
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_mask


def binarize_array(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Returns a binary array by applying a threshold to `arr`. All
    the values of arr greater than the threshold are set to 1. The
    remaining values are set to 0.

    Args:
        arr (np.ndarray): array to binarize
        threshold (float, optional): Defaults to 0.5.

    Returns:
        np.ndarray: binarized array
    """
    return (arr > threshold).astype(np.float32)


def transform_data_and_mask(
    data: np.ndarray, mask: np.ndarray, transformation: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns in a single called the transformed data and mask
    using the signature of Augmentations library functions.

    Args:
        data (np.ndarray): data to be resized/augmented
        mask (np.ndarray): mask to be resized/augmented
        transformation (Callable): Augmentation function

    Returns:
        Tuple[np.ndarray, np.ndarray]: augmented data and mask
    """
    aug_data = transformation(image=data, mask=mask)
    return aug_data["image"], aug_data["mask"]


# Add contours to masks.
# ----------------------------------------------------------------


def get_mask_with_contours(mask: np.ndarray, contour_thickness: int = 1) -> np.ndarray:
    """Takes a 2D or 3D mask and draw contours using OpenCV's findCountours
    function.

    Considerations:
        - Note that contours are drawn at slice-level.
        - If all values of mask are zeros, the returned value is the same mask,
        otherwise, mask will have 3 classes belonging to {0, 1, 2}. Class 0 is
        background, 1 is lesion, 2 is contour.
    """

    if np.count_nonzero(mask) == 0:
        return mask.astype(np.float32)

    mask = mask.astype(np.int32)
    if mask.ndim == 3:

        # Transpose to (slices, H, W) if needed.
        if mask.shape[0] == mask.shape[1]:
            channels_last = True
            mask = mask.transpose(2, 0, 1)
        else:
            channels_last = False

        # Create the mask with contours.
        mask_with_contours = []
        for mask_slice in mask:
            mask_slice_with_contours = _draw_contours_on_mask(
                mask_slice, contour_thickness=contour_thickness
            )
            mask_with_contours.append(mask_slice_with_contours)

        # Create the volume with the slices and transpose if needed.
        mask_with_contours = np.stack(mask_with_contours)
        if channels_last:
            mask_with_contours = mask_with_contours.transpose(1, 2, 0)

    elif mask.ndim == 2:
        mask_with_contours = _draw_contours_on_mask(
            mask, contour_thickness=contour_thickness
        )
    else:
        raise ValueError("Mask must be an array of 2 or 3 dimensions.")

    return mask_with_contours.astype(np.float32)


def _draw_contours_on_mask(mask: np.ndarray, contour_thickness: int = 1) -> np.ndarray:
    """Returns a mask with the highlighted contours"""

    mask_copy = mask.copy()
    contours, hierarchy = cv2.findContours(
        mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(mask_copy, contours, -1, 2, contour_thickness)
    return mask_copy


# Assert integrity and others.
# ----------------------------------------------------------------


def get_idxs_of_annotated_slices(mask: np.ndarray) -> np.ndarray:
    """Return an array of booleans that indicate which slices have
    lesions. Ideal for numpy fancy indexing.

    Args:
        mask (np.ndarray): mask with annotations.

    Returns:
        np.ndarray: Array indicating which slices have lesions.
    """
    if mask.shape[0] == mask.shape[1]:
        mask = mask.transpose(2, 0, 1)
    return np.array([True if np.count_nonzero(s) > 0 else False for s in mask])


def assert_mask_integrity(mask: np.ndarray):
    """Use this function in other functions to verify that the mask
    values are either zeros or zeros and ones.

    Args:
        mask (np.ndarray): mask to verify
    """
    assert mask.min() == 0.0
    assert mask.max() <= 1.0
    assert np.unique(mask).shape[0] <= 2


# !: ------------------------------------------------------------------------------------
# !: Code that needs rewriting/refactoring.


def transform_data_and_mask_cls(
    data: np.ndarray, transformation: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns in a single called the transformed data and mask
    using the signature of Augmentations library functions.

    Args:
        data (np.ndarray): data to be resized/augmented
        mask (np.ndarray): mask to be resized/augmented
        transformation (Callable): Augmentation function

    Returns:
        Tuple[np.ndarray, np.ndarray]: augmented data and mask
    """
    aug_data = transformation(image=data)
    return aug_data["image"]
