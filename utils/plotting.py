"""Miscelaneous functions for visualizing the modalities and masks."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import imageio
from itertools import zip_longest
from typing import Union, Tuple, List, Optional
from matplotlib.patches import Patch

from utils.preprocessing.numpy import min_max_normalization

IntOrFloat = Union[int, float]
Color = Tuple[IntOrFloat, IntOrFloat, IntOrFloat]

RED_COLOR = (1.0, 0.0, 0.0)
GREEN_COLOR = (0.0, 1.0, 0.0)
BLUE_COLOR = (0.0, 0.0, 1.0)

# TODO: replace ot for mask.


def plot_data_with_overlapping_ots(
    data: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    true_mask_with_contours: bool = True,
    pred_mask_with_contours: bool = False,
    true_color: Color = RED_COLOR,
    pred_color: Color = BLUE_COLOR,
    true_label: str = "True mask",
    pred_label: str = "Pred mask",
    alpha: float = 1.0,
    save_path: str = "",
    show_plot: bool = True,
    title: str = "",
) -> None:
    """Creates a plot using one or multiple slices of one modality (e.g. ADC) one or
    multiple slices corresponding to the true and predicted OTs (masks). Firstly,
    all the arrays will be converted to RGB and the masks will get colored. Secondly, all
    the slices will be combined into a single array. Lastly, each slice will be shown.

    Args:
        data (np.ndarray): slices of one modality.
        true_ot (np.ndarray): Masks delineated by an expert.
        pred_ot (np.ndarray): Masks predicted by the model.
        true_color (Tuple[float], optional). Defaults to RED_COLOR.
        pred_color (Tuple[float], optional). Defaults to GREEN_COLOR.
        true_label (str, optional): Text for the legend. Defaults to 'True mask'.
        pred_label (str, optional): Text for the legend. Defaults to 'Pred mask'.
        alpha (float, optional): Opacity of the masks. Defaults to 0.8.
        save_path (str, optional): Animation save path. Defaults to ''.
        show_plot (bool, optional): Wheter the plot will be displayed. Defaults to True.
        title (str, optional): Title for the plot. Defaults to ''.
    """

    # Plotting RGB float values with matplotlib requires values between [0..1].
    if not (data.min() >= 0.0 and data.max() <= 1.0):
        data = min_max_normalization(data)

    # Add axis if data is a single slice.
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
        true_mask = np.expand_dims(true_mask, axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)

    intersection_color = tuple(sum(x) for x in zip(true_color, pred_color))

    # Transpose (H, W, S) -> (S, W, H) to iterate over slices (S).
    data = data.transpose(2, 0, 1)
    true_mask = true_mask.transpose(2, 0, 1)
    pred_mask = pred_mask.transpose(2, 0, 1)

    # Combine the data with the OTs.
    data_w_overlayed_masks = overlay_masks_on_data(
        data,
        true_mask,
        pred_mask,
        true_mask_with_contours=true_mask_with_contours,
        pred_mask_with_contours=pred_mask_with_contours,
        true_color=true_color,
        pred_color=pred_color,
    )

    # Create the plot.
    ncols = data.shape[0]
    fig = plt.figure(figsize=(8 * ncols, 8))
    for idx, slice_ in enumerate(data_w_overlayed_masks):
        ax = fig.add_subplot(1, ncols, idx + 1)
        ax.axis("off")
        ax.imshow(slice_.clip(0.0, 1.0))

    height = ax.get_position().get_points()[1, 1] + 0.125
    num_slices = data_w_overlayed_masks.shape[0]
    fontsize = 12 + 2 * num_slices

    if title != "":
        fig.suptitle(title, fontsize=fontsize)

    add_annotations_legend(
        true_color,
        pred_color,
        intersection_color,
        fig=fig,
        true_label=true_label,
        pred_label=pred_label,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, height),
    )

    fig.tight_layout()
    if save_path != "":
        fig.savefig(save_path, bbox_inches="tight")
    if not show_plot:
        plt.close()


def save_animated_data_with_overlapping_ots(
    data: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    true_mask_with_contours: bool = True,
    pred_mask_with_contours: bool = False,
    true_color: Color = RED_COLOR,
    pred_color: Color = BLUE_COLOR,
    true_label: str = "True mask",
    pred_label: str = "Pred mask",
    alpha: float = 1.0,
    save_path: str = "",
    interval: int = 1200,
    titles: Union[str, List[str]] = "",
):
    """Creates an animation using a stack of slices (volume) of one modality (e.g. ADC)
    and two stacks of slices corresponding to the true and predicted OTs (masks). Firstly,
    all the arrays will be converted to RGB and the masks will get colored. Secondly, all
    the slices will be combined into a single array that will be rendered as a GIF or MP4.

    Args:
        data (np.ndarray): slices of one modality.
        true_ot (np.ndarray): Masks delineated by an expert.
        pred_ot (np.ndarray): Masks predicted by the model.
        true_color (Tuple[float], optional). Defaults to RED_COLOR.
        pred_color (Tuple[float], optional). Defaults to GREEN_COLOR.
        true_label (str, optional): Text for the legend. Defaults to 'True mask'.
        pred_label (str, optional): Text for the legend. Defaults to 'Pred mask'.
        alpha (float, optional): Opacity of the masks. Defaults to 0.8.
        save_path (str, optional): Animation save path. Defaults to ''.
        interval (int, optional): Delay between frames. Defaults to 800.
        titles (Union[str, List[str]], optional). Defaults to ''.

    Raises:
        ValueError: If no save path was given.
        ValueError: save_path extension is not in ['.gif', '.mp4']
        ValueError: If any of the given arrays is not a volume.
    """

    if save_path == "":
        raise ValueError("Expected a path to save the animation.")
    if not (save_path.endswith(".gif") or save_path.endswith(".mp4")):
        raise ValueError("Save path extension must be one .gif or .mp4")
    if data.ndim == 2 or true_mask.ndim == 2 or pred_mask.ndim == 2:
        raise ValueError("Data is a single slice. Expected 3-dimensional array.")

    # Plotting RGB float values with matplotlib requires values between [0..1].
    if not (data.min() >= 0.0 and data.max() <= 1.0):
        data = min_max_normalization(data)

    intersection_color = tuple(sum(x) for x in zip(true_color, pred_color))

    # Transpose (H, W, S) -> (S, W, H) to iterate over slices (S).
    data = data.transpose(2, 0, 1)
    true_mask = true_mask.transpose(2, 0, 1)
    pred_mask = pred_mask.transpose(2, 0, 1)

    # Combine the data with the OTs.
    data_w_overlayed_masks = overlay_masks_on_data(
        data,
        true_mask,
        pred_mask,
        true_mask_with_contours=true_mask_with_contours,
        pred_mask_with_contours=pred_mask_with_contours,
        true_color=true_color,
        pred_color=pred_color,
    )

    # Setup figure.
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot()

    add_annotations_legend(
        true_color,
        pred_color,
        intersection_color,
        true_label=true_label,
        pred_label=pred_label,
        fontsize=16,
        fig=ax,
        bbox_to_anchor=(0.5, 1.0),
    )
    ax.axis("off")

    # Create the matplotlib artists and compose animation.
    artists = []
    if isinstance(titles, str):
        titles = [titles]
    for s, t in zip_longest(data_w_overlayed_masks, titles, fillvalue=titles[0]):

        # Split the title in two lines if it has more than 60 chars.
        num_lines = np.ceil(t.count(",") / 5)
        ax_title = split_title(t, elements_per_line=5)
        text_height = 1.1 + num_lines * 0.05

        im = ax.imshow(s.clip(0.0, 1.0), animated=True)
        title = ax.text(
            0.5,
            text_height,
            ax_title,
            animated=True,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=16,
        )
        artists.append([im, title])

    fig.tight_layout()
    anim = animation.ArtistAnimation(fig, artists, interval=interval, blit=True)
    writergif = animation.PillowWriter(fps=2)
    anim.save(save_path, writer=writergif)
    plt.close()


def plot_data(
    title: str = "",
    save_path: str = "",
    dpi: Union[str, float] = "figure",
    show_plot: bool = True,
    **kwargs,
):
    """Plots the slices of all the ndarrays passed as kwargs.

    Args:
        title (str, optional): Defaults to ''.
        save_path (str, optional): Defaults to ''.
        dpi (Union[str, float], optional): Defaults to 'figure'.
        show_plot (bool, optional): Defaults to True.
    """

    # Kwargs must be ndarrays with the same dimensions.
    first_value = list(kwargs.values())[0]
    for k, v in kwargs.items():
        assert isinstance(v, np.ndarray), f"{k} must be an instance of np.ndarray"
        assert (
            v.shape == first_value.shape
        ), f"{k} shape does not match the other values shape"

    num_slices = first_value.shape[-1]
    fontsize = 12 + 2 * num_slices

    nrows, ncols = len(kwargs), first_value.shape[-1]
    ncols = (
        first_value.shape[-1]
        if first_value.shape[0] == first_value.shape[1]
        else first_value.shape[0]
    )
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
    fig.suptitle(title, fontsize=fontsize)

    for i, (k, v) in enumerate(kwargs.items()):
        vol = v.transpose(2, 0, 1) if v.shape[0] == v.shape[1] else v
        for j in range(ncols):
            if nrows > 1:
                axs[i, j].imshow(vol[j], cmap="gray")
                axs[i, j].set_title(f"{k} Slice: {j+1}", fontsize=32)
            else:
                axs[j].imshow(vol[j], cmap="gray")
                axs[j].set_title(f"{k} Slice: {j+1}", fontsize=32)

    if save_path != "":
        fig.savefig(save_path, dpi=dpi)

    if not show_plot:
        plt.close()


def save_multimedia_in_patient_dir(
    patient_dir,
    norm_data,
    ot,
    resized_mask,
    title,
    save_plots: bool = True,
    save_animation: bool = True,
    draw_contours: bool = True,
    animation_format: str = "mp4",
):

    if save_plots:
        data_and_ots_path = os.path.join(patient_dir, "data_and_ots.png")
        overlapped_ots_path = os.path.join(patient_dir, "data_and_ots_overlapped.png")
        plot_data(
            Data=norm_data,
            GT=ot,
            Pred=resized_mask,
            title=title,
            show_plot=False,
            save_path=data_and_ots_path,
        )
        plot_data_with_overlapping_ots(
            norm_data,
            ot,
            resized_mask,
            draw_contours=draw_contours,
            save_path=overlapped_ots_path,
            show_plot=False,
            title=title,
        )

    if save_animation:
        animation_save_path = os.path.join(patient_dir, f"animation.{animation_format}")
        save_animated_data_with_overlapping_ots(
            norm_data,
            ot,
            resized_mask,
            draw_contours=draw_contours,
            save_path=animation_save_path,
            titles=title,
        )


def plot_slices_and_ot(data: np.ndarray, ot: np.ndarray, axis=False) -> None:
    """Plots a row of images given by data and ot numpy arrays.

    Args:
        data (np.ndarray): array with the modalities.
        ot (np.ndarray): array with the mask.
        axis (bool, optional). Defaults to False.

    Raises:
        ValueError: data is assumed to have 3 dimensions (H, W, C). If
        only one modality is in data, then C should be 1.
    """

    if data.ndim != 3:
        raise ValueError("Expected data of 3 dimensions.")
    if ot.ndim == 2:
        ot = np.expand_dims(ot, axis=-1)

    # Concatenate and transpose the data and ot to iterate them easily.
    slices = np.concatenate([data, ot], axis=-1)
    slices = np.transpose(slices, axes=(2, 0, 1))

    cols = data.shape[-1] + 1
    fig, axs = plt.subplots(nrows=1, ncols=cols, figsize=(8 * cols, 8))
    for ax, slice in zip(axs.flat, slices):
        ax.imshow(slice, cmap="gray")
        if not axis:
            ax.axis("off")

    fig.tight_layout()


def add_annotations_legend(
    true_color: Tuple[float, float, float],
    pred_color: Tuple[float, float, float],
    intersection_color: Tuple[float, float, float],
    true_label: str = "True mask",
    pred_label: str = "Pred mask",
    bbox_to_anchor: Tuple[float, float] = (0.5, 1.1),
    fontsize: int = 16,
    fig=None,
) -> None:
    """Add a legend indicating which color corresponds to each annotation.

    Args:
        true_color (Tuple[float, float, float]): Color of the true mask
        pred_color (Tuple[float, float, float]): Color of the pred mask
        intersection_color (Tuple[float, float, float]): Color for the
        intersection of the two masks.
        true_label (str, optional). Defaults to 'True mask'.
        pred_label (str, optional). Defaults to 'Pred mask'.
        fontsize (int, optional). Defaults to 16.
    """

    legend_elements = [
        Patch(facecolor=true_color, edgecolor=true_color, label=true_label),
        Patch(facecolor=pred_color, edgecolor=pred_color, label=pred_label),
        Patch(
            facecolor=intersection_color,
            edgecolor=intersection_color,
            label="Intersection",
        ),
    ]

    legend_kwargs = dict(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        ncol=3,
        frameon=False,
    )
    fig.legend(**legend_kwargs) if fig is not None else plt.legend(**legend_kwargs)


# Change data display (contours, color, etc.).
# ----------------------------------------------------------------


def convert_slices_from_gray_to_rgb(slices: np.ndarray) -> np.ndarray:
    """Converts stack of grayscale slices (volume) to rgb. Works if slices
    are stacked around the first or third axis.

    Args:
        slices (np.ndarray): grayscale slices.

    Returns:
        np.ndarray: rgb slices.
    """
    if slices.shape[0] == slices.shape[1]:
        slices = slices.transpose(2, 0, 1)
    return np.stack([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in slices])


def replace_color_in_mask(
    mask: np.ndarray, color_to_replace: Color, replacement_color: Color
):
    """Changes `color_to_replace` values on `image` with `replacement_color`
    values. Both colors must be a Tuple of three integer or float values
    representing one color. The color to replace serves to identify the
    values within the image that should be changed. The replacement color is
    the value by which these pixels will be changed."""
    mask_copy = mask.copy()
    colors_to_replace = (mask == color_to_replace).all(axis=-1)
    mask_copy[colors_to_replace] = replacement_color
    return mask_copy


def generate_color_mask(
    mask: np.ndarray, color: Tuple[float] = (1.0, 0.0, 0.0)
) -> np.ndarray:
    """Replace the white pixels with `color` pixels.

    Args:
        mask (np.ndarray): binary array.
        color (Tuple[float], optional): color tuple. Defaults to (1.0, 0.0, 0.0).

    Returns:
        np.ndarray: colored mask.
    """
    black_color, white_color = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    return np.where(mask == white_color, color, black_color).astype(np.float32)


def overlay_masks_on_data(
    data: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    true_mask_with_contours: bool = True,
    pred_mask_with_contours: bool = False,
    true_color: Color = RED_COLOR,
    pred_color: Color = BLUE_COLOR,
) -> np.ndarray:

    if data.ndim == 3:
        data = convert_slices_from_gray_to_rgb(data)

    if true_mask_with_contours:
        true_mask = convert_mask_to_contours(true_mask)
    true_mask = convert_slices_from_gray_to_rgb(true_mask)
    true_mask = generate_color_mask(true_mask, color=true_color)

    if not pred_mask is None:
        if pred_mask_with_contours:
            pred_mask = convert_mask_to_contours(pred_mask)
        pred_mask = convert_slices_from_gray_to_rgb(pred_mask)
        pred_mask = generate_color_mask(pred_mask, color=pred_color)

    return _add_masks_on_data(
        data,
        true_mask,
        pred_mask=pred_mask,
        true_mask_alpha=1.0,
        pred_mask_alpha=0.6,
        alpha=1.0,
    )


def _add_masks_on_data(
    data: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    true_mask_alpha: float = 1.0,
    pred_mask_alpha: float = 1.0,
    alpha: float = 0.8,
) -> np.ndarray:
    """Combines `true_ot` and `pred_ot` and adds the result to `data`.
    The contribution of the masks is weighted by `alpha`.

    Args:
        data (np.ndarray): Medical image.
        true_ot (np.ndarray): Ground truth mask.
        pred_ot (np.ndarray): Predicted mask.
        alpha (float, optional). Defaults to 0.8.

    Returns:
        np.ndarray: Medical image with masks overlayed.
    """

    alpha_msg = "Alpha must be a number between 0 and 1"
    assert 0.0 <= true_mask_alpha <= 1.0, alpha_msg
    assert 0.0 <= pred_mask_alpha <= 1.0, alpha_msg
    assert 0.0 <= alpha <= 1.0, alpha_msg

    # add masks, then the result add it to data.
    if pred_mask is None:
        assert data.ndim == true_mask.ndim
        masks_sum = true_mask
    else:
        assert data.ndim == true_mask.ndim == pred_mask.ndim
        masks_sum = cv2.addWeighted(
            pred_mask, pred_mask_alpha, true_mask, true_mask_alpha, 0.0
        )

    data_w_overlayed_masks = cv2.addWeighted(data, 1.0, masks_sum, alpha, 0.0)

    return data_w_overlayed_masks


def convert_mask_to_contours(mask: np.ndarray) -> np.ndarray:
    """Transforms a binary mask into a mask with only the contours.
    This function assumes that mask is a grayscale image."""

    if np.count_nonzero(mask) == 0:
        return mask

    mask = mask.astype(np.int32)
    if mask.ndim == 3:

        if mask.shape[0] == mask.shape[1]:
            channels_last = True
            mask = mask.transpose(2, 0, 1)
        else:
            channels_last = False

        # Create the mask with contours.
        mask_with_contours = np.stack(
            [_convert_mask_slice_to_contours(mask_slice) for mask_slice in mask]
        )
        if channels_last:
            mask_with_contours = mask_with_contours.transpose(1, 2, 0)

    elif mask.ndim == 2:
        mask_with_contours = _convert_mask_slice_to_contours(mask)
    else:
        raise ValueError("Mask must be an array of 2 or 3 dimensions.")

    return mask_with_contours.astype(np.float32)


def _convert_mask_slice_to_contours(mask: np.ndarray) -> np.ndarray:
    """Transforms a binary mask into a mask with only the contours.
    This function assumes that mask is a grayscale image."""
    if np.count_nonzero(mask) == 0:
        return mask

    mask = mask.astype(np.int32)
    contours_mask = np.zeros(mask.shape).astype(np.float32)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw all (-1) contours, with white (1) color and 1 thickness
    cv2.drawContours(contours_mask, contours, -1, 1, 1)

    return contours_mask


def save_array_as_animation(save_path: str, arr: Union[List[np.array], np.array]):
    """Saves CxHxW as animation. Elements from `arr` have to be normalized between
    0 and 1.

    Args:
        save_path (str): path to save the array
        arr (Union[List[np.array], np.array]): _description_
    """

    if isinstance(arr, np.ndarray):
        assert arr.ndim == 3, "`arr` must have 3 dims."
    else:
        assert arr[0].ndim == 2, "The elements from `arr` list must have 2 dims."
        arr = np.array(arr)

    arr = (arr * 255.0).astype(np.uint8)
    imageio.mimsave(save_path, arr)


def stack_arrays_horizontally(
    arr_list: List[np.array], resize_to: Tuple[int, int] = (224, 224), padding: int = 32
) -> np.array:
    """This function stacks the elements from `arr_list` horizontally. To achieve this,
    arrays are resized to `resize_to` size and their last channels are repeated until
    they have all the same number.

    Args:
        arr_list (List[np.array]): _description_
        resize_to (Tuple[int, int], optional): _description_. Defaults to (224, 224).

    Returns:
        np.array: _description_
    """

    # Make sure the arrays are HxWxC before resizing.
    channels_last = arr_list[0].shape[0] == arr_list[1].shape[1]
    if not channels_last:
        arr_list = [arr.transpose(1, 2, 0) for arr in arr_list]

    # Identify which array has the most channels.
    channels_list = [arr.shape[2] for arr in arr_list]
    max_channels = max(channels_list)
    channels_list.index(max_channels)

    # Resize and repeat until all the arrays have the same number of channels.
    arr_list = [cv2.resize(arr, resize_to) for arr in arr_list]
    arr_list = [
        repeat_last_channel_until_limit(arr, limit=max_channels) for arr in arr_list
    ]

    # Add white frames in between elements to act as padding.
    if padding:
        aux_arr_list = []
        pad_frame = np.ones((resize_to[0], padding, max_channels))
        for i, arr in enumerate(arr_list):
            aux_arr_list.append(arr)
            if i != len(arr_list) - 1:
                aux_arr_list.append(pad_frame)
        arr_list = aux_arr_list

    return np.hstack(arr_list).transpose(2, 0, 1)


def repeat_last_channel_until_limit(arr: np.array, limit: int) -> np.array:
    """Repeats the last channels from `arr` until it reaches `limit` number
    of channels."""
    if arr.shape[2] < limit:
        repetitions = limit - arr.shape[2]
        repeated_channels = np.tile(arr[..., -1:], repetitions)
        arr = np.concatenate([arr, repeated_channels], axis=2)
    return arr


# Text methods for annotations.
# ----------------------------------------------------------------


def create_title_with_metrics(**kwargs):
    """Creates a nicely formatted title for the plots."""
    title_shards = [f"{k}: {v:.3f}" for k, v in kwargs.items()]
    return ", ".join(title_shards)


def split_title(title: str, elements_per_line: int = 5):

    aux_title = title

    shards = []
    while aux_title.count(",") >= elements_per_line:
        idx = find_nth(aux_title, ", ", elements_per_line)
        shards.append(aux_title[:idx])
        aux_title = aux_title[idx:]

        if aux_title.startswith(", "):
            aux_title = aux_title[2:]

    shards.append(aux_title)
    ax_title = "\n".join(shards)

    return ax_title


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start
