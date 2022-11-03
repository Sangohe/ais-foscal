"""Tensorflow functions to normalize, resize, compute weights and check for 
the integrity of the medical images and masks of ischemic stroke patients."""

import tensorflow as tf
from tensorflow_addons.image import euclidean_dist_transform

import sys
from functools import partial
from typing import Callable, Tuple, Union, Dict, List

from .numpy import transform_data_and_mask, transform_data_and_mask_cls

# Mask weights.
# ----------------------------------------------------------------


def class_imbalance_weights(mask: tf.Tensor, class_weights: tf.Tensor) -> tf.Tensor:
    """Create a `sample_weights` array with the same dimensions of
    `mask`. The values of the `sample_weights` array will be determined
    by the `class_weights` array.

    Args:
        mask (tf.Tensor): mask tensor
        class_weights (tf.Tensor): importance for each class

    Returns:
        tf.Tensor: sample weights
    """
    sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))
    return sample_weights


def label_uncertainty(mask: tf.Tensor) -> tf.Tensor:
    """Computes the label uncertainty weights proposed in:
    https://arxiv.org/abs/2102.04566

    Args:
        mask (tf.Tensor): reference mask

    Returns:
        tf.Tensor: uncertainty weights
    """
    std = tf.math.reduce_std(mask)
    exp_den = 2 * (std**2)
    exp_num = (
        euclidean_dist_transform(tf.cast(mask, dtype=tf.uint8), dtype=tf.float32) ** 2
    )
    exp = tf.math.exp(-(exp_num / exp_den))

    return tf.constant(1.0, dtype=tf.float32) - exp


# Normalization and resizing.
# ----------------------------------------------------------------


def z_normalization(
    data: tf.Tensor, min_divisor: float = tf.constant(1e-3)
) -> tf.Tensor:
    """Returns a Z normalized data tensor

    Args:
        data (tf.Tensor): Tensor to be normalized
        min_divisor (float, optional). Defaults to 1e-3.

    Returns:
        tf.Tensor: Normalized data
    """
    mean = tf.math.reduce_mean(data)
    std = tf.math.reduce_std(data)
    if std < min_divisor:
        std = min_divisor
    return (data - mean) / std


def min_max_normalization(data: tf.Tensor) -> tf.Tensor:
    """Returns a min-max normalized data tensor. The values for the
    normalized tensor will lie between 0 and 1.

    Args:
        data (tf.Tensor): Tensor to be normalized

    Returns:
        tf.Tensor: Normalized data
    """
    min = tf.math.reduce_min(data)
    max = tf.math.reduce_max(data)
    return (data - min) / (max - min)


def resize_data(data: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    """Function to resize medical images except the manually delineated
    masks. This function applies a bilinear interpolation to all the
    channels/slices in `data` to obtain a resized version.

    Args:
        data (tf.Tensor): medical images to resize.
        target_size (Tuple[int, int]): target dimensionality

    Returns:
        tf.Tensor: resized data
    """
    resized_data = tf.image.resize(data, target_size)
    return resized_data


def resize_mask(mask: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    """Function to resize the manually delineated masks. This function
    applies a nearest neighbor interpolation to all the channels/slices
    in `mask` to obtain a resized version.

    Args:
        mask (tf.Tensor): mask to resize.
        target_size (Tuple[int, int]): target dimensionality

    Returns:
        tf.Tensor: resized mask
    """
    resized_mask = tf.image.resize(mask, target_size, method="nearest")
    return resized_mask


# !: Deprecate `num_lods` in this function.
def resize_data_multiresolution(
    data: tf.Tensor, num_lods: Union[int, tf.Tensor]
) -> Tuple[tf.Tensor, ...]:
    """Takes `data` and creates `num_lods` resized versions of it.
    The returned tuple contains all the versions, each one has half
    the size.

    Ex:
        input shapes: data -> [224, 224, 3] and num_lods -> 4
        output shapes: resized_data_array -> ([224, 224, 3], [112, 112, 3],
        [56, 56, 3], [26, 26, 3])

    Args:
        data (tf.Tensor): medical images
        num_lods (Union[int, tf.Tensor]): number of levels of detail
        (LODs), i.e. number of resized versions.

    Returns:
        Tuple[tf.Tensor, ...]: resized medical images
    """
    if tf.rank(data) == 3:
        target_size = tf.shape(data)[:2]
    else:
        target_size = tf.shape(data)[1:3]

    resized_data_array = [data]
    for i in range(1, num_lods):
        lod_target_size = target_size[:2] // tf.math.pow(2, i)
        resized_data = resize_data(data, target_size=lod_target_size)
        resized_data_array.append(resized_data)

    return tuple(resized_data_array)


def resize_mask_deep_supervision(
    mask: tf.Tensor, num_lods: Union[int, tf.Tensor]
) -> Tuple[tf.Tensor, ...]:
    """Takes `mask` and creates `num_lods` resized versions of it.
    The returned tuple contains all the versions, each one has half
    the size.

    Ex:
        input shapes: mask -> [224, 224, 1] and num_lods -> 4
        output shapes: resized_mask_array -> ([224, 224, 1], [112, 112, 1],
        [56, 56, 1], [26, 26, 1])

    Args:
        mask (tf.Tensor): mask tensor
        num_lods (Union[int, tf.Tensor]): number of levels of detail
        (LODs), i.e. number of resized versions.

    Returns:
        Tuple[tf.Tensor, ...]: resized masks
    """

    # Avoid selecting the None from the tensor's shape.
    if tf.rank(mask) == 3:
        target_size = tf.shape(mask)
    else:
        target_size = tf.shape(mask)[1:]

    resized_mask_array = [mask]
    for i in range(1, num_lods):
        lod_target_size = target_size[:2] // tf.math.pow(2, i)
        resized_mask = resize_mask(mask, target_size=lod_target_size[:2])
        resized_mask_array.append(resized_mask)

    return tuple(resized_mask_array)


def repeat_tensor(
    tensor: tf.Tensor, num_repetitions: Union[int, tf.Tensor]
) -> Tuple[tf.Tensor, ...]:
    """This function takes a tensor and repeats it `num_repetitions` times.
    This function is useful when used in deep supervised trainings

    Ex:
        input shapes: tensor -> [224, 224, 1] and num_repetitions -> 4
        output shapes: resized_tensor_array -> ([224, 224, 1], [224, 224, 1],
        [224, 224, 1], [224, 224, 1])

    Args:
        tensor (tf.Tensor)
        num_repetitions (Union[int, tf.Tensor])

    Returns:
        Tuple[tf.Tensor, ...]: tuple with `num_repetitions` tensors
    """

    # Avoid selecting the None from the tensor's shape.
    repeated_tensors = []
    for i in range(num_repetitions):
        repeated_tensors.append(tensor)

    return tuple(repeated_tensors)


# Functions for tf.data.Dataset input pipelines.
# ----------------------------------------------------------------


def parse_2d_tf_example(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(mask, [parsed_example["height"], parsed_example["width"], 1])

    return data, mask


def parse_2d_with_contours_tf_example(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask_with_contours"], out_type=tf.float32)
    mask = tf.reshape(mask, [parsed_example["height"], parsed_example["width"], 1])
    return data, mask


def parse_3d_tf_example(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["num_slices"],
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(
        mask,
        [
            parsed_example["num_slices"],
            parsed_example["height"],
            parsed_example["width"],
            1,
        ],
    )
    return data, mask


def parse_3d_with_contours_tf_example(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["num_slices"],
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask_with_contours"], out_type=tf.float32)
    mask = tf.reshape(
        mask,
        [
            parsed_example["num_slices"],
            parsed_example["height"],
            parsed_example["width"],
            1,
        ],
    )
    return data, mask


def add_sample_weights(
    data: tf.Tensor, mask: tf.Tensor, class_weights: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Combines the call of multiple sample weight computation functions to
    create a single `sample_weights` tensor.

    Args:
        image (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        class_weights (tf.Tensor): importance of the classes

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: medical images, mask and
        sample weights tensors
    """

    if isinstance(mask, tuple):
        imbalance_weights = tuple(
            [class_imbalance_weights(m, class_weights) for m in mask]
        )
    else:
        imbalance_weights = class_imbalance_weights(mask, class_weights)

    # Issue: Even on masked samples it's giving NaNs.
    # Issue: STD for non-masked samples is 0, therefore, zero-division
    # is ocurring when computing label_uncertainty.
    # uncertainty_weights = label_uncertainty(mask)
    # sample_weights = imbalance_weights * uncertainty_weights

    sample_weights = imbalance_weights
    return data, mask, sample_weights


def drop_sample_with_probability(
    data: tf.Tensor, mask: tf.Tensor, drop_prob: tf.Tensor
) -> tf.Tensor:
    """This function is meant to be used in a tf.data.Dataset pipeline.
    If the mask has annotations, it will be kept. Otherwise, it will be
    dropped if the sampled value from a random uniform distribution is
    greater than `drop_prob`.

    Args:
        data (tf.Tensor): medical images
        mask (tf.Tensor): annotated mask
        drop_prob (tf.Tensor): probability of dropping the sample

    Returns:
        tf.Tensor: Wheter to keep or not the sample
    """

    # TensorFlow requires to define both branches in conditionals
    # for autograph to work properly.
    has_annotations = tf.math.count_nonzero(mask) > 0
    if not has_annotations:
        drop_sample = tf.random.uniform([]) <= drop_prob
        return drop_sample
    else:
        return has_annotations


def augment_data_and_mask(
    data: tf.Tensor, mask: tf.Tensor, transformation: Callable
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Albumentation augmentations on TensorFlow. Given
    a data and mask tensor, it returns the augmented versions of both tensor.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        transformation (Callable): albumentations function to augment the
        tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: augmented tensors
    """
    aug_data, aug_mask = tf.numpy_function(
        func=partial(transform_data_and_mask, transformation=transformation),
        inp=[data, mask],
        Tout=[tf.float32, tf.float32],
    )
    return aug_data, aug_mask


def multiresolution_resizing(
    data: tf.Tensor, mask: tf.Tensor, num_lods: Union[int, tf.Tensor]
) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to create `num_lods` resized
    versions of the data tensor to.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        num_lods (Union[int, tf.Tensor]): number of levels of detail
        (LODs), i.e. number of resized versions.

    Returns:
        Tuple[Tuple[tf.Tensor, ...], tf.Tensor]: Tuple with resized
        tensors. The first element of the tuple is a tuple of the resized
        medical images tensors of different sizes. The second element is
        the mask.
    """
    resized_data_array = resize_data_multiresolution(data, num_lods=num_lods)
    return resized_data_array, mask


def deep_supervision_resizing(
    data: tf.Tensor, mask: tf.Tensor, num_lods: Union[int, tf.Tensor]
) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to create `num_lods` resized
    versions of the data tensor to.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        num_lods (Union[int, tf.Tensor]): number of levels of detail
        (LODs), i.e. number of resized versions.

    Returns:
        Tuple[Tuple[tf.Tensor, ...], tf.Tensor]: Tuple with resized
        tensors. The first element of the tuple is a tuple of the resized
        medical images tensors of different sizes. The second element is
        the mask.
    """
    resized_mask_array = resize_mask_deep_supervision(mask, num_lods=num_lods)
    return data, resized_mask_array


def deep_supervision_repetition(
    data: tf.Tensor, mask: tf.Tensor, num_repetitions: Union[int, tf.Tensor]
) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to create `num_repetitions` copies
    of the mask.
    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        num_repetitions (Union[int, tf.Tensor]): number of levels of detail
        (LODs), i.e. number of resized versions.

    Returns:
        Tuple[Tuple[tf.Tensor, ...], tf.Tensor]: Tuple with resized
        tensors. All the `num_repetitions` elements are the same.
    """
    repeated_masks = repeat_tensor(mask, num_repetitions=num_repetitions)
    return data, repeated_masks


def resize_data_and_mask(
    data: tf.Tensor, mask: tf.Tensor, target_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to resize both data and
    mask tensors to a `target_size`.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        target_size (Tuple[int, int]): target size of both tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: resized tensors
    """
    resized_data = resize_data(data, target_size=target_size)
    resized_mask = resize_mask(mask, target_size=target_size)
    return resized_data, resized_mask


def set_shapes(
    data: tf.Tensor, mask: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((height, width, data_channels))
    mask.set_shape((height, width, 1))
    return data, mask


def set_shapes_batch(
    data: tf.Tensor, mask: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline. Use this when you have already used .batch() on your input pipeline

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((None, height, width, data_channels))
    mask.set_shape((None, height, width, 1))
    return data, mask


def _parse_2d_tf_example(
    example_proto: tf.Tensor, feature_description: Dict[str, tf.Tensor]
):
    """Used to unserialize examples from TFRecords. Warning: this function is
    kept for compatibility with old experiments. Please use parse_2d_tf_example
    instead."""
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.parse_tensor(parsed_example["data"], out_type=tf.float32)
    data = tf.reshape(
        data,
        [
            parsed_example["height"],
            parsed_example["width"],
            parsed_example["data_channels"],
        ],
    )
    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(mask, [parsed_example["height"], parsed_example["width"], 1])
    return data, mask


def _parse_3d_tf_example(
    example_proto: tf.Tensor, feature_description: Dict[str, tf.Tensor]
):
    """Used to unserialize examples from TFRecords. Warning: this function is
    kept for compatibility with old experiments. Please use parse_3d_tf_example
    instead."""
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.parse_tensor(parsed_example["data"], out_type=tf.float32)
    data = tf.reshape(
        data,
        [
            parsed_example["num_slices"],
            parsed_example["height"],
            parsed_example["width"],
            parsed_example["data_channels"],
        ],
    )
    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(
        mask,
        [
            parsed_example["num_slices"],
            parsed_example["height"],
            parsed_example["width"],
            1,
        ],
    )
    return data, mask


# !: ------------------------------------------------------------------------------------
# !: Code that needs rewriting/refactoring.


def parse_2d_tf_example_cls(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(mask, [parsed_example["height"], parsed_example["width"], 1])
    label = tf.cast(tf.math.count_nonzero(mask) > 0, dtype=tf.float32)

    return data, label


def parse_3d_tf_example_cls(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["num_slices"],
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    mask = tf.io.parse_tensor(parsed_example["mask"], out_type=tf.float32)
    mask = tf.reshape(
        mask,
        [
            parsed_example["num_slices"],
            parsed_example["height"],
            parsed_example["width"],
            1,
        ],
    )
    label = tf.cast(tf.math.count_nonzero(mask, axis=[1, 2, 3]) > 0, dtype=tf.float32)
    return data, label


def resize_data_and_mask_cls(
    data: tf.Tensor, label: tf.Tensor, target_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to resize both data and
    mask tensors to a `target_size`.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        target_size (Tuple[int, int]): target size of both tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: resized tensors
    """
    resized_data = resize_data(data, target_size=target_size)
    return resized_data, label


def augment_data_and_mask_cls(
    data: tf.Tensor, label: tf.Tensor, transformation: Callable
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Albumentation augmentations on TensorFlow. Given
    a data and mask tensor, it returns the augmented versions of both tensor.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        transformation (Callable): albumentations function to augment the
        tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: augmented tensors
    """
    aug_data = tf.numpy_function(
        func=partial(transform_data_and_mask_cls, transformation=transformation),
        inp=[data],
        Tout=tf.float32,
    )
    return aug_data, label


def set_shapes_cls(
    data: tf.Tensor, label: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((height, width, data_channels))
    label.set_shape(())
    return data, label


def set_shapes_cls_batch(
    data: tf.Tensor, label: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline. Use this when you have already used .batch() on your input pipeline

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((None, height, width, data_channels))
    label.set_shape((None,))
    return data, label


def parse_2d_tf_example_den(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)

    return data, data


def parse_3d_tf_example_den(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.Tensor],
    modalities: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used to unserialize examples from TFRecords

    Args:
        example_proto (tf.Tensor): Serialized tensor (bytes)
        feature_description (Dict[str, tf.Tensor]): dictionary with the
        description of each feature, i.e. expected type and length.
        modalities (List[str]): list of modalities to include in the
        dataset

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: medical images and mask tensors.
    """
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    data = []
    for modality in modalities:
        modality_tensor = tf.io.parse_tensor(
            parsed_example[modality], out_type=tf.float32
        )
        modality_tensor = tf.reshape(
            modality_tensor,
            [
                parsed_example["num_slices"],
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        data.append(modality_tensor)
    data = tf.concat(data, axis=-1)
    return data, data


def resize_data_and_mask_den(
    data: tf.Tensor, mask: tf.Tensor, target_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Used in tf.data.Dataset input pipelines to resize both data and
    mask tensors to a `target_size`.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        target_size (Tuple[int, int]): target size of both tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: resized tensors
    """
    resized_data = resize_data(data, target_size=target_size)
    return resized_data, resized_data


def noise_den(data: tf.Tensor, mask: tf.Tensor):
    """Used in tf.data.Dataset input pipelines to resize both data and
    mask tensors to a `target_size`.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        target_size (Tuple[int, int]): target size of both tensors

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: resized tensors
    """

    noise_factor = 0.2
    noisy_array = data + tf.random.normal(tf.shape(data)) * noise_factor

    return tf.clip_by_value(noisy_array, 0.0, 1.0), data


def set_shapes_den(
    data: tf.Tensor, label: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline.

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((height, width, data_channels))
    label.set_shape((height, width, data_channels))
    return data, label


def set_shapes_den_batch(
    data: tf.Tensor, label: tf.Tensor, height: int, width: int, data_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Set shapes for the data and mask tensors on a tf.data.Dataset input
    pipeline. Use this when you have already used .batch() on your input pipeline

    Args:
        data (tf.Tensor): medical images tensor
        mask (tf.Tensor): mask tensor
        height (int): height of the data and mask tensors
        width (int): width of the data and mask tensors
        data_channels (int): numper os channels of the data.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: data and mask tensors with shapes set.
    """
    data.set_shape((None, height, width, data_channels))
    data.set_shape((None, height, width, data_channels))
    return data, label
