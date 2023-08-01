import numpy as np
import tensorflow as tf

from typing import Dict, List, Any


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Feature description
# ----------------------------------------------------------------------------


def get_feature_description_with_modalities(
    modalities: List[str], volumes: bool = False
) -> Dict[str, Any]:
    """Takes a list of modalities and creates a dictionary to deserialize the features
    from an example in a TFRecord file.

    Args:
        modalities (List[str]): modalities inside the TFRecord. Do not specify mask,
        height or width as they are already considered.
        volumes (bool): True to include the number of slices description. Defaults to False.

    Raises:
        ValueError: if `modalities` is empty

    Returns:
        Dict[str, Any]: dictionary with the description for each serialized feature.
    """

    if len(modalities) == 0:
        raise ValueError("`modalities` list cannot be empty.")

    feature_description = {
        "mask": tf.io.FixedLenFeature([], tf.string),
        "mask_with_contours": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    for modality in modalities:
        feature_description[modality] = tf.io.FixedLenFeature([], tf.string)

    if volumes:
        feature_description["num_slices"] = tf.io.FixedLenFeature([], tf.int64)

    return feature_description


# Functions to serialize dict with all the modalities.
# ----------------------------------------------------------------------------


def serialize_2d_example(data_dict: Dict[str, np.ndarray]):
    """Creates a tf.train.Example ready to be written to a file. This
    function does not implement tf.io.encode_jpg/png because images are
    not of dtype uint8 or uint16"""

    modalities_names = list(data_dict.keys())
    first_modality_shape = data_dict[modalities_names[0]].shape[:2]
    if not all(v.shape[:2] == first_modality_shape for v in data_dict.values()):
        print({k: v.shape[:2] for k, v in data_dict.items()})
    assert all(v.shape[:2] == first_modality_shape for v in data_dict.values())

    # Serialize all the images with their corresponding key.
    feature = {
        modality_name: _bytes_feature(tf.io.serialize_tensor(modality))
        for modality_name, modality in data_dict.items()
    }

    # Add the height and width to the feature dictionary.
    feature["height"] = _int64_feature(first_modality_shape[0])
    feature["width"] = _int64_feature(first_modality_shape[1])

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_3d_example(data_dict: Dict[str, np.ndarray]):
    """Creates a tf.train.Example ready to be written to a file. This
    function does not implement tf.io.encode_jpg/png because images are
    not of dtype uint8 or uint16"""

    modalities_names = list(data_dict.keys())
    first_modality_shape = data_dict[modalities_names[0]].shape[:3]
    assert all(v.shape[:3] == first_modality_shape for v in data_dict.values())

    # Serialize all the images with their corresponding key.
    feature = {
        modality_name: _bytes_feature(tf.io.serialize_tensor(modality))
        for modality_name, modality in data_dict.items()
    }

    # Add the height and width to the feature dictionary.
    feature["num_slices"] = _int64_feature(first_modality_shape[0])
    feature["height"] = _int64_feature(first_modality_shape[1])
    feature["width"] = _int64_feature(first_modality_shape[2])

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Old serialization functions. Used the stacked modalities.
# ----------------------------------------------------------------------------


def _serialize_2d_example(data: np.ndarray, mask: np.ndarray):
    """Creates a tf.train.Example ready to be written to a file. This
    function does not implement tf.io.encode_jpg/png because images are
    not of dtype uint8 or uint16"""

    assert data.shape[:2] == mask.shape[:2]
    has_annotations = False if np.count_nonzero(mask) else True
    feature = {
        "data": _bytes_feature(tf.io.serialize_tensor(data)),
        "mask": _bytes_feature(tf.io.serialize_tensor(mask)),
        "height": _int64_feature(data.shape[0]),
        "width": _int64_feature(data.shape[1]),
        "data_channels": _int64_feature(data.shape[-1]),
        "has_annotations": _int64_feature(has_annotations),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _serialize_3d_example(data: np.ndarray, mask: np.ndarray):
    """Creates a tf.train.Example ready to be written to a file. This
    function does not implement tf.io.encode_jpg/png because images are
    not of dtype uint8 or uint16"""

    assert data.shape[:3] == mask.shape[:3]
    has_annotations = False if np.count_nonzero(mask) else True
    feature = {
        "data": _bytes_feature(tf.io.serialize_tensor(data)),
        "mask": _bytes_feature(tf.io.serialize_tensor(mask)),
        "num_slices": _int64_feature(data.shape[0]),
        "height": _int64_feature(data.shape[1]),
        "width": _int64_feature(data.shape[2]),
        "data_channels": _int64_feature(data.shape[-1]),
        "has_annotations": _int64_feature(has_annotations),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
