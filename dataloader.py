"""Loads and pass the data for training and testing pipelines."""

import numpy as np
import tensorflow as tf
import albumentations as A

from functools import partial
from typing import List, Optional, Dict, Any, Tuple, Callable

import utils.preprocessing.tensorflow as tfp

AUTOTUNE = tf.data.AUTOTUNE


class TFSlicesDataloader:
    def __init__(
        self,
        tfrecord_path: str,
        modalities: List[str],
        slice_size: int,
        batch_size: int = 32,
        multiresolution: bool = False,
        mask_with_contours: bool = False,
        deep_supervision: bool = False,
        num_lods: Optional[int] = None,
        repeat_mask_for_deep_supervision: bool = True,
        one_hot_encoding: bool = False,
        augmentations: bool = False,
        sample_weights: bool = False,
        class_weights: Optional[List[float]] = None,
        drop_non_lesion_slices_prob: float = 0.0,
        shuffle_buffer: int = 50_000,
        cache: bool = False,
        prefetch: bool = False,
        **kwargs,
    ):
        """Utility to load the TFRecord datasets

        Args:
            tfrecord_path (str): path to the TFRecord file containing the stroke dataset
            feature_description (Dict[str, Any]): dictionary with the description of each
            serialized featured within `tfrecord_path`.
            modalities (List[str]): list of modalities to include. `modalities` elements
            have to be one or multiple from `feature_descriptions` keys.
            slice_size (int): integer denoting the size to resize the slices within the
            dataset.
            batch_size (int, optional): Defaults to 32.
            multiresolution (bool, optional): True to generate multiple resized version
            of the medical images, num_lods has to be specified if True. Defaults to False.
            mask_with_contours (bool, optional): True to use the multiclass masks with contours
            for training. Defaults to False.
            deep_supervision (bool, optional): True to generate multiple resized version
            of the masks, num_lods has to be specified if True. Defaults to False.
            num_lods (Optional[int], optional): number of resized versions for the data
            or masks.
            Ignored if multiresolution and deep_supervision are False. Defaults to None.
            repeat_mask_for_deep_supervision (bool, optional): If true, the dataloader will
            repeat the mask `num_lods` times instead of resizing the images. Defaults to True.
            one_hot_encoding (bool, optional): True to use one hot encoding on masks.
            Defaults to False.
            augmentations (bool, optional): True to augment the inputs. Defaults to False.
            sample_weights (bool, optional): True to generate sample weights. Defaults to
            False.
            class_weights (Optional[List[float]], optional): class importance to create
            the sample_weights. If None, all classes have the same importance. Defaults
            to None.
            drop_non_lesion_slices_prob (float, optional): probability of dropping one
            sample if the mask has no annotations. If 0.0 no samples are dropped.
            Defaults to 0.0.
            shuffle (bool, optional): Defaults to False.
            shuffle_buffer (int, optional):  Defaults to 1000.
            cache (bool, optional): Defaults to False.
            prefetch (bool, optional): Defaults to False.

        Raises:
            ValueError: if drop_non_lesion_slices_prob is less than 0 or greater than 1
            ValueError: num_lods is not set when multiresolution or deep_supervision are
            True
            ValueError: num_lods is a negative integer
        """

        if "FOSCAL" in tfrecord_path:
            get_feature_description_fn = tfp.get_feature_description_with_modalities
        else:
            get_feature_description_fn = get_feature_description_with_modalities

        feature_description = get_feature_description_fn(modalities, volumes=False)
        medical_image_feature_names = [
            feature_name
            for feature_name in feature_description.keys()
            if feature_name not in ["height", "width", "mask"]
        ]

        for modality in modalities:
            if not modality in medical_image_feature_names:
                raise ValueError(
                    f"modality {modality} is not in the `feature_description`. "
                    f"Please choose one from {medical_image_feature_names}"
                )

        # TODO: add the option to compute the class_weights by itself.
        if class_weights is None:
            class_weights = [1.0, 1.0]

        if not (0 <= drop_non_lesion_slices_prob <= 1):
            raise ValueError("`drop_non_lesion_slices_prob` has to be between 0 and 1")

        if multiresolution or deep_supervision:
            if num_lods is None:
                raise ValueError(
                    "`num_lods` must be an integer greater than zero if "
                    "multiresolution or deep_supervision are set to True"
                )
            elif num_lods <= 0:
                raise ValueError(
                    f"You have set `num_lods` to {num_lods} and it must "
                    "be a positive integer"
                )

        self.tfrecord_path = tfrecord_path
        self.feature_description = feature_description
        self.modalities = modalities
        self.slice_size = slice_size
        self.batch_size = batch_size
        self.multiresolution = multiresolution
        self.mask_with_contours = mask_with_contours
        self.deep_supervision = deep_supervision
        self.repeat_mask_for_deep_supervision = repeat_mask_for_deep_supervision
        self.num_lods = num_lods
        self.one_hot_encoding = one_hot_encoding
        self.augmentations = augmentations
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        self.drop_non_lesion_slices_prob = tf.constant(
            drop_non_lesion_slices_prob, dtype=tf.float32
        )

        self.cache = cache
        self.prefetch = prefetch
        self.shuffle_buffer = shuffle_buffer

        self.target_size = (slice_size, slice_size)

        # Compose functions for tf.data.Dataset.
        if self.mask_with_contours:
            parse_method = tfp.parse_2d_with_contours_tf_example
        else:
            if "FOSCAL" in tfrecord_path:
                parse_method = parse_2d_tf_example
            else:
                parse_method = tfp.parse_2d_tf_example

        self._parse_fn = partial(
            parse_method,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        # * Infer number of channels for the  _set_shape_fn.
        self.infer_data_channels_and_num_samples()

        if self.drop_non_lesion_slices_prob > 0.0:
            self._sampling_fn = partial(
                tfp.drop_sample_with_probability,
                drop_prob=self.drop_non_lesion_slices_prob,
            )

        self._resize_fn = partial(
            tfp.resize_data_and_mask, target_size=self.target_size
        )
        self._set_shape_fn = partial(
            set_shapes if "FOSCAL" in tfrecord_path else tfp.set_shapes,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        if augmentations:
            self.transformation = self.compose_augmentations()
            self._aug_fn = partial(
                augment_data_and_mask
                if "FOSCAL" in tfrecord_path
                else tfp.augment_data_and_mask,
                transformation=self.transformation,
            )

        if sample_weights:
            self._weights_fn = partial(
                tfp.add_sample_weights, class_weights=class_weights
            )

        # Resize functions that create multiple versions of inputs.
        if multiresolution:
            self._multiresolution_fn = partial(
                tfp.multiresolution_resizing, num_lods=self.num_lods
            )
        if deep_supervision:
            if repeat_mask_for_deep_supervision:
                self._deep_supervision_fn = partial(
                    tfp.deep_supervision_repetition, num_repetitions=self.num_lods
                )
            else:
                self._deep_supervision_fn = partial(
                    tfp.deep_supervision_resizing, num_lods=self.num_lods
                )

        # !: -----------------------------------------------------------------------
        # !: Code that needs refactoring.

        self._parse_cls_fn = partial(
            tfp.parse_2d_tf_example_cls,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        self._resize_cls_fn = partial(
            tfp.resize_data_and_mask_cls, target_size=self.target_size
        )

        if augmentations:
            self.transformation = self.compose_augmentations()
            self._aug_cls_fn = partial(
                tfp.augment_data_and_mask_cls, transformation=self.transformation
            )

        self._set_shape_cls_fn = partial(
            tfp.set_shapes_cls,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        self._parse_den_fn = partial(
            tfp.parse_2d_tf_example_den,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        self._resize_den_fn = partial(
            tfp.resize_data_and_mask_den, target_size=self.target_size
        )

        self._set_shape_den_fn = partial(
            tfp.set_shapes_den,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        # !: -----------------------------------------------------------------------

    def get_dataset(self):
        # Reading, parsing and filtering.
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_fn, num_parallel_calls=AUTOTUNE)
        if self.cache:
            dset = dset.cache()
        if self.drop_non_lesion_slices_prob > 0.0:
            dset = dset.filter(self._sampling_fn)

        # Transform the inputs.
        dset = dset.map(self._resize_fn, num_parallel_calls=AUTOTUNE)
        if self.shuffle_buffer > 0:
            dset = dset.shuffle(buffer_size=self.shuffle_buffer)
        if self.augmentations:
            dset = dset.map(self._aug_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_fn, num_parallel_calls=AUTOTUNE)
        if "FOSCAL" in self.tfrecord_path:
            dset = dset.map(split, num_parallel_calls=AUTOTUNE)
        dset = dset.map(binarize_mask, num_parallel_calls=AUTOTUNE)  # !: work around.
        if self.sample_weights:
            dset = dset.map(self._weights_fn, num_parallel_calls=AUTOTUNE)

        # # Prefetch some samples and batch them at the end.
        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        if self.batch_size > 0:
            dset = dset.batch(self.batch_size)
        return dset

    def get_dataset_cls(self):
        # Reading, parsing and filtering.
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_cls_fn, num_parallel_calls=AUTOTUNE)
        if self.cache:
            dset = dset.cache()

        # Transform the inputs.
        dset = dset.map(self._resize_cls_fn, num_parallel_calls=AUTOTUNE)
        if self.shuffle_buffer > 0:
            dset = dset.shuffle(buffer_size=self.shuffle_buffer)
        if self.augmentations:
            dset = dset.map(self._aug_cls_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_cls_fn, num_parallel_calls=AUTOTUNE)

        # Prefetch some samples and batch them at the end.
        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        if self.batch_size > 0:
            dset = dset.batch(self.batch_size)
        return dset

    def get_dataset_denoising(self):
        # Reading, parsing and filtering.
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_den_fn, num_parallel_calls=AUTOTUNE)
        if self.cache:
            dset = dset.cache()

        # Transform the inputs.
        dset = dset.map(self._resize_den_fn, num_parallel_calls=AUTOTUNE)
        if self.shuffle_buffer > 0:
            dset = dset.shuffle(buffer_size=self.shuffle_buffer)
        if self.augmentations:
            dset = dset.map(self._aug_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(tfp.noise_den, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_den_fn, num_parallel_calls=AUTOTUNE)

        # Prefetch some samples and batch them at the end.
        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        if self.batch_size > 0:
            dset = dset.batch(self.batch_size)
        return dset

    def compose_augmentations(self) -> Optional[A.core.composition.Compose]:
        """Returns an albumentations object to transform the data and ots."""
        additional_targets = {"mask1": "mask"} if len(self.modalities) > 1 else None
        return A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(limit=7, p=0.5),
                # A.RandomBrightnessContrast(
                #     brightness_limit=0.1, contrast_limit=0.1, p=0.0
                # ),
                # A.ElasticTransform(alpha=5, sigma=1.5, alpha_affine=0.8, p=0.5),
                # A.GridDistortion(distort_limit=0.1, p=0.5),
                # A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
            ],
            additional_targets=additional_targets,
        )

    def infer_data_channels_and_num_samples(self):
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_fn, num_parallel_calls=AUTOTUNE)
        dset_list = list(dset)

        self.num_samples = len(dset_list)
        self.n_channels = dset_list[0][0].shape[-1]


class TFSlicesValidationDataloader:
    def __init__(
        self,
        tfrecord_path: str,
        modalities: List[str],
        slice_size: int,
        multiresolution: bool = False,
        mask_with_contours: bool = False,
        deep_supervision: bool = False,
        repeat_mask_for_deep_supervision: bool = True,
        num_lods: Optional[int] = False,
        cache: bool = False,
        prefetch: bool = False,
        **kwargs,
    ):
        """This class is meant to be used with TFRecords holding the data from the
        validation/testing split. It loads and pass volumes as batches of slices (this
        is the reason why it does not require a batch size).


        Args:
            tfrecord_path (str): path to the TFRecord file containing the validation split
            split from a stroke dataset.
            feature_description (Dict[str, Any]): dictionary with the description of each
            serialized featured within `tfrecord_path`.
            modalities (List[str]): list of modalities to include. `modalities` elements
            have to be one or multiple from `feature_descriptions` keys.
            slice_size (int): integer denoting the size to resize the slices within the
            dataset.
            multiresolution (bool, optional): True to generate multiple resized version
            of the medical images, num_lods has to be specified if True. Defaults to False.
            mask_with_contours (bool, optional): True to use the multiclass masks with contours
            for training. Defaults to False.
            deep_supervision (bool, optional): True to generate multiple resized version
            of the masks, num_lods has to be specified if True. Defaults to False.
            num_lods (Optional[int], optional): number of resized versions for the data
            or masks.
            repeat_mask_for_deep_supervision (bool, optional): If true, the dataloader will
            repeat the mask `num_lods` times instead of resizing the images. Defaults to True.
            cache (bool, optional): Defaults to False.
            prefetch (bool, optional): Defaults to False.

        Raises:
            ValueError: num_lods is not set when multiresolution or deep_supervision are
            True
            ValueError: num_lods is a negative integer
        """

        if "FOSCAL" in tfrecord_path:
            get_feature_description_fn = (
                get_feature_description_with_modalities_and_masks
            )
        else:
            get_feature_description_fn = get_feature_description_with_modalities

        feature_description = get_feature_description_fn(modalities, volumes=True)

        medical_image_feature_names = [
            feature_name
            for feature_name in feature_description.keys()
            if feature_name not in ["height", "width", "mask"]
        ]

        for modality in modalities:
            if not modality in medical_image_feature_names:
                raise ValueError(
                    f"modality {modality} is not in the `feature_description`. "
                    f"Please choose one from {medical_image_feature_names}"
                )

        if multiresolution or deep_supervision:
            if num_lods is None:
                raise ValueError(
                    "`num_lods` must be an integer greater than zero if "
                    "multiresolution or deep_supervision are set to True"
                )
            elif num_lods <= 0:
                raise ValueError(
                    f"You have set `num_lods` to {num_lods} and it must "
                    "be a positive integer"
                )

        self.tfrecord_path = tfrecord_path
        self.feature_description = feature_description
        self.modalities = modalities
        self.slice_size = slice_size
        self.multiresolution = multiresolution
        self.mask_with_contours = mask_with_contours
        self.deep_supervision = deep_supervision
        self.num_lods = num_lods
        self.cache = cache
        self.prefetch = prefetch
        self.target_size = (slice_size, slice_size)

        # Compose functions for tf.data.Dataset.
        if self.mask_with_contours:
            parse_method = tfp.parse_3d_with_contours_tf_example
        else:
            if "FOSCAL" in tfrecord_path:
                parse_method = parse_3d_tf_example
            else:
                parse_method = tfp.parse_3d_tf_example

        self._parse_fn = partial(
            parse_method,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        # * Infer number of channels for the  _set_shape_fn.
        self.infer_data_channels_and_num_samples()

        self._resize_fn = partial(
            tfp.resize_data_and_mask, target_size=self.target_size
        )
        self._set_shape_fn = partial(
            set_shapes_batch if "FOSCAL" in tfrecord_path else tfp.set_shapes_batch,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )
        if multiresolution:
            self._multiresolution_fn = partial(
                tfp.multiresolution_resizing, num_lods=self.num_lods
            )
        if deep_supervision:
            if repeat_mask_for_deep_supervision:
                self._deep_supervision_fn = partial(
                    tfp.deep_supervision_repetition, num_repetitions=self.num_lods
                )
            else:
                self._deep_supervision_fn = partial(
                    tfp.deep_supervision_resizing, num_lods=self.num_lods
                )

        # !: -----------------------------------------------------------------------
        # !: Code that needs refactoring.

        self._parse_cls_fn = partial(
            tfp.parse_3d_tf_example_cls,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        self._resize_cls_fn = partial(
            tfp.resize_data_and_mask_cls, target_size=self.target_size
        )

        self._set_shape_cls_fn = partial(
            tfp.set_shapes_cls_batch,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        self._parse_den_fn = partial(
            tfp.parse_3d_tf_example_den,
            feature_description=self.feature_description,
            modalities=self.modalities,
        )

        self._resize_den_fn = partial(
            tfp.resize_data_and_mask_den, target_size=self.target_size
        )

        self._set_shape_den_fn = partial(
            tfp.set_shapes_den_batch,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        # !: -----------------------------------------------------------------------

    def get_dataset(self):
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._resize_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_fn, num_parallel_calls=AUTOTUNE)
        if "FOSCAL" in self.tfrecord_path:
            dset = dset.map(split, num_parallel_calls=AUTOTUNE)
        dset = dset.map(binarize_mask, num_parallel_calls=AUTOTUNE)  # !: work around.
        if self.multiresolution:
            dset = dset.map(self._multiresolution_fn, num_parallel_calls=AUTOTUNE)
        if self.deep_supervision:
            dset = dset.map(self._deep_supervision_fn, num_parallel_calls=AUTOTUNE)
        if self.cache:
            dset = dset.cache()
        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        return dset

    def get_dataset_cls(self):
        # Reading, parsing and filtering.
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_cls_fn, num_parallel_calls=AUTOTUNE)

        # Transform the inputs.
        dset = dset.map(self._resize_cls_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_cls_fn, num_parallel_calls=AUTOTUNE)

        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        if self.cache:
            dset = dset.cache()
        return dset

    def get_dataset_denoising(self):
        # Reading, parsing and filtering.
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_den_fn, num_parallel_calls=AUTOTUNE)

        # Transform the inputs.
        dset = dset.map(self._resize_den_fn, num_parallel_calls=AUTOTUNE)
        dset = dset.map(tfp.noise_den, num_parallel_calls=AUTOTUNE)
        dset = dset.map(self._set_shape_den_fn, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dset = dset.cache()
        if self.prefetch:
            dset = dset.prefetch(AUTOTUNE)
        return dset

    def infer_data_channels_and_num_samples(self):
        dset = tf.data.TFRecordDataset(self.tfrecord_path)
        dset = dset.map(self._parse_fn, num_parallel_calls=AUTOTUNE)
        dset_list = list(dset)

        self.num_samples = len(dset_list) - 1
        self.n_channels = dset_list[0][0].shape[-1]


def binarize_mask(data, mask):
    return data, tf.cast(mask > 0, tf.float32)


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
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    for modality in modalities:
        feature_description[modality] = tf.io.FixedLenFeature([], tf.string)
    feature_description[f"mask"] = tf.io.FixedLenFeature([], tf.string)
    feature_description[f"mask_with_contours"] = tf.io.FixedLenFeature([], tf.string)

    if volumes:
        feature_description["num_slices"] = tf.io.FixedLenFeature([], tf.int64)

    return feature_description


def get_feature_description_with_modalities_and_masks(
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
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    for modality in modalities:
        feature_description[modality] = tf.io.FixedLenFeature([], tf.string)
        feature_description[f"{modality}_mask"] = tf.io.FixedLenFeature([], tf.string)
        feature_description[f"{modality}_mask_with_contours"] = tf.io.FixedLenFeature(
            [], tf.string
        )

    if volumes:
        feature_description["num_slices"] = tf.io.FixedLenFeature([], tf.int64)

    return feature_description


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

    data, masks = [], []
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

        mask = tf.io.parse_tensor(
            parsed_example[f"{modality}_mask"], out_type=tf.float32
        )
        mask = tf.reshape(
            mask,
            [
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        masks.append(mask)

    data = tf.concat(data, axis=-1)
    masks = tf.concat(masks, axis=-1)

    return data, masks


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

    data, masks = [], []
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
        mask_tensor = tf.io.parse_tensor(
            parsed_example[f"{modality}_mask"], out_type=tf.float32
        )
        mask_tensor = tf.reshape(
            mask_tensor,
            [
                parsed_example["num_slices"],
                parsed_example["height"],
                parsed_example["width"],
                1,
            ],
        )
        masks.append(mask_tensor)

    data = tf.concat(data, axis=-1)
    masks = tf.concat(masks, axis=-1)
    return data, masks


def split(data, mask):
    num_channels = data.shape[-1]
    if num_channels > 1:
        data = tuple(tf.split(data, num_channels, axis=-1))
        mask = tuple(tf.split(mask, num_channels, axis=-1))
    return data, mask


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
    if isinstance(mask, tuple):
        aug_data = transformation(image=data, mask=mask[..., 0:1], mask1=mask[..., 1:])
        data = aug_data["image"]
        mask = tf.concat([aug_data["mask"], aug_data["mask1"]], axis=-1)
    else:
        aug_data = transformation(image=data, mask=mask)
        data, mask = aug_data["image"], aug_data["mask"]
    return data, mask


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
    mask.set_shape((height, width, data_channels))
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
    mask.set_shape((None, height, width, data_channels))
    return data, mask
