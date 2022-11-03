"""Loads and pass the data for training and testing pipelines."""

import tensorflow as tf
import albumentations as A

from functools import partial
from typing import List, Optional, Dict, Any

import utils.preprocessing.tensorflow as tfp
from utils.datasets.serializers import get_feature_description_with_modalities

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
        feature_description = get_feature_description_with_modalities(
            modalities, volumes=False
        )
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
            tfp.set_shapes,
            height=slice_size,
            width=slice_size,
            data_channels=self.n_channels,
        )

        if augmentations:
            self.transformation = self.compose_augmentations()
            self._aug_fn = partial(
                tfp.augment_data_and_mask, transformation=self.transformation
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
        if self.multiresolution:
            dset = dset.map(self._multiresolution_fn, num_parallel_calls=AUTOTUNE)
        if self.deep_supervision:
            dset = dset.map(self._deep_supervision_fn, num_parallel_calls=AUTOTUNE)
        if self.sample_weights:
            dset = dset.map(self._weights_fn, num_parallel_calls=AUTOTUNE)

        # Prefetch some samples and batch them at the end.
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
        return A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(limit=7, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.0
                ),
                A.ElasticTransform(alpha=5, sigma=1.5, alpha_affine=0.8, p=0.5),
                A.GridDistortion(distort_limit=0.1, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
            ]
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

        feature_description = get_feature_description_with_modalities(
            modalities, volumes=True
        )

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
            tfp.set_shapes_batch,
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
