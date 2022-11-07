import os
import ml_collections

from utils.config import load_dset_config


def get_dataloader_config(dset_dir: str) -> ml_collections.ConfigDict:

    dset_cfg = load_dset_config(dset_dir)
    config = ml_collections.ConfigDict()

    # Feature description to unserialize TFRecord examples.
    config.dataset = dset_cfg["dset_name"]
    config.num_train_samples = dset_cfg["num_train_samples"]
    config.use_full_train_dset = False

    # Paths to TFRecords.
    config.train_tfrecord_path = os.path.join(dset_dir, dset_cfg["train_tfrecord"])
    config.valid_tfrecord_path = os.path.join(dset_dir, dset_cfg["test_tfrecord"])
    if "full_train_tfrecord" in dset_cfg:
        config.full_train_tfrecord_path = os.path.join(
            dset_dir, dset_cfg["full_train_tfrecord"]
        )

    # Paths to patients list.
    config.train_patients_path = os.path.join(dset_dir, dset_cfg["train_patients_path"])
    config.valid_patients_path = os.path.join(dset_dir, dset_cfg["test_patients_path"])
    if "full_train_patients_path" in dset_cfg:
        config.full_train_patients_path = os.path.join(
            dset_dir, dset_cfg["full_train_patients_path"]
        )
    if "test_patients_path" in dset_cfg:
        config.test_patients_path = os.path.join(
            dset_dir, dset_cfg["test_patients_path"]
        )

    # Use all modalities as default.
    # ?: Consider writing the modalities as strings in tools.py
    config.modalities = ",".join(dset_cfg["modalities"])
    config.num_channels = len(dset_cfg["modalities"])
    config.batch_size = 8

    # Preprocessing/augmentations/filtering ops.
    config.augmentations = True
    config.sample_weights = True
    config.class_weights = "1.0,3.0"
    config.drop_non_lesion_slices_prob = 0.0
    config.shuffle_buffer = 80_000
    config.cache = True
    config.prefetch = True
    # *: False if sigmoid/binary, True if softmax/multiclass.
    config.one_hot_encoding = False

    # Should be changed with the common config parameters.
    config.multiresolution = False
    config.mask_with_contours = False
    config.deep_supervision = False
    config.num_lods = ml_collections.config_dict.placeholder(int)
    config.repeat_mask_for_deep_supervision = True

    return config
