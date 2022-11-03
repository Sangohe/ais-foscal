"""Generates the TFRecord datasets needed for training and validation"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import click
import numpy as np

sys.path.append("../..")

from utils.config import save_dict_as_yaml, create_dir_suffix
from utils.datasets.isles2017.create_dataset import (
    create_isles2017_dataset,
    get_train_val_test_patients,
)
from utils.datasets.isles2018.create_dataset import (
    create_isles2018_dataset,
    get_isles2018_train_val_test_patients,
)
from utils.datasets.foscal.create_dataset import (
    create_foscal_dataset,
    get_foscal_train_val_test_patients,
)
from utils.datasets.atlasv1.create_dataset import (
    create_atlasv1_dataset,
    generate_data_and_masks_paths,
    flatten_list_of_str,
)

# Data directory must be inside the data dir inside the root dir.
module_path = os.path.abspath(__file__)
module_shards = module_path.split("/")
if module_shards[-3] == "utils" and module_shards[-2] == "datasets":
    new_shards = module_shards[:-3] + ["data"]
    data_dir = "/".join(new_shards)
else:
    data_dir = "data/"


# Commands
# ----------------------------------------------------------------

cli = click.Group()

# yapf: disable
@cli.command()
@click.option('--source-dset-dir', '-sdset', required=True, help='Path to source dataset')
@click.option('--volumes', '-vol', is_flag=True, help='Serialize volumes to the tfrecord')
@click.option('--slices', '-sli', is_flag=True, help='Serialize slices to the tfrecord')
@click.option('--patches', '-pat', is_flag=True, help='Serialize patches to the tfrecord')
@click.option('--z-norm', '-z', is_flag=True, help='Normalize volumes with z normalization')
@click.option('--min-max-norm', '-min-max', is_flag=True, help='Normalize volumes with min-max normalization')
# yapf: enable
def isles2017_train_val_test(
    source_dset_dir,
    volumes,
    slices,
    patches,
    z_norm,
    min_max_norm,
):

    check_norm_flags(z_norm, min_max_norm)
    check_spatial_flags(volumes, slices, patches)
    modalities = ["ADC", "rCBV", "rCBF", "MTT", "TTP", "Tmax"]

    # Get the patients for each split.
    (
        train_patients_orig,
        train_patients,
        valid_patients,
        test_patients,
    ) = get_train_val_test_patients(source_dset_dir)

    # Create the directory to store all the data.
    dir_suffix = create_dir_suffix(
        volumes=volumes,
        slices=slices,
        patches=patches,
        z_norm=z_norm,
        min_max_norm=min_max_norm,
    )
    target_dset_dir = os.path.join(data_dir, f"ISLES2017_{dir_suffix}")
    print(f"Created directory for the dataset at: {target_dset_dir}\n")
    os.makedirs(target_dset_dir, exist_ok=True)

    # Create training and validation sets.
    if z_norm:
        normalization = "z"
    elif min_max_norm:
        normalization = "min_max"

    # Create the TFRecords for full_training, training, valid and test.
    tfrecords_dir = os.path.join(target_dset_dir, "tfrecords")
    os.makedirs(tfrecords_dir, exist_ok=True)
    full_train_tfrecord_path, num_full_train_samples = create_isles2017_dataset(
        tfrecords_dir,
        train_patients_orig,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="full_train",
    )
    train_tfrecord_path, num_train_samples = create_isles2017_dataset(
        tfrecords_dir,
        train_patients,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="train",
    )

    # Validation and test metrics are always calculated at volume-level.
    # Hence, serialize volumes to the TFRecords. Test TFRecord is not created
    # because it patients do not have masks. Other script will be used.
    valid_tfrecord_path, num_valid_samples = create_isles2017_dataset(
        tfrecords_dir,
        valid_patients,
        volumes=True,
        slices=False,
        patches=False,
        normalization=normalization,
        modalities=modalities,
        dset_split="valid",
    )

    # Save the patient paths to the dataset dir.
    patients_dir = os.path.join(target_dset_dir, "patients")
    os.makedirs(patients_dir, exist_ok=True)
    full_train_patients_path = os.path.join(patients_dir, "full_train_patients.txt")
    train_patients_path = os.path.join(patients_dir, "train_patients.txt")
    valid_patients_path = os.path.join(patients_dir, "valid_patients.txt")
    test_patients_path = os.path.join(patients_dir, "test_patients.txt")
    np.savetxt(full_train_patients_path, train_patients_orig, fmt="%s")
    np.savetxt(train_patients_path, train_patients, fmt="%s")
    np.savetxt(valid_patients_path, valid_patients, fmt="%s")
    np.savetxt(test_patients_path, test_patients, fmt="%s")

    # Save dataset configuration as YAML file.
    dset_config = {
        "dset_name": "ISLES2017",
        "normalization": normalization,
        "full_train_tfrecord": get_path_and_parent_dir(full_train_tfrecord_path),
        "train_tfrecord": get_path_and_parent_dir(train_tfrecord_path),
        "valid_tfrecord": get_path_and_parent_dir(valid_tfrecord_path),
        "full_train_patients_path": get_path_and_parent_dir(full_train_patients_path),
        "train_patients_path": get_path_and_parent_dir(train_patients_path),
        "valid_patients_path": get_path_and_parent_dir(valid_patients_path),
        "test_patients_path": get_path_and_parent_dir(test_patients_path),
        "num_full_train_samples": num_full_train_samples,
        "num_train_samples": num_train_samples,
        "num_valid_samples": num_valid_samples,
        "modalities": modalities,
    }
    dset_config_path = os.path.join(target_dset_dir, "dset_config.yml")
    save_dict_as_yaml(dset_config_path, dset_config)


# yapf: disable
@cli.command()
@click.option('--source-dset-dir', '-sdset', required=True, help='Path to source dataset')
@click.option('--volumes', '-vol', is_flag=True, help='Serialize volumes to the tfrecord')
@click.option('--slices', '-sli', is_flag=True, help='Serialize slices to the tfrecord')
@click.option('--patches', '-pat', is_flag=True, help='Serialize patches to the tfrecord')
@click.option('--z-norm', '-z', is_flag=True, help='Normalize volumes with z normalization')
@click.option('--min-max-norm', '-min-max', is_flag=True, help='Normalize volumes with min-max normalization')
# yapf: enable
def isles2018_train_val_test(
    source_dset_dir,
    volumes,
    slices,
    patches,
    z_norm,
    min_max_norm,
):

    check_norm_flags(z_norm, min_max_norm)
    check_spatial_flags(volumes, slices, patches)
    modalities = ["CT", "CBV", "CBF", "MTT", "Tmax"]

    # Get the patients for each split.
    (
        train_patients_orig,
        train_patients,
        valid_patients,
        test_patients,
    ) = get_isles2018_train_val_test_patients(source_dset_dir)

    # Create the directory to store all the data.
    dir_suffix = create_dir_suffix(
        volumes=volumes,
        slices=slices,
        patches=patches,
        z_norm=z_norm,
        min_max_norm=min_max_norm,
    )
    target_dset_dir = os.path.join(data_dir, f"ISLES2018_{dir_suffix}")
    print(f"Created directory for the dataset at: {target_dset_dir}\n")
    os.makedirs(target_dset_dir, exist_ok=True)

    # Create training and validation sets.
    if z_norm:
        normalization = "z"
    elif min_max_norm:
        normalization = "min_max"

    # Create the TFRecords for full_training, training, valid and test.
    tfrecords_dir = os.path.join(target_dset_dir, "tfrecords")
    os.makedirs(tfrecords_dir, exist_ok=True)
    full_train_tfrecord_path, num_full_train_samples = create_isles2018_dataset(
        tfrecords_dir,
        train_patients_orig,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="full_train",
    )
    train_tfrecord_path, num_train_samples = create_isles2018_dataset(
        tfrecords_dir,
        train_patients,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="train",
    )

    # Validation and test metrics are always calculated at volume-level.
    # Hence, serialize volumes to the TFRecords. Test TFRecord is not created
    # because it patients do not have masks. Other script will be used.
    valid_tfrecord_path, num_valid_samples = create_isles2018_dataset(
        tfrecords_dir,
        valid_patients,
        volumes=True,
        slices=False,
        patches=False,
        normalization=normalization,
        modalities=modalities,
        dset_split="valid",
    )

    # Save the patient paths to the dataset dir.
    patients_dir = os.path.join(target_dset_dir, "patients")
    os.makedirs(patients_dir, exist_ok=True)
    full_train_patients_path = os.path.join(patients_dir, "full_train_patients.txt")
    train_patients_path = os.path.join(patients_dir, "train_patients.txt")
    valid_patients_path = os.path.join(patients_dir, "valid_patients.txt")
    test_patients_path = os.path.join(patients_dir, "test_patients.txt")
    np.savetxt(full_train_patients_path, train_patients_orig, fmt="%s")
    np.savetxt(train_patients_path, train_patients, fmt="%s")
    np.savetxt(valid_patients_path, valid_patients, fmt="%s")
    np.savetxt(test_patients_path, test_patients, fmt="%s")

    # Save dataset configuration as YAML file.
    dset_config = {
        "dset_name": "ISLES2018",
        "normalization": normalization,
        "full_train_tfrecord": get_path_and_parent_dir(full_train_tfrecord_path),
        "train_tfrecord": get_path_and_parent_dir(train_tfrecord_path),
        "valid_tfrecord": get_path_and_parent_dir(valid_tfrecord_path),
        "full_train_patients_path": get_path_and_parent_dir(full_train_patients_path),
        "train_patients_path": get_path_and_parent_dir(train_patients_path),
        "valid_patients_path": get_path_and_parent_dir(valid_patients_path),
        "test_patients_path": get_path_and_parent_dir(test_patients_path),
        "num_full_train_samples": num_full_train_samples,
        "num_train_samples": num_train_samples,
        "num_valid_samples": num_valid_samples,
        "modalities": modalities,
    }
    dset_config_path = os.path.join(target_dset_dir, "dset_config.yml")
    save_dict_as_yaml(dset_config_path, dset_config)


# yapf: disable
@cli.command()
@click.option('--source-dset-dir', '-sdset', required=True, help='Path to source dataset')
@click.option('--volumes', '-vol', is_flag=True, help='Serialize volumes to the tfrecord')
@click.option('--slices', '-sli', is_flag=True, help='Serialize slices to the tfrecord')
@click.option('--patches', '-pat', is_flag=True, help='Serialize patches to the tfrecord')
@click.option('--z-norm', '-z', is_flag=True, help='Normalize volumes with z normalization')
@click.option('--min-max-norm', '-min-max', is_flag=True, help='Normalize volumes with min-max normalization')
# yapf: enable
def foscal_train_val_test(
    source_dset_dir,
    volumes,
    slices,
    patches,
    z_norm,
    min_max_norm,
):

    check_norm_flags(z_norm, min_max_norm)
    check_spatial_flags(volumes, slices, patches)
    modalities = ["ADC"]

    # Get the patients for each split.
    (
        train_patients_orig,
        train_patients,
        valid_patients,
        test_patients,
    ) = get_foscal_train_val_test_patients(source_dset_dir)

    # Create the directory to store all the data.
    dir_suffix = create_dir_suffix(
        volumes=volumes,
        slices=slices,
        patches=patches,
        z_norm=z_norm,
        min_max_norm=min_max_norm,
    )
    target_dset_dir = os.path.join(data_dir, f"FOSCAL_{dir_suffix}")
    print(f"Created directory for the dataset at: {target_dset_dir}\n")
    os.makedirs(target_dset_dir, exist_ok=True)

    # Create training and validation sets.
    if z_norm:
        normalization = "z"
    elif min_max_norm:
        normalization = "min_max"

    # Create the TFRecords for full_training, training, valid and test.
    tfrecords_dir = os.path.join(target_dset_dir, "tfrecords")
    os.makedirs(tfrecords_dir, exist_ok=True)
    full_train_tfrecord_path, num_full_train_samples = create_foscal_dataset(
        tfrecords_dir,
        train_patients_orig,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="full_train",
    )
    train_tfrecord_path, num_train_samples = create_foscal_dataset(
        tfrecords_dir,
        train_patients,
        volumes=volumes,
        slices=slices,
        patches=patches,
        normalization=normalization,
        modalities=modalities,
        dset_split="train",
    )

    # Validation and test metrics are always calculated at volume-level.
    # Hence, serialize volumes to the TFRecords. Test TFRecord is not created
    # because it patients do not have masks. Other script will be used.
    valid_tfrecord_path, num_valid_samples = create_foscal_dataset(
        tfrecords_dir,
        valid_patients,
        volumes=True,
        slices=False,
        patches=False,
        normalization=normalization,
        modalities=modalities,
        dset_split="valid",
    )

    # Save the patient paths to the dataset dir.
    patients_dir = os.path.join(target_dset_dir, "patients")
    os.makedirs(patients_dir, exist_ok=True)
    full_train_patients_path = os.path.join(patients_dir, "full_train_patients.txt")
    train_patients_path = os.path.join(patients_dir, "train_patients.txt")
    valid_patients_path = os.path.join(patients_dir, "valid_patients.txt")
    test_patients_path = os.path.join(patients_dir, "test_patients.txt")
    np.savetxt(full_train_patients_path, train_patients_orig, fmt="%s")
    np.savetxt(train_patients_path, train_patients, fmt="%s")
    np.savetxt(valid_patients_path, valid_patients, fmt="%s")
    np.savetxt(test_patients_path, test_patients, fmt="%s")

    # Save dataset configuration as YAML file.
    dset_config = {
        "dset_name": "FOSCAL",
        "normalization": normalization,
        "full_train_tfrecord": get_path_and_parent_dir(full_train_tfrecord_path),
        "train_tfrecord": get_path_and_parent_dir(train_tfrecord_path),
        "valid_tfrecord": get_path_and_parent_dir(valid_tfrecord_path),
        "full_train_patients_path": get_path_and_parent_dir(full_train_patients_path),
        "train_patients_path": get_path_and_parent_dir(train_patients_path),
        "valid_patients_path": get_path_and_parent_dir(valid_patients_path),
        "test_patients_path": get_path_and_parent_dir(test_patients_path),
        "num_full_train_samples": num_full_train_samples,
        "num_train_samples": num_train_samples,
        "num_valid_samples": num_valid_samples,
        "modalities": modalities,
    }
    dset_config_path = os.path.join(target_dset_dir, "dset_config.yml")
    save_dict_as_yaml(dset_config_path, dset_config)


# yapf: disable
@cli.command()
@click.option('--source-dset-dir', '-sdset', required=True, help='Path to source dataset')
@click.option('--volumes', '-vol', is_flag=True, help='Serialize volumes to the tfrecord')
@click.option('--slices', '-sli', is_flag=True, help='Serialize slices to the tfrecord')
@click.option('--patches', '-pat', is_flag=True, help='Serialize patches to the tfrecord')
@click.option('--z-norm', '-z', is_flag=True, help='Normalize volumes with z normalization')
@click.option('--min-max-norm', '-min-max', is_flag=True, help='Normalize volumes with min-max normalization')
# yapf: enable
def atlasv1_train_val(source_dset_dir, volumes, slices, patches, z_norm, min_max_norm):

    check_norm_flags(z_norm, min_max_norm)
    check_spatial_flags(volumes, slices, patches)

    # Load the data and masks paths from the source dataset.
    data_paths, mask_paths = generate_data_and_masks_paths(source_dset_dir)

    # Divide the paths into train and validation.
    train_data_paths = [p for p in data_paths if "train" in p]
    train_mask_paths = [p for p in mask_paths if "train" in ",".join(p)]
    val_data_paths = [p for p in data_paths if "val" in p]
    val_mask_paths = [p for p in mask_paths if "val" in ",".join(p)]

    # Create the directory to store all the data.
    dir_suffix = create_dir_suffix(
        volumes=volumes,
        slices=slices,
        patches=patches,
        z_norm=z_norm,
        min_max_norm=min_max_norm,
    )
    target_dset_dir = os.path.join(data_dir, f"ATLASv1_{dir_suffix}")
    print(f"Created directory for the dataset at: {target_dset_dir}\n")
    os.makedirs(target_dset_dir, exist_ok=True)

    # Create training and validation sets.
    if z_norm:
        normalization = "z"
    elif min_max_norm:
        normalization = "min_max"

    # Create the TFRecords for full_training, training, valid and test.
    tfrecords_dir = os.path.join(target_dset_dir, "tfrecords")
    os.makedirs(tfrecords_dir, exist_ok=True)
    train_tfrecord_path, num_train_samples = create_atlasv1_dataset(
        tfrecords_dir,
        train_data_paths,
        train_mask_paths,
        volumes,
        slices,
        patches,
        normalization,
        dset_split="train",
    )

    # Validation and test metrics are always calculated at volume-level.
    # Hence, serialize volumes to the TFRecords.
    val_tfrecord_path, num_valid_samples = create_atlasv1_dataset(
        tfrecords_dir,
        val_data_paths,
        val_mask_paths,
        volumes=True,
        slices=False,
        patches=False,
        normalization=normalization,
        dset_split="valid",
    )

    # Save the patient paths to the dataset dir.
    patients_dir = os.path.join(target_dset_dir, "patients")
    os.makedirs(patients_dir, exist_ok=True)
    train_mask_paths_save = flatten_list_of_str(train_mask_paths)
    train_data_path = os.path.join(patients_dir, "train_data.txt")
    train_masks_path = os.path.join(patients_dir, "train_masks.txt")
    np.savetxt(
        train_data_path,
        np.array(train_data_paths, dtype=object),
        fmt="%s",
    )
    np.savetxt(
        train_masks_path,
        np.array(train_mask_paths_save, dtype=object),
        fmt="%s",
    )

    valid_mask_paths_save = flatten_list_of_str(val_mask_paths)
    valid_data_path = os.path.join(patients_dir, "valid_data.txt")
    valid_masks_path = os.path.join(patients_dir, "valid_masks.txt")
    np.savetxt(
        valid_data_path,
        np.array(val_data_paths, dtype=object),
        fmt="%s",
    )
    np.savetxt(
        valid_masks_path,
        np.array(valid_mask_paths_save, dtype=object),
        fmt="%s",
    )

    # Save dataset configuration as YAML file.
    dset_config = {
        "dset_name": "ATLASv1",
        "normalization": normalization,
        "train_tfrecord": get_path_and_parent_dir(train_tfrecord_path),
        "valid_tfrecord": get_path_and_parent_dir(val_tfrecord_path),
        "train_data_path": get_path_and_parent_dir(valid_data_path),
        "train_masks_path": get_path_and_parent_dir(valid_masks_path),
        "num_train_samples": num_train_samples,
        "num_valid_samples": num_valid_samples,
        "modalities": ["T1w"],  # Modalities taken from the official github repo.
    }

    dset_config_path = os.path.join(target_dset_dir, "dset_config.yml")
    save_dict_as_yaml(dset_config_path, dset_config)


# Helpers
# ----------------------------------------------------------------


def check_norm_flags(z_norm: bool, min_max_norm: bool):
    """Normalization argument testing: Check if only one argument
    was marked"""
    if z_norm and min_max_norm:
        raise ValueError(
            "Both z_norm and min_max_norm are True. Only one could "
            "be True per execution."
        )
    elif (not z_norm) and (not min_max_norm):
        raise ValueError(
            "Both z_norm and min_max_norm are False. One of them should True."
        )


def check_spatial_flags(volumes: bool, slices: bool, patches: bool):
    """Dataset type argument testing: Raise an exception if
    the user marked two or more modes."""
    if volumes and slices and patches:
        raise ValueError(
            "All the arguments were set to True. Only one of volumes"
            " slices and patches should be True per execution"
        )
    elif not (volumes ^ slices ^ patches):
        raise ValueError(
            "Set one, and only one, argument of volumes, slices and " "patches to True."
        )


def get_path_and_parent_dir(path: str) -> str:
    """Return only the parent dir and filename of a path

    e.g. 'dir/dir2/dir3/filename.txt' -> 'dir3/filename.txt'
    """
    return "/".join(path.split("/")[-2:])


# CLI
# ----------------------------------------------------------------

if __name__ == "__main__":
    cli()
