import os
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from typing import List, Optional, Tuple

from utils.preprocessing.numpy import get_mask_with_contours
from utils.datasets.isles2017.patient import ISLES2017Patient
from utils.datasets.serializers import serialize_2d_example, serialize_3d_example


def create_isles2017_dataset(
    dset_dir: str,
    patient_paths: np.ndarray,
    volumes: bool,
    slices: bool,
    patches: bool,
    normalization: str,
    modalities: List[str],
    dset_split: Optional[str] = None,
) -> str:
    """Creates a new dataset to store numpy files given a list of paths to
    patients directories. This function returns two lists with the paths to
    the numpy files (modalities and annotations).

    Args:
        dset_dir (str): Path where the dataset will be stored.
        patient_paths (np.ndarray): Paths to all the patient's directories.
        volumes (bool): Save data as volumes (3D).
        slices (bool): Save data as slices (2D).
        patches (bool): Save data as patches of slices (2D).
        z_norm (bool): Whether to apply or not z normalization.
        min_max_norm (bool): Whether to apply or not min max normalization.
        modalities (List[str]): List of modalities to take into account.
        dset_split (Optional[str], optional): Which split/partition the paths
        belongs to, e.g. train, valid, test.

    Raises:
        NotImplementedError: if patches is selected.

    Returns:
        str: path to the created TFRecord
    """

    num_samples = 0
    os.makedirs(dset_dir, exist_ok=True)
    tfrecord_path = os.path.join(dset_dir, f"{dset_split}.tfrecord")
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path)

    for patient_path in tqdm(patient_paths, desc=f"Creating {dset_split} dataset"):
        patient = ISLES2017Patient(str(patient_path))
        patient.load_niftis()

        # Get the normalized data.
        mask = patient.get_mask()
        data = patient.get_data(modalities=modalities, normalization=normalization)

        if volumes:
            # Slices first.
            modalities_volumes = {
                k: expand_last_dim(v.transpose(2, 0, 1)) for k, v in data.items()
            }
            modalities_volumes["mask"] = expand_last_dim(mask.transpose(2, 0, 1))
            mask_with_contours = get_mask_with_contours(mask, contour_thickness=1)
            modalities_volumes["mask_with_contours"] = expand_last_dim(
                mask_with_contours.transpose(2, 0, 1)
            )
            serialized_features = serialize_3d_example(modalities_volumes)
            tfrecord_writer.write(serialized_features)
            num_samples += 1
        elif slices:
            num_slices = data[modalities[0]].shape[-1]
            for slice_idx in range(num_slices):
                modalities_slices = {
                    k: expand_last_dim(v[..., slice_idx]) for k, v in data.items()
                }
                modalities_slices["mask"] = expand_last_dim(mask[..., slice_idx])
                mask_with_contours = get_mask_with_contours(
                    mask[..., slice_idx], contour_thickness=1
                )
                modalities_slices["mask_with_contours"] = expand_last_dim(
                    mask_with_contours
                )
                serialized_features = serialize_2d_example(modalities_slices)
                tfrecord_writer.write(serialized_features)
                num_samples += 1
        elif patches:
            raise NotImplementedError("Not implemented at the time.")

    tfrecord_writer.close()

    return tfrecord_path, num_samples


# Old functions, used with the old serializers.
# ----------------------------------------------------------------------------


def _create_isles2017_dataset(
    dset_dir: str,
    patient_paths: np.ndarray,
    volumes: bool,
    slices: bool,
    patches: bool,
    normalization: str,
    modalities: List[str],
    dset_split: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Creates a new dataset to store numpy files given a list of paths to
    patients directories. This function returns two lists with the paths to
    the numpy files (modalities and annotations).

    Args:
        dset_dir (str): Path where the dataset will be stored.
        patient_paths (np.ndarray): Paths to all the patient's directories.
        volumes (bool): Save data as volumes (3D).
        slices (bool): Save data as slices (2D).
        patches (bool): Save data as patches of slices (2D).
        z_norm (bool): Whether to apply or not z normalization.
        min_max_norm (bool): Whether to apply or not min max normalization.
        modalities (List[str]): List of modalities to take into account.
        dset_split (Optional[str], optional): Which split/partition the paths
        belongs to, e.g. train, valid, test.

    Raises:
        NotImplementedError: if patches is selected.

    Returns:
        Tuple[List[str], List[str]]: [description]
    """

    from utils.datasets.serializers import _serialize_2d_example, _serialize_3d_example

    os.makedirs(dset_dir, exist_ok=True)
    tfrecord_path = os.path.join(dset_dir, f"{dset_split}.tfrecord")
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path)

    for patient_path in tqdm(patient_paths, desc=f"Creating {dset_split} dataset"):
        patient = ISLES2017Patient(str(patient_path))
        patient.load_niftis()

        # Get the normalized data.
        mask = patient.get_mask()
        data = patient.get_data(modalities=modalities, normalization=normalization)

        if volumes:
            data_volume = np.stack(list(data.values()))  # (C, H, W, N)
            data_volume = data_volume.transpose(3, 1, 2, 0)  # (N, H, W, C)
            mask_volume = np.expand_dims(mask, axis=0)
            mask_volume = mask_volume.transpose(3, 1, 2, 0)
            serialized_features = _serialize_3d_example(data_volume, mask_volume)
            tfrecord_writer.write(serialized_features)
        elif slices:
            num_slices = data[modalities[0]].shape[-1]
            for slice_idx in range(num_slices):
                data_slice = [m[..., slice_idx] for m in data.values()]
                data_slice = np.transpose(data_slice, axes=(1, 2, 0))
                data_slice = data_slice.astype(np.float32)
                mask_slice = mask[..., slice_idx]
                mask_slice = np.expand_dims(mask_slice, axis=-1)
                mask_slice = mask_slice.astype(np.float32)
                serialized_features = _serialize_2d_example(data_slice, mask_slice)
                tfrecord_writer.write(serialized_features)
        elif patches:
            raise NotImplementedError("Not implemented at the time.")

    tfrecord_writer.close()


def get_train_val_test_patients(source_dset_dir: str):

    # Load the paths of all patient directories.
    dset_patients = glob(os.path.join(source_dset_dir, "*"))
    dset_patients = [p for p in dset_patients if os.path.isdir(p)]
    dset_patients = [os.path.abspath(p) for p in dset_patients]

    # Create a list for the training and test patients.
    train_patients_orig = [p for p in dset_patients if "training" in p]
    test_patients = [p for p in dset_patients if "test" in p]
    train_patients_orig = np.array(sorted(train_patients_orig))
    test_patients = np.array(sorted(test_patients))

    # Split the training set into train and validation.
    np.random.seed(26)
    num_train_patients_orig = int(len(train_patients_orig) * 0.8)
    idxs = np.arange(len(train_patients_orig))
    train_idxs = np.random.choice(idxs, replace=False, size=num_train_patients_orig)
    valid_idxs = np.delete(idxs, train_idxs)
    train_patients = train_patients_orig[train_idxs]
    valid_patients = train_patients_orig[valid_idxs]

    return train_patients_orig, train_patients, valid_patients, test_patients


def expand_last_dim(arr: np.ndarray) -> np.ndarray:
    """Expands the last dimension of `arr` and converts it to float32"""
    return np.expand_dims(arr, axis=-1).astype(np.float32)
