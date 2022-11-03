import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from utils.datasets.atlasv1.patient import ATLASV1Patient
from utils.datasets.serializers import serialize_2d_example, serialize_3d_example


def create_atlasv1_dataset(
    dset_dir: str,
    data_paths: Union[List[str], np.ndarray],
    mask_paths: Union[List[str], np.ndarray],
    volumes: bool,
    slices: bool,
    patches: bool,
    normalization: str,
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
        dset_split (Optional[str], optional): Which split/partition the paths
        belongs to, e.g. train, valid, test.

    Raises:
        NotImplementedError: if patches is selected.

    Returns:
        str: path to the created TFRecord file
    """

    num_samples = 0
    os.makedirs(dset_dir, exist_ok=True)
    tfrecord_path = os.path.join(dset_dir, f"{dset_split}.tfrecord")
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path)

    assert len(data_paths) == len(mask_paths), "Data and mask paths length differ"
    num_paths = len(data_paths)
    for data_path, mask_path in tqdm(
        zip(data_paths, mask_paths),
        total=num_paths,
        desc=f"Creating {dset_split} dataset",
    ):

        patient = ATLASV1Patient(data_path, mask_path)
        patient.load_niftis()

        # Get the normalized data.
        mask = patient.get_mask()
        data = patient.get_data(normalization=normalization)

        # Create the channel dim and tranpose for ease of iteration.
        # i.e. (H, W, N) -> (C=1, H, W, N) -> (N, H, W, C=1)
        data_volume = np.expand_dims(data, axis=0)
        data_volume = data_volume.transpose(3, 1, 2, 0)  #
        mask_volume = np.expand_dims(mask, axis=0)
        mask_volume = mask_volume.transpose(3, 1, 2, 0)

        if volumes:
            volumes_dict = {"T1w": data_volume, "mask": mask_volume}
            serialized_features = serialize_3d_example(volumes_dict)
            tfrecord_writer.write(serialized_features)
            num_samples += 1
        elif slices:
            for data_slice, mask_slice in zip(data_volume, mask_volume):
                data_slice = data_slice.astype(np.float32)
                mask_slice = mask_slice.astype(np.float32)
                slice_dict = {"T1w": data_slice, "mask": mask_slice}
                serialized_features = serialize_2d_example(slice_dict)
                tfrecord_writer.write(serialized_features)
                num_samples += 1
        elif patches:
            raise NotImplementedError("Not implemented at the time.")

    tfrecord_writer.close()

    return tfrecord_path, num_samples


def generate_data_and_masks_paths(dir: str):
    """Generates a list of data and inputs to MRI objects
    for the ATLASv1 dataset.

    Taken from: https://github.com/BMIRDS/3dMRISegmentation/"""
    inputs, labels = [], []
    for dirpath, dirs, files in os.walk(dir):
        label_list = list()
        for file in files:
            if not file.startswith(".") and file.endswith(".nii.gz"):
                if "Lesion" in file:
                    label_list.append(os.path.join(dirpath, file))
                elif "mask" not in file:
                    inputs.append(os.path.join(dirpath, file))
        if label_list:
            labels.append(label_list)

    return inputs, labels


def split_paths_into_train_and_val(data_paths: List[str], mask_paths: List[str]):
    train_data_paths = [p for p in data_paths if "train" in p]
    train_mask_paths = [p for p in mask_paths if "train" in ",".join(p)]
    val_data_paths = [p for p in data_paths if "val" in p]
    val_mask_paths = [p for p in mask_paths if "val" in ",".join(p)]

    return train_data_paths, train_mask_paths, val_data_paths, val_mask_paths


def flatten_list_of_str(paths):
    save_mask_paths = []
    for p in paths:
        save_mask_paths.append(",".join(p))
    return save_mask_paths
