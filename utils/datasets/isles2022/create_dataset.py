import os
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from typing import List, Optional, Tuple

from utils.preprocessing.numpy import get_mask_with_contours
from utils.datasets.isles2022.patient import ISLES2022Patient
from utils.datasets.serializers import serialize_2d_example, serialize_3d_example


def create_isles2022_dataset(
    dset_dir: str,
    patient_ids: np.ndarray,
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
        patient_ids (np.ndarray): Paths to all the patient's directories.
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

    for patient_id in tqdm(patient_ids, desc=f"Creating {dset_split} dataset"):
        patient = ISLES2022Patient("/data/stroke/ISLES2022/", str(patient_id))

        # Get the normalized data.
        data = patient.get_data(
            modalities=modalities, normalization=normalization, resampled=True
        )
        masks = patient.get_mask(resampled=True)

        if volumes:
            # Slices first.
            modalities_volumes = {
                k: expand_last_dim(v.transpose(2, 0, 1)) for k, v in data.items()
            }
            modalities_masks = {
                "mask": expand_last_dim(masks["mask"].transpose(2, 0, 1))
            }
            modalities_masks_contours = {
                "mask_with_contours": expand_last_dim(
                    get_mask_with_contours(
                        masks["mask"], contour_thickness=1
                    ).transpose(2, 0, 1)
                )
            }
            modalities_volumes.update(modalities_masks)
            modalities_volumes.update(modalities_masks_contours)
            serialized_features = serialize_3d_example(modalities_volumes)
            tfrecord_writer.write(serialized_features)
            num_samples += 1
        elif slices:
            num_slices = data[modalities[0]].shape[-1]
            for slice_idx in range(num_slices):
                modalities_slices = {
                    k: expand_last_dim(v[..., slice_idx]) for k, v in data.items()
                }
                modalities_slices_mask = {
                    "mask": expand_last_dim(masks["mask"][..., slice_idx])
                }
                modalities_slices_mask_contours = {
                    "mask_with_contours": expand_last_dim(
                        get_mask_with_contours(
                            masks["mask"][..., slice_idx], contour_thickness=1
                        )
                    )
                }
                modalities_slices.update(modalities_slices_mask)
                modalities_slices.update(modalities_slices_mask_contours)
                serialized_features = serialize_2d_example(modalities_slices)
                tfrecord_writer.write(serialized_features)
                num_samples += 1
        elif patches:
            raise NotImplementedError("Not implemented at the time.")

    tfrecord_writer.close()

    return tfrecord_path, num_samples


def expand_last_dim(arr: np.ndarray) -> np.ndarray:
    """Expands the last dimension of `arr` and converts it to float32"""
    return np.expand_dims(arr, axis=-1).astype(np.float32)
