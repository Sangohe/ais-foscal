import numpy as np
import nibabel as nib

import os
from typing import List, Optional, Dict

from utils.preprocessing.numpy import min_max_normalization


class ISLES2022Patient:
    """Utility class for loading the information of a patient."""

    def __init__(self, dset_dir: str, patient_id: str) -> None:
        self.dset_dir = dset_dir
        self.patient_id = patient_id
        self.data_dir = os.path.join(dset_dir, "rawdata", patient_id)
        self.derivatives_dir = os.path.join(dset_dir, "derivatives", patient_id)

        if os.path.isdir(self.data_dir) and os.path.isdir(self.derivatives_dir):
            self.load_niftis()
            # same_spacing = np.array_equal(
            #     self.adc.header['pixdim'], [-1.,  2.,  2.,  2.,  0.,  0.,  0.,  0.]
            # )

            # if not same_spacing:
            #     print(f"{patient_id}: {self.adc.header['pixdim']}.")
        else:
            raise ValueError(
                "The patient does not have a `data` or `derivatives` directory."
            )

    def load_niftis(self) -> None:
        """Traverse the directories inside paient_dir and load the nifti data."""

        self.load_modality(self.data_dir, "adc")
        self.load_modality(self.data_dir, "dwi")
        self.load_modality(self.derivatives_dir, "msk")

    def get_data(
        self,
        modalities: List[str] = ["ADC", "DWI"],
        normalization: Optional[str] = None,
        resampled: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Returns a list with the modalities"""

        if normalization not in [None, "z", "min_max"]:
            raise ValueError("normalization kwarg has to be one of None, z or min_max.")

        if normalization == "z":
            norm_fn = z_normalization
        elif normalization == "min_max":
            norm_fn = min_max_normalization

        data = {}
        for modality in modalities:
            attr_name = (
                f"{modality.lower()}_resampled" if resampled else f"{modality.lower()}"
            )
            modality_data = getattr(self, attr_name).get_fdata()
            modality_data = modality_data.astype(np.float32)
            if normalization is not None:
                modality_data = norm_fn(modality_data)
            data[modality] = modality_data
        return data

    def get_mask(
        self,
        resampled: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Returns the OT data within a dictionary. Implemented because the
        normalization kwarg could case problems when returning the OT
        with the get_data function."""
        masks = {}
        attr_name = f"msk_resampled" if resampled else f"msk"
        mask = getattr(self, attr_name).get_fdata()
        mask = mask.astype(np.float32)
        masks["mask"] = mask
        return masks

    def load_modality(self, source_dir: str, modality_name: str):
        data_dir = os.path.join(source_dir, "ses-0001")
        data_filenames = os.listdir(data_dir)
        modality_filename = get_first_match_for_substring(
            data_filenames, f"{modality_name}.nii.gz"
        )
        modality_path = os.path.join(data_dir, modality_filename)

        setattr(self, f"{modality_name}_path", modality_path)
        setattr(self, f"{modality_name}", nib.load(modality_path))
        setattr(self, f"{modality_name}_shape", getattr(self, f"{modality_name}").shape)

        resampled_data_dir = os.path.join(data_dir, "resampled")
        if os.path.exists(resampled_data_dir):
            resampled_data_filenames = os.listdir(resampled_data_dir)
            resampled_modality_filename = get_first_match_for_substring(
                resampled_data_filenames, f"{modality_name}_resampled.nii.gz"
            )
            resampled_modality_path = os.path.join(
                resampled_data_dir, resampled_modality_filename
            )

            setattr(self, f"{modality_name}_resampled_path", resampled_modality_path)
            setattr(
                self, f"{modality_name}_resampled", nib.load(resampled_modality_path)
            )
            setattr(
                self,
                f"{modality_name}_resampled_shape",
                getattr(self, f"{modality_name}_resampled").shape,
            )


def get_first_match_for_substring(list_, substring: str):
    """Returns the first element in `list_` that matches."""
    match = None
    for elem in list_:
        if substring in elem:
            match = elem
            break
    return match
