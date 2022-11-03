import numpy as np
import nibabel as nib

import os
from glob import glob
from typing import List, Optional, Dict

from ...preprocessing.numpy import z_normalization, min_max_normalization


class ISLES2017Patient:
    """ "Utility class for loading the information of a patient."""

    def __init__(self, patient_dir: str, load_pwi: bool = False) -> None:

        if os.path.isdir(patient_dir):
            self.patient_dir = patient_dir
            self.patient_id = patient_dir.split("/")[-1]
            self.content = os.listdir(patient_dir)
            self.load_pwi = load_pwi

            self.load_niftis()
        else:
            raise ValueError("patient_dir does not exist or is not a directory.")

    def load_niftis(
        self,
        modalities: List[str] = [
            "PWI",
            "ADC",
            "rCBV",
            "rCBF",
            "MTT",
            "TTP",
            "Tmax",
            "OT",
        ],
    ) -> None:
        """Traverse the directories inside paient_dir and load the nifti data."""
        if not self.load_pwi and "PWI" in modalities:
            modalities.remove("PWI")

        # Traverse the folders inside patient_dir.
        for modality_dir in self.content:

            # Find out the modality name of the directory.
            if "OT" not in modality_dir:
                modality = self.extract_modality_from_path(modality_dir)
                modality = "PWI" if modality == "4DPWI" else modality
            else:
                modality = "OT"

            if modality in modalities:
                # Get the path for the NifTi file inside each folder.
                modality_dir = os.path.join(self.patient_dir, modality_dir)
                modality_nifti_path = glob(os.path.join(modality_dir, "*.nii.gz"))[0]
                setattr(self, f"{modality.lower()}_path", modality_nifti_path)

                # Load the NifTi using the path.
                modality_nifti = nib.load(modality_nifti_path)
                setattr(self, modality.lower(), modality_nifti)

        self.original_shape = self.adc.get_fdata().shape
        # Add the original shape from dwi.

    def get_data(
        self,
        modalities: List[str] = ["ADC", "rCBV", "rCBF", "MTT", "TTP", "Tmax"],
        normalization: Optional[str] = None,
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
            modality_data = getattr(self, modality.lower()).get_fdata()
            modality_data = modality_data.astype(np.float32)

            # Fix Some PWI slices have 5 dims.
            if modality_data.ndim == 5:
                modality_data = np.squeeze(modality_data, axis=3)
            if normalization is not None:
                modality_data = norm_fn(modality_data)
            data[modality] = modality_data
        return data

    def get_mask(self) -> Dict[str, np.ndarray]:
        """Returns the OT data within a dictionary. Implemented because the
        normalization kwarg could case problems when returning the OT
        with the get_data function."""
        data = self.ot.get_fdata().astype(np.float32)
        return data

    @staticmethod
    def extract_modality_from_path(path: str) -> str:
        path_shards = path.split(".")
        modality = path_shards[-2][3:]
        return modality

    @staticmethod
    def get_smir_id_from_path(path: str) -> str:
        path_shards = path.split("/")[-1]
        return path_shards.split(".")[-3]

    def save_mask_in_smir_format(
        self, mask: np.ndarray, save_dir: str = "", description: str = "patient"
    ) -> None:
        """[summary]

        Args:
            mask (np.ndarray): Array of zeros or ones.
            save_dir (str, optional): Directory to export the NifTi. Defaults to ''.
            description (str, optional): String to elaborate the save path. Defaults
            to 'patient'.

        Raises:
            ValueError: If save_dir is empty.
        """

        if save_dir == "":
            raise ValueError("save_dir cannot be empty")

        # Elaborate the save path as specified in the competition website.
        smir_id = self.get_smir_id_from_path(self.mtt_path)
        fname = f"SMIR.{description}.{smir_id}.nii"
        save_path = os.path.join(save_dir, fname)

        # Create the NifTi object with the mask data and adc header.
        mask_nifti = nib.Nifti1Image(mask, np.eye(4))
        mask_nifti.set_data_dtype(dtype=np.uint8)

        # Taken from https://www.smir.ch/Content/scratch/isles/nibabel_copy_header.py
        # Somehow this changes the data array from the export image but preserves the
        # headers.
        export_image = self.adc
        i = export_image.get_data()
        i[:] = mask_nifti.get_data()

        nib.save(mask_nifti, save_path)

    def save_probs_in_smir_format(
        self, probs: np.ndarray, save_dir: str = "", description: str = "patient"
    ) -> None:
        """[summary]

        Args:
            probs (np.ndarray): Array with floating point numbers symbolizing
            probabilities.
            save_dir (str, optional): Directory to export the NifTi. Defaults to ''.
            description (str, optional): String to elaborate the save path. Defaults
            to 'patient'.

        Raises:
            ValueError: If save_dir is empty.
        """

        if save_dir == "":
            raise ValueError("save_dir cannot be empty")

        # Elaborate the save path as specified in the competition website.
        smir_id = self.get_smir_id_from_path(self.mtt_path)
        fname = f"SMIR.{description}.{smir_id}.nii"
        save_path = os.path.join(save_dir, fname)

        # Create the NifTi object with the mask data and adc header.
        original = self.adc
        original.set_data_dtype(16)

        img = nib.Nifti1Image(probs, original.affine, header=original.header)
        nib.save(img, save_path)
