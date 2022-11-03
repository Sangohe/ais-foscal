import numpy as np
import nibabel as nib

import os
from glob import glob
from typing import List, Optional, Dict

from ...preprocessing.numpy import z_normalization, min_max_normalization


class FOSCALPatient:
    """ "Utility class for loading the information of a patient."""

    def __init__(self, patient_dir: str) -> None:

        if os.path.isdir(patient_dir):
            self.patient_dir = patient_dir
            self.patient_id = patient_dir.split("/")[-1]
            self.content = os.listdir(patient_dir)

            self.load_niftis()
        else:
            raise ValueError("patient_dir does not exist or is not a directory.")

    def load_niftis(self) -> None:
        """Traverse the directories inside paient_dir and load the nifti data."""

        for modality in ["adc", "dwi"]:
            self.load_modality(modality)
            for radiologist in ["Daniel", "Andres"]:
                self.load_mask(modality, radiologist)

    def get_data(
        self,
        modalities: List[str] = ["ADC", "DWI"],
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
            if modality_data.ndim == 4:
                modality_data = modality_data[..., 0]
            modality_data = modality_data.astype(np.float32)
            if normalization is not None:
                modality_data = norm_fn(modality_data)
            data[modality] = modality_data
        return data

    def get_mask(
        self,
        modalities: List[str] = ["ADC", "DWI"],
    ) -> Dict[str, np.ndarray]:
        """Returns the OT data within a dictionary. Implemented because the
        normalization kwarg could case problems when returning the OT
        with the get_data function."""
        masks = {}
        for modality in modalities:
            modality_mask = getattr(self, f"{modality.lower()}_mask").get_fdata()
            modality_mask = modality_mask.astype(np.float32)
            masks[modality] = modality_mask
        return masks

    def load_modality(self, modality_name: str):

        content = [os.path.join(self.patient_dir, p) for p in self.content]
        content = [p for p in content if os.path.isdir(p)]

        modality_matches = [p for p in content if modality_name in p.lower()]
        assert (
            len(modality_matches) > 0
        ), f"Could not found {modality_name.upper()} directory."
        assert (
            len(modality_matches) == 1
        ), f"Found more than one {modality_name.upper()} directory."
        modality_dir = modality_matches[0]

        modality_dir_content = glob(os.path.join(modality_dir, "*.nii*"))
        assert (
            len(modality_dir_content) == 1
        ), f"Found more than one {modality_name.upper()} path."
        modality_nifti = nib.load(modality_dir_content[0])

        setattr(self, f"{modality_name}_path", modality_dir_content[0])
        setattr(self, modality_name, modality_nifti)
        setattr(self, f"{modality_name}_shape", modality_nifti.shape)

    def load_mask(self, modality_name: str, radiologist: str):
        masks_dir = os.path.join(self.patient_dir, "Masks", radiologist)
        masks_dir_content = [os.path.join(masks_dir, p) for p in os.listdir(masks_dir)]

        if modality_name == "dwi":

            def cond(path):
                return "dwi" in path.lower() or "1000" in path.lower()

        elif modality_name == "adc":

            def cond(path):
                return "adc" in path.lower()

        modality_mask_matches = [p for p in masks_dir_content if cond(p)]
        if len(modality_mask_matches) == 0:
            print(f"Could not found {modality_name.upper()} mask for {radiologist}.")
        if len(modality_mask_matches) > 1:
            print(
                f"Found more than one {modality_name.upper()} mask for {radiologist}."
            )
        modality_mask_path = modality_mask_matches[0]

        modality_mask = nib.load(modality_mask_path)
        setattr(self, f"{modality_name}_mask_path", modality_mask_path)
        setattr(self, f"{modality_name}_mask", modality_mask)

        modality = getattr(self, modality_name)
        shape_is_equal = modality.shape == modality_mask.shape
        if not shape_is_equal and modality_name != "dwi":
            print(
                f"{modality_name} and mask shapes do not match -> {modality.shape} != {modality_mask.shape}"
            )
