import numpy as np
import nibabel as nib

from typing import List, Optional

from ...preprocessing.numpy import z_normalization, min_max_normalization


class ATLASV1Patient:
    def __init__(self, data_path: str, mask_paths: List[str]):
        self.patient_id = data_path.split("/")[-1].split("_")[0]
        self.data_path = data_path
        self.mask_paths = mask_paths

    def load_niftis(self):
        self.data = self.correct_dims(nib.load(self.data_path).get_fdata())
        self.mask = self.gen_mask(self.mask_paths)

    def get_data(self, normalization: Optional[str] = None) -> np.ndarray:
        """Returns a list with the modalities"""

        if normalization not in [None, "z", "min_max"]:
            raise ValueError("normalization kwarg has to be one of None, z or min_max.")

        if normalization is None:
            data = self.data
        elif normalization == "z":
            data = z_normalization(self.data)
        elif normalization == "min_max":
            data = min_max_normalization(self.data)

        return data.astype(np.float32)

    def get_mask(self) -> np.ndarray:
        mask = self.mask.astype(np.float32) / 255.0
        return mask

    @staticmethod
    def correct_dims(img):
        """
        Fix the dimension of the image, if necessary
        """
        if len(img.shape) > 3:
            img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
        return img

    @staticmethod
    def gen_mask(lesion_files):
        """
        Given a list of lesion files, generate a mask
        that incorporates data from all of them
        """
        first_lesion = nib.load(lesion_files[0]).get_fdata()
        if len(lesion_files) == 1:
            return first_lesion
        lesion_data = np.zeros(
            (first_lesion.shape[0], first_lesion.shape[1], first_lesion.shape[2])
        )
        for file in lesion_files:
            l_file = ATLASV1Patient.correct_dims(nib.load(file).get_fdata())
            lesion_data = np.maximum(l_file, lesion_data)
        return lesion_data
