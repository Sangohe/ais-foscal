import numpy as np
import pandas as pd

from typing import Tuple, Union, List
from utils.datasets.isles2017.patient import ISLESPatient


def count_pixels_for_mask_and_data(
    ot: np.ndarray, data: np.ndarray
) -> Tuple[int, int, int, int]:
    """Computes a set of useful metrics for a given ot and data pair.

    Args:
        ot (np.ndarray): Mask volume
        data (np.ndarray): Medical image. ADC modality is preferred.

    Returns:
        Tuple[int, int, int, int]: measurements of the data.
    """
    num_pixels = np.prod(ot.shape)
    lesion_pixels = np.count_nonzero(ot)
    bg_pixels = num_pixels - lesion_pixels
    brain_pixels = (data > 0).sum()

    return num_pixels, lesion_pixels, bg_pixels, brain_pixels


def create_df_with_pixel_count_for_patients(
    patient_paths: Union[List[str], np.ndarray]
) -> pd.DataFrame:
    """Creates a DataFrame with the pixel count for the background, lesion
    classes and some other interesting metrics.

    Args:
        patient_paths (Union[List[str], np.ndarray]): List of paths to
        patients directories.

    Returns:
        pd.DataFrame: Summary of pixel count metrics for the given
        patients.
    """

    pixel_metrics_dict = {
        "num_pixels": [],
        "lesion_pixels": [],
        "bg_pixels": [],
        "brain_pixels": [],
        "has_lesion": [],
        "lesion_perc": [],
    }

    for patient_path in patient_paths:

        patient = ISLESPatient(str(patient_path))
        patient.load_niftis()
        ot = patient.get_ot()["OT"]
        adc = patient.get_data(modalities=["ADC"])["ADC"]

        # Pixel level metrics.
        (
            num_pixels,
            lesion_pixels,
            bg_pixels,
            brain_pixels,
        ) = count_pixels_for_mask_and_data(ot=ot, data=adc)
        has_lesion = lesion_pixels > 0
        lesion_perc = lesion_pixels / brain_pixels

        pixel_metrics_dict["num_pixels"].append(num_pixels)
        pixel_metrics_dict["lesion_pixels"].append(lesion_pixels)
        pixel_metrics_dict["bg_pixels"].append(bg_pixels)
        pixel_metrics_dict["brain_pixels"].append(brain_pixels)
        pixel_metrics_dict["has_lesion"].append(has_lesion)
        pixel_metrics_dict["lesion_perc"].append(lesion_perc)

    pixel_metrics_dict["patient"] = [p.split("/")[-1] for p in patient_paths.tolist()]
    pixel_metrics_df = pd.DataFrame(pixel_metrics_dict)

    return pixel_metrics_df
