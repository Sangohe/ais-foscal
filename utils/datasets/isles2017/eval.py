import os
import sys
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional

from .patient import ISLESPatient

sys.path.append("../../..")
from metrics import compute_segmentation_metrics
from models.att_unet import model as att_unet
from models.swin_unet import model as swin_unet
from models.multires_att_unet import model as multires_att_unet
from utils.preprocessing.numpy import binarize_array, resize_data
from utils.preprocessing.numpy import resize_mask
from utils.config import get_path_of_directory_with_id, load_yaml_config
from utils.config import (
    get_weights_path_from_experiment,
    get_model_config_from_experiment,
)
from utils.plotting import create_title_with_metrics, save_multimedia_in_patient_dir
from utils.config import get_dset_config_from_experiment

def multires_data_resizing(data, target_size=224):
    import cv2
    data_versions = []
    for size in [224, 112, 56, 28]:
        if size == 224:
            data_versions.append(data)
        else:
            slices = []
            for data_slice in data:
                slices.append(cv2.resize(data_slice, (size, size)))
            data_versions.append(np.stack(slices))
    return tuple(data_versions)

def evaluate_patients_of_isles2017(
    experiment_id: Union[str, int],
    dset_dir: str,
    dset_split: str = "test",
    submission: bool = False,
    normalization: str = "min_max",
    target_size: int = 224,
    modalities: List[str] = None,
    compute_metrics: bool = False,
    exclude: Optional[List[str]] = None,
    lesion_metrics: bool = False,
    save_plots: bool = True,
    save_animations: bool = True,
    save_multimedia_mode: Optional[str] = None,
    animation_format: str = "mp4",
):
    
    target_size = (224, 224)
    figures_params = []
    if not animation_format in ["mp4", "gif"]:
        raise ValueError("animation_format must be one of mp4 or gif")

    if not save_multimedia_mode in ["best", "all", None]:
        raise ValueError("save_multimedia_mode must be one of best, all or None")

    metric_names = ["sens", "spec", "ppv", "npv", "dsc", "avd", "hd"]
    if lesion_metrics:
        metric_names.extend(["l_tpf", "l_fpf", "l_ppv", "l_f1"])
    if exclude is not None:
        for m in exclude:
            metric_names.remove(m)
    metrics = {m: [] for m in metric_names}

    experiment_dir = get_path_of_directory_with_id(experiment_id)

    # Create the model.
    model_config = get_model_config_from_experiment(experiment_dir)
    if "multires_att_unet" in experiment_dir:
        model_name = "multires_att_unet"
        model = multires_att_unet.get_model(**model_config)
    elif "att_unet" in experiment_dir:
        model_name = "att_unet"
        model = att_unet.get_model(**model_config)
    print(f"Constructed the model: {model_name} with specs:")
    print(model_config)

    # Load the weights
    best_weights_path = get_weights_path_from_experiment(experiment_dir)
    print(best_weights_path)
    model.load_weights(best_weights_path)
    print("Successfully populated model with the weights at:")
    print(f"{best_weights_path}")

    # Load the data.
    dset_config = get_dset_config_from_experiment(experiment_dir)
    patients_path = os.path.join(dset_dir, dset_config[f"{dset_split}_patients_path"])
    patients_paths = np.loadtxt(patients_path, dtype=str)
    assert all(os.path.isdir(str(p)) for p in patients_paths)

    # Create dir to store the evaluation results.
    weights_id = best_weights_path.split("_")[-1]
    eval_dir = os.path.join(
        experiment_dir, "evaluation", f"epoch_{weights_id}", dset_split
    )
    if submission:
        submission_dir = os.path.join(eval_dir, "submission")
        os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Iterate over all the patients.
    patient_ids = []
    patients_with_ot = []
    for patient_path in tqdm(
        patients_paths, desc=f'Evaluating {dset_split} for {dset_config["dset_name"]}'
    ):
        patient = ISLESPatient(str(patient_path))
        patient.load_niftis()

        patient_ids.append(patient.patient_id)
        patient_dir = os.path.join(eval_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Retrieve the data for the DL model to predict.
        norm_data = patient.get_data(
            normalization=dset_config["normalization"],
            modalities=dset_config["modalities"],
        )
        resized_data = {k: resize_data(v, target_size) for k, v in norm_data.items()}
        resized_data = np.asarray(list(resized_data.values()))
        resized_data = resized_data.transpose(3, 1, 2, 0)
        if "multiresolution" in experiment_dir:
            resized_data = multires_data_resizing(resized_data, target_size=target_size)

        # Predict using the resized_data, binarize the probs to obtain
        # the mask and resize it to match the OT shape to compute the metrics.
        probabilities = model.predict(resized_data)
        mask = binarize_array(probabilities, threshold=0.5)
        resized_mask = resize_mask(
            mask[..., 0].transpose(1, 2, 0), (patient.original_shape[0], patient.original_shape[0])
        )

        # Save the predictions and both the true and predicted masks.
        niftis_dir = os.path.join(patient_dir, "niftis")
        os.makedirs(niftis_dir, exist_ok=True)
        squeezed_probs = probabilities[..., 0].transpose(1, 2, 0)
        patient.save_mask_in_smir_format(
            resized_mask, save_dir=niftis_dir, description="mask"
        )
        patient.save_probs_in_smir_format(
            squeezed_probs, save_dir=niftis_dir, description="probabilities"
        )

        if submission:
            if isinstance(experiment_id, int):
                idx = f"{experiment_id:03d}"
            else:
                idx = experiment_id
            patient.save_mask_in_smir_format(
                resized_mask,
                save_dir=submission_dir,
                description=f"{idx}_{patient.patient_id}",
            )

        # Copy the original OT and compute metrics if patient
        # has mask.
        if hasattr(patient, "ot"):
            patients_with_ot.append(patient.patient_id)
            shutil.copy(patient.ot_path, niftis_dir)

            # Compute metrics.
            ot = patient.get_mask()
            resized_ot = resize_mask(ot, target_size=target_size)
            resized_ot = np.expand_dims(resized_ot, axis=0)
            resized_ot = resized_ot.transpose(3, 1, 2, 0)

            vol_metrics = compute_segmentation_metrics(
                ot, resized_mask, lesion_metrics=lesion_metrics, exclude=exclude
            )
            for m in metric_names:
                metrics[m].append(vol_metrics[m])

            # Create the titles and append the arrays for creating plots later.
            title = create_title_with_metrics(**vol_metrics)
            key_list = list(norm_data.keys())
            first_key = "ADC" if "ADC" in key_list else key_list[0]
            figures_params.append(
                (patient_dir, norm_data[first_key], ot, resized_mask, title)
            )

    if len(patients_with_ot) != 0:

        metrics_per_patient = pd.DataFrame(metrics)
        metrics_per_patient = metrics_per_patient.set_index(pd.Index(patient_ids))
        metrics_per_patient.loc["mean"] = metrics_per_patient.mean()
        metrics_per_patient.loc["std"] = metrics_per_patient.std()
        metrics_per_patient.to_csv(os.path.join(eval_dir, "patient_metrics.csv"))

        # Check if current dice is the best and save figures.
        current_dice = np.mean(metrics["dsc"]).astype(np.float32)

        for params in figures_params:
            patient_dir, norm_data, ot, resized_mask, title = params
            save_multimedia_in_patient_dir(
                patient_dir,
                norm_data,
                ot,
                resized_mask,
                title,
                save_plots,
                save_animations,
                animation_format=animation_format,
            )
