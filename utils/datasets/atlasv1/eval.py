import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional

from .patient import ATLASV1Patient

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
from utils.config import get_dset_config_from_experiment
from utils.plotting import create_title_with_metrics, save_multimedia_in_patient_dir

def multires_data_resizing(data, target_size=224):
    import cv2
    resized_list = [
        data,
        cv2.resize(data, (112, 112)),
        cv2.resize(data, (56, 56)),
        cv2.resize(data, (28, 28)),
    ]
    return tuple(resized_list)

def evaluate_studies_of_atlasv1(
    experiment_id: Union[str, int],
    dset_dir: str,
    dset_split: str = "val",
    normalization: str = "min_max",
    target_size: int = 224,
    compute_metrics: bool = False,
    exclude: Optional[List[str]] = None,
    lesion_metrics: bool = False,
    save_multimedia: bool = True,
    animation_format: str = "mp4",
):

    if not animation_format in ["mp4", "gif"]:
        raise ValueError("animation_format must be one of mp4 or gif")

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
    data_path = os.path.join(dset_dir, dset_config[f"{dset_split}_data"])
    masks_path = os.path.join(dset_dir, dset_config[f"{dset_split}_masks"])

    data_paths = np.loadtxt(data_path, dtype=str)
    mask_paths = np.loadtxt(masks_path, dtype=str)
    mask_paths = np.asarray([p.split(",") for p in mask_paths], dtype=object)

    # Create dir to store the evaluation results.
    weights_id = best_weights_path.split("_")[-1]
    eval_dir = os.path.join(
        experiment_dir, "evaluation", f"epoch_{weights_id}", dset_split
    )

    # Iterate over all the patients.
    patient_ids = []
    patients_with_ot = []
    figures_params = []
    for d_path, m_path in tqdm(
        zip(data_paths, mask_paths),
        total=len(data_paths),
        desc=f'Evaluating {dset_split} for {dset_config["dset_name"]}',
    ):
        patient = ATLASV1Patient(d_path, m_path)
        patient.load_niftis()

        patient_ids.append(patient.patient_id)
        patient_dir = os.path.join(eval_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Retrieve the data for the DL model to predict.
        norm_data = patient.get_data(normalization=normalization)
        ot = patient.get_mask()

        # Resize the volume slices to the input size ot the model.
        resized_data = resize_data(norm_data, target_size)
        resized_data = np.expand_dims(resized_data, axis=0)
        resized_ot = resize_mask(ot, target_size)
        resized_ot = np.expand_dims(resized_ot, axis=0)

        # Transpose axes to use them for
        resized_data = resized_data.transpose(3, 1, 2, 0)
        resized_ot = resized_ot.transpose(3, 1, 2, 0)

        # Predict using the resized_data, binarize the probs to obtain
        # the mask and resize it to match the OT shape to compute the metrics.
        probabilities = model.predict(resized_data)
        mask = binarize_array(probabilities, threshold=0.5)
        resized_mask = resize_mask(mask[..., 0].transpose(1, 2, 0), norm_data.shape[0])

        # Compute the metrics.
        vol_metrics = compute_segmentation_metrics(
            resized_ot, mask, lesion_metrics=lesion_metrics, exclude=exclude
        )
        for m in metric_names:
            metrics[m].append(vol_metrics[m])
        print(vol_metrics)
        # Create the titles and append the arrays for creating plots later.
        title = create_title_with_metrics(**vol_metrics)
        figures_params.append((patient_dir, resized_data, resized_ot, mask, title))

    # Summarize metrics.
    metrics_per_patient = pd.DataFrame(metrics)
    metrics_per_patient = metrics_per_patient.set_index(pd.Index(patient_ids))
    metrics_per_patient.loc["mean"] = metrics_per_patient.mean()
    metrics_per_patient.loc["std"] = metrics_per_patient.std()
    metrics_per_patient.to_csv(os.path.join(eval_dir, "patient_metrics.csv"))

    # Check if current dice is the best and save figures.
    current_dice = np.mean(metrics_per_patient["dsc"]).astype(np.float32)
    print(metrics)
    if save_multimedia:
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
