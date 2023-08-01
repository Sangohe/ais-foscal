import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

import os
import shutil
from typing import Optional

from utils import plotting
from metrics import compute_segmentation_metrics
from utils.preprocessing.numpy import binarize_array
from utils.preprocessing.tensorflow import resize_data, resize_mask

# TODO: SHOULDN'T IMPORT AN SPECIFIC DATASET MODULE.
from utils.datasets.isles2017.patient import ISLES2017Patient
from utils.datasets.isles2018.patient import ISLES2018Patient
from utils.datasets.foscal.patient import FOSCALPatient


def segmentation_eval(
    model: tf.keras.Model,
    config: ml_collections.ConfigDict,
    subdir: Optional[str] = None,
    radiologist: str = "Andres",
    submission_desc: str = "",
):

    if subdir is None:
        save_dir = config.evaluation_dir
    else:
        save_dir = os.path.join(config.evaluation_dir, subdir)
    valid_dir = os.path.join(save_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)

    input_shape = tuple(int(x) for x in config.model.input_shape.split(","))
    modalities = config.dataloader.modalities.split(",")

    metric_names = ["sens", "spec", "ppv", "npv", "dsc", "avd", "hd"]
    results = {m: [] for m in metric_names}
    results["patient"] = []
    patient_paths = np.loadtxt(config.dataloader.valid_patients_path, dtype=str)
    for patient_path in patient_paths:
        patient = FOSCALPatient(str(patient_path))
        original_shape = (patient.original_shape[0], patient.original_shape[1])
        patient_dir = os.path.join(valid_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        data_dict = patient.get_data(modalities, normalization="min_max")
        data = {
            k: np.expand_dims(v.transpose(2, 0, 1), -1) for k, v in data_dict.items()
        }
        data = np.concatenate(list(data.values()), axis=-1)
        resized_data = resize_data(data, input_shape[:2])

        # Predict the lesion. Keep the last output if the model is deeply supervised.
        probabilities = model.predict(resized_data)
        if isinstance(probabilities, (list, tuple)):
            probabilities = probabilities[-1]
        pred_mask = binarize_array(probabilities, threshold=0.5)
        pred_mask = pred_mask[..., 0].transpose(1, 2, 0)
        resized_pred_mask = resize_mask(pred_mask, original_shape).numpy()

        # Save each prediction and mask.
        niftis_dir = os.path.join(patient_dir, "niftis")
        os.makedirs(niftis_dir, exist_ok=True)

        mask_attr = modalities[0].lower()
        if hasattr(patient, mask_attr):
            shutil.copy(getattr(patient, mask_attr + "_path"), niftis_dir)

            # Compute metrics.
            mask = patient.get_mask(modalities=modalities, radiologist=radiologist)[
                modalities[0]
            ]
            vol_metrics = compute_segmentation_metrics(mask, resized_pred_mask)
            results["patient"].append(patient.patient_id)
            for m in metric_names:
                results[m].append(vol_metrics[m])

            # Create the titles and append the arrays for creating plots later.
            title = plotting.create_title_with_metrics(**vol_metrics)
            key_list = list(data_dict.keys())
            first_key = key_list[0]

            overlapped_ots_path = os.path.join(
                patient_dir, "data_and_ots_overlapped.png"
            )
            animation_save_path = os.path.join(patient_dir, f"animation.gif")

            plotting.plot_data_with_overlapping_ots(
                data_dict[first_key],
                mask,
                resized_pred_mask,
                save_path=overlapped_ots_path,
                show_plot=False,
                title=title,
            )

            plotting.save_animated_data_with_overlapping_ots(
                data_dict[first_key],
                mask,
                resized_pred_mask,
                save_path=animation_save_path,
                titles=title,
            )

    if len(results["patient"]) != 0:
        metrics_per_patient = pd.DataFrame(results)
        metrics_per_patient = metrics_per_patient.set_index("patient")
        metrics_per_patient.loc["mean"] = metrics_per_patient.mean()
        metrics_per_patient.loc["std"] = metrics_per_patient.std()
        metrics_per_patient.to_csv(os.path.join(save_dir, "patient_metrics.csv"))


def dual_segmentation_eval(
    model: tf.keras.Model,
    config: ml_collections.ConfigDict,
    subdir: Optional[str] = None,
    radiologist: str = "Andres",
    submission_desc: str = "",
):

    if subdir is None:
        save_dir = config.evaluation_dir
    else:
        save_dir = os.path.join(config.evaluation_dir, subdir)
    valid_dir = os.path.join(save_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)

    input_shape = tuple(int(x) for x in config.model.input_shape.split(","))
    modalities = config.dataloader.modalities.split(",")

    metric_names = ["sens", "spec", "ppv", "npv", "dsc", "avd", "hd"]
    adc_results = {m: [] for m in metric_names}
    adc_results["patient"] = []
    dwi_results = {m: [] for m in metric_names}
    dwi_results["patient"] = []

    patient_paths = np.loadtxt(config.dataloader.valid_patients_path, dtype=str)
    for patient_path in patient_paths:
        patient = FOSCALPatient(str(patient_path))
        original_shape = (patient.original_shape[0], patient.original_shape[1])
        patient_dir = os.path.join(valid_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        data_dict = patient.get_data(modalities, normalization="min_max")
        data = {
            k: np.expand_dims(v.transpose(2, 0, 1), -1) for k, v in data_dict.items()
        }
        data = np.concatenate(list(data.values()), axis=-1)
        resized_data = resize_data(data, input_shape[:2])
        resized_data = (
            resized_data[..., 0:1],
            resized_data[..., 1:],
        )

        # Predict the lesion. Keep the last output if the model is deeply supervised.
        adc_probabilities, dwi_probabilities = model.predict(resized_data)

        adc_pred_mask = binarize_array(adc_probabilities, threshold=0.5)
        adc_pred_mask = adc_pred_mask[..., 0].transpose(1, 2, 0)
        adc_resized_pred_mask = resize_mask(adc_pred_mask, original_shape).numpy()

        dwi_pred_mask = binarize_array(dwi_probabilities, threshold=0.5)
        dwi_pred_mask = dwi_pred_mask[..., 0].transpose(1, 2, 0)
        dwi_resized_pred_mask = resize_mask(dwi_pred_mask, original_shape).numpy()

        # Save each prediction and mask.
        niftis_dir = os.path.join(patient_dir, "niftis")
        os.makedirs(niftis_dir, exist_ok=True)
        shutil.copy(getattr(patient, "adc_andres_mask_path"), niftis_dir)
        shutil.copy(getattr(patient, "dwi_andres_mask_path"), niftis_dir)

        # Compute metrics.
        adc_mask = patient.get_mask(modalities=modalities, radiologist=radiologist)[
            "ADC"
        ]
        adc_vol_metrics = compute_segmentation_metrics(adc_mask, adc_resized_pred_mask)
        adc_results["patient"].append(patient.patient_id)
        for m in metric_names:
            adc_results[m].append(adc_vol_metrics[m])

        # Create the titles and append the arrays for creating plots later.
        title = plotting.create_title_with_metrics(**adc_vol_metrics)
        overlapped_ots_path = os.path.join(
            patient_dir, "adc_data_and_ots_overlapped.png"
        )
        animation_save_path = os.path.join(patient_dir, f"adc_animation.gif")
        plotting.plot_data_with_overlapping_ots(
            data_dict["ADC"],
            adc_mask,
            adc_resized_pred_mask,
            save_path=overlapped_ots_path,
            show_plot=False,
            title=title,
        )
        plotting.save_animated_data_with_overlapping_ots(
            data_dict["ADC"],
            adc_mask,
            adc_resized_pred_mask,
            save_path=animation_save_path,
            titles=title,
        )

        dwi_mask = patient.get_mask(modalities=modalities, radiologist=radiologist)[
            "DWI"
        ]
        dwi_vol_metrics = compute_segmentation_metrics(dwi_mask, dwi_resized_pred_mask)
        dwi_results["patient"].append(patient.patient_id)
        for m in metric_names:
            dwi_results[m].append(dwi_vol_metrics[m])

        # Create the titles and append the arrays for creating plots later.
        title = plotting.create_title_with_metrics(**dwi_vol_metrics)
        overlapped_ots_path = os.path.join(
            patient_dir, "dwi_data_and_ots_overlapped.png"
        )
        animation_save_path = os.path.join(patient_dir, f"dwi_animation.gif")
        plotting.plot_data_with_overlapping_ots(
            data_dict["DWI"],
            dwi_mask,
            dwi_resized_pred_mask,
            save_path=overlapped_ots_path,
            show_plot=False,
            title=title,
        )
        plotting.save_animated_data_with_overlapping_ots(
            data_dict["DWI"],
            dwi_mask,
            dwi_resized_pred_mask,
            save_path=animation_save_path,
            titles=title,
        )

    if len(adc_results["patient"]) != 0:
        metrics_per_patient = pd.DataFrame(adc_results)
        metrics_per_patient = metrics_per_patient.set_index("patient")
        metrics_per_patient.loc["mean"] = metrics_per_patient.mean()
        metrics_per_patient.loc["std"] = metrics_per_patient.std()
        metrics_per_patient.to_csv(os.path.join(save_dir, "adc_patient_metrics.csv"))

    if len(dwi_results["patient"]) != 0:
        metrics_per_patient = pd.DataFrame(dwi_results)
        metrics_per_patient = metrics_per_patient.set_index("patient")
        metrics_per_patient.loc["mean"] = metrics_per_patient.mean()
        metrics_per_patient.loc["std"] = metrics_per_patient.std()
        metrics_per_patient.to_csv(os.path.join(save_dir, "dwi_patient_metrics.csv"))


def decoder_denoising_eval(
    model: tf.keras.Model,
    config: ml_collections.ConfigDict,
    subdir: Optional[str] = None,
):

    input_shape = tuple(int(x) for x in config.model.input_shape.split(","))
    modalities = config.dataloader.modalities.split(",")
    patient_paths = np.loadtxt(config.dataloader.valid_patients_path, dtype=str)

    results = {"patient": [], "mse": [], "images": []}
    for patient_path in patient_paths:
        patient = FOSCALPatient(str(patient_path))
        data = patient.get_data(modalities, normalization="min_max")
        data = {k: np.expand_dims(v.transpose(2, 0, 1), -1) for k, v in data.items()}
        data = np.concatenate(list(data.values()), axis=-1)
        resized_data = resize_data(data, input_shape[:2])

        noise_factor = 0.2
        noisy_data = (
            resized_data + tf.random.normal(tf.shape(resized_data)) * noise_factor
        )
        noisy_data = tf.clip_by_value(noisy_data, 0.0, 1.0)
        reconstruction = model.predict(noisy_data)
        mse = ((resized_data.numpy() - reconstruction) ** 2).mean()

        results["patient"].append(patient.patient_id)
        results["mse"].append(mse)
        results["images"].append(
            plotting.stack_arrays_horizontally(
                [
                    noisy_data.numpy()[..., 0].astype(np.float32),
                    resized_data.numpy()[..., 0].astype(np.float32),
                    reconstruction[..., 0].astype(np.float32),
                ],
                resize_to=input_shape[:2],
            )
        )

    results_copy = results.copy()
    del results_copy["images"]
    per_patient_metrics_df = pd.DataFrame(results_copy)

    # Save the results.
    if subdir is None:
        save_dir = config.evaluation_dir
    else:
        save_dir = os.path.join(config.evaluation_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    per_patient_metrics_df.to_csv(
        os.path.join(save_dir, "per_patient_metrics.csv"), index=False
    )

    for patient_id, image in zip(results["patient"], results["images"]):
        plotting.save_array_as_animation(
            os.path.join(save_dir, f"{patient_id}.mp4"), image
        )


def encoder_classification_eval(
    model: tf.keras.Model,
    config: ml_collections.ConfigDict,
    subdir: Optional[str] = None,
):

    input_shape = tuple(int(x) for x in config.model.input_shape.split(","))
    modalities = config.dataloader.modalities.split(",")
    patient_paths = np.loadtxt(config.dataloader.valid_patients_path, dtype=str)

    results = {"patient": [], "label": [], "probability": []}
    for patient_path in patient_paths:
        patient = FOSCALPatient(str(patient_path))
        data = patient.get_data(modalities, normalization="min_max")
        data = {k: np.expand_dims(v.transpose(2, 0, 1), -1) for k, v in data.items()}
        data = np.concatenate(list(data.values()), axis=-1)
        mask = np.expand_dims(patient.get_mask().transpose(2, 0, 1), -1)

        resized_data = resize_data(data, input_shape[:2])
        label = tf.cast(
            tf.math.count_nonzero(mask, axis=[1, 2, 3]) > 0, dtype=tf.float32
        ).numpy()
        probabilities = model.predict(resized_data)

        results["patient"].extend([patient.patient_id] * data.shape[0])
        results["label"].extend(label)
        results["probability"].extend(np.squeeze(probabilities))

    results_df = results
    results_df = pd.DataFrame(results)
    results_df["prediction"] = (results_df.probability > 0.5).astype(int)

    # Compute metrics.
    metrics_df = pd.DataFrame(
        data={
            "accuracy": [
                metrics.accuracy_score(results_df.label, results_df.prediction)
            ],
            "precision": [
                metrics.precision_score(results_df.label, results_df.prediction)
            ],
            "recall": [metrics.recall_score(results_df.label, results_df.prediction)],
            "f1": [metrics.f1_score(results_df.label, results_df.prediction)],
            "auroc": [metrics.roc_auc_score(results_df.label, results_df.probability)],
        },
        columns=["accuracy", "precision", "recall", "f1", "auroc"],
    )

    patient_results_df = results_df.groupby("patient")
    per_patient_metrics_df = patient_results_df.apply(
        lambda x: pd.Series(
            {
                "accuracy": metrics.accuracy_score(x.label, x.prediction),
                "precision": metrics.precision_score(x.label, x.prediction),
                "recall": metrics.recall_score(x.label, x.prediction),
                "f1": metrics.f1_score(x.label, x.prediction),
            }
        )
    )
    per_patient_metrics_df.insert(0, "patient", patient_results_df.groups.keys())

    cm = metrics.confusion_matrix(results_df.label, results_df.prediction)
    cm_disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Without lesion", "With lesion"]
    )
    cm_disp.plot(cmap="Blues")

    # Save the results.
    if subdir is None:
        save_dir = config.evaluation_dir
    else:
        save_dir = os.path.join(config.evaluation_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, "confusion_matrix.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    results_df.to_csv(os.path.join(save_dir, "results.csv"), index=False)
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    per_patient_metrics_df.to_csv(
        os.path.join(save_dir, "per_patient_metrics.csv"), index=False
    )
