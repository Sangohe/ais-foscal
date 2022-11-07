import ml_collections

import os

from configs import base
from configs import models
from configs import dataloaders


def get_config(model_dset_dir_task: str) -> ml_collections.ConfigDict:

    model, dset_dir, task = model_dset_dir_task.split(",")
    get_task_config = getattr(base, f"get_{task}_config")
    config = get_task_config()
    config.task = task

    # Load the dataloader config.
    config.dataloader = dataloaders.get_dataloader_config(os.path.abspath(dset_dir))

    # Load the model config.
    #! define the number of channels
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    config.model.name = model

    # Define placeholders to keep attributes in sync.
    config.dataloader.slice_size = config.get_ref("slice_size")
    if task == "segmentation":
        config.dataloader.multiresolution = config.get_ref("multiresolution")
        config.model.multiresolution = config.get_ref("multiresolution")
        config.dataloader.mask_with_contours = config.get_ref("mask_with_contours")
        config.dataloader.deep_supervision = config.get_ref("deep_supervision")
        config.model.deep_supervision = config.get_ref("deep_supervision")

    return config
