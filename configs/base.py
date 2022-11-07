import ml_collections


def get_segmentation_train_config():
    """Returns config values other than model parameters."""

    config = get_common_config()
    config.encoder_weights_path = ""
    config.unet_weights_path = ""
    config.restart_weights_path = ""

    # Shared.
    config.base_lr = 5e-4
    config.mask_with_contours = False
    config.deep_supervision = False
    config.multiresolution = False

    # Optimization.
    config.loss_name = "binary_focal_crossentropy"

    return config


def get_dual_segmentation_train_config():
    """Returns config values other than model parameters."""

    config = get_common_config()
    config.encoder_weights_path = ""
    config.unet_weights_path = ""
    config.restart_weights_path = ""

    # Shared.
    config.base_lr = 5e-4
    config.mask_with_contours = False
    config.deep_supervision = False
    config.multiresolution = False

    # Optimization.
    config.loss_name = "binary_focal_crossentropy"

    return config


def get_decoder_denoising_pretrain_config():
    """Returns config values other than model parameters."""

    config = get_common_config()
    config.encoder_weights_path = ""

    # Optimization.
    config.base_lr = 3e-5
    config.loss_name = "binary_crossentropy"

    return config


def get_encoder_classification_pretrain_config():
    """Returns config values other than model parameters."""

    config = get_common_config()

    # Optimization.
    config.base_lr = 3e-5
    config.loss_name = "binary_crossentropy"

    return config


def get_common_config():
    """Returns config values other than model parameters."""

    config = ml_collections.ConfigDict()
    config.root_dir = "/home/sangohe/projects/isbi2023-foscal/results/"

    config.use_ema = True
    config.slice_size = 224
    config.mixed_precision = False

    # Optimization.
    config.epochs = 600
    config.optimizer_name = "adamw"
    config.base_lr = 5e-4
    config.weight_decay = 1e-5
    config.grad_norm_clip = 1.0
    config.warmup_epoch_percentage = 0.4

    # Will be set from ./models.py and ./dataloaders.py
    config.model = None
    config.dataloader = None

    return config
