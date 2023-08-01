import ml_collections
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow_addons.optimizers import AdamW

import os
import importlib
from typing import Callable, Optional

from utils.logging import Logger
from utils.callbacks import EMACallback
from utils.schedulers import WarmUpCosine
from metrics import MulticlassDiceScore, BinaryDiceScore
from dataloader import TFSlicesDataloader, TFSlicesValidationDataloader
from eval import (
    encoder_classification_eval,
    decoder_denoising_eval,
    segmentation_eval,
    dual_segmentation_eval,
)
from utils.config import (
    create_run_dir,
    create_experiment_desc,
    save_dict_as_yaml,
    save_md_with_experiment_config,
)


def create_directory(path: str) -> str:
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def segmentation_train(
    config: ml_collections.ConfigDict,
) -> tf.keras.Model:

    tf.keras.utils.set_random_seed(42)

    experiment_desc = create_experiment_desc(config)
    run_dir = create_run_dir(experiment_desc, root_dir=config.root_dir)
    logs_fname = os.path.join(run_dir, "execution_logs.txt")
    _ = Logger(file_name=logs_fname, file_mode="a", should_flush=True)

    # Experiment directories.
    config.logs_dir = create_directory(os.path.join(run_dir, "logs"))
    config.config_dir = create_directory(os.path.join(run_dir, "config"))
    config.weights_dir = create_directory(os.path.join(run_dir, "weights"))
    config.evaluation_dir = create_directory(os.path.join(run_dir, "evaluation"))

    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Finish the configuration, make sure to define model's input shape as a string.
    num_modalities = len(config.dataloader.modalities.split(","))
    config.model.input_shape = (
        f"{config.slice_size},{config.slice_size},{num_modalities}"
    )
    config.best_loss_weights_path = os.path.join(
        config.weights_dir, "best_loss", "weights"
    )
    config.best_weights_path = os.path.join(config.weights_dir, "best", "weights")
    config.last_epoch_weights_path = os.path.join(
        config.weights_dir, "last_epoch", "weights"
    )
    if config.use_ema:
        config.ema_weights_path = os.path.join(config.weights_dir, "ema", "weights")
    config.dataloader.num_lods = len(config.model.filters_per_level.split(",")) - 1
    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Experiment setup.
    if config.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        print("Training with mixed precision...")

    # Process the arguments from the YAML config.
    modalities = config.dataloader.modalities.replace(" ", "").split(",")
    input_shape = tuple(int(s) for s in config.model.input_shape.split(","))
    filters_per_level = [int(f) for f in config.model.filters_per_level.split(",")]
    blocks_depth = [int(f) for f in config.model.blocks_depth.split(",")]
    class_weights = [float(c) for c in config.dataloader.class_weights.split(",")]

    # Optimization.
    total_steps = (
        int(config.dataloader.num_train_samples / config.dataloader.batch_size)
        * config.epochs
    )
    warmup_steps = int(total_steps * config.warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=config.base_lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    # Model and optimizer creation.
    model_module = importlib.import_module(f"models.{config.model.model_module}")
    model_config = config.model.to_dict()
    model_config["input_shape"] = input_shape
    model_config["filters_per_level"] = filters_per_level
    model_config["blocks_depth"] = blocks_depth
    model_config["norm_layer"] = import_method_from_config(config.model, "norm_layer")
    model_config["upsample_layer"] = import_method_from_config(
        config.model, "upsample_layer"
    )
    model_config["attention_layer"] = import_method_from_config(
        config.model, "attention_layer"
    )
    model_config["pooling_layer"] = import_method_from_config(
        config.model, "pooling_layer"
    )

    encoder = model_module.UNetEncoder(**model_config)
    skip_names = model_module.get_skip_names_from_encoder(encoder)

    # Load the encoder weights only if no unet weights are provided.
    if not config.encoder_weights_path == "" and config.unet_weights_path == "":
        print(f"Loading encoder weights from: {config.encoder_weights_path}...")
        encoder = model_module.add_head_to_encoder(
            encoder, num_classes=1, out_activation="sigmoid"
        )
        encoder.load_weights(config.encoder_weights_path).expect_partial()
        encoder = model_module.remove_head_from_encoder(encoder)

    # Load the unet weights if provided.
    if not config.unet_weights_path == "":
        print(f"Loading U-Net weights from: {config.unet_weights_path}...")

        # Unets with trained weights have the same shape at input and output.
        model_config["num_classes"] = num_modalities
        model = model_module.UNet(encoder, skip_names, **model_config)
        model.load_weights(config.unet_weights_path).expect_partial()
        model = model_module.remove_prediction_layer_from_unet(model)
        model = model_module.add_prediction_layer_to_unet(
            model, config.model.num_classes, config.model.out_activation
        )
    else:
        model = model_module.UNet(encoder, skip_names, **model_config)

    # if config.deep_supervision:
    #     hidden_layer_names = model_module.get_output_names_for_deep_supervision(model)
    #     model = model_module.add_deep_supervision_to_unet(model, hidden_layer_names)
    model.trainable = True

    opt = AdamW(
        learning_rate=scheduled_lrs,
        weight_decay=config.weight_decay,
        clipnorm=config.grad_norm_clip,
    )

    if config.dataloader.deep_supervision:
        loss_weights = [0.045, 0.08, 0.125, 0.25, 0.5]
    else:
        loss_weights = None
    compile_args = dict(
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        loss_weights=loss_weights,
        metrics=[BinaryDiceScore(threshold=0.5, name="binary_dice")],
    )
    model.compile(optimizer=opt, **compile_args)

    if not config.restart_weights_path == "":
        print(f"Restarting weights from checkpoint {config.restart_weights_path}...")
        model.load_weights(config.restart_weights_path)

    # Data loading.
    train_dataloader_config = config.dataloader.to_dict()
    train_dataloader_config["modalities"] = modalities
    train_dataloader_config["tfrecord_path"] = config.dataloader.train_tfrecord_path
    train_dataloader_config["deep_supervision"] = config.deep_supervision
    train_dataloader_config["class_weights"] = class_weights
    train_dataloader = TFSlicesDataloader(**train_dataloader_config)
    train_dset = train_dataloader.get_dataset()

    if not config.dataloader.use_full_train_dset:
        print("Using a train-validation split")
        valid_dataloader_config = config.dataloader.to_dict()
        valid_dataloader_config["modalities"] = modalities
        valid_dataloader_config["tfrecord_path"] = config.dataloader.valid_tfrecord_path
        valid_dataloader = TFSlicesValidationDataloader(**valid_dataloader_config)
        valid_dset = valid_dataloader.get_dataset()
    else:
        print("Using the full training dataset")

    # Training.
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=config.logs_dir),
    ]

    if not config.dataloader.use_full_train_dset:
        if config.dataloader.num_lods is None:
            metric_to_monitor = "val_binary_dice"
        else:
            # metric_to_monitor = f"val_{config.model.name}_last_probs_binary_dice"
            metric_to_monitor = "val_binary_dice"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_weights_path,
                monitor=metric_to_monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_loss_weights_path,
                monitor=f"val_{config.model.name}_last_probs_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
            )
        )

        if config.use_ema:
            ema_model = create_model_copy(model)
            ema_model.compile(**compile_args)
            callbacks.append(
                EMACallback(
                    ema_model,
                    validation_dset=valid_dset,
                    log_dir=config.logs_dir,
                    monitor=metric_to_monitor[
                        4:
                    ],  # Not necessary to prefix it with _val.
                    mode="max",
                    filepath=config.ema_weights_path,
                    save_weights_only=True,
                )
            )

    if config.dataloader.use_full_train_dset:
        history = model.fit(
            train_dset, epochs=config.epochs, callbacks=callbacks, verbose=2
        )
    else:
        history = model.fit(
            train_dset,
            validation_data=valid_dset,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        # Append the epoch number that got the best accuracy.
        with open(md_config_path, "r+") as f:
            lines = f.readlines()
            best_dice = max(history.history[metric_to_monitor])
            best_dice_idx = history.history[metric_to_monitor].index(best_dice) + 1
            lines.insert(-2, f"- Best epoch: {best_dice_idx}\r")
            f.writelines(lines)

    model.save_weights(config.last_epoch_weights_path)

    # Evaluation of the last model and the model with the best dice score.
    print("Evaluating the last model...")
    exp_id = os.path.basename(run_dir).split("-")[0]
    segmentation_eval(model, config, subdir="last_epoch", submission_desc=exp_id)

    if not config.dataloader.use_full_train_dset:
        print("Evaluating the model with the best dice score...")
        best_dice_model = tf.keras.models.clone_model(model)
        best_dice_model.load_weights(config.best_weights_path).expect_partial()
        segmentation_eval(
            best_dice_model, config, subdir="best", submission_desc=exp_id
        )

        print("Evaluating the model with the best loss...")
        best_loss_model = tf.keras.models.clone_model(model)
        best_loss_model.load_weights(config.best_loss_weights_path).expect_partial()
        segmentation_eval(
            best_loss_model, config, subdir="best_loss", submission_desc=exp_id
        )

        if config.use_ema:
            print("Evaluating the EMA model with the best dice score...")
            best_dice_ema_model = tf.keras.models.clone_model(model)
            best_dice_ema_model.load_weights(config.ema_weights_path).expect_partial()
            segmentation_eval(
                best_dice_ema_model, config, subdir="ema", submission_desc=exp_id
            )


def dual_segmentation_train(
    config: ml_collections.ConfigDict,
) -> tf.keras.Model:

    tf.keras.utils.set_random_seed(42)

    experiment_desc = create_experiment_desc(config)
    run_dir = create_run_dir(experiment_desc, root_dir=config.root_dir)
    logs_fname = os.path.join(run_dir, "execution_logs.txt")
    _ = Logger(file_name=logs_fname, file_mode="a", should_flush=True)

    # Experiment directories.
    config.logs_dir = create_directory(os.path.join(run_dir, "logs"))
    config.config_dir = create_directory(os.path.join(run_dir, "config"))
    config.weights_dir = create_directory(os.path.join(run_dir, "weights"))
    config.evaluation_dir = create_directory(os.path.join(run_dir, "evaluation"))

    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Finish the configuration, make sure to define model's input shape as a string.
    num_modalities = len(config.dataloader.modalities.split(","))
    config.model.input_shape = f"{config.slice_size},{config.slice_size},{1}"
    config.best_loss_weights_path = os.path.join(
        config.weights_dir, "best_loss", "weights"
    )
    config.best_weights_path = os.path.join(config.weights_dir, "best", "weights")
    config.last_epoch_weights_path = os.path.join(
        config.weights_dir, "last_epoch", "weights"
    )
    if config.use_ema:
        config.ema_weights_path = os.path.join(config.weights_dir, "ema", "weights")
    config.dataloader.num_lods = len(config.model.filters_per_level.split(",")) - 1
    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Experiment setup.
    if config.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        print("Training with mixed precision...")

    # Process the arguments from the YAML config.
    modalities = config.dataloader.modalities.replace(" ", "").split(",")
    input_shape = tuple(int(s) for s in config.model.input_shape.split(","))
    filters_per_level = [int(f) for f in config.model.filters_per_level.split(",")]
    blocks_depth = [int(f) for f in config.model.blocks_depth.split(",")]
    class_weights = [float(c) for c in config.dataloader.class_weights.split(",")]

    # Optimization.
    total_steps = (
        int(config.dataloader.num_train_samples / config.dataloader.batch_size)
        * config.epochs
    )
    warmup_steps = int(total_steps * config.warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=config.base_lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    # Model and optimizer creation.
    model_module = importlib.import_module(f"models.{config.model.model_module}")
    model_config = config.model.to_dict()
    model_config["input_shape"] = input_shape
    model_config["filters_per_level"] = filters_per_level
    model_config["blocks_depth"] = blocks_depth
    model_config["norm_layer"] = import_method_from_config(config.model, "norm_layer")
    model_config["upsample_layer"] = import_method_from_config(
        config.model, "upsample_layer"
    )
    if config.model.attention_layer is not None:
        model_config["attention_layer"] = import_method_from_config(
            config.model, "attention_layer"
        )
    else:
        model_config["attention_layer"] = None
    model_config["pooling_layer"] = import_method_from_config(
        config.model, "pooling_layer"
    )

    model = model_module.DualUnet(**model_config)
    opt = AdamW(
        learning_rate=scheduled_lrs,
        weight_decay=config.weight_decay,
        clipnorm=config.grad_norm_clip,
    )

    if config.dataloader.deep_supervision:
        loss_weights = [0.045, 0.08, 0.125, 0.25, 0.5]
    else:
        loss_weights = None
    compile_args = dict(
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        loss_weights=loss_weights,
        metrics=[BinaryDiceScore(threshold=0.5, name="binary_dice")],
    )
    model.compile(optimizer=opt, **compile_args)

    if not config.restart_weights_path == "":
        print(f"Restarting weights from checkpoint {config.restart_weights_path}...")
        model.load_weights(config.restart_weights_path)

    # Data loading.
    train_dataloader_config = config.dataloader.to_dict()
    train_dataloader_config["modalities"] = modalities
    train_dataloader_config["tfrecord_path"] = config.dataloader.train_tfrecord_path
    train_dataloader_config["deep_supervision"] = config.deep_supervision
    train_dataloader_config["class_weights"] = class_weights
    train_dataloader = TFSlicesDataloader(**train_dataloader_config)
    train_dset = train_dataloader.get_dataset()

    for x, y, z in train_dset.take(1):
        if isinstance(x, tuple):
            for i, x_i in enumerate(x):
                print(f"Input {i} shape: {x_i.shape}")
        if isinstance(y, tuple):
            for i, y_i in enumerate(y):
                print(f"Input {i} shape: {y_i.shape}")

    if not config.dataloader.use_full_train_dset:
        print("Using a train-validation split")
        valid_dataloader_config = config.dataloader.to_dict()
        valid_dataloader_config["modalities"] = modalities
        valid_dataloader_config["tfrecord_path"] = config.dataloader.valid_tfrecord_path
        valid_dataloader = TFSlicesValidationDataloader(**valid_dataloader_config)
        valid_dset = valid_dataloader.get_dataset()
    else:
        print("Using the full training dataset")

    # Training.
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=config.logs_dir),
    ]

    if not config.dataloader.use_full_train_dset:
        if config.dataloader.num_lods is None:
            metric_to_monitor = "val_binary_dice"
        else:
            metric_to_monitor = f"val_adc_unet_last_probs_binary_dice"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_weights_path,
                monitor=config.metric_to_monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_loss_weights_path,
                monitor=f"val_adc_unet_last_probs_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
            )
        )

        if config.use_ema:
            ema_model = create_model_copy(model)
            ema_model.compile(**compile_args)
            callbacks.append(
                EMACallback(
                    ema_model,
                    validation_dset=valid_dset,
                    log_dir=config.logs_dir,
                    monitor=metric_to_monitor[
                        4:
                    ],  # Not necessary to prefix it with _val.
                    mode="max",
                    filepath=config.ema_weights_path,
                    save_weights_only=True,
                )
            )

    if config.dataloader.use_full_train_dset:
        history = model.fit(
            train_dset, epochs=config.epochs, callbacks=callbacks, verbose=2
        )
    else:
        history = model.fit(
            train_dset,
            validation_data=valid_dset,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        # Append the epoch number that got the best accuracy.
        with open(md_config_path, "r+") as f:
            lines = f.readlines()
            best_dice = max(history.history[metric_to_monitor])
            best_dice_idx = history.history[metric_to_monitor].index(best_dice) + 1
            lines.insert(-2, f"- Best epoch: {best_dice_idx}\r")
            f.writelines(lines)

    model.save_weights(config.last_epoch_weights_path)

    # Evaluation of the last model and the model with the best dice score.
    print("Evaluating the last model...")
    exp_id = os.path.basename(run_dir).split("-")[0]
    dual_segmentation_eval(model, config, subdir="last_epoch", submission_desc=exp_id)

    if not config.dataloader.use_full_train_dset:
        print("Evaluating the model with the best dice score...")
        best_dice_model = tf.keras.models.clone_model(model)
        best_dice_model.load_weights(config.best_weights_path).expect_partial()
        dual_segmentation_eval(
            best_dice_model, config, subdir="best", submission_desc=exp_id
        )

        print("Evaluating the model with the best loss...")
        best_loss_model = tf.keras.models.clone_model(model)
        best_loss_model.load_weights(config.best_loss_weights_path).expect_partial()
        dual_segmentation_eval(
            best_loss_model, config, subdir="best_loss", submission_desc=exp_id
        )

        if config.use_ema:
            print("Evaluating the EMA model with the best dice score...")
            best_dice_ema_model = tf.keras.models.clone_model(model)
            best_dice_ema_model.load_weights(config.ema_weights_path).expect_partial()
            dual_segmentation_eval(
                best_dice_ema_model, config, subdir="ema", submission_desc=exp_id
            )


def decoder_denoising_pretrain(
    config: ml_collections.ConfigDict,
) -> tf.keras.Model:

    tf.keras.utils.set_random_seed(42)

    experiment_desc = create_experiment_desc(config)
    run_dir = create_run_dir(experiment_desc, root_dir=config.root_dir)
    logs_fname = os.path.join(run_dir, "execution_logs.txt")
    _ = Logger(file_name=logs_fname, file_mode="a", should_flush=True)

    # Experiment directories.
    # Experiment directories.
    config.logs_dir = create_directory(os.path.join(run_dir, "logs"))
    config.config_dir = create_directory(os.path.join(run_dir, "config"))
    config.weights_dir = create_directory(os.path.join(run_dir, "weights"))
    config.evaluation_dir = create_directory(os.path.join(run_dir, "evaluation"))

    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Finish the configuration, make sure to define model's input shape as a string.
    num_modalities = len(config.dataloader.modalities.split(","))
    config.model.input_shape = (
        f"{config.slice_size},{config.slice_size},{num_modalities}"
    )
    config.best_weights_path = os.path.join(config.weights_dir, "best", "weights")
    config.last_epoch_weights_path = os.path.join(
        config.weights_dir, "last_epoch", "weights"
    )
    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Experiment setup.
    if config.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        print("Training with mixed precision...")

    modalities = config.dataloader.modalities.replace(" ", "").split(",")
    input_shape = tuple(int(s) for s in config.model.input_shape.split(","))
    filters_per_level = [int(f) for f in config.model.filters_per_level.split(",")]
    blocks_depth = [int(f) for f in config.model.blocks_depth.split(",")]

    # Optimization.
    total_steps = (
        int(config.dataloader.num_train_samples / config.dataloader.batch_size)
        * config.epochs
    )
    warmup_steps = int(total_steps * config.warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=config.base_lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )
    opt = AdamW(learning_rate=scheduled_lrs, weight_decay=config.weight_decay)

    # Model and optimizer creation.
    model_module = importlib.import_module(f"models.{config.model.model_module}")
    model_config = config.model.to_dict()
    model_config["input_shape"] = input_shape
    model_config["num_classes"] = num_modalities
    model_config["filters_per_level"] = filters_per_level
    model_config["blocks_depth"] = blocks_depth
    model_config["norm_layer"] = import_method(config.model.norm_layer)
    model_config["upsample_layer"] = import_method(config.model.upsample_layer)
    model_config["attention_layer"] = import_method(config.model.attention_layer)
    model_config["pooling_layer"] = import_method(config.model.pooling_layer)

    encoder = model_module.UNetEncoder(**model_config)
    if not config.encoder_weights_path == "":
        print("Loading encoder weights...")
        encoder = model_module.add_head_to_encoder(
            encoder, num_classes=1, out_activation="sigmoid"
        )
        encoder.load_weights(
            config.encoder_weights_path
        ).expect_partial()  # to avoid warnings.
        encoder = model_module.remove_head_from_encoder(encoder)
        print("Freezing the encoder weights...")
        encoder.trainable = False

    skip_names = model_module.get_skip_names_from_encoder(encoder)
    model = model_module.UNet(encoder, skip_names, **model_config)
    model.compile(optimizer=opt, loss=config.loss_name, metrics=["mae"])

    # Data loading.
    train_dataloader_config = config.dataloader.to_dict()
    train_dataloader_config["modalities"] = modalities
    train_dataloader_config["tfrecord_path"] = config.dataloader.train_tfrecord_path
    train_dataloader = TFSlicesDataloader(**train_dataloader_config)
    train_dset = train_dataloader.get_dataset_denoising()

    if not config.dataloader.use_full_train_dset:
        print("Using a train-validation split")
        valid_dataloader_config = config.dataloader.to_dict()
        valid_dataloader_config["modalities"] = modalities
        valid_dataloader_config["multiresolution"] = False
        valid_dataloader_config["deep_supervision"] = False
        valid_dataloader_config["tfrecord_path"] = config.dataloader.valid_tfrecord_path
        valid_dataloader = TFSlicesValidationDataloader(**valid_dataloader_config)
        valid_dset = valid_dataloader.get_dataset_denoising()
    else:
        print("Using the full training dataset")

    # Training.
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=config.logs_dir),
    ]

    if not config.dataloader.use_full_train_dset:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_weights_path,
                monitor="val_mae",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
            )
        )

    if config.dataloader.use_full_train_dset:
        history = model.fit(
            train_dset, epochs=config.epochs, callbacks=callbacks, verbose=2
        )
    else:
        history = model.fit(
            train_dset,
            validation_data=valid_dset,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )
    model.save_weights(config.last_epoch_weights_path)

    # Append the epoch number that got the best accuracy.
    with open(md_config_path, "r+") as f:
        lines = f.readlines()
        best_mae = max(history.history["val_mae"])
        best_mae_idx = history.history["val_mae"].index(best_mae) + 1
        lines.insert(-2, f"- Best epoch: {best_mae_idx}\r")
        f.writelines(lines)

    # Evaluation of the last model and the model with the best mae.
    decoder_denoising_eval(model, config, subdir="last_epoch")
    if not config.dataloader.use_full_train_dset:
        best_mae_model = tf.keras.models.clone_model(model)
        best_mae_model.load_weights(config.best_weights_path).expect_partial()
        decoder_denoising_eval(best_mae_model, config, subdir="best")


def encoder_classification_pretrain(
    config: ml_collections.ConfigDict,
) -> tf.keras.Model:

    tf.keras.utils.set_random_seed(42)

    experiment_desc = create_experiment_desc(config)
    run_dir = create_run_dir(experiment_desc, root_dir=config.root_dir)
    logs_fname = os.path.join(run_dir, "execution_logs.txt")
    _ = Logger(file_name=logs_fname, file_mode="a", should_flush=True)

    # Experiment directories.
    config.logs_dir = create_directory(os.path.join(run_dir, "logs"))
    config.config_dir = create_directory(os.path.join(run_dir, "config"))
    config.weights_dir = create_directory(os.path.join(run_dir, "weights"))
    config.evaluation_dir = create_directory(os.path.join(run_dir, "evaluation"))

    # Finish the configuration, make sure to define model's input shape as a string.
    num_modalities = len(config.dataloader.modalities.split(","))
    config.model.input_shape = (
        f"{config.slice_size},{config.slice_size},{num_modalities}"
    )
    config.best_weights_path = os.path.join(config.weights_dir, "best", "weights")
    config.last_epoch_weights_path = os.path.join(
        config.weights_dir, "last_epoch", "weights"
    )
    yaml_config_path = os.path.join(config.config_dir, "config.yml")
    md_config_path = os.path.join(run_dir, "README.md")
    save_dict_as_yaml(yaml_config_path, config.to_dict())
    save_md_with_experiment_config(md_config_path, config, config.task)

    # Experiment setup.
    if config.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        print("Training with mixed precision...")

    modalities = config.dataloader.modalities.replace(" ", "").split(",")
    input_shape = tuple(int(s) for s in config.model.input_shape.split(","))
    filters_per_level = [int(f) for f in config.model.filters_per_level.split(",")]
    blocks_depth = [int(f) for f in config.model.blocks_depth.split(",")]

    # Optimization.
    total_steps = (
        int(config.dataloader.num_train_samples / config.dataloader.batch_size)
        * config.epochs
    )
    warmup_steps = int(total_steps * config.warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=config.base_lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    # Model and optimizer creation.
    model_module = importlib.import_module(f"models.{config.model.model_module}")
    model_config = config.model.to_dict()
    model_config["input_shape"] = input_shape
    model_config["filters_per_level"] = filters_per_level
    model_config["blocks_depth"] = blocks_depth
    model_config["norm_layer"] = import_method(config.model.norm_layer)
    model_config["pooling_layer"] = import_method(config.model.pooling_layer)

    encoder = model_module.UNetEncoder(**model_config)
    model = model_module.add_head_to_encoder(
        encoder, num_classes=1, out_activation="sigmoid"
    )
    opt = AdamW(learning_rate=scheduled_lrs, weight_decay=config.weight_decay)
    model.compile(
        optimizer=opt,
        loss=config.loss_name,
        metrics=[tf.keras.metrics.BinaryAccuracy()],
        jit_compile=True,
    )

    # Data loading.
    train_dataloader_config = config.dataloader.to_dict()
    train_dataloader_config["modalities"] = modalities
    train_dataloader_config["multiresolution"] = False
    train_dataloader_config["deep_supervision"] = False
    train_dataloader_config["tfrecord_path"] = config.dataloader.train_tfrecord_path
    train_dataloader = TFSlicesDataloader(**train_dataloader_config)
    train_dset = train_dataloader.get_dataset_cls()

    if not config.dataloader.use_full_train_dset:
        print("Using a train-validation split")
        valid_dataloader_config = config.dataloader.to_dict()
        valid_dataloader_config["modalities"] = modalities
        valid_dataloader_config["multiresolution"] = False
        valid_dataloader_config["deep_supervision"] = False
        valid_dataloader_config["tfrecord_path"] = config.dataloader.valid_tfrecord_path
        valid_dataloader = TFSlicesValidationDataloader(**valid_dataloader_config)
        valid_dset = valid_dataloader.get_dataset_cls()
    else:
        print("Using the full training dataset")

    # Training.
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=config.logs_dir),
    ]

    if not config.dataloader.use_full_train_dset:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.best_weights_path,
                monitor="val_binary_accuracy",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            )
        )

    if config.dataloader.use_full_train_dset:
        history = model.fit(
            train_dset, epochs=config.epochs, callbacks=callbacks, verbose=2
        )
    else:
        history = model.fit(
            train_dset,
            validation_data=valid_dset,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )
    model.save_weights(config.last_epoch_weights_path)

    # Append the epoch number that got the best accuracy.
    with open(md_config_path, "r+") as f:
        lines = f.readlines()
        best_acc = max(history.history["val_binary_accuracy"])
        best_acc_idx = history.history["val_binary_accuracy"].index(best_acc) + 1
        lines.insert(-2, f"- Best epoch: {best_acc_idx}\r")
        f.writelines(lines)

    # Evaluation of the last model and the model with the best accuracy.
    encoder_classification_eval(model, config, subdir="last_epoch")
    if not config.dataloader.use_full_train_dset:
        best_acc_model = tf.keras.models.clone_model(model)
        best_acc_model.load_weights(config.best_weights_path).expect_partial()
        encoder_classification_eval(best_acc_model, config, subdir="best")


# ------------------------------------------------------------------------------
# Train utilities.


def create_model_copy(model: tf.keras.Model) -> tf.keras.Model:
    """Creates a model with the same structure and weights of `model`."""
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    return model_copy


def import_method_from_config(
    config: ml_collections.ConfigDict, attr: str
) -> Optional[Callable]:
    """Imports a method from a config attribute."""
    if attr in config:
        return import_method(config[attr])


def import_method(method_path: str) -> Callable:
    """Import a method from a module path."""
    method_shards = method_path.split(".")
    method_shards[0] = {
        "np": "numpy",
        "tf": "tensorflow",
        "tfa": "tensorflow_addons",
    }.get(method_shards[0], method_shards[0])

    module_path = ".".join(method_shards[:-1])
    method_name = method_shards[-1]

    module = importlib.import_module(module_path)
    return getattr(module, method_name)
