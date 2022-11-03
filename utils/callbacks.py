import numpy as np
import tensorflow as tf

import os
from typing import Optional


class EMACallback(tf.keras.callbacks.Callback):
    """This callback updates the weights of `ema_model` (a copy of `self.model`)
    following a Exponential Moving Average strategy. Additionally, it keeps track
    of the metrics of the `ema_model` to save the best model and log them to TensorBoard."""

    def __init__(
        self,
        ema_model: tf.keras.Model,
        ema: float = 0.999,
        validation_dset: Optional[tf.data.Dataset] = None,
        log_dir: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        filepath: Optional[str] = None,
        save_weights_only: bool = True,
    ):
        assert mode in ["min", "max"], "mode must be 'min' or 'max'"

        self.ema = ema
        self.ema_model = ema_model
        self.validation_dset = validation_dset
        self.log_dir = log_dir
        self.monitor = monitor
        self.mode = mode
        self.filepath = filepath
        self.save_weights_only = save_weights_only

        if validation_dset is not None:
            if log_dir is not None:
                self.log_to_tensorboard = True
                valid_writer_path = os.path.join(log_dir, "ema_validation")
                self._valid_file_writer = tf.summary.create_file_writer(
                    valid_writer_path
                )
            if monitor is not None and mode is not None and filepath is not None:
                self.save_best_model = True
                self.best = {"min": np.Inf, "max": -np.Inf}[mode]
                self.monitor_op = {"min": np.less, "max": np.greater}[mode]

    def on_train_batch_end(self, batch, logs):
        for weight, ema_weight in zip(self.model.weights, self.ema_model.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_dset is not None:
            metrics_values = self.ema_model.evaluate(self.validation_dset, verbose=0)
            metrics = dict(zip(self.model.metrics_names, metrics_values))

            # Log metrics to TensorBoard.
            if self.log_to_tensorboard:
                with tf.summary.record_if(True):
                    with self._valid_file_writer.as_default():
                        for name, value in metrics.items():
                            tf.summary.scalar("epoch_" + name, value, step=epoch)

            # If the metric being monitored improved, save the model.
            if self.save_best_model:
                current = metrics.get(self.monitor)
                if self.monitor_op(current, self.best):
                    print(
                        f"Epoch {epoch + 1}: ema_{self.monitor} improved from "
                        f"{self.best:.5f} to {current:.5f}, saving model to {self.filepath}"
                    )
                    self.best = current
                    if self.save_weights_only:
                        self.ema_model.save_weights(self.filepath, overwrite=True)
                    else:
                        self.ema_model.save(self.filepath, overwrite=True)
                else:
                    print(
                        f"Epoch {epoch + 1}: ema_{self.monitor} did not improve from "
                        f"{self.best:.5f}"
                    )
