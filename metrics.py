"""Metrics to measure the performance of models"""

import numpy as np
import tensorflow as tf

from keras import backend
from scipy import ndimage
from tensorflow.keras.metrics import Metric
from medpy.metric.binary import hd as haussdorf_dist

# -----------------------------------------------------------------------
# Custom keras metrics.

# *: refactor this class to take multiclass inputs and parse them to binary masks.
class BinaryDiceScore(Metric):
    def __init__(self, name="dice", threshold=0.5, dtype=None):
        super(BinaryDiceScore, self).__init__(name=name, dtype=dtype)
        self.total_dice = self.add_weight("total_dice", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        # self.threshold = tf.constant(threshold, dtype=tf.float32, name="threshold")
        self.threshold = tf.Variable(
            initial_value=threshold, dtype="float32", trainable=False
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        if not tf.is_tensor(y_true):
            y_true = tf.convert_to_tensor(y_true)
        if not tf.is_tensor(y_pred):
            y_pred = tf.convert_to_tensor(y_pred)

        # Convert multiclass/one-hot encoded inputs to one channel.
        if y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.bool)

        if y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.cast(y_pred, tf.bool)
        else:
            # threshold = tf.cast(self.threshold, dtype=y_pred.dtype)
            y_pred = tf.greater_equal(y_pred, self.threshold)

        current_dice = self.compute_dice_for_one_sample(y_true, y_pred)
        self.total_dice.assign_add(current_dice)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total_dice, self.count)

    def reset_state(self):
        backend.set_value(self.total_dice, 0.0)
        backend.set_value(self.count, 0)

    def compute_dice_for_one_sample(self, y_true, y_pred):
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        intersection = tf.math.logical_and(y_true, y_pred)
        intersection_sum = tf.cast(
            tf.math.count_nonzero(intersection), dtype=self._dtype
        )

        if intersection_sum == 0:
            current_dice = 0.0
        else:
            true_size = tf.cast(tf.math.count_nonzero(y_true), dtype=self._dtype)
            pred_size = tf.cast(tf.math.count_nonzero(y_pred), dtype=self._dtype)
            current_dice = 2 * intersection_sum / (true_size + pred_size)

        return current_dice


class MulticlassDiceScore(Metric):
    def __init__(self, num_classes, name="multiclass_dice", dtype=None):
        super(MulticlassDiceScore, self).__init__(name=name, dtype=None)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name="total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """This metric expects `y_pred` to be a Tensor with the last dimension
        shape equal to `self.num_classes`.

        Args:
            y_true (_type_): Tensor with the GT values
            y_pred (_type_): Tensor with predictions.
            sample_weight (_type_, optional): Tensor with the weights for each
            sample. Defaults to None.
        """

        # Convert the one hot encoded tensor if needed.
        if y_true.shape[-1] == self.num_classes:
            y_true = tf.argmax(y_true, axis=-1)
        if y_pred.shape[-1] == self.num_classes:
            y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # Do the same for the weights if given.
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        current_cm = tf.math.confusion_matrix(
            y_true, y_pred, self.num_classes, weights=sample_weight, dtype=self._dtype
        )

        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean dice score via the confusion matrix."""
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )

        dice = tf.math.divide_no_nan(2 * true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(dice, name="mean_dice"), num_valid_entries
        )

    def reset_state(self):
        backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes}
        base_config = super(MulticlassDiceScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# -----------------------------------------------------------------------
# Numpy metrics.


def compute_confusion_matrix(y_true, y_pred):
    """
    Returns tuple tp, tn, fp, fn
    """

    assert y_true.size == y_pred.size

    true_pos = np.sum(np.logical_and(y_true, y_pred))
    true_neg = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    false_pos = np.sum(np.logical_and(y_true == 0, y_pred))
    false_neg = np.sum(np.logical_and(y_true, y_pred == 0))

    return true_pos, true_neg, false_pos, false_neg


def compute_lesion_confusion_matrix(y_true, y_pred):
    # True positives
    lesions_true, num_lesions_true = ndimage.label(y_true)
    lesions_pred, num_lesions_pred = ndimage.label(y_pred)

    true_pos = 0.0
    for i in range(num_lesions_true):
        lesion_detected = np.logical_and(y_pred, lesions_true == (i + 1)).any()
        if lesion_detected:
            true_pos += 1
    true_pos = np.min([true_pos, num_lesions_pred])

    # False positives
    tp_labels = np.unique(y_true * lesions_pred)
    fp_labels = np.unique(np.logical_not(y_true) * lesions_pred)

    # [label for label in fp_labels if label not in tp_labels]
    false_pos = 0.0
    for fp_label in fp_labels:
        if fp_label not in tp_labels:
            false_pos += 1

    return true_pos, false_pos, num_lesions_true, num_lesions_pred


def dice_coef(y_true, y_pred, smooth=0.01):
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection + smooth) / (
            np.sum(y_true) + np.sum(y_pred) + smooth
        )
    else:
        return 0.0


def compute_segmentation_metrics(y_true, y_pred, lesion_metrics=False, exclude=None):
    metrics = {}
    eps = np.finfo(np.float32).eps

    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)

    # Sensitivity and specificity
    metrics["sens"] = tp / (tp + fn + eps)  # Correct % of the real lesion
    metrics["spec"] = tn / (tn + fp + eps)  # Correct % of the healthy area identified

    # Predictive value
    metrics["ppv"] = tp / (tp + fp + eps)  # Of all lesion voxels, % of really lesion
    metrics["npv"] = tn / (tn + fn + eps)  # Of all lesion voxels, % of really lesion

    # Lesion metrics
    if lesion_metrics:
        tpl, fpl, num_lesions_true, num_lesions_pred = compute_lesion_confusion_matrix(
            y_true, y_pred
        )
        metrics["l_tpf"] = tpl / num_lesions_true if num_lesions_true > 0 else np.nan
        metrics["l_fpf"] = fpl / num_lesions_pred if num_lesions_pred > 0 else np.nan

        metrics["l_ppv"] = tpl / (tpl + fpl + eps)
        metrics["l_f1"] = (2.0 * metrics["l_ppv"] * metrics["l_tpf"]) / (
            metrics["l_ppv"] + metrics["l_tpf"] + eps
        )

    # Dice coefficient
    metrics["dsc"] = dice_coef(y_true, y_pred)

    # Relative volume difference
    metrics["avd"] = (
        2.0
        * np.abs(np.sum(y_pred) - np.sum(y_true))
        / (np.sum(y_pred) + np.sum(y_true) + eps)
    )

    # Haussdorf distance
    try:
        metrics["hd"] = haussdorf_dist(
            y_pred, y_true, connectivity=3
        )  # Why connectivity 3?
    except Exception:
        metrics["hd"] = np.nan

    if exclude is not None:
        [metrics.pop(metric, None) for metric in exclude]

    return metrics


def compute_segmentation_metrics_for_slices(
    true_slices: np.ndarray, pred_slices: np.ndarray, lesion_metrics=False, exclude=None
):
    """Computes segmentation metrics for all the slices of `true_slices` and
    `pred_slices`."""

    assert true_slices.shape == pred_slices.shape

    if true_slices.shape[0] == true_slices.shape[1]:
        true_slices = true_slices.transpose(2, 0, 1)
        pred_slices = pred_slices.transpose(2, 0, 1)

    metrics = []
    for true_slice, pred_slice in zip(true_slices, pred_slices):
        slice_metrics = compute_segmentation_metrics(
            true_slice, pred_slice, lesion_metrics=lesion_metrics, exclude=exclude
        )
        metrics.append(slice_metrics)
    return metrics


def compute_avg_std_metrics_list(metrics_list):
    metrics_avg_std = dict()

    assert len(metrics_list) > 0

    for metric_name in metrics_list[0].keys():
        metric_values = [metrics[metric_name] for metrics in metrics_list]

        metrics_avg_std.update(
            {
                "{}_avg".format(metric_name): np.nanmean(metric_values),
                "{}_std".format(metric_name): np.nanstd(metric_values),
            }
        )

    return metrics_avg_std
