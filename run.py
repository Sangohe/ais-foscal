import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from absl import app, flags
from ml_collections import config_flags

import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    help_string="Path to the configuration file in configs/.",
    lock_config=False,
)
flags.mark_flags_as_required(["config"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    train_fn = getattr(train, FLAGS.config.task)
    train_fn(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
