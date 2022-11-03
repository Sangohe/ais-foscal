import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import re
from typing import List, Optional, Tuple, Literal, Any, Union

from .conv_layers import Prediction, DeepSupervision

IntOrIntTuple = Union[int, Tuple[int, int, int]]


def UNet(
    encoder: tf.keras.Model,
    skip_names: List[str],
    num_classes: int,
    out_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    layer_scale_init_value: float = 1e-6,
    drop_path_rate: float = 0.0,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    blocks_depth: Union[int, List[int]] = 2,
    name: str = "unet_decoder",
    **kwargs,
) -> tf.keras.Model:

    # Expects the skip names to be ordered from earlier to deeper levels.
    x = encoder.output

    if isinstance(blocks_depth, int):
        blocks_depth = [blocks_depth] * len(skip_names)
    else:
        # Cut the last two elements because they belong to the bottleneck.
        blocks_depth = blocks_depth[:-2]

    skip_names = list(reversed(skip_names))
    blocks_depth = list(reversed(blocks_depth))
    for i, (skip_name, depth) in enumerate(zip(skip_names, blocks_depth)):
        x_skip = encoder.get_layer(skip_name).output
        x = UpBlock(
            x_skip.shape[-1],
            layer_scale_init_value=layer_scale_init_value,
            drop_path_rate=drop_path_rate,
            upsample_layer=upsample_layer,
            attention_layer=attention_layer,
            factor=2,
            depth=depth,
            name=name + f"_up_{i}",
        )([x, x_skip])
    x = Prediction(num_classes, out_activation, name=name + "_last")(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x, name=name)


def UNetEncoder(
    input_shape: Tuple[int, int, int],
    filters_per_level: List[int],
    layer_scale_init_value: float = 1e-6,
    drop_path_rate: float = 0.0,
    blocks_depth: Union[int, List[int]] = 2,
    name="unet_encoder",
    **kwargs,
) -> tf.keras.Model:

    if isinstance(blocks_depth, int):
        blocks_depth = [blocks_depth] * len(filters_per_level)

    inputs = tf.keras.Input(input_shape, name=name + "_inputs")
    x = Stem(filters_per_level[0], name=name)(inputs)
    x = Block(
        filters_per_level[0],
        layer_scale_init_value=layer_scale_init_value,
        drop_path_rate=drop_path_rate,
        depth=blocks_depth[0],
        name=name + "_first_block",
    )(x)

    filters_and_blocks = zip(filters_per_level[1:-1], blocks_depth[1:-1])
    for i, (filters, depth) in enumerate(filters_and_blocks):
        x = DownBlock(
            filters,
            layer_scale_init_value=layer_scale_init_value,
            drop_path_rate=drop_path_rate,
            factor=2,
            depth=depth,
            name=name + f"_down_{i}",
        )(x)

    x = Block(
        filters_per_level[-1],
        layer_scale_init_value=layer_scale_init_value,
        drop_path_rate=drop_path_rate,
        depth=blocks_depth[-1],
        name=name + "_bottleneck",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


# ---------------------------------------------------------------------------------
# Main blocks.


def UpBlock(
    filters: int,
    layer_scale_init_value: float = 1e-6,
    drop_path_rate: float = 0.0,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    factor: int = 2,
    depth: int = 2,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):

        x, x_skip = inputs

        x = upsample_layer(size=factor, name=name + "_up")(x)
        if not attention_layer is None:
            if "Self" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_sa")(x_skip)
            elif "Cross" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_ca")([x, x_skip])
        x = layers.Concatenate(name=name + "_concat")([x, x_skip])

        x = Block(
            filters,
            layer_scale_init_value=layer_scale_init_value,
            drop_path_rate=drop_path_rate,
            depth=depth,
            name=name,
        )(x)

        return x

    return apply


def DownBlock(
    filters: int,
    layer_scale_init_value: float = 1e-6,
    drop_path_rate: float = 0.0,
    factor: int = 2,
    depth: int = 2,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):

        x = Downsample(filters, factor=factor, name=name)(inputs)
        x = Block(
            filters,
            layer_scale_init_value=layer_scale_init_value,
            drop_path_rate=drop_path_rate,
            depth=depth,
            name=name,
        )(x)
        return x

    return apply


def Block(
    filters: int,
    layer_scale_init_value: float = 1e-6,
    drop_path_rate: float = 0.0,
    depth: int = 2,
    name: Optional[str] = None,
) -> tf.Tensor:

    assert depth > 0, "Conv block depth must be greater than 0."

    def apply(inputs):

        x = inputs

        for i in range(depth):

            input_width = x.shape[3]
            if input_width == filters:
                residual = x
            else:
                residual = layers.Conv2D(
                    filters, kernel_size=1, name=name + f"_d{i}_res_conv"
                )(x)

            x = layers.Conv2D(
                filters=filters,
                kernel_size=7,
                padding="same",
                groups=filters,
                name=name + f"_d{i}_depthwise_conv",
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6, name=name + f"_d{i}_layernorm")(
                x
            )
            x = layers.Dense(4 * filters, name=name + f"_d{i}_pointwise_conv_1")(x)
            x = layers.Activation("gelu", name=name + f"_d{i}_gelu")(x)
            x = layers.Dense(filters, name=name + f"_d{i}_pointwise_conv_2")(x)

            if layer_scale_init_value is not None:
                x = LayerScale(
                    layer_scale_init_value,
                    filters,
                    name=name + f"_d{i}_layer_scale",
                )(x)
            if drop_path_rate:
                layer = StochasticDepth(
                    drop_path_rate, name=name + f"_d{i}_stochastic_depth"
                )
            else:
                layer = layers.Activation("linear", name=name + f"_d{i}_identity")

            x = layers.Add(name=name + f"_d{i}_residual")([layer(x), residual])

        return x

    return apply


class LayerScale(layers.Layer):
    """Taken from:
    https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/convnext.py

    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.
    References:
      - https://github.com/rwightman/pytorch-image-models
    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


def Stem(filters, name: Optional[str] = None):
    def apply(inputs):
        x = inputs
        x = layers.Conv2D(
            filters,
            kernel_size=1,
            name=name + "_stem_conv",
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_stem_layernorm")(x)
        return x

    return apply


def Downsample(filters: int, factor: int = 2, name: Optional[str] = None):
    def apply(inputs):
        x = inputs
        x = layers.LayerNormalization(
            epsilon=1e-6,
            name=name + "_downsampling_layernorm",
        )(x)
        x = layers.Conv2D(
            filters,
            kernel_size=factor,
            strides=factor,
            name=name + "_downsampling_conv",
        )(x)

        return x

    return apply


# ---------------------------------------------------------------------------------
# Model utils.


def add_prediction_layer_to_unet(
    unet: tf.keras.Model,
    num_classes: int,
    out_activation: Literal["sigmoid", "softmax"] = "sigmoid",
) -> tf.keras.Model:
    x = unet.output
    x = Prediction(num_classes, out_activation, name=unet.name + "_last")(x)

    return tf.keras.Model(inputs=unet.inputs, outputs=x, name=unet.name)


def remove_prediction_layer_from_unet(unet: tf.keras.Model) -> tf.keras.Model:
    last_layer = unet.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    suffix = "residual"

    candidate_layers = [
        layer.name
        for layer in unet.layers
        if (
            layer.name.endswith(suffix)
            and "up" in layer.name
            and layer.output.shape[1:-1] == output_shape[1:-1]
        )
    ]

    last_block_output = unet.get_layer(candidate_layers[-1]).output
    return tf.keras.Model(inputs=unet.inputs, outputs=last_block_output, name=unet.name)


def get_output_names_for_deep_supervision(unet: tf.keras.Model) -> List[str]:
    last_layer = unet.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    suffix = "residual"

    layer_candidates = [
        layer.name
        for layer in unet.layers
        if (
            layer.name.endswith(suffix)
            and not "down" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]

    return get_deepest_layer_per_block(layer_candidates)


def add_deep_supervision_to_unet(
    unet: tf.keras.Model,
    hidden_layer_names: List[str],
    resize_features: bool = True,
) -> tf.keras.Model:

    # Get the activation layer at the end of the U-Net.
    last_layer = unet.layers[-1]
    out_activation = last_layer.activation
    last_output = last_layer.output
    num_classes = last_output.shape[-1]

    outputs = []
    for i, hidden_layer_name in enumerate(hidden_layer_names):
        hidden_layer_output = unet.get_layer(hidden_layer_name).output
        x = DeepSupervision(
            num_classes,
            out_activation,
            resize_features=resize_features,
            target_shape=last_output.shape[1],
            name=unet.name + f"_deep_supervision_{i}",
        )(hidden_layer_output)
        outputs.append(x)

    outputs.append(last_output)
    model = tf.keras.Model(inputs=unet.inputs, outputs=outputs, name=unet.name)
    return model


def get_skip_names_from_encoder(encoder: tf.keras.Model) -> List[str]:
    last_layer = encoder.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    suffix = "residual"

    layer_candidates = [
        layer.name
        for layer in encoder.layers
        if (
            layer.name.endswith(suffix)
            and not "bottleneck" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]

    return get_deepest_layer_per_block(layer_candidates)


def add_head_to_encoder(
    encoder: tf.keras.Model,
    num_classes: int,
    out_activation: Literal["sigmoid", "softmax"],
) -> tf.keras.Model:
    x = encoder.output
    x = layers.GlobalAveragePooling2D(name=encoder.name + "_global_pooling")(x)
    x = layers.Dense(num_classes, activation=None, name=encoder.name + "_logits")(x)
    x = layers.Activation(
        out_activation, dtype="float32", name=encoder.name + "_probs"
    )(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x, name=encoder.name)


def remove_head_from_encoder(encoder: tf.keras.Model) -> tf.keras.Model:

    suffix = "residual"
    candidate_layers = sorted(
        [
            layer.name
            for layer in encoder.layers
            if layer.name.endswith(suffix) and "bottleneck" in layer.name
        ]
    )
    last_bottleneck_layer_name = sorted(candidate_layers)[-1]

    output = encoder.get_layer(last_bottleneck_layer_name).output
    return tf.keras.Model(inputs=encoder.inputs, outputs=output, name=encoder.name)


def get_string_with_max_numeric_value(strings: List[str]) -> int:
    max_num = 0
    p = re.compile(r"\d+")
    for string in strings:
        num_str = p.search(string).group()
        num = int(num_str)
        if num > max_num:
            max_num = num
            max_num_str = string
    return max_num_str


def get_deepest_layer_per_block(layer_names: List[str]) -> List[str]:

    layer_name_shards = [s.rsplit("_", maxsplit=2) for s in layer_names]
    return (
        pd.DataFrame(layer_name_shards, columns=["block", "depth", "suffix"])
        .groupby("block", sort=False)
        .agg({"depth": get_string_with_max_numeric_value, "suffix": "first"})
        .assign(
            last_block_layer=lambda df: df.index + "_" + df.depth + "_" + df.suffix
        )["last_block_layer"]
        .tolist()
    )
