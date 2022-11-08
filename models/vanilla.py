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
    activation: str = "relu",
    out_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Optional[Any] = layers.BatchNormalization,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    dropout_rate: float = 0.0,
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

    skip_names = reversed(skip_names)
    blocks_depth = reversed(blocks_depth)
    for i, (skip_name, depth) in enumerate(zip(skip_names, blocks_depth)):
        x_skip = encoder.get_layer(skip_name).output
        x = UpBlock(
            x_skip.shape[-1],
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            upsample_layer=upsample_layer,
            attention_layer=attention_layer,
            dropout_rate=dropout_rate,
            depth=depth,
            name=name + f"_up_{i}",
        )([x, x_skip])
    x = Prediction(num_classes, out_activation, name=name + "_last")(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x, name=name)


def UNetEncoder(
    input_shape: Tuple[int, int, int],
    filters_per_level: List[int],
    activation: str = "relu",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Optional[Any] = layers.BatchNormalization,
    pooling_layer: Any = layers.MaxPooling2D,
    blocks_depth: Union[int, List[int]] = 2,
    dropout_rate: float = 0.0,
    name="unet_encoder",
    **kwargs,
) -> tf.keras.Model:

    if isinstance(blocks_depth, int):
        blocks_depth = [blocks_depth] * len(filters_per_level)

    inputs = tf.keras.Input(input_shape, name=name + "_inputs")
    x = Block(
        filters_per_level[0],
        activation=activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        dropout_rate=dropout_rate,
        depth=blocks_depth[0],
        name=name + "_first_block",
    )(inputs)

    filters_and_blocks = zip(filters_per_level[1:-1], blocks_depth[1:-1])
    for i, (filters, depth) in enumerate(filters_and_blocks):
        x = DownBlock(
            filters,
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            pooling_layer=pooling_layer,
            dropout_rate=dropout_rate,
            depth=depth,
            name=name + f"_down_{i}",
        )(x)

    x = Block(
        filters_per_level[-1],
        activation=activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        dropout_rate=dropout_rate,
        depth=blocks_depth[-1],
        name=name + "_bottleneck",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def DualUnet(
    input_shape: Tuple[int, int, int],
    filters_per_level: List[int],
    num_classes: int,
    activation: str = "relu",
    out_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    pooling_layer: Any = layers.MaxPooling2D,
    norm_layer: Optional[Any] = layers.BatchNormalization,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    dropout_rate: float = 0.0,
    blocks_depth: Union[int, List[int]] = 2,
    **kwargs,
):

    e1 = UNetEncoder(
        input_shape,
        filters_per_level,
        activation=activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        pooling_layer=pooling_layer,
        blocks_depth=blocks_depth,
        dropout_rate=dropout_rate,
        name="adc_unet",
    )
    e2 = UNetEncoder(
        input_shape,
        filters_per_level,
        activation=activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        pooling_layer=pooling_layer,
        blocks_depth=blocks_depth,
        dropout_rate=dropout_rate,
        name="dwi_unet",
    )

    e1_skips = get_skip_names_from_encoder(e1)
    e2_skips = get_skip_names_from_encoder(e2)

    # Create a model that shares the embedding.
    joint_bottleneck = layers.Concatenate(name="joint_bottleneck")(
        [e1.output, e2.output]
    )
    m = tf.keras.Model(inputs=[e1.input, e2.input], outputs=joint_bottleneck)

    u1 = UNet(
        m,
        e1_skips,
        num_classes,
        activation=activation,
        out_activation=out_activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        upsample_layer=upsample_layer,
        attention_layer=attention_layer,
        dropout_rate=dropout_rate,
        blocks_depth=blocks_depth,
        name="adc_unet",
    )
    u2 = UNet(
        m,
        e2_skips,
        num_classes,
        activation=activation,
        out_activation=out_activation,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        upsample_layer=upsample_layer,
        attention_layer=attention_layer,
        dropout_rate=dropout_rate,
        blocks_depth=blocks_depth,
        name="dwi_unet",
    )

    multimodal_unet = tf.keras.Model(inputs=m.inputs, outputs=[u1.output, u2.output])
    return multimodal_unet


# ---------------------------------------------------------------------------------
# Main blocks.


def UpBlock(
    filters: int,
    activation: str = "relu",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    dropout_rate: float = 0.0,
    depth: int = 2,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):

        x, x_skip = inputs

        # Upsample and block with depth=2 to match our paper implementation.
        x = upsample_layer(name=name + "_up")(x)
        x = Block(
            filters,
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            depth=2,
            name=name + "_upconv",
        )(x)

        # Filter skip connection values with attention.
        if not attention_layer is None:
            if "Self" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_sa")(x_skip)
            elif "Cross" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_ca")([x, x_skip])

        x = layers.Concatenate(name=name + "_concat")([x, x_skip])

        x = Block(
            filters,
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            depth=depth,
            name=name,
        )(x)

        return x

    return apply


def DownBlock(
    filters: int,
    activation: str = "relu",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    pooling_layer: Any = layers.MaxPooling2D,
    dropout_rate: float = 0.0,
    depth: int = 2,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):
        x = pooling_layer(pool_size=2, name=name + "_pool")(inputs)
        x = Block(
            filters,
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            depth=depth,
            name=name,
        )(x)
        return x

    return apply


def Block(
    filters: int,
    activation: str = "relu",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
    depth: int = 2,
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
                filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding=padding,
                name=name + f"_d{i}_conv",
            )(x)
            if not norm_layer is None:
                if norm_layer.__name__ == "GroupNormalization":
                    x = norm_layer(groups=8, name=name + f"_d{i}_bn")(x)
                else:
                    x = norm_layer(name=name + f"_d{i}_bn")(x)
            x = layers.Activation(activation, name=name + f"_d{i}_{activation}")(x)

            x = layers.Add(name=name + f"_d{i}_residual")([x, residual])

        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=name + "_drop")(x)
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
    has_dropout = any(layer.name.endswith("_drop") for layer in encoder.layers)
    suffix = "drop" if has_dropout else "residual"

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


def remove_prediction_layer_from_unet(unet: tf.keras.Model) -> tf.keras.Model:
    last_layer = unet.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    has_dropout = any(layer.name.endswith("drop") for layer in unet.layers)
    suffix = "drop" if has_dropout else "residual"

    candidate_layers = [
        layer.name
        for layer in unet.layers
        if (
            layer.name.endswith(suffix)
            and "up" in layer.name
            and layer.output.shape[1:-1] == output_shape[1:-1]
            and "upconv" not in layer.name
        )
    ]

    last_block_output = unet.get_layer(candidate_layers[-1]).output
    return tf.keras.Model(inputs=unet.inputs, outputs=last_block_output, name=unet.name)


def get_skip_names_from_encoder(encoder: tf.keras.Model) -> List[str]:
    last_layer = encoder.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    has_dropout = any(layer.name.endswith("_drop") for layer in encoder.layers)
    suffix = "drop" if has_dropout else "residual"

    layer_candidates = [
        layer.name
        for layer in encoder.layers
        if (
            layer.name.endswith(suffix)
            and not "bottleneck" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]

    return (
        get_deepest_layer_per_block(layer_candidates)
        if not has_dropout
        else layer_candidates
    )


def get_output_names_for_deep_supervision(unet: tf.keras.Model) -> List[str]:
    last_layer = unet.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    has_dropout = any(layer.name.endswith("_drop") for layer in unet.layers)
    suffix = "drop" if has_dropout else "residual"

    layer_candidates = [
        layer.name
        for layer in unet.layers
        if (
            layer.name.endswith(suffix)
            and not "down" in layer.name
            and not "upconv" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]

    return (
        get_deepest_layer_per_block(layer_candidates)
        if not has_dropout
        else layer_candidates
    )


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
