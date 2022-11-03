import tensorflow as tf
from tensorflow.keras import layers
from einops.layers.keras import Rearrange

from typing import Optional


def Prediction(
    num_classes: int, out_activation: str, name: Optional[str] = None
) -> tf.Tensor:
    def apply(inputs):
        x = layers.Conv2D(num_classes, kernel_size=1, name=name + "_logits")(inputs)
        x = layers.Activation(out_activation, dtype="float32", name=name + "_probs")(x)
        return x

    return apply


def DeepSupervision(
    num_classes: int,
    out_activation: str,
    resize_features: bool = True,
    target_shape: Optional[int] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):

        if resize_features:
            resize_factor = target_shape // inputs.shape[1]
            x = layers.UpSampling2D(
                size=resize_factor, interpolation="bilinear", name=name + "_up"
            )(inputs)
        else:
            x = inputs

        x = layers.Conv2D(num_classes, kernel_size=1, name=name + "_logits")(x)
        x = layers.Activation(out_activation, dtype="float32", name=name + "_probs")(x)
        return x

    return apply


# ---------------------------------------------------------------------------------
# Attention.


def CrossAttention(filters: int, name: Optional[str] = None) -> tf.Tensor:
    def apply(inputs):
        g, x = inputs

        q = layers.Conv2D(filters, kernel_size=1, name=name + "_q_proj")(g)
        q = Rearrange("b h w d -> b d h w")(q)
        k = layers.Conv2D(filters, kernel_size=1, name=name + "_k_proj")(x)
        k = Rearrange("b h w d -> b d h w")(k)
        v = layers.Conv2D(filters, kernel_size=1, name=name + "_v_proj")(x)
        v = Rearrange("b h w d -> b d h w")(v)

        sim = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (filters**-0.5)

        att_map = layers.Activation("softmax", name=name + "_att_map")(sim)
        filtered_value = tf.cast(att_map, dtype=v.dtype) @ v
        value_back = Rearrange("b d h w -> b h w d")(filtered_value)
        out = layers.Conv2D(
            filters, kernel_size=1, use_bias=False, name=name + "_proj_conv"
        )(value_back)
        return out

    return apply


def SelfAttention(filters: int, name: Optional[str] = None) -> tf.Tensor:
    def apply(inputs):
        x = inputs

        q = layers.Conv2D(filters, kernel_size=1, name=name + "_q_proj")(x)
        q = Rearrange("b h w d -> b d h w")(q)
        k = layers.Conv2D(filters, kernel_size=1, name=name + "_k_proj")(x)
        k = Rearrange("b h w d -> b d h w")(k)
        v = layers.Conv2D(filters, kernel_size=1, name=name + "_v_proj")(x)
        v = Rearrange("b h w d -> b d h w")(v)

        sim = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (filters**-0.5)

        att_map = layers.Activation("softmax", name=name + "_att_map")(sim)
        filtered_value = tf.cast(att_map, dtype=v.dtype) @ v
        value_back = Rearrange("b d h w -> b h w d")(filtered_value)
        out = layers.Conv2D(
            filters, kernel_size=1, use_bias=False, name=name + "_proj_conv"
        )(value_back)
        return out

    return apply


def AdditiveCrossAttention(filters: int, name: Optional[str] = None) -> tf.Tensor:
    def apply(inputs):
        g, x = inputs

        q = layers.Conv2D(filters, kernel_size=1, name=name + "_q_proj")(g)
        k = layers.Conv2D(filters, kernel_size=1, name=name + "_k_proj")(x)

        # Compute the similarities.
        sim = layers.Add(name=name + "_sim")([q, k])
        sim = layers.Activation("relu", name=name + "_pos_sim")(sim)

        # Project the similarities to a single channel.
        conv_sim = layers.Conv2D(1, kernel_size=1, name=name + "_pos_sim_conv")(sim)
        attention_map = layers.Activation("sigmoid", name=name + "_att_map")(conv_sim)

        filtered_features = layers.Multiply(name=name + "_ref_v")([x, attention_map])
        return filtered_features

    return apply


def AdditiveSelfAttention(filters: int, name: Optional[str] = None) -> tf.Tensor:
    def apply(inputs):
        x = inputs

        q = layers.Conv2D(filters, kernel_size=1, name=name + "_q_proj")(x)
        k = layers.Conv2D(filters, kernel_size=1, name=name + "_k_proj")(x)

        # Compute the similarities.
        sim = layers.Add(name=name + "_sim")([q, k])
        sim = layers.Activation("relu", name=name + "_pos_sim")(sim)

        # Project the similarities to a single channel.
        conv_sim = layers.Conv2D(1, kernel_size=1, name=name + "_pos_sim_conv")(sim)
        attention_map = layers.Activation("sigmoid", name=name + "_att_map")(conv_sim)

        filtered_features = layers.Multiply(name=name + "_ref_v")([x, attention_map])
        return filtered_features

    return apply
