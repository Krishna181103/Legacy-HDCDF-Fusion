"""Reusable HDCDF-inspired layers for hyperspectral image classification.

These modules are adapted for this TensorFlow/Keras project. They implement the
paper's useful building blocks without claiming an exact reproduction of the
published PyTorch architecture.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers



@tf.keras.utils.register_keras_serializable(package="HDCDF")
class ChannelReduceMean(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="HDCDF")
class ChannelReduceMax(layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def activation_layer(use_mish=True, name=None):
    if use_mish:
        return layers.Activation(mish, name=name)
    return layers.Activation("relu", name=name)


def depthwise_separable_conv3d(
    x,
    filters,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    padding="same",
    depth_multiplier=1,
    weight_decay=1e-4,
    name=None,
):
    """3D depth-wise separable convolution.

    Shape flow: (B, H, W, S, C) -> grouped 3D DWC -> (B, H, W, S, C*m)
    -> 1x1x1 PWC -> (B, H, W, S, filters).
    """
    in_channels = int(x.shape[-1])
    if in_channels is None:
        raise ValueError("The input channel dimension must be statically known.")

    prefix = "" if name is None else name + "_"
    x = layers.Conv3D(
        in_channels * depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=in_channels,
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "dw3d",
    )(x)
    x = layers.Conv3D(
        filters,
        kernel_size=(1, 1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "pw3d",
    )(x)
    return x


def dsc3d_bn_act(
    x,
    filters,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    use_mish=True,
    weight_decay=1e-4,
    name=None,
):
    x = depthwise_separable_conv3d(
        x,
        filters,
        kernel_size=kernel_size,
        strides=strides,
        weight_decay=weight_decay,
        name=name,
    )
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    x = activation_layer(use_mish, name=None if name is None else name + "_act")(x)
    return x


def dsc2d_bn_act(
    x,
    filters,
    kernel_size=3,
    strides=1,
    use_mish=True,
    weight_decay=1e-4,
    name=None,
):
    prefix = "" if name is None else name + "_"
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(weight_decay),
        name=prefix + "dw2d",
    )(x)
    x = layers.BatchNormalization(name=prefix + "dw_bn")(x)
    x = activation_layer(use_mish, name=prefix + "dw_act")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "pw2d",
    )(x)
    x = layers.BatchNormalization(name=prefix + "pw_bn")(x)
    x = activation_layer(use_mish, name=prefix + "pw_act")(x)
    return x


def channel_attention(x, ratio=8, name=None):
    """CBAM channel attention.

    Shape flow: (B, H, W, C) -> pooled descriptors (B, C) -> shared MLP
    -> sigmoid weights (B, 1, 1, C) -> weighted feature map.
    """
    channels = int(x.shape[-1])
    hidden = max(channels // ratio, 8)
    prefix = "" if name is None else name + "_"

    shared_1 = layers.Dense(hidden, activation="relu", name=prefix + "mlp_1")
    shared_2 = layers.Dense(channels, name=prefix + "mlp_2")

    avg = layers.GlobalAveragePooling2D(name=prefix + "gap")(x)
    maxv = layers.GlobalMaxPooling2D(name=prefix + "gmp")(x)
    avg = shared_2(shared_1(avg))
    maxv = shared_2(shared_1(maxv))

    weights = layers.Add(name=prefix + "sum")([avg, maxv])
    weights = layers.Activation("sigmoid", name=prefix + "sigmoid")(weights)
    weights = layers.Reshape((1, 1, channels), name=prefix + "reshape")(weights)
    return layers.Multiply(name=prefix + "scale")([x, weights])


def spatial_attention(x, kernel_size=7, name=None):
    """CBAM spatial attention.

    Shape flow: (B, H, W, C) -> avg/max channel maps (B, H, W, 1)
    -> concat (B, H, W, 2) -> 7x7 conv -> sigmoid spatial mask.
    """
    prefix = "" if name is None else name + "_"
    avg = ChannelReduceMean(name=prefix + "avg_pool")(x)
    maxv = ChannelReduceMax(name=prefix + "max_pool")(x)
    weights = layers.Concatenate(axis=-1, name=prefix + "concat")([avg, maxv])
    weights = layers.Conv2D(
        1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        use_bias=False,
        name=prefix + "conv7",
    )(weights)
    return layers.Multiply(name=prefix + "scale")([x, weights])


def dc_attention_2d(
    x,
    filters=None,
    ratio=8,
    use_mish=True,
    weight_decay=1e-4,
    name="dc_attention_2d",
):
    """2D-DCAttention: DWC -> BN -> Mish -> CAM -> SAM -> PWC."""
    filters = filters or int(x.shape[-1])
    prefix = name + "_"

    # (B, H, W, C) -> depth-wise spatial filtering with C preserved.
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(weight_decay),
        name=prefix + "dw",
    )(x)
    x = layers.BatchNormalization(name=prefix + "bn")(x)
    x = activation_layer(use_mish, name=prefix + "mish")(x)

    # CBAM-style spectral/channel and spatial weighting.
    x = channel_attention(x, ratio=ratio, name=prefix + "channel")
    x = spatial_attention(x, kernel_size=7, name=prefix + "spatial")

    # Point-wise channel mixing: (B, H, W, C) -> (B, H, W, filters).
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "pw",
    )(x)
    x = layers.BatchNormalization(name=prefix + "pw_bn")(x)
    x = activation_layer(use_mish, name=prefix + "pw_act")(x)
    return x


def hybrid_residual_3d(
    x,
    filters,
    pool_size=(2, 2, 1),
    use_mish=True,
    use_depthwise=True,
    weight_decay=1e-4,
    name="hybrid_res3d",
):
    """3D residual block with paper-inspired DSC main path and pooled shortcut."""
    prefix = name + "_"
    shortcut = x

    if use_depthwise:
        y = dsc3d_bn_act(
            x,
            filters,
            kernel_size=(3, 3, 3),
            use_mish=use_mish,
            weight_decay=weight_decay,
            name=prefix + "main",
        )
    else:
        y = layers.Conv3D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=prefix + "conv",
        )(x)
        y = layers.BatchNormalization(name=prefix + "conv_bn")(y)
        y = activation_layer(use_mish, name=prefix + "conv_act")(y)

    y = layers.MaxPool3D(pool_size=pool_size, padding="same", name=prefix + "pool")(y)

    shortcut = layers.Conv3D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "shortcut_conv",
    )(shortcut)
    shortcut = layers.BatchNormalization(name=prefix + "shortcut_bn")(shortcut)
    shortcut = layers.MaxPool3D(
        pool_size=pool_size,
        padding="same",
        name=prefix + "shortcut_pool",
    )(shortcut)

    y = layers.Add(name=prefix + "add")([shortcut, y])
    return activation_layer(use_mish, name=prefix + "out_act")(y)


def hybrid_residual_2d(
    x,
    filters,
    pool_size=(2, 2),
    use_mish=True,
    use_depthwise=True,
    weight_decay=1e-4,
    name="hybrid_res2d",
):
    """2D residual block with DWC -> BN -> Mish -> PWC -> BN -> Mish -> MaxPool."""
    prefix = name + "_"
    shortcut = x

    if use_depthwise:
        y = dsc2d_bn_act(
            x,
            filters,
            kernel_size=3,
            use_mish=use_mish,
            weight_decay=weight_decay,
            name=prefix + "main",
        )
    else:
        y = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=prefix + "conv",
        )(x)
        y = layers.BatchNormalization(name=prefix + "conv_bn")(y)
        y = activation_layer(use_mish, name=prefix + "conv_act")(y)

    y = layers.MaxPool2D(pool_size=pool_size, padding="same", name=prefix + "pool")(y)

    shortcut = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "shortcut_conv",
    )(shortcut)
    shortcut = layers.BatchNormalization(name=prefix + "shortcut_bn")(shortcut)
    shortcut = layers.MaxPool2D(
        pool_size=pool_size,
        padding="same",
        name=prefix + "shortcut_pool",
    )(shortcut)

    y = layers.Add(name=prefix + "add")([shortcut, y])
    return activation_layer(use_mish, name=prefix + "out_act")(y)
