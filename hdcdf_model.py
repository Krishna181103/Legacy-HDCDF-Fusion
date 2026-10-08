"""HDCDF-inspired model variants for HyperLeaf hyperspectral classification."""

from tensorflow.keras import layers, models, regularizers

from models.hdcdf_components import (
    activation_layer,
    dc_attention_2d,
    dsc2d_bn_act,
    dsc3d_bn_act,
    hybrid_residual_2d,
    hybrid_residual_3d,
)


ABLATION_VARIANTS = (
    "baseline",
    "dsc",
    "hybrid",
    "attention",
    "residual",
    "mish",
    "full",
    "2d_only",
    "3d_only",
)


def reshape_3d_to_2d(x, spatial_size, spectral_size, channels, name):
    """Flatten spectral-depth and channels into a 2D feature channel axis."""
    return layers.Reshape((spatial_size, spatial_size, spectral_size * channels), name=name)(x)


def conv3d_bn_act(x, filters, kernel_size=(3, 3, 3), use_mish=False, weight_decay=1e-4, name=None):
    x = layers.Conv3D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=None if name is None else name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    return activation_layer(use_mish, name=None if name is None else name + "_act")(x)


def conv2d_bn_act(x, filters, kernel_size=3, use_mish=False, weight_decay=1e-4, name=None):
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=None if name is None else name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    return activation_layer(use_mish, name=None if name is None else name + "_act")(x)


def build_hdcdf_model(
    input_shape=(48, 48, 24, 1),
    num_classes=4,
    variant="full",
    dropout=0.4,
    weight_decay=1e-4,
):
    """Build a configurable HDCDF-inspired classifier.

    Variants are intentionally additive for ablation:
    baseline -> standard conv/ReLU single path
    dsc -> depth-wise separable conv
    hybrid -> 2D + 3D feature extraction
    attention -> adds 2D-DCAttention
    residual -> adds hybrid residual blocks
    mish/full -> Mish activation, full dual-branch fusion
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError("Unknown HDCDF variant: %s" % variant)

    h, w, spectral_bands, _ = input_shape
    if h != w:
        raise ValueError("This model expects square spatial patches.")

    use_mish = variant in ("mish", "full", "2d_only", "3d_only")
    use_dsc = variant in ("dsc", "hybrid", "attention", "residual", "mish", "full", "2d_only", "3d_only")
    use_hybrid = variant in ("hybrid", "attention", "residual", "mish", "full")
    use_attention = variant in ("attention", "full", "2d_only")
    use_residual = variant in ("residual", "full", "3d_only")

    inputs = layers.Input(shape=input_shape, name="hsi_patch")

    if variant == "2d_only":
        x = layers.Reshape((h, w, spectral_bands), name="flatten_input_spectral")(inputs)
        x = dsc2d_bn_act(x, 96, use_mish=use_mish, weight_decay=weight_decay, name="only2d_dsc1")
        x = dc_attention_2d(x, 128, use_mish=use_mish, weight_decay=weight_decay, name="only2d_dcattn")
        x = hybrid_residual_2d(x, 160, use_mish=use_mish, weight_decay=weight_decay, name="only2d_res")
        x = layers.GlobalAveragePooling2D(name="only2d_gap")(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay), name="only2d_fc")(x)
        x = activation_layer(use_mish, name="only2d_fc_act")(x)
        x = layers.Dropout(dropout, name="only2d_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)
        return models.Model(inputs, outputs, name="hdcdf_2d_only")

    if variant == "3d_only":
        x = dsc3d_bn_act(inputs, 24, use_mish=use_mish, weight_decay=weight_decay, name="only3d_dsc1")
        x = hybrid_residual_3d(x, 48, pool_size=(2, 2, 2), use_mish=use_mish, weight_decay=weight_decay, name="only3d_res1")
        x = hybrid_residual_3d(x, 64, pool_size=(2, 2, 2), use_mish=use_mish, weight_decay=weight_decay, name="only3d_res2")
        x = layers.GlobalAveragePooling3D(name="only3d_gap")(x)
        x = layers.Dropout(dropout, name="only3d_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)
        return models.Model(inputs, outputs, name="hdcdf_3d_only")

    conv3d = dsc3d_bn_act if use_dsc else conv3d_bn_act
    conv2d = dsc2d_bn_act if use_dsc else conv2d_bn_act

    # Upper branch: 3D spectral-spatial extraction -> 2D-DCAttention.
    upper = conv3d(inputs, 24, kernel_size=(3, 3, 3), use_mish=use_mish, weight_decay=weight_decay, name="upper_3d_1")
    upper = conv3d(upper, 32, kernel_size=(3, 3, 5), use_mish=use_mish, weight_decay=weight_decay, name="upper_3d_2")
    upper = layers.MaxPool3D(pool_size=(2, 2, 2), padding="same", name="upper_pool")(upper)
    upper = conv3d(upper, 48, kernel_size=(1, 1, 7), use_mish=use_mish, weight_decay=weight_decay, name="upper_3d_3")
    upper_size = h // 2
    upper_bands = (spectral_bands + 1) // 2
    upper = reshape_3d_to_2d(upper, upper_size, upper_bands, 48, name="upper_reshape_2d")
    if use_attention:
        upper = dc_attention_2d(upper, 160, use_mish=use_mish, weight_decay=weight_decay, name="upper_dcattention")
    else:
        upper = conv2d(upper, 160, use_mish=use_mish, weight_decay=weight_decay, name="upper_2d")

    if not use_hybrid and variant != "baseline":
        x = layers.GlobalAveragePooling2D(name="single_gap")(upper)
        x = layers.Dropout(dropout, name="single_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)
        return models.Model(inputs, outputs, name="hdcdf_%s" % variant)

    # Lower branch: residual 3D path -> reshape -> residual 2D path -> fusion.
    if use_residual:
        lower = hybrid_residual_3d(
            inputs,
            32,
            pool_size=(2, 2, 2),
            use_mish=use_mish,
            use_depthwise=use_dsc,
            weight_decay=weight_decay,
            name="lower_res3d_1",
        )
        lower = hybrid_residual_3d(
            lower,
            48,
            pool_size=(2, 2, 1),
            use_mish=use_mish,
            use_depthwise=use_dsc,
            weight_decay=weight_decay,
            name="lower_res3d_2",
        )
    else:
        lower = conv3d(inputs, 32, kernel_size=(3, 3, 3), use_mish=use_mish, weight_decay=weight_decay, name="lower_3d_1")
        lower = layers.MaxPool3D(pool_size=(2, 2, 2), padding="same", name="lower_pool1")(lower)
        lower = conv3d(lower, 48, kernel_size=(3, 3, 3), use_mish=use_mish, weight_decay=weight_decay, name="lower_3d_2")
        lower = layers.MaxPool3D(pool_size=(2, 2, 1), padding="same", name="lower_pool2")(lower)

    lower_size = h // 4
    lower_bands = (spectral_bands + 1) // 2
    lower = reshape_3d_to_2d(lower, lower_size, lower_bands, 48, name="lower_reshape_2d")
    if use_residual:
        lower = hybrid_residual_2d(
            lower,
            192,
            pool_size=(1, 1),
            use_mish=use_mish,
            use_depthwise=use_dsc,
            weight_decay=weight_decay,
            name="lower_res2d",
        )
    else:
        lower = conv2d(lower, 192, use_mish=use_mish, weight_decay=weight_decay, name="lower_2d")

    lower = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="lower_upsample")(lower)
    lower = layers.Conv2D(160, 1, padding="same", use_bias=False, name="lower_fusion_projection")(lower)
    lower = layers.BatchNormalization(name="lower_fusion_bn")(lower)
    lower = activation_layer(use_mish, name="lower_fusion_act")(lower)

    fused = layers.Concatenate(name="dual_branch_fusion")([upper, lower])
    fused = conv2d(fused, 192, kernel_size=1, use_mish=use_mish, weight_decay=weight_decay, name="fusion_1x1")
    fused = layers.GlobalAveragePooling2D(name="global_pool")(fused)
    fused = layers.Dense(160, kernel_regularizer=regularizers.l2(weight_decay), name="fc")(fused)
    fused = activation_layer(use_mish, name="fc_act")(fused)
    fused = layers.Dropout(dropout, name="dropout")(fused)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(fused)

    return models.Model(inputs, outputs, name="hdcdf_%s" % variant)
