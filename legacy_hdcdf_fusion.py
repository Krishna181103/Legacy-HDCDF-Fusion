"""Legacy-HDCDF fusion model for high-accuracy HyperLeaf classification.

This architecture keeps the project's strongest legacy 3D PCA backbone and
adds selected HDCDF ideas in the late 2D refinement stage: Mish activation,
2D-DCAttention, and depth-wise separable convolution. It is intentionally not a
pure HDCDF reproduction; it is a dataset-specific fusion methodology designed
to improve the existing 87.74% baseline without discarding its proven feature
extractor.
"""

from tensorflow.keras import layers, models, regularizers

from models.hdcdf_components import activation_layer, dc_attention_2d, dsc2d_bn_act


DEFAULT_BRANCH_KERNELS = {
    "full": [(3, 3, 3), (3, 3, 5), (1, 1, 7)],
    "k333": [(3, 3, 3)],
    "k335": [(3, 3, 5)],
    "k117": [(1, 1, 7)],
}


def se3d_block(x, ratio=8, name="se3d"):
    channels = int(x.shape[-1])
    hidden = max(channels // ratio, 8)
    s = layers.GlobalAveragePooling3D(name=name + "_gap")(x)
    s = layers.Dense(hidden, activation="relu", name=name + "_fc1")(s)
    s = layers.Dense(channels, activation="sigmoid", name=name + "_fc2")(s)
    s = layers.Reshape((1, 1, 1, channels), name=name + "_reshape")(s)
    return layers.Multiply(name=name + "_scale")([x, s])


def conv3d_bn_act(x, filters, kernel_size, use_mish=False, weight_decay=1e-4, name=None):
    prefix = "" if name is None else name + "_"
    x = layers.Conv3D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=prefix + "conv3d",
    )(x)
    x = layers.BatchNormalization(name=prefix + "bn")(x)
    x = activation_layer(use_mish, name=prefix + "act")(x)
    return x


def residual3d_legacy(x, filters, use_mish=False, weight_decay=1e-4, name="res3d"):
    shortcut = x
    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv3D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=name + "_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=name + "_shortcut_bn")(shortcut)

    y = conv3d_bn_act(
        x,
        filters,
        (3, 3, 3),
        use_mish=use_mish,
        weight_decay=weight_decay,
        name=name + "_main1",
    )
    y = layers.Conv3D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + "_main2_conv3d",
    )(y)
    y = layers.BatchNormalization(name=name + "_main2_bn")(y)
    y = layers.Add(name=name + "_add")([shortcut, y])
    return activation_layer(use_mish, name=name + "_out_act")(y)


def build_legacy_hdcdf_fusion_model(
    input_shape=(48, 48, 24, 1),
    num_classes=4,
    branch_variant="full",
    dropout=0.45,
    weight_decay=1e-4,
    use_mish=True,
):
    """Build the proposed legacy + HDCDF fusion model.

    Shape flow:
    (B, 48, 48, 24, 1) -> legacy multi-kernel 3D stem/residual backbone
    -> (B, 12, 12, 12, 64) -> reshape to 2D channels
    -> HDCDF 2D-DCAttention -> DSC refinement -> classifier.
    """
    if branch_variant not in DEFAULT_BRANCH_KERNELS:
        raise ValueError("Unknown branch_variant=%s" % branch_variant)

    crop_size = input_shape[0]
    pca_components = input_shape[2]
    if crop_size % 4 != 0 or pca_components % 2 != 0:
        raise ValueError("Expected crop_size divisible by 4 and PCA components divisible by 2.")

    inputs = layers.Input(shape=input_shape, name="hsi_patch")

    branches = [
        conv3d_bn_act(
            inputs,
            16,
            kernel,
            use_mish=use_mish,
            weight_decay=weight_decay,
            name="legacy_stem_%d" % idx,
        )
        for idx, kernel in enumerate(DEFAULT_BRANCH_KERNELS[branch_variant])
    ]
    x = branches[0] if len(branches) == 1 else layers.Concatenate(name="legacy_multikernel_concat")(branches)

    x = conv3d_bn_act(x, 32, (3, 3, 3), use_mish=use_mish, weight_decay=weight_decay, name="legacy_3d_expand")
    x = layers.MaxPool3D(pool_size=(2, 2, 1), name="legacy_pool_spatial")(x)
    x = residual3d_legacy(x, 48, use_mish=use_mish, weight_decay=weight_decay, name="legacy_res3d_1")
    x = layers.MaxPool3D(pool_size=(2, 2, 2), name="legacy_pool_spatial_spectral")(x)
    x = residual3d_legacy(x, 64, use_mish=use_mish, weight_decay=weight_decay, name="legacy_res3d_2")
    x = se3d_block(x, ratio=8, name="legacy_se3d")

    spatial_after = crop_size // 4
    spectral_after = pca_components // 2
    x = layers.Reshape((spatial_after, spatial_after, spectral_after * 64), name="spectral_to_2d_channels")(x)

    # HDCDF-inspired late refinement: spatial/spectral attention followed by DSC.
    x = layers.Conv2D(160, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name="fusion_reduce_conv")(x)
    x = layers.BatchNormalization(name="fusion_reduce_bn")(x)
    x = activation_layer(use_mish, name="fusion_reduce_act")(x)
    x = dc_attention_2d(x, filters=160, ratio=8, use_mish=use_mish, weight_decay=weight_decay, name="hdcdf_dcattention")
    x = dsc2d_bn_act(x, 192, kernel_size=3, use_mish=use_mish, weight_decay=weight_decay, name="hdcdf_dsc_refine")

    shortcut = layers.Conv2D(192, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name="refine_shortcut_conv")(x)
    shortcut = layers.BatchNormalization(name="refine_shortcut_bn")(shortcut)
    y = dsc2d_bn_act(x, 192, kernel_size=3, use_mish=use_mish, weight_decay=weight_decay, name="hdcdf_dsc_residual")
    x = layers.Add(name="hdcdf_refine_add")([shortcut, y])
    x = activation_layer(use_mish, name="hdcdf_refine_out_act")(x)

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(192, kernel_regularizer=regularizers.l2(weight_decay), name="classifier_fc1")(x)
    x = activation_layer(use_mish, name="classifier_fc1_act")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    x = layers.Dense(96, kernel_regularizer=regularizers.l2(weight_decay), name="classifier_fc2")(x)
    x = activation_layer(use_mish, name="classifier_fc2_act")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    return models.Model(inputs, outputs, name="legacy_hdcdf_fusion")
