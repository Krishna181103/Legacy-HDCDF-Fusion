import tensorflow as tf
from tensorflow.keras import layers, models


def convnext_block(x, filters, drop_rate=0.2):
    shortcut = x

    x = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(4 * filters, activation="gelu")(x)
    x = layers.Dense(filters)(x)
    x = layers.Dropout(drop_rate)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding="same")(shortcut)

    return layers.Add()([shortcut, x])


def build_convnext(input_shape=(128, 128, 16), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    for filters in [32, 64, 128]:
        x = convnext_block(x, filters)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    reg_out = layers.Dense(4, name="regression")(x)
    cls_out = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    model = models.Model(inputs, [reg_out, cls_out])
    return model
