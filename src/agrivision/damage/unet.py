from __future__ import annotations

import tensorflow as tf


def _up_block(x, skip, filters: int):
    x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x


def build_unet(
    num_classes: int = 3, image_size: int = 256, weights: str | None = "imagenet"
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """U-Net with a MobileNetV2 encoder (light enough for phones).

    Input RGB float32 [0, 255]; output per-pixel softmax over (background, leaf, lesion).
    Returns (model, encoder).
    """
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    skip_names = [
        "block_1_expand_relu",  # 1/2
        "block_3_expand_relu",  # 1/4
        "block_6_expand_relu",  # 1/8
        "block_13_expand_relu",  # 1/16
        "block_16_project",  # 1/32 (bottleneck)
    ]
    encoder = tf.keras.Model(base.input, [base.get_layer(n).output for n in skip_names], name="encoder")
    s1, s2, s3, s4, bottleneck = encoder(x)

    d = _up_block(bottleneck, s4, 256)
    d = _up_block(d, s3, 128)
    d = _up_block(d, s2, 64)
    d = _up_block(d, s1, 32)
    d = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(d)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", name="mask")(d)
    return tf.keras.Model(inputs, outputs, name="unet_mobilenetv2"), encoder


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """y_true: int mask (B,H,W) or (B,H,W,1); y_pred: softmax (B,H,W,C)."""
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.one_hot(tf.cast(tf.reshape(y_true, tf.shape(y_pred)[:-1]), tf.int32), num_classes)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (union + smooth))


def dice_ce_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(ce) + dice_loss(y_true, y_pred)
