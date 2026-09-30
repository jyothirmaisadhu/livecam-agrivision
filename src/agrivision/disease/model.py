from __future__ import annotations

import tensorflow as tf


def build_classifier(
    num_classes: int,
    image_size: int = 224,
    variant: str = "large",
    weights: str | None = "imagenet",
    dropout: float = 0.3,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """MobileNetV3 backbone + new softmax head. Input: RGB float32 in [0, 255].

    Returns (model, backbone) so the caller can freeze / unfreeze the backbone.
    """
    app = (
        tf.keras.applications.MobileNetV3Large
        if variant == "large"
        else tf.keras.applications.MobileNetV3Small
    )
    backbone = app(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
        include_preprocessing=True,
    )
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease")(x)
    return tf.keras.Model(inputs, outputs, name=f"mobilenetv3_{variant}_disease"), backbone


def augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="augment",
    )
