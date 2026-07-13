"""Custom CNN branch for local texture and cell morphology features."""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from backend.config import IMAGE_SIZE


def build_cnn_feature_extractor(input_shape=(*IMAGE_SIZE, 3), feature_dim: int = 256) -> Model:
    """Build a custom CNN that outputs a compact feature vector."""
    inputs = Input(shape=input_shape, name="cnn_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.45)(x)
    features = Dense(feature_dim, activation="relu", name="cnn_features")(x)

    return Model(inputs, features, name="custom_cnn_feature_extractor")
