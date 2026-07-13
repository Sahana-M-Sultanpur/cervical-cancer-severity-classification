"""ResNet50 transfer-learning branch for high-level visual representation."""

from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from backend.config import IMAGE_SIZE


def build_resnet_feature_extractor(
    input_shape=(*IMAGE_SIZE, 3),
    feature_dim: int = 256,
    fine_tune_layers: int = 25,
) -> Model:
    """Build a ResNet50 feature extractor with a custom projection head."""
    inputs = Input(shape=input_shape, name="resnet_input")
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    features = Dense(feature_dim, activation="relu", name="resnet_features")(x)

    return Model(inputs, features, name="resnet50_feature_extractor")
