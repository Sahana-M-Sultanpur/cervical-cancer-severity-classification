"""Hybrid CNN + ResNet classifier used after YOLOv5 region detection."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from backend.config import IMAGE_SIZE
from backend.models.cnn_model import build_cnn_feature_extractor
from backend.models.resnet_model import build_resnet_feature_extractor


def build_hybrid_model(
    num_classes: int,
    input_shape=(*IMAGE_SIZE, 3),
    learning_rate: float = 1e-4,
) -> Model:
    """Build and compile the hybrid classifier."""
    inputs = Input(shape=input_shape, name="pap_smear_image")

    cnn_branch = build_cnn_feature_extractor(input_shape=input_shape)
    resnet_branch = build_resnet_feature_extractor(input_shape=input_shape)

    cnn_features = cnn_branch(inputs)
    resnet_features = resnet_branch(inputs)

    x = Concatenate(name="hybrid_feature_fusion")([cnn_features, resnet_features])
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation="softmax", name="severity_prediction")(x)

    model = Model(inputs, outputs, name="cnn_resnet_hybrid_classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
