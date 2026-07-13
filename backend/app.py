"""Flask API for hybrid cervical cancer severity classification."""

from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.config import ALLOWED_IMAGE_EXTENSIONS, STORAGE_DIR, UPLOAD_DIR, ensure_directories
from backend.utils.dataset_utils import prepare_dataset_from_zip
from backend.utils.logger import get_logger


ensure_directories()
app = Flask(__name__)
@app.route('/')
def home():
    return {
        "message": "Cervical Cancer Detection Backend Running"
    }
CORS(app)
logger = get_logger(__name__)


def error_response(message: str, status_code: int = 400):
    """Return a consistent JSON error response."""
    logger.error(message)
    return jsonify({"error": message}), status_code


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    """Catch unexpected errors and prevent raw tracebacks in API responses."""
    logger.exception("Unhandled backend error: %s", exc)
    return jsonify({"error": str(exc)}), 500


@app.get("/health")
def health_check():
    """Simple health endpoint for deployment checks."""
    return jsonify(
        {
            "status": "ok",
            "message": "Backend is running.",
            "storage_dir": str(STORAGE_DIR),
            "dataset_mode": "zip_stream_no_raw_extract",
        }
    )


@app.post("/upload-dataset")
def upload_dataset():
    """Upload or reference a ZIP dataset, then extract, clean, and split it."""
    zip_path = None

    if "dataset" in request.files:
        dataset_file = request.files["dataset"]
        if not dataset_file.filename:
            return error_response("No dataset filename provided.")

        filename = secure_filename(dataset_file.filename)
        zip_path = UPLOAD_DIR / filename
        dataset_file.save(zip_path)
        logger.info("Received uploaded dataset: %s", zip_path)
    else:
        payload = request.get_json(silent=True) or {}
        local_zip_path = payload.get("zip_path")
        if local_zip_path:
            zip_path = Path(local_zip_path)

    if zip_path is None:
        return error_response(
            "Provide a multipart file named 'dataset' or JSON {'zip_path': 'D:/archive.zip'}."
        )

    result = prepare_dataset_from_zip(Path(zip_path))
    return jsonify(result)


@app.post("/train")
def train():
    """Train the hybrid CNN + ResNet classifier."""
    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get("epochs", 20))
    batch_size = int(payload.get("batch_size", 16))

    from backend.train import train_model

    result = train_model(epochs=epochs, batch_size=batch_size)
    return jsonify(result)


@app.post("/predict")
def predict():
    """Predict severity from a Pap smear image."""
    if "image" not in request.files:
        return error_response("Upload an image file using multipart field name 'image'.")

    image_file = request.files["image"]
    if not image_file.filename:
        return error_response("No image filename provided.")

    extension = Path(image_file.filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return error_response(f"Unsupported image extension: {extension}")

    filename = f"{uuid4().hex}_{secure_filename(image_file.filename)}"
    image_path = UPLOAD_DIR / filename
    image_file.save(image_path)

    from backend.predict import predict_image

    result = predict_image(image_path)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
