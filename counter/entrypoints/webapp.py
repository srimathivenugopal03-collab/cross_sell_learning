import logging
import os
from dataclasses import asdict
from io import BytesIO
from typing import Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image

from counter import config
from counter.domain.actions import CountDetectedObjects, PredictObjects

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)


def _extract_and_validate_request() -> (
    Tuple[Optional[BytesIO], Optional[float], Optional[str]]
):
    """Extract and validate file and threshold parameters from the multipart form request."""
    if "file" not in request.files:
        return None, None, "Missing 'file' in multipart form data"

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return None, None, "No file selected for upload"

    threshold_raw = request.form.get("threshold", 0.5)
    try:
        threshold = float(threshold_raw)
    except (ValueError, TypeError):
        return (
            None,
            None,
            f"Invalid threshold value '{threshold_raw}'. Must be a float.",
        )

    if not (0.0 <= threshold <= 1.0):
        return None, None, f"Threshold must be between 0.0 and 1.0, got {threshold}"

    image_bytes = BytesIO()
    uploaded_file.save(image_bytes)
    image_bytes.seek(0)

    # Validate that the file is indeed a valid readable image
    try:
        with Image.open(image_bytes) as img:
            img.verify()
    except Exception as e:
        return None, None, f"Invalid image file: {e}"

    image_bytes.seek(0)
    return image_bytes, threshold, None


def create_app(
    count_action: Optional[CountDetectedObjects] = None,
    predict_action: Optional[PredictObjects] = None,
    rate_limit_override: Optional[str] = None,
) -> Flask:
    """Application factory for the Object Counter Flask service with Rate Limiting."""
    app = Flask(__name__)
    config.setup_logging()

    _count_action = count_action or config.get_count_action()
    _predict_action = predict_action or config.get_predict_action()

    # Rate Limiting Configuration (Entrypoint Protection for ML Compute)
    rate_limit_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    default_limit = rate_limit_override or os.environ.get(
        "RATE_LIMIT_DEFAULT", "60 per minute"
    )
    storage_uri = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[default_limit] if rate_limit_enabled else [],
        storage_uri=storage_uri,
        headers_enabled=True,
    )

    @app.route("/health", methods=["GET"])
    @limiter.exempt
    def health():
        return jsonify({"status": "ok", "service": "object-counter"}), 200

    @app.route("/ready", methods=["GET"])
    @limiter.exempt
    def ready():
        return jsonify({"status": "ready", "service": "object-counter"}), 200

    @app.route("/object-count", methods=["POST"])
    def object_detection():
        image, threshold, error_msg = _extract_and_validate_request()
        if error_msg:
            return jsonify({"error": "Bad Request", "message": error_msg}), 400

        try:
            count_response = _count_action.execute(image, threshold)
            return jsonify(asdict(count_response)), 200
        except RuntimeError as e:
            logger.error(f"Inference error in /object-count: {e}")
            return jsonify({"error": "Service Unavailable", "message": str(e)}), 503
        except Exception as e:
            logger.exception(f"Unexpected error in /object-count: {e}")
            return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

    @app.route("/predictions", methods=["POST"])
    def predictions():
        image, threshold, error_msg = _extract_and_validate_request()
        if error_msg:
            return jsonify({"error": "Bad Request", "message": error_msg}), 400

        try:
            preds = _predict_action.predict(image, threshold)
            return (
                jsonify([asdict(prediction) for prediction in preds]),
                200,
            )
        except RuntimeError as e:
            logger.error(f"Inference error in /predictions: {e}")
            return jsonify({"error": "Service Unavailable", "message": str(e)}), 503
        except Exception as e:
            logger.exception(f"Unexpected error in /predictions: {e}")
            return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return (
            jsonify(
                {"error": "Not Found", "message": "The requested URL was not found"}
            ),
            404,
        )

    @app.errorhandler(429)
    def ratelimit_handler(error):
        return (
            jsonify(
                {
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded: {error.description}",
                }
            ),
            429,
        )

    @app.errorhandler(500)
    def internal_error(error):
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An internal error occurred",
                }
            ),
            500,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host, port=port, debug=True)
