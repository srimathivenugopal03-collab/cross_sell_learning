import io
from pathlib import Path

import pytest
from PIL import Image

from counter.domain.actions import PredictObjects
from counter.entrypoints.webapp import create_app


@pytest.fixture
def image_path():
    ref_dir = Path(__file__).parent
    return ref_dir.parent.parent / "resources" / "images" / "boy.jpg"


@pytest.fixture
def sample_image_stream():
    img = Image.new("RGB", (50, 50), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestWebAppEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "object-counter"

    def test_ready_endpoint(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ready"

    def test_object_count_success(self, client, image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()

        data = {
            "threshold": "0.9",
            "file": (io.BytesIO(image_data), "boy.jpg"),
        }
        response = client.post(
            "/object-count", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 200
        res_json = response.get_json()
        assert "current_objects" in res_json
        assert "total_objects" in res_json

    def test_predictions_endpoint_success(self, client, image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()

        data = {
            "threshold": "0.5",
            "file": (io.BytesIO(image_data), "boy.jpg"),
        }
        response = client.post(
            "/predictions", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 200
        res_json = response.get_json()
        assert isinstance(res_json, list)
        if res_json:
            first = res_json[0]
            assert "class_name" in first
            assert "score" in first
            assert "box" in first
            assert "xmin" in first["box"]

    def test_missing_file_returns_400(self, client):
        response = client.post(
            "/predictions",
            data={"threshold": "0.5"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        res_json = response.get_json()
        assert "error" in res_json
        assert "Missing 'file'" in res_json["message"]

    def test_invalid_threshold_string_returns_400(self, client, sample_image_stream):
        data = {
            "threshold": "not-a-valid-float",
            "file": (sample_image_stream, "test.jpg"),
        }
        response = client.post(
            "/predictions", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400
        res_json = response.get_json()
        assert "Invalid threshold" in res_json["message"]

    def test_threshold_out_of_range_returns_400(self, client, sample_image_stream):
        data = {
            "threshold": "1.5",
            "file": (sample_image_stream, "test.jpg"),
        }
        response = client.post(
            "/predictions", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400
        res_json = response.get_json()
        assert "Threshold must be between 0.0 and 1.0" in res_json["message"]

    def test_non_image_file_returns_400(self, client):
        text_data = io.BytesIO(b"Hello world, I am a plain text file, not an image.")
        data = {
            "threshold": "0.5",
            "file": (text_data, "document.txt"),
        }
        response = client.post(
            "/predictions", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400
        res_json = response.get_json()
        assert "Invalid image file" in res_json["message"]

    def test_not_found_returns_404(self, client):
        response = client.get("/non-existent-route")
        assert response.status_code == 404
        assert response.get_json()["error"] == "Not Found"

    def test_inference_error_returns_503(self, sample_image_stream):
        from unittest.mock import Mock

        failing_detector = Mock()
        failing_detector.predict.side_effect = RuntimeError("TFS connection timeout")

        failing_predict_action = PredictObjects(failing_detector)
        app = create_app(predict_action=failing_predict_action)
        app.config["TESTING"] = True

        with app.test_client() as test_client:
            data = {
                "threshold": "0.5",
                "file": (sample_image_stream, "test.jpg"),
            }
            response = test_client.post(
                "/predictions", data=data, content_type="multipart/form-data"
            )
            assert response.status_code == 503
            assert response.get_json()["error"] == "Service Unavailable"

    def test_rate_limit_exceeded_returns_429(self, sample_image_stream):
        # Create app with strict limit of 1 per minute for testing
        app = create_app(rate_limit_override="1 per minute")
        app.config["TESTING"] = True

        with app.test_client() as test_client:
            data = {
                "threshold": "0.5",
                "file": (sample_image_stream, "test.jpg"),
            }
            # 1st request -> 200 OK
            r1 = test_client.post(
                "/predictions", data=data, content_type="multipart/form-data"
            )
            assert r1.status_code == 200

            # 2nd request -> 429 Too Many Requests
            r2 = test_client.post(
                "/predictions",
                data={"threshold": "0.5", "file": (io.BytesIO(b"fake"), "test.jpg")},
                content_type="multipart/form-data",
            )
            assert r2.status_code == 429
            assert r2.get_json()["error"] == "Too Many Requests"
