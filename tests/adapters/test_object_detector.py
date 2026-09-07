import io
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from counter.adapters.object_detector import (
    FakeObjectDetector,
    ONNXObjectDetector,
    TFSObjectDetector,
    TorchVisionObjectDetector,
    YOLOObjectDetector,
)
from counter.domain.models import Box, Prediction


def _create_sample_image_bytes():
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestObjectDetectorAdapters:
    def test_fake_detector_default(self):
        detector = FakeObjectDetector()
        predictions = detector.predict(None)
        assert len(predictions) == 1
        assert predictions[0].class_name == "cat"
        assert predictions[0].score > 0.9

    def test_fake_detector_custom_predictions(self):
        custom = [Prediction("dog", 0.88, Box(0, 0, 1, 1))]
        detector = FakeObjectDetector(predictions=custom)
        predictions = detector.predict(None)
        assert len(predictions) == 1
        assert predictions[0].class_name == "dog"

    @patch("requests.post")
    def test_tfs_detector_successful_prediction(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {
                    "num_detections": 2,
                    "detection_boxes": [
                        [0.1, 0.2, 0.8, 0.9],
                        [0.0, 0.0, 0.5, 0.5],
                    ],
                    "detection_scores": [0.95, 0.80],
                    "detection_classes": [1, 2],  # person, bicycle
                }
            ]
        }
        mock_post.return_value = mock_response

        detector = TFSObjectDetector(host="localhost", port=8501, model="ssd_mobilenet_v2")
        img_bytes = _create_sample_image_bytes()
        results = detector.predict(img_bytes)

        assert len(results) == 2
        assert results[0].class_name == "person"
        assert results[0].score == 0.95
        assert results[0].box.ymin == 0.1
        assert results[0].box.xmin == 0.2
        assert results[0].box.ymax == 0.8
        assert results[0].box.xmax == 0.9

    @patch("requests.post")
    def test_tfs_detector_empty_predictions(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": []}
        mock_post.return_value = mock_response

        detector = TFSObjectDetector(host="localhost", port=8501, model="ssd_mobilenet_v2")
        img_bytes = _create_sample_image_bytes()
        results = detector.predict(img_bytes)
        assert results == []

    @patch("requests.post")
    def test_tfs_detector_network_failure(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        detector = TFSObjectDetector(host="localhost", port=8501, model="ssd_mobilenet_v2")
        img_bytes = _create_sample_image_bytes()

        with pytest.raises(RuntimeError, match="TensorFlow Serving inference failed"):
            detector.predict(img_bytes)

    def test_torchvision_detector_missing_torch_raises_runtime_error(self):
        detector = TorchVisionObjectDetector()
        img_bytes = _create_sample_image_bytes()
        with patch.dict("sys.modules", {"torch": None}):
            # if torch cannot be imported
            with pytest.raises(RuntimeError, match="PyTorch is not installed"):
                detector.predict(img_bytes)

    def test_onnx_detector_missing_runtime_raises_runtime_error(self):
        detector = ONNXObjectDetector(model_path="dummy.onnx")
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
                detector._get_session()

    def test_yolo_detector_missing_ultralytics_raises_runtime_error(self):
        detector = YOLOObjectDetector(weights_path="dummy.pt")
        img_bytes = _create_sample_image_bytes()
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(RuntimeError, match="Ultralytics is not installed"):
                detector.predict(img_bytes)
