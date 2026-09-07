import os
from unittest.mock import patch

from counter import config
from counter.adapters.count_repo import (
    CountInMemoryRepo,
    CountMongoDBRepo,
    CountPostgresRepo,
)
from counter.adapters.object_detector import (
    FakeObjectDetector,
    ONNXObjectDetector,
    TFSObjectDetector,
    TorchVisionObjectDetector,
    YOLOObjectDetector,
)


class TestConfigFactories:
    def test_default_detector_is_fake(self):
        with patch.dict(os.environ, {}, clear=True):
            detector = config.get_object_detector()
            assert isinstance(detector, FakeObjectDetector)

    def test_tfs_detector_selection(self):
        with patch.dict(
            os.environ,
            {"DETECTOR_TYPE": "tfs", "TFS_HOST": "tfs-server", "TFS_PORT": "8501"},
            clear=True,
        ):
            detector = config.get_object_detector()
            assert isinstance(detector, TFSObjectDetector)
            assert "tfs-server:8501" in detector.url

    def test_torch_detector_selection(self):
        with patch.dict(os.environ, {"DETECTOR_TYPE": "torch"}, clear=True):
            detector = config.get_object_detector()
            assert isinstance(detector, TorchVisionObjectDetector)

    def test_onnx_detector_selection(self):
        with patch.dict(
            os.environ,
            {"DETECTOR_TYPE": "onnx", "ONNX_MODEL_PATH": "test.onnx"},
            clear=True,
        ):
            detector = config.get_object_detector()
            assert isinstance(detector, ONNXObjectDetector)

    def test_yolo_detector_selection(self):
        with patch.dict(
            os.environ, {"DETECTOR_TYPE": "yolo", "YOLO_WEIGHTS": "yolov8n.pt"}, clear=True
        ):
            detector = config.get_object_detector()
            assert isinstance(detector, YOLOObjectDetector)

    def test_default_repo_is_memory(self):
        with patch.dict(os.environ, {}, clear=True):
            repo = config.get_count_repo()
            assert isinstance(repo, CountInMemoryRepo)

    def test_postgres_repo_selection_with_url(self):
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/test_db"},
            clear=True,
        ):
            repo = config.get_count_repo()
            assert isinstance(repo, CountPostgresRepo)

    def test_mongo_repo_selection(self):
        with patch.dict(
            os.environ,
            {"DB_TYPE": "mongo", "MONGO_HOST": "mongo-host", "MONGO_PORT": "27017"},
            clear=True,
        ):
            with patch("counter.adapters.count_repo.MongoClient"):
                repo = config.get_count_repo()
                assert isinstance(repo, CountMongoDBRepo)

    def test_get_count_action_and_predict_action(self):
        with patch.dict(os.environ, {}, clear=True):
            count_action = config.get_count_action()
            predict_action = config.get_predict_action()
            assert count_action is not None
            assert predict_action is not None
