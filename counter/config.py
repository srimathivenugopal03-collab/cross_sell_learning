import logging
import os
from typing import Optional

from dotenv import load_dotenv

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
from counter.debug import debug_image_hook
from counter.domain.actions import CountDetectedObjects, PredictObjects
from counter.domain.ports import ObjectCountRepo, ObjectDetector

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(level: Optional[str] = None):
    log_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_object_detector() -> ObjectDetector:
    detector_type = os.environ.get("DETECTOR_TYPE")
    env = os.environ.get("ENV", "dev").lower()

    if detector_type == "tfs" or (not detector_type and env == "prod"):
        tfs_host = os.environ.get("TFS_HOST", "localhost")
        tfs_port = int(os.environ.get("TFS_PORT", 8501))
        model_name = os.environ.get("MODEL_NAME", "ssd_mobilenet_v2")
        timeout = int(os.environ.get("TFS_TIMEOUT", 15))
        return TFSObjectDetector(
            host=tfs_host, port=tfs_port, model=model_name, timeout=timeout
        )

    elif detector_type == "torch":
        model_name = os.environ.get("MODEL_NAME", "fasterrcnn_resnet50_fpn")
        device = os.environ.get("TORCH_DEVICE", "cpu")
        return TorchVisionObjectDetector(model_name=model_name, device=device)

    elif detector_type == "onnx":
        model_path = os.environ.get("ONNX_MODEL_PATH", "models/detector.onnx")
        return ONNXObjectDetector(model_path=model_path)

    elif detector_type == "yolo":
        weights = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
        return YOLOObjectDetector(weights_path=weights)

    else:
        return FakeObjectDetector()


def get_count_repo() -> ObjectCountRepo:
    db_type = os.environ.get("DB_TYPE", "").lower()
    env = os.environ.get("ENV", "dev").lower()
    has_db_url = bool(os.environ.get("DATABASE_URL"))

    if db_type == "postgres" or (not db_type and (env == "prod" or has_db_url)):
        uri = os.environ.get("DATABASE_URL")
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = int(os.environ.get("POSTGRES_PORT", 5432))
        db = os.environ.get("POSTGRES_DB", "counter_db")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "postgres")
        return CountPostgresRepo(
            host=host, port=port, database=db, user=user, password=password, uri=uri
        )

    elif db_type == "mongo" or (not db_type and env == "prod"):
        mongo_uri = os.environ.get("MONGO_URI")
        mongo_host = os.environ.get("MONGO_HOST", "localhost")
        mongo_port = int(os.environ.get("MONGO_PORT", 27017))
        mongo_db = os.environ.get("MONGO_DB", "prod_counter")
        return CountMongoDBRepo(
            host=mongo_host, port=mongo_port, database=mongo_db, uri=mongo_uri
        )

    else:
        return CountInMemoryRepo()


def get_debug_hook():
    enable_debug_draw = os.environ.get("ENABLE_DEBUG_DRAW", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    return debug_image_hook if enable_debug_draw else None


def get_count_action() -> CountDetectedObjects:
    detector = get_object_detector()
    repo = get_count_repo()
    return CountDetectedObjects(detector, repo, debug_hook=get_debug_hook())


def get_predict_action() -> PredictObjects:
    detector = get_object_detector()
    return PredictObjects(detector, debug_hook=get_debug_hook())


# Backward-compatibility helpers
def dev_count_action() -> CountDetectedObjects:
    return CountDetectedObjects(FakeObjectDetector(), CountInMemoryRepo())
