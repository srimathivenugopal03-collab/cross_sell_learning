import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from counter.domain.models import Box, Prediction
from counter.domain.ports import ObjectDetector

logger = logging.getLogger(__name__)


class FakeObjectDetector(ObjectDetector):
    """Deterministic fake detector for testing and local development."""

    def __init__(self, predictions: Optional[List[Prediction]] = None) -> None:
        self._predictions = predictions or [
            Prediction(
                class_name="cat",
                score=0.999190748,
                box=Box(
                    xmin=0.367288858,
                    ymin=0.278333426,
                    xmax=0.735821366,
                    ymax=0.6988855,
                ),
            ),
        ]

    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        return self._predictions


class TFSObjectDetector(ObjectDetector):
    """TensorFlow Serving Object Detector adapter via REST API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8501,
        model: str = "ssd_mobilenet_v2",
        timeout: int = 15,
        label_map_path: Optional[Path] = None,
    ) -> None:
        self.url = f"http://{host}:{port}/v1/models/{model}:predict"
        self.timeout = timeout
        label_path = label_map_path or (Path(__file__).parent / "mscoco_label_map.json")
        self.classes_dict = self.__build_classes_dict(label_path)

    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        if image is None:
            return []
        np_image = self.__to_np_array(image)
        predict_request = '{"instances" : %s}' % np.expand_dims(np_image, 0).tolist()
        logger.info(f"Sending inference request to TensorFlow Serving: {self.url}")
        try:
            response = requests.post(
                self.url,
                data=predict_request,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with TensorFlow Serving: {e}")
            raise RuntimeError(f"TensorFlow Serving inference failed: {e}") from e

        data = response.json()
        if "predictions" not in data or not data["predictions"]:
            logger.warning("Empty predictions response received from TensorFlow Serving")
            return []

        raw_predictions = data["predictions"][0]
        return self.__raw_predictions_to_domain(raw_predictions)

    @staticmethod
    def __build_classes_dict(label_path: Path) -> Dict[int, str]:
        with open(label_path, "r", encoding="utf-8") as json_file:
            labels = json.load(json_file)
            return {label["id"]: label["display_name"] for label in labels}

    @staticmethod
    def __to_np_array(image: BinaryIO) -> np.ndarray:
        current_pos = image.tell() if hasattr(image, "tell") else None
        image_ = Image.open(image).convert("RGB")
        arr = np.array(image_, dtype=np.uint8)
        if current_pos is not None and hasattr(image, "seek"):
            image.seek(current_pos)
        return arr

    def __raw_predictions_to_domain(self, raw_predictions: dict[str, Any]) -> List[Prediction]:
        num_detections = int(raw_predictions.get("num_detections", 0))
        predictions = []
        for i in range(0, num_detections):
            detection_box = raw_predictions["detection_boxes"][i]
            box = Box(
                xmin=float(detection_box[1]),
                ymin=float(detection_box[0]),
                xmax=float(detection_box[3]),
                ymax=float(detection_box[2]),
            )
            detection_score = float(raw_predictions["detection_scores"][i])
            detection_class = int(raw_predictions["detection_classes"][i])
            class_name = self.classes_dict.get(detection_class, f"class_{detection_class}")
            predictions.append(Prediction(class_name=class_name, score=detection_score, box=box))
        return predictions


class TorchVisionObjectDetector(ObjectDetector):
    """PyTorch / TorchVision Object Detector Adapter."""

    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._classes: Dict[int, str] = {}

    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        if image is None:
            return []
        try:
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Install torch and torchvision to use TorchVisionObjectDetector."
            )

        current_pos = image.tell() if hasattr(image, "tell") else None
        pil_img = Image.open(image).convert("RGB")
        if current_pos is not None and hasattr(image, "seek"):
            image.seek(current_pos)

        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        if self._model is None:
            logger.warning("TorchVision model not initialized with weights, returning empty.")
            return []

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(img_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        w, h = pil_img.size
        predictions = []
        for b, s, lbl in zip(boxes, scores, labels):
            box = Box(
                xmin=float(b[0] / w),
                ymin=float(b[1] / h),
                xmax=float(b[2] / w),
                ymax=float(b[3] / h),
            )
            class_name = self._classes.get(int(lbl), f"class_{lbl}")
            predictions.append(Prediction(class_name=class_name, score=float(s), box=box))
        return predictions


class ONNXObjectDetector(ObjectDetector):
    """ONNX Runtime Object Detector Adapter for high-performance cross-platform inference."""

    def __init__(
        self,
        model_path: str,
        classes: Optional[Dict[int, str]] = None,
        input_shape: Tuple[int, int] = (300, 300),
    ) -> None:
        self.model_path = model_path
        self.classes = classes or {}
        self.input_shape = input_shape
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort  # type: ignore

                self._session = ort.InferenceSession(self.model_path)
            except ImportError:
                raise RuntimeError(
                    "onnxruntime is not installed. Install onnxruntime to use ONNXObjectDetector."
                )
        return self._session

    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        if image is None:
            return []
        session = self._get_session()
        current_pos = image.tell() if hasattr(image, "tell") else None
        pil_img = Image.open(image).convert("RGB")
        if current_pos is not None and hasattr(image, "seek"):
            image.seek(current_pos)

        resized = pil_img.resize(self.input_shape)
        input_data = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        predictions: List[Prediction] = []
        if outputs and len(outputs) >= 3:
            boxes, scores, labels = outputs[0], outputs[1], outputs[2]
            for b, s, lbl in zip(boxes[0], scores[0], labels[0]):
                class_name = self.classes.get(int(lbl), f"class_{lbl}")
                predictions.append(
                    Prediction(
                        class_name=class_name,
                        score=float(s),
                        box=Box(
                            xmin=float(b[0]), ymin=float(b[1]), xmax=float(b[2]), ymax=float(b[3])
                        ),
                    )
                )
        return predictions


class YOLOObjectDetector(ObjectDetector):
    """YOLO / Ultralytics Framework Adapter."""

    def __init__(self, weights_path: str = "yolov8n.pt") -> None:
        self.weights_path = weights_path
        self._model: Any = None

    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        if image is None:
            return []
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Ultralytics is not installed. Install ultralytics to use YOLOObjectDetector."
            )

        if self._model is None:
            self._model = YOLO(self.weights_path)

        current_pos = image.tell() if hasattr(image, "tell") else None
        pil_img = Image.open(image).convert("RGB")
        if current_pos is not None and hasattr(image, "seek"):
            image.seek(current_pos)

        predictions: List[Prediction] = []
        if self._model is not None:
            results = self._model(pil_img)
            for r in results:
                for box, score, cls_id in zip(
                    r.boxes.xyxyn.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                ):
                    predictions.append(
                        Prediction(
                            class_name=r.names[int(cls_id)],
                            score=float(score),
                            box=Box(
                                xmin=float(box[0]),
                                ymin=float(box[1]),
                                xmax=float(box[2]),
                                ymax=float(box[3]),
                            ),
                        )
                    )
        return predictions
