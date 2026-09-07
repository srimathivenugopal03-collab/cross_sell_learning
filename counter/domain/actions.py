from typing import BinaryIO, Callable, List, Optional

from counter.domain.models import CountResponse, Prediction
from counter.domain.ports import ObjectCountRepo, ObjectDetector
from counter.domain.predictions import count, over_threshold


class PredictObjects:
    """Domain action to detect objects in an image and filter by confidence threshold."""

    def __init__(
        self,
        object_detector: ObjectDetector,
        debug_hook: Optional[Callable[[Optional[BinaryIO], List[Prediction], str], None]] = None,
    ):
        self.__object_detector = object_detector
        self.__debug_hook = debug_hook

    def predict(self, image: Optional[BinaryIO], threshold: float) -> List[Prediction]:
        predictions = self.__object_detector.predict(image)
        valid_predictions = list(over_threshold(predictions, threshold=threshold))
        if self.__debug_hook and image is not None:
            self.__debug_hook(
                image, valid_predictions, f"valid_predictions_with_threshold_{threshold}.jpg"
            )
        return valid_predictions


class CountDetectedObjects:
    """Domain action to count detected objects above a threshold and record occurrences in the repository."""

    def __init__(
        self,
        object_detector: ObjectDetector,
        object_count_repo: ObjectCountRepo,
        debug_hook: Optional[Callable[[Optional[BinaryIO], List[Prediction], str], None]] = None,
    ):
        self.__object_count_repo = object_count_repo
        self.__predict_objects = PredictObjects(object_detector, debug_hook=debug_hook)

    def execute(self, image: Optional[BinaryIO], threshold: float) -> CountResponse:
        valid_predictions = self.__predict_objects.predict(image, threshold)
        object_counts = count(valid_predictions)
        self.__object_count_repo.update_values(object_counts)
        total_objects = self.__object_count_repo.read_values()
        return CountResponse(current_objects=object_counts, total_objects=total_objects)
