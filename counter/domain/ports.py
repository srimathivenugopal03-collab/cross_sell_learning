from abc import ABC, abstractmethod
from typing import BinaryIO, List, Optional

from counter.domain.models import ObjectCount, Prediction


class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image: Optional[BinaryIO]) -> List[Prediction]:
        """Run object detection on the provided image binary stream."""
        raise NotImplementedError


class ObjectCountRepo(ABC):
    @abstractmethod
    def read_values(
        self, object_classes: Optional[List[str]] = None
    ) -> List[ObjectCount]:
        """Read current stored counts, optionally filtered by a list of object classes."""
        raise NotImplementedError

    @abstractmethod
    def update_values(self, new_values: List[ObjectCount]) -> None:
        """Increment count values for the specified object classes."""
        raise NotImplementedError
