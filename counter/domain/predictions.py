from functools import reduce
from typing import Dict, Iterable, List

from counter.domain.models import ObjectCount, Prediction


def over_threshold(predictions: List[Prediction], threshold: float) -> Iterable[Prediction]:
    """Filter a list of predictions whose confidence score is greater than or equal to threshold."""
    return filter(lambda prediction: prediction.score >= threshold, predictions)


def count(predictions: List[Prediction]) -> List[ObjectCount]:
    """Count occurrences of each object class in predictions."""
    object_classes = map(lambda prediction: prediction.class_name, predictions)
    initial_dict: Dict[str, int] = {}
    object_classes_counter: Dict[str, int] = reduce(
        __count_object_classes, object_classes, initial_dict
    )
    return [
        ObjectCount(object_class, occurrences)
        for object_class, occurrences in object_classes_counter.items()
    ]


def __count_object_classes(class_counter: Dict[str, int], object_class: str) -> Dict[str, int]:
    class_counter[object_class] = class_counter.get(object_class, 0) + 1
    return class_counter
