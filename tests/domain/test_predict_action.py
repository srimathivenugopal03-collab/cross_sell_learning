from unittest.mock import Mock

import pytest

from counter.domain.actions import PredictObjects
from tests.domain.helpers import generate_prediction


class TestPredictObjects:
    @pytest.fixture
    def object_detector(self) -> Mock:
        detector = Mock()
        detector.predict.return_value = [
            generate_prediction("cat", 0.95),
            generate_prediction("cat", 0.40),
            generate_prediction("dog", 0.85),
            generate_prediction("bird", 0.10),
        ]
        return detector

    def test_predict_filters_by_threshold(self, object_detector: Mock) -> None:
        action = PredictObjects(object_detector)
        results = action.predict(None, threshold=0.80)

        assert len(results) == 2
        classes = [p.class_name for p in results]
        assert "cat" in classes
        assert "dog" in classes
        assert "bird" not in classes

    def test_predict_returns_all_when_threshold_zero(
        self, object_detector: Mock
    ) -> None:
        action = PredictObjects(object_detector)
        results = action.predict(None, threshold=0.0)

        assert len(results) == 4

    def test_predict_returns_empty_when_threshold_unreachable(
        self, object_detector: Mock
    ) -> None:
        action = PredictObjects(object_detector)
        results = action.predict(None, threshold=0.99)

        assert len(results) == 0

    def test_predict_with_debug_hook(self, object_detector: Mock) -> None:
        hook_called = False

        def mock_hook(image, predictions, filename):
            nonlocal hook_called
            hook_called = True

        mock_image = Mock()
        action = PredictObjects(object_detector, debug_hook=mock_hook)
        action.predict(mock_image, threshold=0.5)

        assert hook_called is True
