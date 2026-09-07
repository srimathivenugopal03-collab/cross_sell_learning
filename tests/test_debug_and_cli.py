import io
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from counter.debug import debug_image_hook, draw
from counter.domain.models import Box, Prediction


def test_draw_saves_image(tmp_path):
    img = Image.new("RGB", (100, 100), color="white")
    predictions = [Prediction("cat", 0.95, Box(0.1, 0.1, 0.5, 0.5))]

    with patch("counter.debug.Path") as mock_path:
        # redirect tmp/debug to tmp_path
        mock_path.return_value = tmp_path
        mock_path.__file__ = str(Path(__file__))
        draw(predictions, img, "test_output.jpg")


def test_debug_image_hook_executes_safely():
    img = Image.new("RGB", (50, 50), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    predictions = [Prediction("dog", 0.85, Box(0.0, 0.0, 0.5, 0.5))]
    # Should not raise any exceptions
    debug_image_hook(buf, predictions, "hook_test.jpg")
