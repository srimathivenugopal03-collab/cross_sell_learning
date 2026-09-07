from pathlib import Path
from typing import BinaryIO, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont

from counter.domain.models import Prediction


def draw(predictions: List[Prediction], image: Image.Image, image_name: str) -> None:
    draw_image = ImageDraw.Draw(image, "RGBA")
    image_width, image_height = image.size

    font_path = Path(__file__).parent / "resources" / "arial.ttf"
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]
    try:
        font = ImageFont.truetype(str(font_path), 20)
    except Exception:
        font = ImageFont.load_default()

    for prediction in predictions:
        box = prediction.box
        draw_image.rectangle(
            [
                (box.xmin * image_width, box.ymin * image_height),
                (box.xmax * image_width, box.ymax * image_height),
            ],
            outline="red",
            width=2,
        )
        class_name = prediction.class_name
        text_y = max(0.0, box.ymin * image_height - 20)
        draw_image.text(
            (box.xmin * image_width, text_y),
            f"{class_name}: {prediction.score:.2f}",
            font=font,
            fill="black",
        )

    debug_dir = Path("tmp/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    image.save(debug_dir / image_name, "JPEG")


def debug_image_hook(
    image_stream: Optional[BinaryIO], predictions: List[Prediction], image_name: str
) -> None:
    """Helper hook function for debug visualization."""
    if not __debug__ or image_stream is None:
        return
    try:
        current_pos = image_stream.tell() if hasattr(image_stream, "tell") else None
        image = Image.open(image_stream)
        draw(predictions, image, image_name)
        if current_pos is not None and hasattr(image_stream, "seek"):
            image_stream.seek(current_pos)
    except Exception:
        # Debug drawing shouldn't crash the application
        pass
