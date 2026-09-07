import io

import pytest
from PIL import Image

from counter.adapters.count_repo import CountInMemoryRepo
from counter.adapters.object_detector import FakeObjectDetector
from counter.domain.actions import CountDetectedObjects, PredictObjects
from counter.domain.models import Box, Prediction
from counter.entrypoints.webapp import create_app


@pytest.fixture
def test_image():
    img = Image.new("RGB", (64, 64), color="yellow")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestEndToEndWorkflow:
    def test_e2e_predictions_and_cumulative_counts(self, test_image):
        # Setup multi-object fake detector
        predictions_mock = [
            Prediction("cat", 0.95, Box(0.1, 0.1, 0.4, 0.4)),
            Prediction("cat", 0.85, Box(0.5, 0.5, 0.8, 0.8)),
            Prediction("dog", 0.70, Box(0.2, 0.2, 0.6, 0.6)),
            Prediction("bird", 0.30, Box(0.0, 0.0, 0.2, 0.2)),
        ]
        detector = FakeObjectDetector(predictions=predictions_mock)
        repo = CountInMemoryRepo()

        count_action = CountDetectedObjects(detector, repo)
        predict_action = PredictObjects(detector)

        app = create_app(count_action=count_action, predict_action=predict_action)
        app.config["TESTING"] = True

        with app.test_client() as client:
            # 1. Test Prediction endpoint at threshold 0.8
            resp1 = client.post(
                "/predictions",
                data={
                    "threshold": "0.8",
                    "file": (io.BytesIO(test_image.getvalue()), "img1.jpg"),
                },
                content_type="multipart/form-data",
            )
            assert resp1.status_code == 200
            pred_data = resp1.get_json()
            assert isinstance(pred_data, list)
            assert len(pred_data) == 2
            classes = [p["class_name"] for p in pred_data]
            assert classes == ["cat", "cat"]

            # 2. First count request at threshold 0.65 -> counts 2 cats and 1 dog
            resp2 = client.post(
                "/object-count",
                data={
                    "threshold": "0.65",
                    "file": (io.BytesIO(test_image.getvalue()), "img2.jpg"),
                },
                content_type="multipart/form-data",
            )
            assert resp2.status_code == 200
            count_data1 = resp2.get_json()
            cur_map1 = {
                item["object_class"]: item["count"]
                for item in count_data1["current_objects"]
            }
            tot_map1 = {
                item["object_class"]: item["count"]
                for item in count_data1["total_objects"]
            }
            assert cur_map1["cat"] == 2
            assert cur_map1["dog"] == 1
            assert tot_map1["cat"] == 2
            assert tot_map1["dog"] == 1

            # 3. Second count request at threshold 0.65 -> accumulates counts
            resp3 = client.post(
                "/object-count",
                data={
                    "threshold": "0.65",
                    "file": (io.BytesIO(test_image.getvalue()), "img3.jpg"),
                },
                content_type="multipart/form-data",
            )
            assert resp3.status_code == 200
            count_data2 = resp3.get_json()
            tot_map2 = {
                item["object_class"]: item["count"]
                for item in count_data2["total_objects"]
            }
            assert tot_map2["cat"] == 4
            assert tot_map2["dog"] == 2
