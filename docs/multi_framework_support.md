# Multi-Framework Deep Learning Support in Hexagonal Architecture

## 1. Overview
The Object Counter application leverages **Hexagonal Architecture (Ports and Adapters)** to decouple core domain logic from the underlying deep learning frameworks. 

By defining the `ObjectDetector` port interface, any machine learning framework (PyTorch, TensorFlow, ONNX Runtime, Ultralytics YOLO, TensorRT) can be integrated as an independent adapter without altering business rules, counting logic, or API routing.

---

## 2. The `ObjectDetector` Port Interface

The domain layer depends strictly on the abstract port:

```python
# counter/domain/ports.py
from abc import ABC, abstractmethod
from typing import BinaryIO, List
from counter.domain.models import Prediction

class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image: BinaryIO) -> List[Prediction]:
        """Run object detection on the provided image binary stream and return standardized Prediction domain objects."""
        raise NotImplementedError
```

Every adapter is responsible for:
1. Decoding the binary image stream into the tensor/array representation required by the target framework.
2. Executing inference.
3. Normalizing framework-specific bounding boxes (`[ymin, xmin, ymax, xmax]`, `[x, y, w, h]`, or normalized `[0.0, 1.0]`) into the domain `Box(xmin, ymin, xmax, ymax)` standard.
4. Mapping numeric class IDs to string class names.

---

## 3. Supported Framework Adapters

### 3.1 TensorFlow / TensorFlow Serving (`TFSObjectDetector`)
- **Protocol**: HTTP/REST or gRPC.
- **Input Tensor**: `[1, height, width, 3]` `uint8`.
- **Output**: Returns `detection_boxes` (normalized `[ymin, xmin, ymax, xmax]`), `detection_scores`, `detection_classes`.
- **Use Case**: Production deployment with dedicated TensorFlow Serving daemon.

### 3.2 PyTorch / TorchVision (`TorchVisionObjectDetector`)
- **Protocol**: In-process Python C++ engine (or TorchScript / TorchServe).
- **Input Tensor**: `[1, 3, height, width]` normalized `float32` `[0.0, 1.0]` on CPU or CUDA GPU.
- **Supported Architectures**: Faster R-CNN, RetinaNet, Mask R-CNN, SSD.
- **Transformation Pipeline**:
  ```python
  img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
  outputs = model(img_tensor.unsqueeze(0))
  # outputs contain: 'boxes' [x1, y1, x2, y2] in pixels, 'scores', 'labels'
  ```

### 3.3 ONNX Runtime (`ONNXObjectDetector`)
- **Protocol**: High-performance, cross-platform C++ inference runtime via Python bindings.
- **Input Tensor**: Standardized NCHW or NHWC tensor formats.
- **Use Case**: Edge deployments, microservices requiring minimal memory footprint, cross-hardware portability (Intel OpenVINO, AMD ROCm, NVIDIA TensorRT execution providers).

### 3.4 Ultralytics YOLO (`YOLOObjectDetector`)
- **Protocol**: In-process inference using YOLOv8 / YOLOv9 / YOLOv10 / YOLO11.
- **Input**: Raw PIL Image or NumPy array.
- **Output**: Direct extraction of normalized bounding box coordinates (`xyxyn`), class names, and confidence scores.

---

## 4. Normalization and Coordinate Space Mapping

Different frameworks output bounding boxes in varying coordinate spaces. The adapter layer ensures uniform conversion to the domain `Box`:

| Framework | Raw Coordinate Format | Transformation to Domain `Box` |
| :--- | :--- | :--- |
| **TensorFlow Detection API** | Normalized `[ymin, xmin, ymax, xmax]` | `Box(xmin=b[1], ymin=b[0], xmax=b[3], ymax=b[2])` |
| **PyTorch / TorchVision** | Absolute Pixels `[x1, y1, x2, y2]` | `Box(xmin=x1/W, ymin=y1/H, xmax=x2/W, ymax=y2/H)` |
| **YOLO (Ultralytics)** | Normalized `[x1, y1, x2, y2]` via `xyxyn` | `Box(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3])` |
| **OpenCV DNN / SSD** | `[batch, class, score, x1, y1, x2, y2]` | `Box(xmin=b[3], ymin=b[4], xmax=b[5], ymax=b[6])` |

---

## 5. Extensibility Guide: Adding a New Framework Adapter

To add a new framework (e.g., Google MediaPipe, HuggingFace Transformers, or AWS Rekognition):

1. **Create Adapter Class**:
   ```python
   # counter/adapters/mediapipe_detector.py
   from counter.domain.ports import ObjectDetector
   from counter.domain.models import Prediction, Box

   class MediaPipeObjectDetector(ObjectDetector):
       def __init__(self, model_path: str):
           # initialize MediaPipe Vision detector
           pass

       def predict(self, image: BinaryIO) -> List[Prediction]:
           # 1. Decode image
           # 2. Run inference
           # 3. Convert to List[Prediction]
           return predictions
   ```

2. **Register in Configuration**:
   Add factory resolution in `counter/config.py` under `get_object_detector()`.

3. **Write Unit Tests**:
   Create a test suite in `tests/adapters/test_object_detector.py` mocking inference and verifying output matches the `Prediction` domain specification.
