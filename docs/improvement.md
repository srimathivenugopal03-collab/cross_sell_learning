# Object Counter & ML Platform – Improvements

## 1. Assignment Requirements

The following improvements were implemented in the Object Counter & ML Platform:

| Requirement            | Status     | Summary                                                                                                    |
| ---------------------- | ---------- | ---------------------------------------------------------------------------------------------------------- |
| Prediction API         | ✅ Complete | Added `POST /predictions` to return detected objects, confidence scores, and bounding boxes.               |
| PostgreSQL Support     | ✅ Complete | Added a PostgreSQL repository using SQLAlchemy with atomic upsert support.                                 |
| Code Review            | ✅ Complete | Reviewed the existing codebase and identified validation, coupling, configuration, and reliability issues. |
| Code Improvements      | ✅ Complete | Added validation, error handling, logging, health checks, configuration management, and rate limiting.     |
| Multiple ML Models     | ✅ Complete | Added support for PyTorch, ONNX Runtime, and YOLO through detector adapters.                               |
| Testing                | ✅ Complete | Expanded the test suite to 53 automated tests covering APIs, adapters, domain logic, and end-to-end flows. |
| Packaging & Deployment | ✅ Complete | Added `pyproject.toml`, Docker, Docker Compose, Makefile, and CI configuration.                            |

## 2. Main Improvements

### Prediction API

Added a new:

`POST /predictions`

The endpoint:

* Accepts an image.
* Validates the uploaded file.
* Runs object detection.
* Applies the confidence threshold.
* Returns the detected class, confidence score, and bounding box.

Example response:

```json
{
  "predictions": [
    {
      "class": "person",
      "confidence": 0.92,
      "box": {
        "xmin": 10,
        "ymin": 20,
        "xmax": 100,
        "ymax": 200
      }
    }
  ]
}
```

### PostgreSQL Repository

Added `CountPostgresRepo` using SQLAlchemy.

The repository supports:

* PostgreSQL persistence.
* Connection pooling.
* Atomic count updates using PostgreSQL upsert.
* The existing repository interface, so the domain logic does not depend directly on PostgreSQL.

MongoDB and in-memory repositories are still supported.

### Input Validation & Error Handling

Added validation for:

* Missing image files.
* Empty filenames.
* Invalid image formats.
* Invalid confidence thresholds.
* Invalid API routes.

Added HTTP error handling for common errors including:

`400`, `404`, `429`, `500`, and `503`.

### Health & Readiness Checks

Added:

* `GET /health` – checks that the application is running.
* `GET /ready` – checks whether the application is ready to serve requests.

These are useful for Docker and container-based deployments.

### Configuration

Configuration is now managed through environment variables and `.env`.

Factory functions are used to create:

* Object detectors.
* Count repositories.
* Count actions.
* Prediction actions.

This makes it easier to switch between implementations without changing the main application logic.

### ML Framework Support

The object detector interface allows different ML frameworks to be used without changing the core counting logic.

Added adapters for:

* TensorFlow Serving
* PyTorch / TorchVision
* ONNX Runtime
* YOLO

A new detector can be added by implementing the existing detector interface.

### Code Quality

The domain logic was separated from framework-specific and infrastructure code.

For example:

* Domain layer handles object detection/counting logic.
* Adapters handle databases and ML frameworks.
* Web layer handles HTTP requests and validation.
* Configuration handles application setup.

Also added:

* Type annotations.
* Structured logging.
* Better path handling.
* Removal of hardcoded configuration where possible.

## 3. Testing

Tests cover:

* Domain logic.
* Prediction functionality.
* Repository implementations.
* Object detector adapters.
* API endpoints.
* Validation and error handling.
* Rate limiting.
* Configuration.
* CLI functionality.
* End-to-end prediction and counting workflows.

The goal was to verify both individual components and the complete application flow.

## 4. Docker & Deployment

Added:

* `Dockerfile`
* `docker-compose.yml`
* `Makefile`
* `run.ps1`
* GitHub Actions CI
* `.env.example`

Docker Compose can be used to run the application together with the required services such as PostgreSQL, MongoDB, and TensorFlow Serving.

## 5. Before vs After

| Area           | Before              | After                            |
| -------------- | ------------------- | -------------------------------- |
| Prediction API | Not available       | `POST /predictions`              |
| Database       | MongoDB             | MongoDB + PostgreSQL + In-memory |
| ML Framework   | TensorFlow Serving  | TFS + PyTorch + ONNX + YOLO      |
| Validation     | Limited             | Image and request validation     |
| Error Handling | Basic               | Standard HTTP error handling     |
| Logging        | `print()`           | Python logging                   |
| Health Checks  | Not available       | `/health` and `/ready`           |
| Rate Limiting  | Not available       | Configurable rate limiting       |
| Testing        | 5 tests             | 53 automated tests               |
| Configuration  | Hardcoded/limited   | Environment-based configuration  |
| Deployment     | Manual Docker setup | Docker Compose                   |
| CI             | Not available       | GitHub Actions                   |

## 6. Summary

The main focus of the improvements was to make the application:

* Easier to test.
* Easier to maintain.
* More reliable.
* Easier to configure.
* Easier to extend with new ML frameworks and databases.
* Better suited for containerized deployment.

The core business logic remains independent of the web framework, database, and ML framework, allowing these components to be changed without rewriting the counting logic.
