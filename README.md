# NielsenIQ Innovation Enablement – Object Counter

A modular Python service for **object detection, prediction, and object counting**, implemented using **Hexagonal Architecture (Ports and Adapters)**.

The application exposes REST APIs for image prediction and object counting, with support for pluggable detector and repository implementations.

---

## Architecture

The application separates business logic from external technologies using a Ports and Adapters approach.

```text
                  ┌─────────────────────────┐
                  │       Flask API         │
                  │      Entry Point        │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │        Domain           │
                  │                         │
                  │  Prediction / Counting  │
                  │  Business Logic         │
                  │                         │
                  │  ObjectDetector Port    │
                  │  ObjectCountRepo Port   │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Detector        │         │ Repository      │
        │ Adapters        │         │ Adapters        │
        │                 │         │                 │
        │ Fake            │         │ In-Memory       │
        │ TensorFlow      │         │ MongoDB         │
        │ Serving         │         │ PostgreSQL      │
        └─────────────────┘         └─────────────────┘
```

This structure keeps the domain layer independent of databases, model-serving technologies, and frameworks.

---

## Features

* Object prediction from uploaded images
* Confidence-threshold filtering
* Object counting and aggregation
* PostgreSQL repository adapter
* MongoDB repository adapter
* In-memory repository for development/testing
* TensorFlow Serving detector integration
* Input validation and structured API error responses
* Unit, integration, and E2E tests
* Environment-based configuration
* Makefile for common development tasks

---

## Quick Start

### Prerequisites

* Python 3.10+
* `uv`
* Git

### Install dependencies

```bash
uv sync --all-extras
```

### Run tests

```bash
uv run pytest
```

### Start the application

```bash
uv run python -m counter.entrypoints.webapp
```

The API will be available at:

```text
http://127.0.0.1:5000
```

---

## API Usage

### 1. Predictions

**POST `/predictions`**

Accepts an image and confidence threshold and returns predictions above the specified threshold.

```bash
curl -F "threshold=0.5" \
     -F "file=@resources/images/boy.jpg" \
     http://127.0.0.1:5000/predictions
```

Example response:

```json
[
  {
    "class_name": "cat",
    "score": 0.99919,
    "box": {
      "xmin": 0.3672,
      "ymin": 0.2783,
      "xmax": 0.7358,
      "ymax": 0.6988
    }
  }
]
```

---

### 2. Object Count

**POST `/object-count`**

Detects objects and updates the configured repository with cumulative counts.

```bash
curl -F "threshold=0.5" \
     -F "file=@resources/images/boy.jpg" \
     http://127.0.0.1:5000/object-count
```

Example response:

```json
{
  "current_objects": [
    {
      "object_class": "cat",
      "count": 1
    }
  ],
  "total_objects": [
    {
      "object_class": "cat",
      "count": 4
    }
  ]
}
```

---

## Configuration

Application configuration is controlled through environment variables.

Typical configuration includes:

| Variable            | Description               |
| ------------------- | ------------------------- |
| `ENV`               | Application environment   |
| `REPOSITORY`        | Repository implementation |
| `MONGO_HOST`        | MongoDB host              |
| `MONGO_PORT`        | MongoDB port              |
| `MONGO_DATABASE`    | MongoDB database          |
| `POSTGRES_HOST`     | PostgreSQL host           |
| `POSTGRES_PORT`     | PostgreSQL port           |
| `POSTGRES_DATABASE` | PostgreSQL database       |
| `POSTGRES_USER`     | PostgreSQL user           |
| `POSTGRES_PASSWORD` | PostgreSQL password       |
| `TFS_HOST`          | TensorFlow Serving host   |
| `TFS_PORT`          | TensorFlow Serving port   |
| `MODEL_NAME`        | Model name                |

A `.env` file can be used for local configuration.

---

## Database Support

The repository layer provides multiple implementations behind the same domain interface:

* **In-memory** – useful for development and testing
* **MongoDB** – document-based persistence
* **PostgreSQL** – relational persistence using SQLAlchemy

The domain layer does not directly depend on any database implementation.

---

## Testing

The project includes:

* Unit tests
* Integration tests
* End-to-end API tests

Run the complete test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=counter tests/
```

Run code-quality checks:

```bash
uv run ruff check counter tests
uv run ruff format --check counter tests
```

---

## Makefile

Common development tasks are available through the Makefile.

```bash
make install
make test
make lint
make format
make typecheck
make check
make run-dev
```

Use:

```bash
make help
```

to see the available commands, if supported by the Makefile.

---

## Key Design Decisions

### Hexagonal Architecture

Business logic is kept independent from Flask, databases, and model-serving technologies. This makes individual components easier to test and replace.

### Repository Pattern

Persistence is abstracted behind `ObjectCountRepo`, allowing the application to switch between in-memory, MongoDB, and PostgreSQL implementations without changing domain logic.

### Detector Abstraction

Object detection is accessed through the `ObjectDetector` interface. This allows different detection implementations to be introduced without modifying the core counting logic.

### PostgreSQL Adapter

The PostgreSQL implementation uses SQLAlchemy to provide database access, transaction handling, and object-relational mapping while keeping database-specific code inside the adapter layer.

---

## Improvements Implemented

The following improvements were made as part of the assignment:

* Added `/predictions` API endpoint
* Added PostgreSQL repository implementation
* Improved request validation and error handling
* Added integration/E2E test coverage
* Added environment-based configuration
* Improved MongoDB client reuse
* Added automated development commands through Makefile

Further architectural improvements and design considerations are documented in:

* `docs/code_review_and_improvements.md`
* `docs/internal_models_architecture.md`
* `docs/multi_framework_support.md`

---

## Future Improvements

Potential next steps include:

* OpenAPI/Swagger documentation
* Structured application logging
* Authentication and authorization
* Additional model-framework adapters
* Improved production monitoring and observability
* Support for multiple internally trained models through a model registry

---

## Project Structure

```text
object-counter/
├── counter/
│   ├── domain/
│   ├── adapters/
│   ├── entrypoints/
│   └── config.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── resources/
├── docs/
├── Makefile
├── ASSESSMENT.md
├── README.md
└── pyproject.toml
```

---

## Assignment

This repository contains the implementation and design proposals for the **NielsenIQ Innovation Enablement Machine Learning | Generative AI take-home assignment**.

Detailed implementation decisions and proposed improvements are available in `ASSESSMENT.md`.
