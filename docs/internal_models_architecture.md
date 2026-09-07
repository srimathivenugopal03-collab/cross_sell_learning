# Supporting Multiple Private Models

## Current setup

The service currently selects one detector when the application starts. The choice is made by environment variables such as `DETECTOR_TYPE`, `MODEL_NAME`, and `TFS_HOST`. Both `/predictions` and `/object-count` then use that detector.

This works for one model, but changing models requires changing configuration and restarting the service.

## Proposed design

Keep the existing Flask API and domain actions, and add a small model registry/router between the API and the detector adapters:

```text
Client
  -> Flask endpoint
  -> model registry (name + version)
  -> ObjectDetector adapter
  -> TensorFlow Serving or another model server
```

Introducing a `ModelRegistryPort` in `counter/domain/ports.py` would keep model selection out of the business logic and allow different serving technologies later.

## Project changes

### 1. Store models outside the application image

Keep private model files out of Git and out of the Docker image. Store each model version in a private, access-controlled registry or object store:

```text
models/
  sku_detector/1.0.0/
    model files
    label_map.json
    metadata.json
  shelf_counter/2.0.0/
    model files
    label_map.json
    metadata.json
```

`metadata.json` should contain the model name, version, framework, serving endpoint, and label-map location. The storage credentials should come from the deployment environment or a secret manager.

### 2. Add a model registry adapter

Add an adapter, for example `counter/adapters/model_registry.py`, that:

- Maps an allowed model name and version to a detector configuration.
- Creates the correct existing adapter, such as `TFSObjectDetector` or `ONNXObjectDetector`.
- Caches loaded detectors so a model is not recreated for every request.
- Returns a clear error for an unknown or unavailable model.

The registry can initially be backed by configuration or a small metadata file. A managed registry can be introduced later without changing the domain actions.

### 3. Let requests select a model

Add optional `model` and `version` parameters to both endpoints:

```text
POST /predictions?model=sku_detector&version=1.0.0
POST /object-count?model=shelf_counter&version=2.0.0
```

When the parameters are omitted, use the configured default model. The API should validate the requested model before running inference and return `400` or `404` for an invalid selection.

The endpoint would resolve the detector and pass it to the existing `PredictObjects` or `CountDetectedObjects` action. The counting and prediction rules do not need to know which model was selected.

### 4. Update deployment configuration

Replace the single `MODEL_NAME` setting in `docker-compose.yml` with settings for the registry and default model, for example:

```text
MODEL_REGISTRY_URL=...
DEFAULT_MODEL_NAME=sku_detector
DEFAULT_MODEL_VERSION=1.0.0
```

For a small deployment, each model can run as a separate TensorFlow Serving instance. For a larger deployment, use one multi-model serving system such as Triton or TensorFlow Serving with multiple model configurations. The Flask application should call the serving system rather than load private weights itself.

## Security and operations

- Require authentication before allowing model selection.
- Authorize which clients may use each private model.
- Keep model storage and serving endpoints on the private network.
- Use secrets or workload identities for registry and serving credentials.
- Record the selected model name and version in request logs and metrics.
- Test every model version for input shape, labels, accuracy, and latency before promotion.
- Deploy a new version beside the old version, then switch the default after validation. This supports rollback without rebuilding the API.

## Recommended implementation order

1. Add a registry adapter using the existing detector adapters.
2. Add model and version request parameters with a default fallback.
3. Add unit tests for routing, unknown models, and version selection.
4. Move private artifacts to protected storage.
5. Add authentication, model permissions, metrics, and controlled model promotion.

This approach supports multiple internally trained models while preserving the project's current Hexagonal Architecture and keeping the change understandable and incremental.
