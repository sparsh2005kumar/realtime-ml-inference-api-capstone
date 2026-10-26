# Real-Time ML Inference REST API & Capstone

A FastAPI microservice scaffold for serving a trained scikit-learn classification model. Includes `/predict`, health/model-info endpoints, request validation, tests, pinned dependencies, and a Dockerfile.

## Project structure

```text
.
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── model/
│   └── README.txt
├── tests/
│   └── test_api.py
└── README.md
```

## 1. Install and run locally

Use Python 3.11 in a virtual environment (recommended). From this directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## 2. Add your trained model

Copy a compatible scikit-learn model/pipeline artifact named `champion_model.joblib` into `model/`.

**Important:** The API cannot make real predictions until this artifact is supplied. Use the champion model from Task 04 only if it is a scikit-learn model and the input schema matches. A PyTorch `.pt` model from Task 05 is not compatible with this `joblib` loader without a separate PyTorch inference implementation.

## 3. Example request

The feature names must match the model's training columns. For a model trained on columns `feature_01`, `feature_02`, etc.:

```json
{
  "features": {
    "feature_01": 0.2,
    "feature_02": -1.1
  }
}
```

Send this to `POST /predict`. The response includes a prediction and class probabilities when the model supports `predict_proba`.

## 4. Tests

```bash
pytest -q
```

The tests check service health, request validation, and predictable behavior when the model artifact is absent.

## 5. Build and run with Docker

First add the model artifact to `model/champion_model.joblib`, then:

```bash
docker build -t realtime-ml-inference-api .
docker run --rm -p 8000:8000 realtime-ml-inference-api
```

Open `http://127.0.0.1:8000/docs`.

## API endpoints

- `GET /` — service status
- `GET /health` — health check
- `GET /model-info` — model loading/input guidance
- `POST /predict` — model inference

## Security and production notes

This is an educational starter, not a hardened production service. Before deployment, add authentication/authorization, rate limiting, structured logging, monitoring, input bounds/schema validation, artifact provenance, and secret management. Do not load untrusted serialized model files. Pin and test dependencies before release.

## Task checklist

- [x] FastAPI microservice with JSON prediction endpoint
- [x] Request/response validation and API docs
- [x] Dockerfile and dependency list
- [x] Unit tests
- [ ] Add compatible trained model artifact
- [ ] Run tests and verify real predictions
- [ ] Capture public repository link for final certification
