# NFL Model Platform

Platform to train and serve NFL draft analytics models (career value, draft optimizer, health risk, etc.) backed by the `nfl-data-platform` API and a versioned model registry. This README tracks how to run what already exists and what remains per `files/plan.md`.

## Repo Layout (selected)
- `serving/data_lake_client/` async client + named queries
- `serving/models/` base classes, registry, and per-model packages (`player_projection`, etc.)
- `artifacts/{model}/{version}/` saved model.pkl, metadata.json, features
- `notebook/` analysis notebooks (e.g., `artifact_analysis.ipynb`)
- `config/data_lake.yaml` connection settings
- `tests/test_client.py` basic client checks

## Quickstart
1) Install deps: `pip install -r requirements.txt`
2) Ensure data lake API reachable (defaults in `config/data_lake.yaml`).
3) Run client test: `pytest tests/test_client.py -v`

## Train (player_projection example)
```bash
python training/scripts/train_player_projection.py
```
Outputs go to `artifacts/player_projection/<version>/` (model.pkl, metadata.json, feature cache). Artifacts are git-ignored by default.

## Serve (FastAPI skeleton)
```bash
uvicorn serving.api.main:app --port 8001 --reload
```
Routes (per plan): `POST /predict/{model}`, `POST /nullclaw/chat`, `GET /models`.

## API Endpoints
- Health: `GET /health` (liveness), `GET /health/ready` (readiness incl. redis + registry), `GET /health/models` (registered models)
- Player Projection: `POST /player-projection/predict`, `GET /player-projection/schema`
- Positional Flexibility: `POST /positional-flexibility/predict`, `GET /positional-flexibility/schema`
- Team Diagnosis: `POST /team-diagnosis/predict`, `GET /team-diagnosis/schema`
- Draft Optimizer: `POST /draft-optimizer/predict` (sync), `POST /draft-optimizer/jobs` (enqueue async), `GET /draft-optimizer/jobs/{job_id}` (poll), `GET /draft-optimizer/schema`
- Career Simulator: `POST /career-simulator/predict`, `GET /career-simulator/schema`
- Roster Fit: `POST /roster-fit/predict`, `GET /roster-fit/schema`
- Health Analyzer: `POST /health-analyzer/predict`, `GET /health-analyzer/schema`

## Architecture (serving layer)
- FastAPI app boots with a lifespan hook that loads config from `config/api.yaml`, initializes a Redis pool, and instantiates the model registry.
- Model registry lazily loads model artifacts and exposes `predict` plus per-model schemas. Each endpoint uses `run_prediction` to standardize inputs, caching, and error handling.
- Redis backs prediction result caching and async job tracking (draft optimizer jobs use a job key with TTL and background task execution).
- Routers are split per model under `serving/api/routers/models/` and mounted in `serving/api/main.py`; health endpoints live in `serving/api/routers/health.py`.
- Container build uses `docker/Dockerfile.api`; runtime config/env wired via `docker-compose.yml` (API + Redis). GH Actions workflow builds/pushes to GHCR and deploys to the VPS compose stack.
- Diagram: see [files/architecture-diagram.md](files/architecture-diagram.md) for a visual of clients → CI/CD → GHCR → VPS (compose), FastAPI, Redis, registry, and artifacts.

## Analyze Artifacts
- Open `notebook/artifact_analysis.ipynb` to inspect metadata, metrics, SHAP, residuals, and fold alignment for a given artifact directory.

## Next Steps (per plan)
- Implement remaining models (Draft Optimizer, Team Diagnosis, Career Simulator, Roster Fit, Positional Flexibility, Health Analyzer) under `serving/models/` + training scripts.
- Wire NullClaw assistant tools in `serving/nullclaw/`.
- Expand FastAPI routers to expose all models and assistant.
- Add temporal validation case studies and per-position metrics.
