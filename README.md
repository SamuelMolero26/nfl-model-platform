# NFL Model Platform

Platform to train and serve NFL draft analytics models (career value, draft optimizer, health risk, etc.) backed by the `nfl-data-platform` API and a versioned model registry. This README tracks how to run what already exists and what remains per `files/plan.md`.

## Status (from plan.md)
- ✅ Data lake client
- ✅ Model infrastructure (BaseModel + Registry)
- ✅ Player Projection (XGBoost, temporal CV, artifacts under `artifacts/player_projection/v1`)
- 🔲 Draft Optimizer, Team Diagnosis, Career Simulator, Roster Fit, Positional Flexibility, Health Analyzer
- 🔲 NullClaw (assistant + tools)
- 🔲 FastAPI serving layer
- 🔲 Validation case studies

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

## Analyze Artifacts
- Open `notebook/artifact_analysis.ipynb` to inspect metadata, metrics, SHAP, residuals, and fold alignment for a given artifact directory.

## Next Steps (per plan)
- Implement remaining models (Draft Optimizer, Team Diagnosis, Career Simulator, Roster Fit, Positional Flexibility, Health Analyzer) under `serving/models/` + training scripts.
- Wire NullClaw assistant tools in `serving/nullclaw/`.
- Expand FastAPI routers to expose all models and assistant.
- Add temporal validation case studies and per-position metrics.