# NFL Model Platform — Implementation Plan

**Goal**: Identify undervalued NFL draft picks using 7 ML models + NullClaw (Claude-powered assistant).
**Data source**: `nfl-data-platform` REST API at `http://localhost:8000`
**Scope**: ML models, training, tuning, serving only. No frontend (separate repo).

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Lake Client | ✅ Done |
| 2 | Model Infrastructure (BaseModel + Registry) | ✅ Done |
| 3 | Player Projection model | ✅ Done |
| 4 | Draft Optimizer model | 🔲 Pending |
| 5 | Team Diagnosis, Career Simulator, Roster Fit, Positional Flexibility, Health Analyzer | 🔲 Pending |
| 6 | NullClaw (Claude assistant — models as tools) | 🔲 Pending |
| 7 | FastAPI serving layer | 🔲 Pending |
| 8 | Temporal validation + case studies | 🔲 Pending |

---

## Repository Structure

```
nfl-model-platform/
├── config/
│   └── data_lake.yaml              # Base URL, timeouts, retry config
├── serving/
│   ├── data_lake_client/
│   │   ├── __init__.py
│   │   ├── client.py               # DataLakeClient (httpx async + retries)
│   │   └── queries.py              # Named query helpers → DataFrames
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseModel abstract class
│   │   ├── registry.py             # ModelRegistry singleton (lazy-load artifacts)
│   │   ├── player_projection/
│   │   │   ├── model.py            # XGBoost regressor → career value score
│   │   │   ├── features.py         # combine + college stats + draft context
│   │   │   └── train.py            # Optuna-tuned, temporal split
│   │   ├── draft_optimizer/
│   │   │   ├── model.py            # CVXPY constrained optimizer
│   │   │   └── features.py         # team needs, pick value, projections
│   │   ├── team_diagnosis/
│   │   │   ├── model.py            # Multi-task XGBoost → positional weakness scores
│   │   │   └── features.py         # snap counts, depth charts, injuries
│   │   ├── career_simulator/
│   │   │   ├── model.py            # Cox PH (lifelines) → games-played trajectory
│   │   │   └── features.py         # injury history, workload, age
│   │   ├── roster_fit/
│   │   │   ├── model.py            # Cosine similarity + learned weights → fit score
│   │   │   └── features.py         # team scheme profile, roster athletic profile
│   │   ├── positional_flexibility/
│   │   │   ├── model.py            # XGBoost multi-label → secondary positions
│   │   │   └── features.py         # athletic profile, snap distribution by position
│   │   └── health_analyzer/
│   │       ├── model.py            # Cox PH (lifelines) → injury risk probability
│   │       └── features.py         # injury freq, workload per snap, position risk
│   ├── nullclaw/
│   │   ├── __init__.py
│   │   ├── assistant.py            # Claude claude-sonnet-4-6 + tool routing
│   │   └── tools.py                # Tool schemas — one per model
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app (:8001)
│       └── routers/
│           ├── __init__.py
│           ├── predictions.py      # POST /predict/{model}
│           └── nullclaw.py         # POST /nullclaw/chat
├── training/
│   ├── feature_extractors/         # Shared feature builders reused across models
│   └── scripts/                    # Per-model training + Optuna tuning scripts
├── artifacts/
│   └── {model}/{version}/          # model.pkl + metadata.json + shap_values.pkl
│       └── features/               # Cached feature Parquet files (speed up retraining)
├── notebooks/                      # EDA, validation, case studies
├── tests/
│   ├── __init__.py
│   └── test_client.py              # Integration tests vs live :8000 API
├── plan.md                         # This file
└── requirements.txt
```

---

## Phase 1 — Data Lake Client ✅

**`serving/data_lake_client/client.py`** — `DataLakeClient`:
- `httpx.AsyncClient` — async-first; sync shim via `asyncio.run()` for training scripts
- Retries: `tenacity`, exponential backoff, 3 attempts, 5xx + connection errors only
- Timeouts: 10s connect / 30s read (from `config/data_lake.yaml`)
- Custom exceptions: `DataLakeConnectionError`, `DataLakeQueryError`, `DataLakeNotFoundError`
- `POST /query` (DuckDB SQL) as primary; graph endpoints for relationship traversal

Key methods:
```python
async def query(self, sql: str) -> pd.DataFrame
async def health_check(self) -> bool
async def list_tables(self) -> list[str]
async def get_player(self, name: str) -> dict
async def list_players(self, position=None, year=None) -> pd.DataFrame
async def get_team_stats(self, abbr, year_start=None, year_end=None) -> pd.DataFrame
async def get_player_graph_profile(self, name: str) -> dict
async def get_college_pipeline(self, college: str) -> pd.DataFrame
async def list_datasets(self) -> list[str]
async def preview_dataset(self, dataset: str) -> pd.DataFrame
```

**`serving/data_lake_client/queries.py`** — named domain helpers:
```python
# Staged tables (raw data)
async def get_combine_data(client, year_start, year_end) -> pd.DataFrame
async def get_draft_picks(client, year_start, year_end) -> pd.DataFrame
async def get_player_career_stats(client, player_name) -> pd.DataFrame
async def get_injury_history(client, position) -> pd.DataFrame
async def get_snap_counts(client, position, season) -> pd.DataFrame
async def get_depth_charts(client, team, season) -> pd.DataFrame
async def get_ngs_stats(client, stat_type, season) -> pd.DataFrame
async def get_college_stats(client, player_name, year) -> pd.DataFrame

# Gold profile tables (lake/curated/ — pre-computed features)
async def get_athletic_profiles(client, player_id=None) -> pd.DataFrame   # speed/agility/burst/strength/size scores
async def get_production_profiles(client, player_id, season) -> pd.DataFrame  # nfl_production_score, snap_share
async def get_durability_profiles(client, player_id=None) -> pd.DataFrame  # durability_score, games_played_rate
async def get_draft_value_history(client, year_start, year_end) -> pd.DataFrame  # draft_value_score, percentile
async def get_player_full_profile(client, player_id) -> dict               # single-player join of all 4 gold tables
```

---

## Phase 2 — Model Infrastructure 🔲

**`serving/models/base.py`** — `BaseModel` abstract class:
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    @abstractmethod
    def predict(self, inputs: dict) -> dict: ...
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict: ...
    def save(self, path: Path) -> None: ...   # pickle + metadata.json
    def load(self, path: Path) -> None: ...
```

**`serving/models/registry.py`** — `ModelRegistry` singleton:
- Scans `artifacts/` on init
- Lazy-loads `.pkl` on first `predict()` call, caches in memory
- Supports versioned artifacts (`v1`, `v2`, …) — defaults to latest
- `metadata.json` per artifact: version, train date, val metrics, feature list, SHAP summary

---

## Phase 3 — Player Projection 🔧

- Algorithm: XGBoost regressor
- Target: `car_av` (Career Approximate Value from Pro Football Reference, in `draft_picks` table)
- Training: 3-fold walk-forward CV + same fold for early stopping and Optuna objective
- Tuning: Optuna, 100 trials, TPESampler + MedianPruner, minimize mean(fold RMSEs)

```
Fold 1: train 2000–2014 → val 2015–2016   (~1,400 train, ~200 val)
Fold 2: train 2000–2016 → val 2017–2018   (~1,800 train, ~200 val)
Fold 3: train 2000–2018 → val 2019–2020   (~2,200 train, ~200 val)
Holdout: 2021–2022                         (touched once, final score)
```

- XGBoost early stopping (50 rounds) on the same val fold — finds n_estimators automatically
- Optuna tunes structural params only: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma
- Overfit-val risk diluted by averaging across 3 independent time windows
- Outputs: `career_value_score`, `confidence`, `top_comparables`, `shap_values`

### Feature sources (gold-first approach)

All features pulled via `POST /query` against the data lake curated tables:

| Source table | Features used |
|---|---|
| `player_athletic_profiles` (gold) | `speed_score`, `agility_score`, `burst_score`, `strength_score`, `size_score` |
| `draft_value_history` (gold) | `draft_value_score`, `draft_value_percentile` (pick slot z-score vs historical AV) |
| `combine` (staged) | raw: `forty_yard`, `vertical_in`, `bench_reps`, `broad_jump_in`, `three_cone`, `shuttle`, `height_in`, `weight_lbs` |
| `draft_picks` (staged) | `draft_round`, `draft_pick`, `age`, `position` + `car_av` target |

**Engineered on top of gold:**
- Position group one-hot (QB / SKILL / OL / DL / LB / DB / SPEC)
- Missing drill flags — skipping a drill is a scouting signal
- `round × draft_value_score` interaction term

**Inference path (single player):** `GET /players/id/{player_id}/profile` joins all 4 gold tables in one call.
**Training path (all prospects):** parallel `asyncio.gather()` across 4 `POST /query` calls.

### Files
- `serving/models/player_projection/features.py` ✅ — `fetch_feature_matrix()`, `build_features()`, `FEATURE_COLS`
- `serving/models/player_projection/model.py` ✅ — XGBoost + SHAP + comparables lookup
- `serving/models/player_projection/train.py` ✅ — Optuna study, 3-fold walk-forward CV, artifact save

---

## Phase 4 — Draft Optimizer 🔲

- Algorithm: CVXPY constrained optimization
- Inputs: Player Projection scores for available prospects + team positional needs + remaining draft picks
- Constraints: positional quotas, pick value budget, need priority weights
- Output: ranked list of recommendations with value-over-ADP scores

---

## Phase 5 — Remaining 5 Models 🔲

| Model | Algorithm | Key Output |
|-------|-----------|-----------|
| Team Diagnosis | Multi-task XGBoost | Positional weakness scores (0–1 per position group) |
| Career Simulator | Cox PH (lifelines) | Projected games played + career arc percentiles |
| Roster Fit | Cosine similarity + Ridge weights | Fit score (0–100) per prospect-team pair |
| Positional Flexibility | XGBoost multi-label | Secondary positions player can learn |
| Health Analyzer | Cox PH (lifelines) | Injury risk probability per season |

All share feature extractors from `training/feature_extractors/`.

---

## Phase 6 — NullClaw 🔲

NullClaw is the **primary interface**. The 7 models are its tools.

```
User message
    │
    ▼
NullClaw (claude-sonnet-4-6)
    │  tool_use routing
    ├──▶ player_projection(player_name, draft_year)
    ├──▶ draft_optimizer(team_abbr, draft_year, available_picks)
    ├──▶ team_diagnosis(team_abbr, year)
    ├──▶ career_simulator(player_name)
    ├──▶ roster_fit(player_name, team_abbr)
    ├──▶ positional_flexibility(player_name)
    └──▶ health_analyzer(player_name)
    │
    ▼
Response with SHAP-driven plain-English explanation + comparables
```

- `tools.py`: one tool schema per model (name, description, input_schema)
- `assistant.py`: maintains conversation history, routes tool calls to `ModelRegistry`, formats SHAP explanations

---

## Phase 7 — FastAPI Serving 🔲

- `POST /nullclaw/chat` — `{message, conversation_id}` — primary endpoint
- `POST /predict/{model_name}` — direct model access for testing/programmatic use
- `GET /models` — list registered models, versions, and validation metrics

---

## Phase 8 — Validation + Case Studies 🔲

- Temporal holdout: must beat naive ADP baseline on seasons Y-2 and Y-1
- SHAP feature importance audit per model
- 5+ undervalued player case studies (historical hindsight validation)
- Ensemble draft score: weighted blend of Player Projection + Health Analyzer + Positional Flexibility

---

## Tuning Principles (all models)

1. **No leakage**: hard temporal cutoff — test data is always future relative to train
2. **SHAP on every model**: stored in `artifacts/{model}/{version}/shap_values.pkl`
3. **Feature caching**: extracted features saved as Parquet in `artifacts/{model}/features/`
4. **Baseline comparison**: every model measured against a simple rule-based or ADP baseline
5. **Optuna for XGBoost models**: 100 trials, `TPESampler`, pruning with `MedianPruner`

---

## Verification Commands

```bash
# Phase 1 — client tests (requires data lake running)
pytest tests/test_client.py -v

# Phase 2+ — train a model
python training/scripts/train_player_projection.py

# Phase 7 — start serving layer
uvicorn serving.api.main:app --port 8001 --reload
```
