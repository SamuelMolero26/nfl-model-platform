# NFL Model Platform — FastAPI + Redis API Layer

## Context
The platform has 7 ML models behind a `ModelRegistry` singleton and an async `DataLakeClient`. The goal is to expose these as a production-ready REST API on a VPS, with Redis for prediction caching, data lake response caching, rate limiting, and async job queuing for heavy models. Nanoclaw AI agent instances will consume this API alongside the data lake.

The API and data lake are co-located on the same VPS. The data lake runs as a separate Docker Compose stack — this API stack joins its internal network. Nanoclaw connects over HTTPS via nginx reverse proxy.

---

## File Structure to Create

```
serving/api/
├── main.py                          # App factory + lifespan (startup/shutdown)
├── dependencies.py                  # Shared FastAPI Depends (registry, lake, redis)
├── auth.py                          # API key verification + rate limiting
├── errors.py                        # Structured error responses + exception handlers
├── cache/
│   ├── __init__.py
│   ├── client.py                    # Redis connection pool wrapper
│   ├── decorators.py                # @redis_cache(ttl, key_fn) decorator
│   └── keys.py                      # Cache key schema functions
└── routers/
    ├── health.py                    # GET /health, /health/ready, /health/models
    ├── models/
    │   ├── __init__.py
    │   ├── player_projection.py     # POST /models/player-projection/predict
    │   ├── positional_flexibility.py
    │   ├── team_diagnosis.py
    │   ├── draft_optimizer.py       # POST /models/draft-optimizer/jobs + GET job status
    │   ├── career_simulator.py
    │   ├── roster_fit.py
    │   └── health_analyzer.py
    └── data_lake/
        ├── __init__.py
        ├── players.py               # GET /data/players/{name}, /data/players/id/{id}/profile
        ├── teams.py                 # GET /data/teams/{abbr}/stats
        └── graph.py                 # GET /data/graph/player/{name}/profile, /college/{c}/pipeline

worker/
├── __init__.py
└── tasks.py                         # ARQ worker — draft_optimizer_predict task

config/
└── api.yaml                         # NEW — API, auth, redis, cache TTLs, job config

docker/
├── Dockerfile.api
├── Dockerfile.worker
└── nginx.conf                       # Reverse proxy + SSL termination

docker-compose.yml
```

---

## Endpoint Map

| Method | Path | Handler | Cache |
|--------|------|---------|-------|
| GET | `/health` | liveness | none |
| GET | `/health/ready` | checks registry + redis + lake | none |
| GET | `/health/models` | list registered models | none |
| POST | `/models/{model}/predict` | inline inference (6 sync models) | `nfl:pred:{model}:{input_hash}` TTL 1hr |
| POST | `/models/draft-optimizer/jobs` | enqueue ARQ job | none |
| GET | `/models/draft-optimizer/jobs/{id}` | poll result from Redis | `nfl:job:{id}` TTL 24hr |
| GET | `/models/{model}/schema` | input key/type metadata | none |
| GET | `/data/players/{name}` | lake proxy | `nfl:lake:player:{name}` TTL 24hr |
| GET | `/data/teams/{abbr}/stats` | lake proxy | `nfl:lake:team_stats:{abbr}` TTL 1hr |
| GET | `/data/graph/player/{name}/profile` | lake proxy | `nfl:lake:graph_profile:{name}` TTL 12hr |
| GET | `/data/graph/college/{college}/pipeline` | lake proxy | `nfl:lake:college_pipeline:{college}` TTL 12hr |
| DELETE | `/admin/cache/flush/{model_name}` | SCAN+DEL `nfl:pred:{model}:*` | — |

---

## Redis Cache Key Schema (`cache/keys.py`)

```
nfl:pred:{model_name}:{sha256(sorted_json(inputs))[:16]}
nfl:lake:player:{name_lower}
nfl:lake:team_stats:{abbr}:{season}
nfl:lake:graph_profile:{name_lower}
nfl:lake:college_pipeline:{college_lower}
nfl:rl:{api_key_prefix}:{utc_minute_bucket}
nfl:job:{job_id}
```

### TTL Config (in `api.yaml`)
- Predictions: 3600s
- Player profile: 86400s
- Team stats: 3600s
- Graph: 43200s
- Jobs: 86400s

---

## Auth (`auth.py`)

- Header: `X-API-Key`
- Keys stored **hashed (sha256)** in `api.yaml` — never plaintext
- Two tiers: `agent` (nanoclaw, 60 req/min), `admin` (600 req/min)
- Rate limit via Redis `INCR` + `EXPIRE` on `nfl:rl:{prefix}:{minute_bucket}`
- Returns `403` invalid key, `429` + `Retry-After` on limit exceeded

---

## Async Strategy

- **6 sync models** (XGBoost, lifelines, sklearn): run via `run_in_threadpool()` — never block event loop
- **Draft Optimizer**: offload to ARQ worker via Redis queue; API returns `{job_id}`, client polls
- **DataLakeClient**: already async (httpx) — `await` directly in endpoint handlers

---

## Docker Compose (`docker-compose.yml`)

Four services + external network:
1. **redis** — `redis:7-alpine`, LRU eviction, 512MB max, append-only persistence
2. **api** — `Dockerfile.api`, uvicorn 2 workers, mounts `artifacts/` read-only
3. **worker** — `Dockerfile.worker`, ARQ worker, profile-gated (`--profile async`) — opt-in only
4. **nginx** — SSL termination via Let's Encrypt (certbot); nanoclaw hits the public domain, nginx reverse-proxies to `api:8000`

**Network topology:**
- Declare the data lake's Compose network as `external: true` and attach `api` + `worker` to it
- `DataLakeClient` base URL in `api.yaml` points to the internal Docker service name (e.g. `http://nfl-datalake:8080`) — no public egress
- Nanoclaw connects to `https://your-vps-domain.com` → nginx → `api:8000`

```yaml
networks:
  datalake_net:
    external: true      # join existing data lake Compose stack network
  internal:
    driver: bridge      # redis + api + worker internal only
```

---

## Implementation Order

1. `config/api.yaml` — schema first
2. `serving/api/cache/` — client, keys, decorators
3. `serving/api/auth.py` + `errors.py`
4. `serving/api/dependencies.py`
5. `serving/api/main.py` — lifespan wiring
6. `serving/api/routers/health.py`
7. Model routers (6 sync, then draft optimizer async)
8. Data lake proxy routers
9. `worker/tasks.py` (ARQ)
10. `docker/` + `docker-compose.yml` (join `datalake_net` external network)
11. `docker/nginx.conf` — reverse proxy + SSL
12. Add `redis`, `arq` to `requirements.txt`

---

## Verification

- `docker compose up` → `GET /health/ready` returns 200 with all checks green
- `POST /models/player-projection/predict` with valid inputs → prediction + SHAP values
- Same call twice → second response has `X-Cache: HIT` header, Redis key visible via `redis-cli`
- `POST /models/draft-optimizer/jobs` → poll job endpoint until `status: complete`
- Invalid API key → 403; rapid fire requests → 429
- `DELETE /admin/cache/flush/player-projection` → Redis keys cleared
