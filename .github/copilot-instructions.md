# GitHub Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Purpose

This is an NFL-focused data lake platform with a REST API, SQL query engine (DuckDB), graph database (Neo4j), interactive data management UI, and an ML model training pipeline. The goal is to ingest data from various sources, transform it into a clean, queryable format, and serve it via multiple interfaces.

**Always refer to `files/plan.md` for the full implementation roadmap and architecture expansion plan.** It is the source of truth for all development.

## Key Libraries & Frameworks

- **Data Ingestion**: `nflreadpy` is the primary library for fetching NFL data. It returns Polars DataFrames. `cfbd` is used for college football data.
- **Data Manipulation**: Primarily use `pandas` and `polars`. Be mindful of converting between them (`.to_pandas()`, `polars.from_pandas()`).
- **API**: `fastapi` is used for the REST API. Routers are separated by domain in `api/routers/`.
- **Database**:
    - `duckdb` for SQL queries on the curated Parquet files.
    - `neo4j` for graph-based data and relationships.
- **ML**: `scikit-learn` for model training.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j (required before running graph builder or API)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# Run the full ingestion pipeline
python ingestion/pipeline.py

# Start the API server
uvicorn api.main:app --reload

# Populate Neo4j from curated Parquet files
python graph/builder.py
```

## Architecture Overview

The platform follows a **medallion architecture**:

1.  **`lake/raw/`**: Immutable source files. Data is downloaded here by loaders.
2.  **`lake/staged/`**: Cleaned, typed Parquet files. Each source is transformed into one or more Parquet files here. This is the output of the `loaders` in `ingestion/loaders/`.
3.  **`lake/curated/`**: Join-ready, query-optimized Parquet files. This is the primary data source for the API and analysis. These are built by the transforms and feature engineering steps in `ingestion/transforms/` and `ingestion/features/`.

**Data Flow**: `raw (nflreadpy, cfbd) → ingestion/loaders → staged (parquet) → ingestion/transforms + features → curated (master tables, feature tables) → DuckDB + Neo4j → FastAPI → UI/ML`

## Development Workflow & Key Concepts

- **Player Identity is Critical**: The `gsis_id` from `nflreadpy` is the canonical `player_id`. A key task is resolving player identities from different sources against the master player list. The `ingestion/player_id_resolver.py` is responsible for this, using fuzzy matching where necessary.
- **Idempotency**: Ingestion and graph building processes should be idempotent. Use `MERGE` in Cypher for Neo4j and overwrite Parquet files.
- **Configuration**: All file paths, credentials, and settings are managed in `config.py`. When adding new data sources or tables, add corresponding path variables there.
- **DuckDB Dynamic Registration**: The DuckDB client in `db/duckdb_client.py` must dynamically register all Parquet files in the `curated` directory as virtual tables. No manual registration should be needed.
- **API Routers**: When adding new API endpoints, find the appropriate router in `api/routers/` based on the domain (e.g., `players.py`, `teams.py`, `graph.py`).
- **Follow the Plan**: The `files/plan.md` document outlines the full architecture, including new data sources, feature engineering logic, and pipeline stages. Refer to it for any major feature implementation.

## Data Source Quirks

- **`nflreadpy`**: This is the main data source. It has many `load_*` functions. A wrapper, `ingestion/loaders/nflreadpy_loader.py`, should be used to handle fetching and saving data to the `raw` and `staged` layers.
- **`cfbd`**: Requires an API key (`CFBD_API_KEY` environment variable).
- **Fuzzy Matching**: Use the `rapidfuzz` library for matching player names during identity resolution.
- **`nfl-combine.xls`**: This file is HTML, not a true Excel file. It must be parsed with `pd.read_html()`.

## PR Review Expectations

When GitHub Copilot performs pull request reviews or posts review comments, it must explicitly evaluate code quality in addition to correctness.

- **Dead code detection is required**: flag and suggest removal of unused imports, unused variables, unused functions/classes, unreachable branches, stale feature flags, and duplicated obsolete logic.
- **Complexity checks are required**: flag overly complex logic such as deep nesting, very long functions, repeated conditional chains, and unnecessary abstraction layers.
- **Prefer simplification**: recommend the simplest implementation that preserves behavior, schema compatibility, and existing API contracts.
- **Actionable feedback only**: review comments should include concrete refactoring suggestions (what to remove/simplify and why), not only generic warnings.
- **Scope discipline**: focus comments on maintainability, readability, and testability improvements that are directly relevant to the changed code.