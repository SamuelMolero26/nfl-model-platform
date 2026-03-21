"""
run_team_diagnostic.py
─────────────────────────────────────────────────────────────────────────────
Runner script for the Team Diagnostic & ROI Model.

Data is sourced from the NFL Data Platform REST API (data repo).
Falls back to local staged/raw files when the API is unreachable.

Run from the project root:
    python run_team_diagnostic.py

Optional flags:
    --season 2022          Only print summary for a specific season
    --team KC              Print full trend for a specific team
    --save                 Save fitted model to ml/models/team_diagnostic.pkl
    --contracts            Include cap ROI (fetched from API or staged parquet)
    --api-url URL          Override API base URL (default: http://localhost:8000)
    --no-api               Skip API and load from local files only
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Locate this file and the model class ──────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Team_diagnostic import TeamDiagnosticModel  # noqa: E402

# ── Find config.py by walking up the directory tree ───────────────────────────
#    Works whether run_team_diagnostic.py lives in the model repo or the
#    data repo (nfl-data-platform).  Falls back to safe defaults so the
#    script never crashes on import when config.py is absent.
def _find_config_dir(start: Path, filename: str = "config.py") -> Path | None:
    """Walk upward from *start* until *filename* is found; return its directory."""
    for parent in [start, *start.parents]:
        if (parent / filename).exists():
            return parent
    return None

_config_dir = _find_config_dir(HERE)
if _config_dir:
    sys.path.insert(0, str(_config_dir))
    try:
        import config as _cfg
        logger.debug("Loaded config from %s", _config_dir / "config.py")
    except ImportError:
        _cfg = None  # type: ignore[assignment]
else:
    _cfg = None  # type: ignore[assignment]
    logger.warning(
        "config.py not found anywhere in the directory tree. "
        "Using built-in defaults. Set --api-url or --no-api explicitly."
    )

# ── Config accessors with defaults ────────────────────────────────────────────
import os

def _cfg_path(attr: str, default: Path) -> Path:
    return Path(getattr(_cfg, attr, default)) if _cfg else default

def _cfg_val(attr: str, default):
    return getattr(_cfg, attr, default) if _cfg else default

_API_PORT        = _cfg_val("API_PORT", 8000)
STAGED_TEAM_STATS = _cfg_path("STAGED_TEAM_STATS",
                               HERE / "lake" / "staged" / "teams" / "team_statistics.parquet")
RAW_TEAM_STATS    = _cfg_path("RAW_TEAM_STATS",
                               HERE / "lake" / "raw" / "team_stats" / "nfl-team-statistics.csv")
STAGED_CONTRACTS  = _cfg_path("STAGED_CONTRACTS",
                               HERE / "lake" / "staged" / "players" / "contracts.parquet")
ML_MODELS_DIR     = _cfg_path("ML_MODELS_DIR", HERE / "ml" / "models")
LAKE_CURATED_DIR  = _cfg_path("LAKE_CURATED_DIR", HERE / "lake" / "curated")

# ── Default API base URL ───────────────────────────────────────────────────────
DEFAULT_API_URL = os.getenv("NFL_API_URL", f"http://localhost:{_API_PORT}")

# ── Optional: path to the data repo for local-file fallback ──────────────────
#    Set via env var or --data-repo CLI flag.
#    Example: set NFL_DATA_REPO=C:\Users\Areen\Python\nfl-data-platform
_ENV_DATA_REPO = os.getenv("NFL_DATA_REPO")
DATA_REPO_ROOT: Path | None = Path(_ENV_DATA_REPO) if _ENV_DATA_REPO else None


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_query(base_url: str, sql: str, timeout: int = 30) -> pd.DataFrame:
    """
    Execute a SQL query against the data repo's DuckDB endpoint.

    POST /query
    Body: { "query": "<sql>" }
    Returns a DataFrame built from the JSON response.

    Raises:
        requests.RequestException  — network / HTTP error
        ValueError                 — unexpected response shape
    """
    url = f"{base_url.rstrip('/')}/query"
    logger.debug("API query → %s | SQL: %s", url, sql.strip())

    resp = requests.post(url, json={"sql": sql}, timeout=timeout)
    resp.raise_for_status()

    payload = resp.json()

    # The /query endpoint returns either:
    #   { "results": [ {col: val, ...}, ... ] }   ← list of row dicts
    #   or a bare list
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
    elif isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
    else:
        raise ValueError(f"Unexpected /query response shape: {list(payload.keys())}")

    if not rows:
        logger.warning("API returned 0 rows for query: %s", sql.strip())
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _api_team_stats(base_url: str) -> pd.DataFrame:
    """
    Fetch all team statistics via POST /query (full table pull).

    Falls back to GET /teams/{abbr}/stats per-team if the SQL route
    returns an empty frame (e.g. table not yet registered in DuckDB).
    """
    sql = "SELECT * FROM team_stats ORDER BY season, team"
    df = _api_query(base_url, sql)

    if df.empty:
        logger.warning(
            "team_statistics table empty via /query — "
            "falling back to per-team GET /teams/{abbr}/stats"
        )
        df = _api_team_stats_per_team(base_url)

    logger.info("Fetched %d team-season rows from API.", len(df))
    return df


def _api_team_stats_per_team(base_url: str) -> pd.DataFrame:
    """
    Fallback: hit GET /teams to list abbreviations, then
    GET /teams/{abbr}/stats for each and concatenate.
    """
    teams_url = f"{base_url.rstrip('/')}/teams"
    resp = requests.get(teams_url, timeout=15)
    resp.raise_for_status()

    teams = resp.json()  # expects list of abbreviation strings
    if isinstance(teams, dict) and "teams" in teams:
        teams = teams["teams"]

    frames = []
    for abbr in teams:
        stats_url = f"{base_url.rstrip('/')}/teams/{abbr}/stats"
        r = requests.get(stats_url, timeout=15)
        if r.ok:
            data = r.json()
            rows = data if isinstance(data, list) else data.get("stats", [])
            if rows:
                frames.append(pd.DataFrame(rows))
        else:
            logger.warning("Could not fetch stats for %s (HTTP %s)", abbr, r.status_code)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _api_contracts(base_url: str) -> pd.DataFrame | None:
    """Fetch contracts table via POST /query."""
    sql = "SELECT * FROM contracts ORDER BY season, team"
    try:
        df = _api_query(base_url, sql)
        if df.empty:
            logger.info("No contracts data returned from API.")
            return None
        logger.info("Fetched %d contract rows from API.", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch contracts from API: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Local-file fallbacks (original behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def _local_team_stats(
    data_repo: Path | None = None,
    team_stats_file: Path | None = None,
) -> pd.DataFrame:
    """Load team statistics from staged Parquet, a direct file path, or raw CSV.

    Search order:
        1. team_stats_file  — explicit path supplied via --team-stats
        2. Staged Parquet in this repo's lake/
        3. data_repo paths  — if --data-repo supplied, searches staged + raw
    """
    # 1. Explicit file supplied — trust it completely, no guessing
    if team_stats_file:
        if not team_stats_file.exists():
            raise FileNotFoundError(f"--team-stats file not found: {team_stats_file}")
        logger.info("Loading team stats from explicit path: %s", team_stats_file)
        if team_stats_file.suffix == ".parquet":
            return pd.read_parquet(team_stats_file)
        return pd.read_csv(team_stats_file)

    # 2. Staged Parquet in this repo
    if STAGED_TEAM_STATS.exists():
        logger.info("(local) Loading staged team stats from %s", STAGED_TEAM_STATS)
        return pd.read_parquet(STAGED_TEAM_STATS)

    # 3. data_repo search
    fallbacks: list[Path] = []
    if data_repo:
        fallbacks += [
            data_repo / "lake" / "staged" / "teams" / "team_statistics.parquet",
            data_repo / "lake" / "raw" / "team_stats" / "nfl-team-statistics.csv",
            data_repo / "nflteamstatistics.csv",
            data_repo / "nfl-team-statistics.csv",
        ]

    for candidate in fallbacks:
        if candidate.exists():
            logger.info("(local) Loading from %s", candidate)
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        "Cannot find team stats.\n"
        "  Pass the file directly:  --team-stats <path/to/nflteamstatistics.csv>\n"
        + (
            "  Tried these paths:\n" + "\n".join(f"    {p}" for p in fallbacks)
            if fallbacks else
            "  No --data-repo supplied and no staged Parquet found in this repo."
        )
    )


def _local_contracts(data_repo: Path | None = None) -> pd.DataFrame | None:
    """Load contracts from staged Parquet, or return None."""
    candidates = [STAGED_CONTRACTS]
    if data_repo:
        candidates.append(data_repo / "lake" / "staged" / "players" / "contracts.parquet")
    for candidate in candidates:
        if candidate.exists():
            logger.info("(local) Loading contracts from %s", candidate)
            return pd.read_parquet(candidate)
    logger.info("No staged contracts found locally — cap ROI columns will be NaN.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders — try API first, fall back to local
# ─────────────────────────────────────────────────────────────────────────────

def load_team_stats(api_url: str | None, use_api: bool = True, data_repo: Path | None = None, team_stats_file: Path | None = None) -> pd.DataFrame:
    """
    Load team statistics.

    Priority:
        1. API  →  POST /query  (SELECT * FROM team_statistics)
        2. API  →  GET /teams/{abbr}/stats  (per-team fallback)
        3. Local staged Parquet
        4. Local raw CSV
    """
    if use_api and api_url:
        try:
            return _api_team_stats(api_url)
        except requests.RequestException as exc:
            logger.warning(
                "API unreachable (%s) — falling back to local files.", exc
            )

    return _local_team_stats(data_repo, team_stats_file)


def load_contracts(api_url: str | None, use_api: bool = True, data_repo: Path | None = None) -> pd.DataFrame | None:
    """
    Load contract / cap-hit data.

    Priority:
        1. API  →  POST /query  (SELECT * FROM contracts)
        2. Local staged Parquet
        3. None (cap ROI will be skipped)
    """
    if use_api and api_url:
        try:
            df = _api_contracts(api_url)
            if df is not None:
                return df
        except requests.RequestException as exc:
            logger.warning(
                "API unreachable for contracts (%s) — falling back to local.", exc
            )

    return _local_contracts(data_repo)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    width = 70
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Team Diagnostic & ROI Model runner")
    parser.add_argument("--season",   type=int,  default=None,
                        help="Season to summarize (e.g. 2022)")
    parser.add_argument("--team",     type=str,  default=None,
                        help="Team abbreviation for trend view (e.g. KC)")
    parser.add_argument("--save",     action="store_true",
                        help="Save fitted model to ml/models/team_diagnostic.pkl")
    parser.add_argument("--contracts", action="store_true",
                        help="Include cap ROI (fetched from API or staged parquet)")
    parser.add_argument("--api-url",  type=str,  default=DEFAULT_API_URL,
                        help=f"Data repo API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--team-stats", type=str, default=None,
                        help="Direct path to team stats CSV or Parquet "
                             "(e.g. C:\\path\\to\\nflteamstatistics.csv). "
                             "Skips all path guessing.")
    parser.add_argument("--data-repo", type=str, default=None,
                        help="Path to nfl-data-platform repo root for local CSV fallback "
                             "(overrides NFL_DATA_REPO env var)")
    parser.add_argument("--no-api",   action="store_true",
                        help="Skip API and load from local files only")
    args = parser.parse_args()

    api_url         = None if args.no_api else args.api_url
    use_api         = not args.no_api
    data_repo       = Path(args.data_repo) if args.data_repo else DATA_REPO_ROOT
    team_stats_file = Path(args.team_stats) if args.team_stats else None

    # ── Load data ──────────────────────────────────────────────────────────────
    team_stats = load_team_stats(api_url, use_api=use_api, data_repo=data_repo, team_stats_file=team_stats_file)
    contracts  = load_contracts(api_url, use_api=use_api, data_repo=data_repo) if args.contracts else None

    logger.info(
        "Loaded %d team-seasons (%d–%d)",
        len(team_stats),
        int(team_stats["season"].min()),
        int(team_stats["season"].max()),
    )

    # ── Fit + score ────────────────────────────────────────────────────────────
    model = TeamDiagnosticModel()
    model.fit(team_stats)
    results = model.score_teams(team_stats, contracts_df=contracts)

    # ── Summary table ──────────────────────────────────────────────────────────
    season = args.season or int(team_stats["season"].max())
    print_section(f"Team Diagnostic Summary — {season}")
    summary = model.summary(results, season=season)
    with pd.option_context("display.max_columns", 20, "display.width", 120,
                           "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))

    # ── Top / bottom units ─────────────────────────────────────────────────────
    print_section(f"Top 5 Pass Offenses — {season}")
    print(model.top_units(results, "pass_offense", season=season, n=5).to_string(index=False))

    print_section(f"Bottom 5 Pass Defenses — {season}")
    print(model.top_units(results, "pass_defense", season=season, n=5, bottom=True).to_string(index=False))

    print_section(f"Most Over-Performing Teams — {season}  (actual wins > expected)")
    print(model.top_units(results, "win_delta", season=season, n=5).to_string(index=False))

    print_section(f"Most Under-Performing Teams — {season}  (actual wins < expected)")
    print(model.top_units(results, "win_delta", season=season, n=5, bottom=True).to_string(index=False))

    # ── Team trend (optional) ──────────────────────────────────────────────────
    if args.team:
        team = args.team.upper()
        print_section(f"Team Trend — {team} (all seasons)")
        trend = model.unit_trends(results, team=team, unit="team_efficiency")
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(trend.to_string(index=False))

    # ── Cap ROI (if contracts loaded) ──────────────────────────────────────────
    if contracts is not None and "cap_efficiency_score" in results.columns:
        print_section(f"Top 5 Cap Efficiency — {season}")
        cap_cols = ["season", "team", "offense_cap_epa_roi",
                    "defense_cap_epa_roi", "cap_efficiency_score"]
        cap_summary = (
            results[results["season"] == season][cap_cols]
            .sort_values("cap_efficiency_score", ascending=False)
            .head(5)
        )
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(cap_summary.to_string(index=False))

    # ── Save model ─────────────────────────────────────────────────────────────
    if args.save:
        save_path = ML_MODELS_DIR / "team_diagnostic.pkl"
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

    # ── Write results parquet ──────────────────────────────────────────────────
    output_path = LAKE_CURATED_DIR / "team_diagnostic_results.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)
    logger.info("Full results written to %s", output_path)


if __name__ == "__main__":
    main()