"""
Team Diagnosis — Feature Engineering
=====================================

DATA SOURCES (from the data lake staged tables):
┌──────────────────────────────┬────────────────────────────────────────────────────┐
│ Table                        │ What we use                                        │
├──────────────────────────────┼────────────────────────────────────────────────────┤
│ team_statistics              │ EPA, success rate, yards, WPA, turnover columns    │
│  (lake/staged/teams/)        │ per team-season; also wins / losses                │
├──────────────────────────────┼────────────────────────────────────────────────────┤
│ contracts                    │ season, team, position, cap_hit                    │
│  (lake/staged/players/)      │ Used for cap-ROI columns (optional)                │
└──────────────────────────────┴────────────────────────────────────────────────────┘

FEATURE MATRIX for train():
  X = team_stats_df  (one row per team-season, FEATURE_COLS columns)
  y = wins Series    (actual regular-season wins)

The expected-wins RidgeCV model uses only EW_FEATURE_COLS (4 EPA totals).
Unit z-score computation is fully deterministic — no training needed there.
"""

import logging
from typing import Optional

import pandas as pd

from serving.data_lake_client import DataLakeClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# The 4 total-EPA columns consumed by the expected-wins RidgeCV model.
EW_FEATURE_COLS = [
    "offense_total_epa_pass",
    "offense_total_epa_run",
    "defense_total_epa_pass",
    "defense_total_epa_run",
]

# All per-play unit statistics used by the deterministic unit z-score layer.
UNIT_STAT_COLS = [
    # Pass offense
    "offense_ave_epa_pass",
    "offense_success_rate_pass",
    "offense_ave_yards_gained_pass",
    "offense_ave_wpa_pass",
    "offense_n_interceptions",
    "offense_n_plays_pass",
    # Run offense
    "offense_ave_epa_run",
    "offense_success_rate_run",
    "offense_ave_yards_gained_run",
    "offense_ave_wpa_run",
    "offense_n_fumbles_lost_run",
    "offense_n_plays_run",
    # Pass defense
    "defense_ave_epa_pass",
    "defense_success_rate_pass",
    "defense_ave_yards_gained_pass",
    "defense_ave_wpa_pass",
    "defense_n_interceptions",
    "defense_n_plays_pass",
    # Run defense
    "defense_ave_epa_run",
    "defense_success_rate_run",
    "defense_ave_yards_gained_run",
    "defense_ave_wpa_run",
    # Turnover components (net turnovers = def gained − off given)
    "defense_n_fumbles_lost_pass",
    "defense_n_fumbles_lost_run",
    "offense_n_fumbles_lost_pass",
]

# Full feature list: EW model inputs first, then unit stats.
# Any column missing from a given dataset is simply absent — scoring degrades
# gracefully rather than raising.
FEATURE_COLS = EW_FEATURE_COLS + [c for c in UNIT_STAT_COLS if c not in EW_FEATURE_COLS]


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_features(
    team_stats_df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Validate, coerce, and select feature columns from a raw team_stats DataFrame.

    Parameters
    ----------
    team_stats_df : DataFrame
        One row per (team, season).  Must contain 'team' and 'season' columns.
    include_target : bool
        When True, appends the 'wins' column.  Raises if wins is absent.

    Returns
    -------
    DataFrame indexed by (team, season) with FEATURE_COLS columns (+ 'wins').
    """
    df = team_stats_df.copy()

    if "season" not in df.columns or "team" not in df.columns:
        raise ValueError("team_stats_df must contain 'season' and 'team' columns.")

    df["season"] = df["season"].astype(int)
    df["team"] = df["team"].str.upper().str.strip()

    for c in df.columns:
        if c not in ("season", "team") and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    available = [c for c in FEATURE_COLS if c in df.columns]
    result = df[["season", "team"] + available].copy()

    if include_target:
        if "wins" not in df.columns:
            raise ValueError("'wins' column is required when include_target=True.")
        result["wins"] = df["wins"]

    result = result.set_index(["team", "season"])
    return result


async def fetch_feature_matrix(
    client: DataLakeClient,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Fetch team_statistics from the data lake and return the feature matrix.

    Sequential fetch — avoids saturating the remote data lake over Tailscale.
    Used by train.py (include_target=True) and model.py at inference time.
    """
    where_clauses = []
    if year_start is not None:
        where_clauses.append(f"season >= {int(year_start)}")
    if year_end is not None:
        where_clauses.append(f"season <= {int(year_end)}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"SELECT * FROM team_statistics {where} ORDER BY season, team"
    log.info("Fetching team_statistics …")
    df = await client.query(sql)
    log.info("  team_statistics: %d rows", len(df))

    return build_features(df, include_target=include_target)


async def fetch_contracts(
    client: DataLakeClient,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch cap-hit contracts from the data lake.

    Returns columns: season, team, position, cap_hit.
    Passed as contracts_df to score_teams() / predict() for cap-ROI columns.
    """
    where_clauses = []
    if year_start is not None:
        where_clauses.append(f"season >= {int(year_start)}")
    if year_end is not None:
        where_clauses.append(f"season <= {int(year_end)}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"SELECT season, team, position, cap_hit FROM contracts {where}"
    log.info("Fetching contracts …")
    df = await client.query(sql)
    log.info("  contracts: %d rows", len(df))
    return df
