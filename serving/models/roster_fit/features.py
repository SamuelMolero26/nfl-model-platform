"""
Roster Fit — Feature Engineering
==================================

Builds two profile DataFrames that feed directly into RosterFitModel:

  Team scheme profile   — what a team's scheme demands, per (team, season)
  Player roster profile — what a player brings, per (player_id, season)

DATA SOURCES (all fetched via DataLakeClient → POST /query):
┌──────────────────────────────────┬────────────────────────────────────────────┐
│ Table                            │ What we use                                │
├──────────────────────────────────┼────────────────────────────────────────────┤
│ team_stats  (lake/curated/)      │ pass_rate, air_yards, yac, EPA splits,     │
│                                  │ success rates — scheme fingerprint          │
├──────────────────────────────────┼────────────────────────────────────────────┤
│ player_athletic_profiles         │ speed_score, agility_score, burst_score,   │
│  (lake/curated/)                 │ strength_score, size_score                 │
├──────────────────────────────────┼────────────────────────────────────────────┤
│ player_production_profiles       │ snap_share, epa_per_game,                  │
│  (lake/curated/)                 │ nfl_production_score, target_share,        │
│                                  │ passing_cpoe                               │
├──────────────────────────────────┼────────────────────────────────────────────┤
│ snap_counts  (lake/staged/)      │ per-position snap share per team-season    │
│  [optional — Tier 2]             │ (requires Stage 0 nflreadpy ingestion)     │
└──────────────────────────────────┴────────────────────────────────────────────┘

WHY use gold profiles instead of raw columns?
  The data lake already computed speed_score, agility_score, etc. with the
  correct formulas and normalisation. Same for nfl_production_score. Re-
  implementing them here would risk drift. We pull the finished scores and
  use them directly as features — identical to the Player Projection model.

DIMENSION PAIRS  (player_dim, scheme_dim)
  Each pair is one compatibility axis in the cosine similarity. A WR's
  speed_score is paired with air_yards_scheme because a fast receiver is most
  valuable in a scheme that attacks vertically. See DIMENSION_PAIRS below.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from serving.data_lake_client import DataLakeClient
from serving.data_lake_client.queries import (
    get_athletic_profiles,
    get_production_profiles,
    get_snap_counts,
)

log = logging.getLogger(__name__)

# ── Position group mapping ────────────────────────────────────────────────────

_POS_GROUPS: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "G": "OL",
    "T": "OL",
    "OL": "OL",
    "DE": "EDGE",
    "OLB": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "DL": "IDL",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "CB": "CB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "K": "SPEC",
    "P": "SPEC",
    "LS": "SPEC",
}


def _pos_group(pos: str) -> str:
    if not isinstance(pos, str):
        return "UNK"
    return _POS_GROUPS.get(pos.strip().upper(), "UNK")


# ── Scheme feature columns ────────────────────────────────────────────────────

# Tier 1: always available from team_stats
SCHEME_FEATURES_T1 = [
    "pass_rate",  # fraction of plays that are passes
    "air_yards_scheme",  # avg depth-of-target — vertical vs. short game
    "yac_scheme",  # avg yards after catch — YAC vs. contested-catch
    "pass_epa_efficiency",  # avg EPA per pass play
    "run_epa_efficiency",  # avg EPA per run play
    "pass_success_rate",  # passing consistency
    "run_success_rate",  # running consistency
    "def_pass_scheme",  # pass-defense emphasis (sign-flipped EPA)
    "def_run_scheme",  # run-defense emphasis  (sign-flipped EPA)
]

# Tier 2: added when snap_counts is available (Stage 0)
SCHEME_FEATURES_T2 = [
    "wr_snap_share",
    "rb_snap_share",
    "te_snap_share",
    "ol_snap_share",
    "dl_snap_share",
    "lb_snap_share",
    "db_snap_share",
]

# ── Player dimension columns ──────────────────────────────────────────────────

# From player_athletic_profiles (gold, pre-computed by athletic_scores.py)
ATHLETIC_DIMS = [
    "speed_score",  # Barnwell formula
    "agility_score",  # −three_cone_z + −shuttle_z  (pos-group z-scored)
    "burst_score",  # vertical_z + broad_jump_z   (pos-group z-scored)
    "strength_score",  # bench_reps_z                (pos-group z-scored)
    "size_score",  # height × weight / pos_avg
]

# From player_production_profiles (gold, pre-computed by production_scores.py)
PRODUCTION_DIMS = [
    "snap_share",  # mean offense_pct across reg-season weeks
    "epa_per_game",  # total EPA / games played
    "nfl_production_score",  # composite z-score (already normalised)
    "target_share",  # WR / TE / RB only — NaN for others
    "passing_cpoe",  # QB only — NaN for all others
]

# Raw dimensions that need z-scoring here (not pre-normalised in gold tables)
_NEEDS_ZSCORE = {"speed_score", "snap_share", "epa_per_game", "size_score"}

# ── Dimension pair definitions ────────────────────────────────────────────────
#
# (player_dim, scheme_dim) → one compatibility axis in the cosine similarity.
# Higher on BOTH sides = stronger fit.

DIMENSION_PAIRS: dict[str, list[tuple[str, str]]] = {
    "WR": [
        ("speed_score", "air_yards_scheme"),
        ("agility_score", "yac_scheme"),
        ("size_score", "pass_success_rate"),
        ("burst_score", "air_yards_scheme"),
        ("nfl_production_score", "pass_rate"),
        ("target_share", "pass_rate"),
    ],
    "RB": [
        ("speed_score", "run_epa_efficiency"),
        ("agility_score", "yac_scheme"),
        ("strength_score", "run_success_rate"),
        ("burst_score", "run_epa_efficiency"),
        ("nfl_production_score", "pass_rate"),
    ],
    "TE": [
        ("size_score", "run_epa_efficiency"),
        ("strength_score", "run_success_rate"),
        ("speed_score", "air_yards_scheme"),
        ("agility_score", "yac_scheme"),
        ("nfl_production_score", "pass_rate"),
    ],
    "QB": [
        ("epa_per_game", "pass_epa_efficiency"),
        ("passing_cpoe", "air_yards_scheme"),
        ("nfl_production_score", "pass_rate"),
        ("snap_share", "pass_rate"),
    ],
    "OL": [
        ("strength_score", "run_epa_efficiency"),
        ("size_score", "run_success_rate"),
        ("agility_score", "pass_epa_efficiency"),
    ],
    "EDGE": [
        ("speed_score", "def_pass_scheme"),
        ("burst_score", "def_pass_scheme"),
        ("strength_score", "def_run_scheme"),
        ("size_score", "def_run_scheme"),
    ],
    "IDL": [
        ("strength_score", "def_run_scheme"),
        ("size_score", "def_run_scheme"),
        ("burst_score", "def_pass_scheme"),
        ("agility_score", "def_pass_scheme"),
    ],
    "LB": [
        ("speed_score", "def_pass_scheme"),
        ("agility_score", "def_pass_scheme"),
        ("strength_score", "def_run_scheme"),
        ("size_score", "def_run_scheme"),
    ],
    "CB": [
        ("speed_score", "def_pass_scheme"),
        ("agility_score", "def_pass_scheme"),
        ("burst_score", "def_pass_scheme"),
        ("size_score", "def_run_scheme"),
    ],
    "DB": [
        ("speed_score", "def_pass_scheme"),
        ("agility_score", "def_pass_scheme"),
        ("strength_score", "def_run_scheme"),
        ("size_score", "def_run_scheme"),
    ],
}

DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("speed_score", "pass_rate"),
    ("agility_score", "pass_success_rate"),
    ("nfl_production_score", "pass_epa_efficiency"),
]


# ── Private helpers ───────────────────────────────────────────────────────────


def _zscore_within(df: pd.DataFrame, col: str, groups: list[str]) -> pd.Series:
    """Z-score *col* within each combination of group columns (min 3 obs)."""

    def _z(x: pd.Series) -> pd.Series:
        if x.notna().sum() < 3:
            return pd.Series(np.nan, index=x.index)
        mu, sigma = x.mean(), x.std(ddof=0)
        return (x - mu) / sigma if sigma > 0 else pd.Series(0.0, index=x.index)

    return df.groupby(groups)[col].transform(_z)


def _zscore_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    return _zscore_within(df, col, ["season"])


# ── build_team_scheme_profile ─────────────────────────────────────────────────


def build_team_scheme_profile(
    team_stats_df: pd.DataFrame,
    snap_counts_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a scheme fingerprint per (team, season) from raw team statistics.

    All output features are z-scored within season so a 2015 team is compared
    only to 2015 peers — same convention as TeamDiagnosticModel.

    Parameters
    ----------
    team_stats_df : DataFrame
        From get_team_stats() — one row per (team, season).
    snap_counts_df : DataFrame, optional
        From get_snap_counts() — adds Tier 2 positional snap share columns.

    Returns
    -------
    DataFrame — one row per (team, season).
    """
    if team_stats_df.empty:
        raise ValueError("team_stats_df is empty.")

    df = team_stats_df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    out = df[["season", "team"]].copy()

    # ── Tier 1 ───────────────────────────────────────────────────────────────
    n_pass = df.get("offense_n_plays_pass", pd.Series(np.nan, index=df.index))
    n_run = df.get("offense_n_plays_run", pd.Series(np.nan, index=df.index))
    out["_pass_rate"] = n_pass / (n_pass + n_run).replace(0, np.nan)
    out["_air_yards_scheme"] = df.get(
        "offense_ave_air_yards", pd.Series(np.nan, index=df.index)
    )
    out["_yac_scheme"] = df.get("offense_ave_yac", pd.Series(np.nan, index=df.index))
    out["_pass_epa_efficiency"] = df.get(
        "offense_ave_epa_pass", pd.Series(np.nan, index=df.index)
    )
    out["_run_epa_efficiency"] = df.get(
        "offense_ave_epa_run", pd.Series(np.nan, index=df.index)
    )
    out["_pass_success_rate"] = df.get(
        "offense_success_rate_pass", pd.Series(np.nan, index=df.index)
    )
    out["_run_success_rate"] = df.get(
        "offense_success_rate_run", pd.Series(np.nan, index=df.index)
    )
    # Sign-flip: higher = better / more emphatic defense
    out["_def_pass_scheme"] = -df.get(
        "defense_ave_epa_pass", pd.Series(np.nan, index=df.index)
    )
    out["_def_run_scheme"] = -df.get(
        "defense_ave_epa_run", pd.Series(np.nan, index=df.index)
    )

    for raw in [c for c in out.columns if c.startswith("_")]:
        out[raw[1:]] = _zscore_within_season(out, raw)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])

    # ── Tier 2 ───────────────────────────────────────────────────────────────
    _PG_SNAP: dict[str, str] = {
        "WR": "wr_snap_share",
        "RB": "rb_snap_share",
        "TE": "te_snap_share",
        "OL": "ol_snap_share",
        "EDGE": "dl_snap_share",
        "IDL": "dl_snap_share",
        "LB": "lb_snap_share",
        "CB": "db_snap_share",
        "DB": "db_snap_share",
    }
    if snap_counts_df is not None and not snap_counts_df.empty:
        sc = snap_counts_df.copy()
        if "game_type" in sc.columns:
            sc = sc[sc["game_type"] == "REG"]
        if "position" in sc.columns and "offense_snaps" in sc.columns:
            sc["_pg"] = sc["position"].map(_pos_group)
            sc["_col"] = sc["_pg"].map(_PG_SNAP)
            totals = (
                sc.groupby(["team", "season"])["offense_snaps"]
                .sum()
                .reset_index(name="_total")
            )
            pos_snaps = (
                sc[sc["_col"].notna()]
                .groupby(["team", "season", "_col"])["offense_snaps"]
                .sum()
                .reset_index()
                .merge(totals, on=["team", "season"], how="left")
            )
            pos_snaps["_share"] = pos_snaps["offense_snaps"] / pos_snaps[
                "_total"
            ].replace(0, np.nan)
            wide = pos_snaps.pivot_table(
                index=["team", "season"],
                columns="_col",
                values="_share",
                aggfunc="sum",
            ).reset_index()
            wide.columns.name = None
            for col in SCHEME_FEATURES_T2:
                if col not in wide.columns:
                    wide[col] = np.nan
            out = out.merge(
                wide[["team", "season"] + SCHEME_FEATURES_T2],
                on=["team", "season"],
                how="left",
            )
            for col in SCHEME_FEATURES_T2:
                out[col] = _zscore_within_season(out, col)
            log.info("Tier 2 snap share features added.")
        else:
            log.warning("snap_counts_df missing required columns — Tier 2 skipped.")
    else:
        log.info("snap_counts_df not supplied — Tier 2 skipped.")

    n_feats = len(
        [c for c in SCHEME_FEATURES_T1 + SCHEME_FEATURES_T2 if c in out.columns]
    )
    log.info(
        "build_team_scheme_profile → %s rows | %d features", f"{len(out):,}", n_feats
    )
    return out


# ── build_player_roster_profile ───────────────────────────────────────────────


def build_player_roster_profile(
    athletic_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join athletic + production gold profiles into one player feature DataFrame.

    Parameters
    ----------
    athletic_df : DataFrame
        From get_athletic_profiles() — one row per player (combine-level).
    production_df : DataFrame
        From get_production_profiles() — one row per (player_id, season).

    Returns
    -------
    DataFrame — one row per (player_id, season) with ATHLETIC_DIMS +
    PRODUCTION_DIMS columns. Raw dimensions are z-scored within
    (position_group, season).
    """
    ath = athletic_df.copy()
    prod = production_df.copy()

    for df in [ath, prod]:
        if "position_group" not in df.columns and "position" in df.columns:
            df["position_group"] = df["position"].map(_pos_group).fillna("UNK")

    if "season" in prod.columns:
        prod["season"] = pd.to_numeric(prod["season"], errors="coerce").astype("Int64")

    ath_slim = (
        ath[
            ["player_id", "position_group"]
            + [c for c in ATHLETIC_DIMS if c in ath.columns]
        ]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"], keep="last")
    )

    result = prod.merge(ath_slim, on="player_id", how="outer", suffixes=("", "_ath"))

    if "position_group_ath" in result.columns:
        result["position_group"] = result["position_group"].fillna(
            result.pop("position_group_ath")
        )

    result["position_group"] = result["position_group"].fillna("UNK")
    result["season"] = (
        result.get("season", pd.Series(dtype="Int64")).fillna(0).astype("Int64")
    )

    for col in _NEEDS_ZSCORE:
        if col in result.columns:
            result[col] = _zscore_within(result, col, ["position_group", "season"])

    before = len(result)
    result = result.dropna(subset=["player_id"])
    if dropped := before - len(result):
        log.warning("Dropped %d rows with no player_id.", dropped)

    meta = ["player_id", "player_name", "position", "position_group", "season"]
    dims = ATHLETIC_DIMS + PRODUCTION_DIMS
    keep = [c for c in meta if c in result.columns] + [
        c for c in dims if c in result.columns
    ]
    result = result[result["position"].notna() | result["nfl_production_score"].notna()]
    result = result[keep].reset_index(drop=True)

    log.info(
        "build_player_roster_profile → %s player-seasons | "
        "%s with athletic dims | %s with production dims",
        f"{len(result):,}",
        (
            f"{result['speed_score'].notna().sum():,}"
            if "speed_score" in result.columns
            else "0"
        ),
        (
            f"{result['nfl_production_score'].notna().sum():,}"
            if "nfl_production_score" in result.columns
            else "0"
        ),
    )
    return result


# ── Data lake query helpers ───────────────────────────────────────────────────


async def get_team_stats(
    client: DataLakeClient,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch team_stats from the data lake via POST /query."""
    sql = "SELECT * FROM team_stats"
    if season:
        sql += f" WHERE season = {season}"
    return await client.query(sql)


# ── fetch_feature_matrices ────────────────────────────────────────────────────


async def fetch_team_scheme_profiles(
    client: DataLakeClient,
    season: Optional[int] = None,
    include_snap_counts: bool = True,
) -> pd.DataFrame:
    """
    Fetch all source tables and return team scheme profiles.

    Sequential fetches to avoid saturating the remote data lake — same
    pattern as fetch_feature_matrix() in player_projection/features.py.

    Parameters
    ----------
    client : DataLakeClient
    season : int, optional
        Filter to a single season. None returns all seasons.
    include_snap_counts : bool
        When True, attempts Tier 2 snap share features. Skips gracefully
        if snap_counts table is not yet available.
    """
    log.info("Fetching team stats …")
    team_stats = await get_team_stats(client, season=season)
    log.info("  team_stats: %d rows", len(team_stats))

    snap_counts = None
    if include_snap_counts:
        try:
            log.info("Fetching snap counts …")
            snap_counts = await get_snap_counts(client, season=season)
            snap_counts = snap_counts[snap_counts["game_type"] == "REG"]
            log.info("  snap_counts: %d rows", len(snap_counts))
        except Exception as exc:
            log.info("snap_counts not available yet (%s) — Tier 2 skipped.", exc)

    return build_team_scheme_profile(team_stats, snap_counts_df=snap_counts)


async def fetch_player_roster_profiles(
    client: DataLakeClient,
    season: Optional[int] = None,
) -> pd.DataFrame:
    log.info("Fetching athletic profiles …")
    athletic = await get_athletic_profiles(client)
    log.info("  athletic_profiles: %d rows", len(athletic))

    log.info("Fetching production profiles …")
    production = await get_production_profiles(client, season=season)
    log.info("  production_profiles: %d rows", len(production))

    # Back-fill player_name into production from athletic profiles
    if (
        "player_name" not in production.columns
        or production["player_name"].isna().all()
    ):
        name_map = (
            athletic.dropna(subset=["player_id"])[["player_id", "player_name"]]
            .drop_duplicates(subset=["player_id"])
            .set_index("player_id")["player_name"]
        )
        production["player_name"] = production["player_id"].map(name_map)

    return build_player_roster_profile(athletic, production)
