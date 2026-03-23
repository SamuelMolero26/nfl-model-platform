"""
Player Projection — Feature Engineering
========================================

Builds the feature matrix for predicting Career Approximate Value (car_av).

DATA SOURCES (all from the data lake gold tables):
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│ Table                           │ What we use                                  │
├─────────────────────────────────┼──────────────────────────────────────────────┤
│ player_athletic_profiles        │ speed_score, agility_score, burst_score,     │
│  (lake/curated/)                │ strength_score, size_score                   │
├─────────────────────────────────┼──────────────────────────────────────────────┤
│ draft_value_history             │ draft_value_score, draft_value_percentile    │
│  (lake/curated/)                │ (z-score of historical AV by pick slot)      │
├─────────────────────────────────┼──────────────────────────────────────────────┤
│ combine (lake/staged/)          │ raw measurables: height, weight, forty, etc. │
│                                 │ + missing-drill flags                         │
├─────────────────────────────────┼──────────────────────────────────────────────┤
│ draft_picks (lake/staged/)      │ round, pick, age, position  +  car_av target │
└─────────────────────────────────┴──────────────────────────────────────────────┘

WHY use the gold profiles instead of re-computing from raw combine?
  The data lake already computed speed_score, agility_score, etc. with the correct
  formulas and normalisation. Re-implementing them here would risk drift.
  We pull the finished scores and use them directly as features.

TARGET: car_av (Career Approximate Value, from Pro Football Reference)
  0–10   = bust / career backup
  10–30  = serviceable starter
  30–60  = Pro Bowl-calibre
  60+    = elite / All-Pro / HOF track

WHAT WE STILL ENGINEER HERE:
  - Position group one-hot encoding (broad groups — enough sample per group)
  - Missing drill flags (skipping a drill is itself a scouting signal)
  - Round × draft_value_score interaction (pick slot modulates the signal)
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import pandas as pd

from serving.data_lake_client import DataLakeClient
from serving.data_lake_client.queries import (
    get_athletic_profiles,
    get_combine_data,
    get_draft_picks,
    get_draft_value_history,
)

# Position grouping
POSITION_GROUPS = {
    "QB": "QB",
    "RB": "SKILL",
    "FB": "SKILL",
    "HB": "SKILL",
    "WR": "SKILL",
    "TE": "SKILL",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "OL": "OL",
    "G": "OL",
    "T": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "ILB": "LB",
    "OLB": "LB",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "SS": "DB",
    "FS": "DB",
    "DB": "DB",
    "K": "SPEC",
    "P": "SPEC",
    "LS": "SPEC",
}

POSITION_GROUP_ORDER = ["QB", "SKILL", "OL", "DL", "LB", "DB", "SPEC"]

# Raw combine columns we keep as features alongside the gold scores
COMBINE_RAW_COLS = [
    "forty_yard",
    "vertical_in",
    "bench_reps",
    "broad_jump_in",
    "three_cone",
    "shuttle",
    "height_in",
    "weight_lbs",
]

# Gold athletic score columns (pre-computed by the data lake)
GOLD_ATHLETIC_COLS = [
    "speed_score",
    "agility_score",
    "burst_score",
    "strength_score",
    "size_score",
]

# Gold draft value columns
GOLD_DRAFT_VALUE_COLS = ["draft_value_score", "draft_value_percentile"]

# Drills for which we generate missing-value flags
DRILL_COLS = [
    "forty_yard",
    "vertical_in",
    "bench_reps",
    "broad_jump_in",
    "three_cone",
    "shuttle",
]

# Full ordered feature list (used by model.py and train.py)
FEATURE_COLS = (
    COMBINE_RAW_COLS
    + GOLD_ATHLETIC_COLS
    + GOLD_DRAFT_VALUE_COLS
    + ["draft_round", "draft_pick", "age"]
    + [f"pos_{g}" for g in POSITION_GROUP_ORDER]
    + [f"missing_{c}" for c in DRILL_COLS]
    + ["round_x_draft_value"]
)


# Core join + engineering


def build_features(
    combine_df: pd.DataFrame,
    draft_df: pd.DataFrame,
    athletic_df: pd.DataFrame,
    draft_value_df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Join all four source DataFrames and return the feature matrix.

    Parameters
    ----------
    combine_df      : from get_combine_data()   — raw measurables
    draft_df        : from get_draft_picks()    — round/pick/age/car_av
    athletic_df     : from get_athletic_profiles() — gold scores
    draft_value_df  : from get_draft_value_history() — pick-level z-scores
    include_target  : if True, append 'car_av' column

    Returns
    -------
    DataFrame indexed by (player_name, draft_year) with FEATURE_COLS columns.
    """
    combine = combine_df.copy()
    draft = draft_df.copy()
    athletic = athletic_df.copy() if athletic_df is not None else pd.DataFrame()
    draft_value = (
        draft_value_df.copy() if draft_value_df is not None else pd.DataFrame()
    )

    # --- normalise draft_picks column names ---
    # Real schema: season → draft_year, round → draft_round, pick → draft_pick
    draft = draft.rename(
        columns={
            "pfr_player_name": "player_name",
            "season": "draft_year",
            "pick": "draft_pick",
            "round": "draft_round",
        },
        errors="ignore",
    )

    for col in ["draft_round", "draft_pick", "age", "car_av", "w_av"]:
        if col in draft.columns:
            draft[col] = pd.to_numeric(draft[col], errors="coerce")

    # draft_picks already contains draft_value_score and draft_value_percentile
    for col in GOLD_DRAFT_VALUE_COLS:
        if col in draft.columns:
            draft[col] = pd.to_numeric(draft[col], errors="coerce")

    combine["draft_year"] = pd.to_numeric(combine["draft_year"], errors="coerce")
    combine = combine.dropna(subset=["draft_year"])

    combine["draft_year"] = combine["draft_year"].astype(int)
    draft["draft_year"] = draft["draft_year"].astype(int)

    combine = combine.sort_values(
        ["draft_year", "draft_round", "draft_pick"],
        na_position="last",
    ).drop_duplicates(subset=["player_name", "draft_year"], keep="first")
    draft = draft.sort_values(
        ["draft_year", "draft_round", "draft_pick"],
        na_position="last",
    ).drop_duplicates(subset=["player_name", "draft_year"], keep="first")

    draft_cols = [
        "player_name",
        "draft_year",
        "draft_round",
        "draft_pick",
        "age",
        "position",
        "car_av",
    ] + [c for c in GOLD_DRAFT_VALUE_COLS if c in draft.columns]
    base = pd.merge(
        combine,
        draft[draft_cols],
        on=["player_name", "draft_year"],
        how="inner",
        suffixes=("_combine", "_draft"),
    )
    if base.empty:
        raise ValueError(
            "combine × draft_picks join is empty — "
            "player_name or draft_year types/values don't align between tables."
        )

    # Use draft position (more standardised) over combine position
    if "position_draft" in base.columns:
        base["position"] = base["position_draft"].fillna(
            base.get("position_combine", pd.Series(dtype=str))
        )
    elif "position_combine" in base.columns:
        base["position"] = base["position_combine"]

    # Mark any that are still missing as NaN
    for col in GOLD_DRAFT_VALUE_COLS:
        if col not in base.columns:
            base[col] = np.nan

    if (
        not athletic.empty
        and "player_id" in base.columns
        and "player_id" in athletic.columns
    ):
        athletic_slim = (
            athletic[
                ["player_id"] + [c for c in GOLD_ATHLETIC_COLS if c in athletic.columns]
            ]
            .copy()
            .drop_duplicates(subset=["player_id"], keep="first")
        )
        base = pd.merge(base, athletic_slim, on="player_id", how="left")
    else:
        for col in GOLD_ATHLETIC_COLS:
            if col not in base.columns:
                base[col] = np.nan

    # --- feature engineering ---
    # Position group one-hot
    base["pos_group"] = base["position"].map(POSITION_GROUPS).fillna("SPEC")
    for grp in POSITION_GROUP_ORDER:
        base[f"pos_{grp}"] = (base["pos_group"] == grp).astype(int)

    # Missing drill flags (1 = player skipped that drill)
    for col in DRILL_COLS:
        if col in base.columns:
            base[f"missing_{col}"] = base[col].isna().astype(int)
        else:
            base[f"missing_{col}"] = 1  # treat entire missing column as skipped

    # Interaction: round × draft_value_score
    # High draft_value_score in round 1 is more meaningful than the same score in round 7
    base["round_x_draft_value"] = base.get(
        "draft_round", pd.Series(np.nan, index=base.index)
    ) * base.get("draft_value_score", pd.Series(np.nan, index=base.index))

    available_features = [c for c in FEATURE_COLS if c in base.columns]
    result = base[available_features].copy()

    if include_target:
        if "car_av" not in base.columns:
            raise ValueError(
                "car_av not found in draft_picks — cannot build training target."
            )
        result["car_av"] = base["car_av"]

    # Multi-index: (player_name, draft_year) for easy slicing during validation
    result.index = pd.MultiIndex.from_arrays(
        [base["player_name"], base["draft_year"]],
        names=["player_name", "draft_year"],
    )

    return result


async def fetch_feature_matrix(
    client: DataLakeClient,
    year_start: Optional[int] = 2000,
    year_end: Optional[int] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Fetch all source tables sequentially and return the complete feature matrix.

    Sequential (not parallel) to avoid saturating the remote data lake over
    Tailscale. Each query can be a full table scan — parallel requests risk
    timeouts and server contention.

    Used by:
      - train.py  (include_target=True,  batch of historical draft classes)
      - model.py  (include_target=False, single prospect at inference time)
    """
    log = logging.getLogger(__name__)

    log.info("Fetching combine data …")
    combine_df = await get_combine_data(
        client, year_start=year_start, year_end=year_end
    )
    log.info(f"  combine: {len(combine_df)} rows")

    log.info("Fetching draft picks …")
    draft_df = await get_draft_picks(client, year_start=year_start, year_end=year_end)
    log.info(f"  draft_picks: {len(draft_df)} rows")
    # Note: draft_picks already contains draft_value_score + draft_value_percentile

    log.info("Fetching athletic profiles …")
    athletic_df = await get_athletic_profiles(client)
    log.info(f"  athletic_profiles: {len(athletic_df)} rows")

    return build_features(
        combine_df,
        draft_df,
        athletic_df,
        draft_value_df=None,
        include_target=include_target,
    )
