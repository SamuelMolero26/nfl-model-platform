"""
serving/models/draft_optimizer/features.py

Step 1 of 3 — prospect score assembly
======================================
Assembles the scored prospect DataFrame consumed by the CVXPY optimizer.

TRANSPORT LAYER (two paths, intentionally different)
─────────────────────────────────────────────────────
  Inference   fetch_prospect_scores(client, draft_year, ...)
              Async DataLakeClient → POST /query on the HTTP API.
              Same pattern as Player Projection so ModelRegistry /
              NullClaw see a consistent interface across all models.

  Calibration fetch_prospect_scores_batch(year_start, year_end)
              Sync DuckDB → local Parquet directly.
              Called once during optimizer calibration (building
              expected_av_curve + quota_defaults), not at inference
              time. Reads 20+ years of data in a single scan —
              the HTTP API was not designed for bulk pipeline work.

DATA SOURCES
─────────────────────────────────────────────────────
  draft_value_history      (curated)  round / pick / w_av / percentile
  player_athletic_profiles (curated)  speed / agility / burst / strength / size

OUTPUT SCHEMA (inference path)
─────────────────────────────────────────────────────
  player_id               str    gsis_id (NaN for pre-draft prospects)
  player_name             str
  season                  int    draft year
  draft_team              str
  position                str    e.g. "WR"
  position_group          str    e.g. "SKILL"  — from shared positions module
  round                   Int64
  pick                    Int64  overall pick number
  age                     float
  college                 str
  career_value_score      float  0–100, normalised (model or percentile fallback)
  draft_value_percentile  float  0–100, ADP historical baseline
  draft_value_score       float  z-score within round (diagnostics only)
  car_av                  float  historical ground truth (NaN for future drafts)
  w_av                    float
  speed_score             float  NaN if player skipped combine drill
  agility_score           float
  burst_score             float
  strength_score          float
  size_score              float
  projection_source       str    "model" | "percentile_fallback"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path bootstrap — CWD-first since python -m is run from the project root.
# Falls back to walking upward from __file__ looking for known root markers.
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "config.py").exists() or (cwd / "duckdb_client.py").exists():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.py").exists() or (parent / "duckdb_client.py").exists():
            return parent
    return cwd  # last resort — let the import error surface naturally


_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Imports — shared position module (single source of truth)
# ---------------------------------------------------------------------------
from serving.models.shared.positions import (
    pos_group,
    POSITION_GROUP_ORDER,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Imports — transport layers
# ---------------------------------------------------------------------------
from serving.data_lake_client import DataLakeClient  # noqa: E402
from serving.data_lake_client.queries import (  # noqa: E402
    get_athletic_profiles,
    get_draft_value_history,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL — calibration path (DuckDB, batch, local Parquet)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Projection model helpers
# ---------------------------------------------------------------------------


def _load_player_projection_model():
    """
    Load Player Projection from ModelRegistry.
    Returns None silently if the artifact is not yet available.
    """
    try:
        from serving.models.registry import ModelRegistry  # noqa: PLC0415

        model = ModelRegistry().get("player_projection")
        if model is None:
            logger.warning(
                "player_projection not in ModelRegistry — "
                "career_value_score falls back to draft_value_percentile."
            )
        return model
    except Exception as exc:
        logger.warning(
            "Could not load Player Projection model (%s) — "
            "career_value_score falls back to draft_value_percentile.",
            exc,
        )
        return None


def _apply_projection_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Attach career_value_score (0–100) and projection_source to df.

    If model is None → percentile fallback for the whole frame.
    If model.predict() fails on a row → that row falls back individually,
    the rest continue. A single bad prospect never aborts the full run.

    Normalisation: raw model output is AV units (≈0–120). We rescale to
    0–100 so it sits on the same scale as draft_value_percentile, keeping
    the CVXPY objective function dimensionally consistent.
    """
    if model is None:
        df = df.copy()
        df["career_value_score"] = df["draft_value_percentile"].fillna(50.0)
        df["projection_source"] = "percentile_fallback"
        logger.warning(
            "Using draft_value_percentile fallback for all %s prospects.",
            f"{len(df):,}",
        )
        return df

    scores: list[float] = []
    sources: list[str] = []
    fallback_count = 0

    for _, row in df.iterrows():
        try:
            result = model.predict(row.to_dict())
            raw = float(result.get("career_value_score", np.nan))
            if np.isnan(raw):
                raise ValueError("model returned NaN")
            scores.append(raw)
            sources.append("model")
        except Exception as exc:
            logger.debug(
                "Projection failed for %s (%s): %s — percentile fallback.",
                row.get("player_name", "unknown"),
                row.get("player_id", "?"),
                exc,
            )
            fallback = float(row.get("draft_value_percentile") or 50.0)
            scores.append(fallback)
            sources.append("percentile_fallback")
            fallback_count += 1

    if fallback_count:
        logger.warning(
            "%s / %s prospects used percentile fallback.",
            f"{fallback_count:,}",
            f"{len(df):,}",
        )

    df = df.copy()
    df["career_value_score"] = scores
    df["projection_source"] = sources

    # Normalise to 0–100 (min-max within this draft class)
    col = df["career_value_score"]
    lo, hi = col.min(), col.max()
    if hi > lo:
        df["career_value_score"] = (col - lo) / (hi - lo) * 100.0
    else:
        df["career_value_score"] = 50.0

    return df


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

_FINAL_COLS = [
    "player_id",
    "player_name",
    "season",
    "draft_team",
    "position",
    "position_group",
    "round",
    "pick",
    "age",
    "college",
    "career_value_score",
    "draft_value_percentile",
    "draft_value_score",
    "car_av",
    "w_av",
    "speed_score",
    "agility_score",
    "burst_score",
    "strength_score",
    "size_score",
    "projection_source",
]


def _attach_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add position_group column using the shared canonical mapping."""
    df = df.copy()
    df["position_group"] = df["position"].map(pos_group).fillna("UNK")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column order and numeric types. Keeps NaN (not 0) for missing."""
    present = [c for c in _FINAL_COLS if c in df.columns]
    df = df[present].copy()
    for col in ("pick", "round"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    float_cols = [
        "career_value_score",
        "draft_value_percentile",
        "draft_value_score",
        "car_av",
        "w_av",
        "speed_score",
        "agility_score",
        "burst_score",
        "strength_score",
        "size_score",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _empty_schema() -> pd.DataFrame:
    return pd.DataFrame(columns=_FINAL_COLS)


def _merge_athletic(dvh: pd.DataFrame, athletic: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join athletic profiles onto draft_value_history on player_id.
    Players who skipped the combine or whose player_id didn't resolve
    get NaN athletic scores — the optimizer handles this gracefully.
    """
    if athletic.empty or "player_id" not in athletic.columns:
        logger.warning(
            "Athletic profiles empty or missing player_id — "
            "all combine scores will be NaN."
        )
        return dvh

    athletic_cols = [
        "player_id",
        "speed_score",
        "agility_score",
        "burst_score",
        "strength_score",
        "size_score",
    ]
    slim = athletic[
        [c for c in athletic_cols if c in athletic.columns]
    ].drop_duplicates(subset=["player_id"], keep="first")
    return dvh.merge(slim, on="player_id", how="left")


# ---------------------------------------------------------------------------
# Public API — INFERENCE (async, DataLakeClient)
# ---------------------------------------------------------------------------


async def fetch_prospect_scores(
    client: DataLakeClient,
    draft_year: int,
    model=None,
    *,
    min_pick: int = 1,
    max_pick: int = 262,
    positions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return the scored prospect DataFrame for draft_year.

    Uses the async DataLakeClient (POST /query) so it is consistent with
    the Player Projection model and callable from NullClaw / ModelRegistry
    without special-casing the transport layer.

    Parameters
    ----------
    client      : DataLakeClient instance (injected by model.py / NullClaw).
    draft_year  : Target draft class, e.g. 2024.
    model       : Player Projection model instance.
                  None  → auto-load from ModelRegistry.
                  False → force percentile fallback, skip registry lookup.
    min_pick / max_pick : Narrow to a pick range (e.g. team only has picks
                          33–100 available). Applied after the join.
    positions   : Optional position allow-list, e.g. ["QB", "WR"].
                  Filtered after position_group is attached.

    Returns
    -------
    pd.DataFrame  one row per prospect, sorted by pick ascending.
                  Empty DataFrame with schema columns on any data failure.
    """
    logger.info(
        "Fetching prospect scores for %s (picks %s–%s) via DataLakeClient.",
        draft_year,
        min_pick,
        max_pick,
    )

    # -- Fetch draft_value_history for the target year ----------------------
    # Sequential, not concurrent — mirrors Player Projection's comment about
    # avoiding server contention over Tailscale.
    logger.info("  [1/2] draft_value_history …")
    try:
        dvh = await get_draft_value_history(
            client, year_start=draft_year, year_end=draft_year
        )
    except Exception as exc:
        logger.error("draft_value_history fetch failed: %s", exc)
        return _empty_schema()

    if dvh.empty:
        logger.warning(
            "No rows in draft_value_history for year %s. "
            "Has the ingestion pipeline run for this draft class?",
            draft_year,
        )
        return _empty_schema()

    # -- Fetch athletic profiles (all players — join on player_id) ----------
    logger.info("  [2/2] player_athletic_profiles …")
    try:
        athletic = await get_athletic_profiles(client)
    except Exception as exc:
        logger.warning(
            "Athletic profiles fetch failed (%s) — combine scores will be NaN.", exc
        )
        athletic = pd.DataFrame()

    # -- Python-side join (same as Player Projection's build_features) ------
    df = _merge_athletic(dvh, athletic)

    # -- Pick range filter ---------------------------------------------------
    if "pick" in df.columns:
        df = df[pd.to_numeric(df["pick"], errors="coerce").between(min_pick, max_pick)]

    if df.empty:
        logger.warning(
            "No prospects remain after pick filter %s–%s for year %s.",
            min_pick,
            max_pick,
            draft_year,
        )
        return _empty_schema()

    # -- Attach position group (shared canonical mapping) -------------------
    df = _attach_position_group(df)

    # -- Optional position filter -------------------------------------------
    if positions:
        pos_upper = {p.strip().upper() for p in positions}
        df = df[df["position"].str.upper().isin(pos_upper)].copy()
        if df.empty:
            logger.warning(
                "No prospects remain after position filter %s for year %s.",
                positions,
                draft_year,
            )
            return _empty_schema()

    # -- Attach career_value_score via Player Projection model --------------
    if model is None:
        model = _load_player_projection_model()
    elif model is False:
        model = None  # explicit force-fallback

    df = _apply_projection_scores(df, model)

    # -- Finalise -----------------------------------------------------------
    df = _coerce_types(df)

    logger.info(
        "Prospect scores ready: %s prospects | %s model scores | "
        "%s with speed_score.",
        f"{len(df):,}",
        f"{(df.get('projection_source', pd.Series()) == 'model').sum():,}",
        (
            f"{df['speed_score'].notna().sum():,}"
            if "speed_score" in df.columns
            else "n/a"
        ),
    )
    return df


# ---------------------------------------------------------------------------
# Public API — CALIBRATION (async, HTTP API)
# ---------------------------------------------------------------------------


async def fetch_prospect_scores_batch(
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Fetch historical prospect data for a range of draft years via the data lake API.

    Used exclusively during optimizer calibration — building expected_av_curve,
    positional_quota_defaults, and need_score_thresholds from historical data.
    NOT called at inference time.

    A single JOIN query is used to minimise round trips over Tailscale.
    Does NOT attach career_value_score or projection_source — calibration
    uses car_av / w_av as historical ground truth directly.

    Parameters
    ----------
    year_start / year_end : inclusive range, e.g. 2000, 2020.

    Returns
    -------
    pd.DataFrame  one row per drafted player across all years.
    """
    sql = f"""
        SELECT
            dvh.player_id,
            dvh.player_name,
            dvh.season,
            dvh.team          AS draft_team,
            dvh.position,
            dvh.round,
            dvh.pick,
            dvh.age,
            dvh.college,
            dvh.car_av,
            dvh.w_av,
            dvh.draft_value_score,
            dvh.draft_value_percentile,
            ap.speed_score,
            ap.agility_score,
            ap.burst_score,
            ap.strength_score,
            ap.size_score
        FROM draft_value_history dvh
        LEFT JOIN player_athletic_profiles ap
               ON dvh.player_id = ap.player_id
        WHERE dvh.season BETWEEN {int(year_start)} AND {int(year_end)}
        ORDER BY dvh.season ASC, dvh.pick ASC NULLS LAST
    """
    logger.info(
        "Calibration batch fetch: draft years %s–%s via data lake API.",
        year_start,
        year_end,
    )
    try:
        async with DataLakeClient() as client:
            df = await client.query(sql)
    except Exception as exc:
        logger.error("Calibration batch query failed: %s", exc)
        return pd.DataFrame()

    if df.empty:
        logger.warning(
            "No historical draft data found for %s–%s. "
            "Run ingestion pipeline Stage 3 first.",
            year_start,
            year_end,
        )
        return pd.DataFrame()

    df = _attach_position_group(df)

    # Coerce numeric columns only — no career_value_score column here
    for col in ("pick", "round"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("car_av", "w_av", "draft_value_score", "draft_value_percentile"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Calibration batch ready: %s picks across %s draft classes.",
        f"{len(df):,}",
        f"{df['season'].nunique():,}",
    )
    return df
