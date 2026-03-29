"""
Positional Flexibility — Feature Engineering
=============================================

Pure physical / athletic features only — draft value and pick signals are
deliberately excluded so the model measures raw athletic qualification,
not draft consensus.

DATA SOURCES
------------
  combine                  raw measurables  (height, weight, drills)
  player_athletic_profiles gold scores      (speed_score, agility_score, …)
  snap_counts              career snaps by position  (optional, snap-share labels only)
  draft_picks              car_av           (career quality weight for labels)

FEATURES
--------
  Physical raw   : forty_yard, vertical_in, bench_reps, broad_jump_in,
                   three_cone, shuttle, height_in, weight_lbs
  Gold athletic  : speed_score, agility_score, burst_score,
                   strength_score, size_score
  Derived        : bmi, speed_to_weight, relative_size
  Missingness    : missing_{drill} flags + n_drills_completed
                   (skipping a drill is itself an informative signal)

LABELS (training only)
--------------------------------------------------------------
Three label strategies are available:

  "archetype" (default)
  ---------------------
  For each position group G, compute the mean feature vector of all
  career-qualified (car_av ≥ QUALIFIED_THRESHOLD) primary-G players
  (the "archetype"). Then for every player P:

      dist_G         = Euclidean distance from P to archetype_G (standardised space)
      sigma_G        = mean within-group distance (position spread scale)
      affinity_G     = exp(−dist_G / sigma_G)
      label_G        = affinity_G × career_quality

  A player scores high at position G if their athletic profile RESEMBLES
  typical G players — regardless of whether they ever played there.  This
  identifies cross-position athletic fits (e.g. an LB who has DB athleticism
  will get an elevated DB score even with zero DB snaps).

  "snap_share"
  ------------
  Option B (legacy): snap_share_G × career_quality weighted by actual career
  snaps at each position.  Captures observed flex, but cannot surface latent
  positional versatility for players who never got the opportunity.

  "declared_position"
  -------------------
  Soft label = career_quality for primary declared position, 0 elsewhere.
  Useful as a training sanity check; rarely the best choice.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from serving.data_lake_client import DataLakeClient
from serving.data_lake_client.queries import (
    get_athletic_profiles,
    get_combine_data,
    get_draft_picks,
    get_snap_counts,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------

POSITION_GROUPS: dict[str, str] = {
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

POSITION_GROUP_ORDER: list[str] = ["QB", "SKILL", "OL", "DL", "LB", "DB", "SPEC"]
LABEL_COLS: list[str] = [f"label_{g}" for g in POSITION_GROUP_ORDER]

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

# career quality: car_av >= threshold → quality = 1.0
QUALIFIED_THRESHOLD: float = 8.0

# snap-share label thresholds (Option B)
# snap_share >= SNAP_SHARE_FULL   → raw label = 1.0 (played position extensively)
# snap_share >= MIN_SNAP_SHARE    → eligible for non-zero label
# snap_share <  MIN_SNAP_SHARE    → noise / garbage time — label zeroed out
SNAP_SHARE_FULL: float = 0.30  # 30% of career snaps = full label
MIN_SNAP_SHARE: float = 0.05  # 5% minimum — filters positional packages

# SPEC-specific minimum snap-share threshold.
# st_snaps includes PAT unit participation, so most NFL players cross the 5%
# general threshold even without being dedicated ST players. 30% captures
# genuine ST contributors (core coverage units, K/P/LS) while excluding
# players who merely line up on PAT snaps a few times per game.
SPEC_MIN_SNAP_SHARE: float = 0.30

# Small-class groups — used by train.py to choose calibration strategy
SMALL_CLASSES: list[str] = ["QB", "SPEC"]

# Archetype label tuning
# Only physical/athletic/derived features are used to define positional archetypes.
# Missingness flags are excluded — "skipped the 40" is positionally circular and
# dominates distance computation when included.
ARCHETYPE_FEATURE_COLS: list[str] = []  # populated after PHYSICAL_COLS etc. are defined

# Target fraction of players labeled as archetype-positive per position group.
# Bottom (1 - ARCHETYPE_POSITIVE_RATE) of affinity scores are zeroed out,
# creating genuine negatives and reducing base_rate from ~90% to ~25%.
ARCHETYPE_POSITIVE_RATE: float = 0.25

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

PHYSICAL_COLS: list[str] = [
    "forty_yard",
    "vertical_in",
    "bench_reps",
    "broad_jump_in",
    "three_cone",
    "shuttle",
    "height_in",
    "weight_lbs",
]

GOLD_ATHLETIC_COLS: list[str] = [
    "speed_score",
    "agility_score",
    "burst_score",
    "strength_score",
    "size_score",
]

DERIVED_COLS: list[str] = [
    "bmi",  # body mass index — distinguishes OL/DL from SKILL
    "speed_to_weight",  # speed score per unit body weight — pocket players vs burners
    "relative_size",  # weight / height — OL threshold signal
]

DRILL_COLS: list[str] = [
    "forty_yard",
    "vertical_in",
    "bench_reps",
    "broad_jump_in",
    "three_cone",
    "shuttle",
]

FEATURE_COLS: list[str] = (
    PHYSICAL_COLS
    + GOLD_ATHLETIC_COLS
    + DERIVED_COLS
    + [f"missing_{c}" for c in DRILL_COLS]
    + ["n_drills_completed"]
)

# Populated here after column lists are defined (referenced in ARCHETYPE_FEATURE_COLS above)
ARCHETYPE_FEATURE_COLS = PHYSICAL_COLS + GOLD_ATHLETIC_COLS + DERIVED_COLS


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------


def _size_bucket(weight_lbs: pd.Series) -> pd.Series:
    """Quartile-based weight bucket for group-median imputation."""
    return pd.cut(
        weight_lbs,
        bins=[0, 200, 245, 295, 9999],
        labels=["light", "medium", "heavy", "very_heavy"],
        right=True,
    ).astype(str)


def _impute_drills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing drill values using median by (pos_group, size_bucket).

    Falls back to pos_group median, then global median if a cell is empty.
    Imputation is applied ONLY to raw drill columns — missing_* flags are
    set before this step and are not altered here.
    """
    df = df.copy()
    df["_size_bucket"] = _size_bucket(
        df["weight_lbs"].fillna(df["weight_lbs"].median())
    )
    df["_pos_group"] = df["pos_group"]

    for col in DRILL_COLS:
        if col not in df.columns:
            continue
        group_med = df.groupby(["_pos_group", "_size_bucket"])[col].transform("median")
        pos_med = df.groupby("_pos_group")[col].transform("median")
        global_med = df[col].median()
        df[col] = df[col].fillna(group_med).fillna(pos_med).fillna(global_med)

    df = df.drop(columns=["_size_bucket", "_pos_group"])
    return df


# ---------------------------------------------------------------------------
# Snap-share label construction (Option B)
# ---------------------------------------------------------------------------


def _add_archetype_labels(
    base: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Add LABEL_COLS to ``base`` using athletic archetype affinity scoring.

    Requires ``base`` to have: ``pos_group``, ``car_av``, and all columns in
    ``feature_cols``.

    For each position group G:
        archetype_G = mean feature vector of career-qualified primary-G players
                      (in standardised feature space)
        sigma_G     = mean within-group distance to archetype (spread scale)
        label_G     = exp(−dist(P, archetype_G) / sigma_G) × career_quality(P)

    Players who look athletic like a G player score high at G even if they
    never played a snap there — enabling the model to surface latent
    cross-position athletic fits for coaching packages.
    """
    df = base.copy()
    career_quality = (df["car_av"] / QUALIFIED_THRESHOLD).clip(upper=1.0)

    # Use only physical/athletic/derived features for archetype distance.
    # Missingness flags are excluded: "skipped the 40" is a proxy for position
    # group membership and dominates distance computation when included.
    arch_cols = [c for c in ARCHETYPE_FEATURE_COLS if c in feature_cols]
    X_raw = df[arch_cols].astype(float)
    X_filled = X_raw.fillna(X_raw.median()).fillna(0.0)
    col_means = X_filled.mean()
    col_stds = X_filled.std().replace(0, 1.0)
    X_scaled = ((X_filled - col_means) / col_stds).values  # (n_players × n_feats)

    # Build position archetypes (mean of qualified primary players at each group)
    archetypes: dict[str, np.ndarray] = {}
    sigmas: dict[str, float] = {}

    for grp in POSITION_GROUP_ORDER:
        qualified_mask = (df["pos_group"] == grp) & (
            df["car_av"] >= QUALIFIED_THRESHOLD
        )
        fallback_mask = df["pos_group"] == grp

        if qualified_mask.sum() >= 10:
            grp_X = X_scaled[qualified_mask.values]
        elif fallback_mask.sum() > 0:
            grp_X = X_scaled[fallback_mask.values]
        else:
            archetypes[grp] = X_scaled.mean(axis=0)
            sigmas[grp] = 1.0
            continue

        arch = grp_X.mean(axis=0)
        within_dists = np.linalg.norm(grp_X - arch, axis=1)
        archetypes[grp] = arch
        sigmas[grp] = float(max(within_dists.mean(), 0.5))

    # Assign affinity labels with percentile floor.
    # Zero out the bottom (1 - ARCHETYPE_POSITIVE_RATE) of affinity scores so
    # only the top ~25% of players are labeled positive for each position.
    # This creates genuine negatives and reduces base_rate from ~90% → ~25%.
    floor_pct = (1.0 - ARCHETYPE_POSITIVE_RATE) * 100
    base_rates = {}
    for grp in POSITION_GROUP_ORDER:
        arch = archetypes[grp]
        sigma = sigmas[grp]
        dists = np.linalg.norm(X_scaled - arch, axis=1)
        affinity = np.exp(-dists / sigma)
        floor = float(np.percentile(affinity, floor_pct))
        affinity_floored = np.where(affinity >= floor, affinity, 0.0)
        df[f"label_{grp}"] = affinity_floored * career_quality.values
        base_rates[grp] = float((df[f"label_{grp}"] > 0).mean())

    log.info(
        f"  archetype labels: {len(df)} players  "
        f"base_rates: { {g: f'{r:.1%}' for g, r in base_rates.items()} }"
    )
    return df


def build_snap_labels(
    snap_df: pd.DataFrame,
    draft_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build multi-position soft labels from career snap shares.

    For each player × position group:
        snap_share_G   = career snaps at G / total career snaps
        career_quality = clip(car_av / QUALIFIED_THRESHOLD, 0, 1)
        label_G        = clip(snap_share_G / SNAP_SHARE_FULL, 0, 1) * career_quality
        label_G = 0    if snap_share_G < MIN_SNAP_SHARE

    Parameters
    ----------
    snap_df   : from get_snap_counts() — must have player_name, position,
                and a snaps column (offense_snaps | defense_snaps | snaps)
    draft_df  : from get_draft_picks() — must have player_name, draft_year, car_av

    Returns
    -------
    DataFrame indexed by player_name with LABEL_COLS + draft_year + car_av columns.
    """
    snap = snap_df.copy()

    # ── normalise player name column ──────────────────────────────────────
    for candidate in ("player_display_name", "player_name", "player"):
        if candidate in snap.columns:
            snap = snap.rename(columns={candidate: "player_name"})
            break
    if "player_name" not in snap.columns:
        raise ValueError(
            f"snap_counts table has no player name column. Found: {snap.columns.tolist()}"
        )

    # ── normalise offense/defense snaps column ───────────────────────────
    # nflreadr snap data can have offense_snaps, defense_snaps, or a combined snaps col
    if "offense_snaps" in snap.columns and "defense_snaps" in snap.columns:
        snap["_snaps"] = snap["offense_snaps"].fillna(0) + snap["defense_snaps"].fillna(
            0
        )
    elif "snaps" in snap.columns:
        snap["_snaps"] = snap["snaps"].fillna(0)
    elif "snap_count" in snap.columns:
        snap["_snaps"] = snap["snap_count"].fillna(0)
    else:
        raise ValueError(
            f"snap_counts table has no snaps column. Found: {snap.columns.tolist()}"
        )

    # ── normalise ST snaps (drives SPEC label) ────────────────────────────
    # nflreadr provides st_snaps separately. Any player — LB, DB, SKILL — who
    # participates heavily on special teams earns a positive SPEC label, not
    # just K/P/LS. K/P/LS themselves have near-zero offense/defense snaps, so
    # they only get SPEC credit here.
    if "st_snaps" in snap.columns:
        snap["_st_snaps"] = snap["st_snaps"].fillna(0)
    else:
        snap["_st_snaps"] = 0.0
        log.warning(
            "  st_snaps column not found in snap_counts — "
            "SPEC labels will be zero for all players. "
            "Verify get_snap_counts() returns st_snaps."
        )

    # ── normalise position column ─────────────────────────────────────────
    pos_col = "position" if "position" in snap.columns else "pos"
    # Unrecognised position strings → "OTHER" (not SPEC).
    # SPEC snap share is computed from _st_snaps, independent of position mapping.
    snap["pos_group"] = snap[pos_col].map(POSITION_GROUPS).fillna("OTHER")

    # ── aggregate career offense/defense snaps per (player_name, pos_group) ─
    career = (
        snap.groupby(["player_name", "pos_group"])["_snaps"]
        .sum()
        .reset_index()
        .rename(columns={"_snaps": "group_snaps"})
    )
    # Drop unrecognised-position rows — they don't belong to any group
    career = career[career["pos_group"] != "OTHER"]

    # ── aggregate ST snaps per player (drives SPEC label) ─────────────────
    st_career = (
        snap.groupby("player_name")["_st_snaps"]
        .sum()
        .reset_index()
        .rename(columns={"_st_snaps": "spec_snaps"})
    )

    # Total snaps = offense + defense + ST — shared denominator for all groups
    total_od = (
        career.groupby("player_name")["group_snaps"]
        .sum()
        .reset_index()
        .rename(columns={"group_snaps": "total_od_snaps"})
    )
    total = total_od.merge(st_career, on="player_name", how="left")
    total["spec_snaps"] = total["spec_snaps"].fillna(0.0)
    total["total_snaps"] = total["total_od_snaps"] + total["spec_snaps"]

    career = career.merge(total[["player_name", "total_snaps"]], on="player_name")
    career["snap_share"] = career["group_snaps"] / career["total_snaps"].clip(lower=1)

    # ── pivot to wide: one row per player, one column per non-SPEC group ──
    wide = career.pivot_table(
        index="player_name", columns="pos_group", values="snap_share", fill_value=0.0
    )
    wide.columns = [f"_share_{c}" for c in wide.columns]
    wide = wide.reset_index()

    # ── add _share_SPEC from ST snaps (replaces position-mapped approach) ─
    # K/P/LS → SPEC in POSITION_GROUPS, so the pivot may already contain
    # _share_SPEC based on their near-zero O/D snaps. Drop it first to avoid
    # a suffix conflict (_share_SPEC_x / _share_SPEC_y) during the merge.
    if "_share_SPEC" in wide.columns:
        wide = wide.drop(columns=["_share_SPEC"])

    spec_share = total[["player_name", "total_snaps"]].merge(
        st_career, on="player_name", how="left"
    )
    spec_share["spec_snaps"] = spec_share["spec_snaps"].fillna(0.0)
    spec_share["_share_SPEC"] = spec_share["spec_snaps"] / spec_share[
        "total_snaps"
    ].clip(lower=1)
    wide = wide.merge(
        spec_share[["player_name", "_share_SPEC"]], on="player_name", how="left"
    )
    wide["_share_SPEC"] = wide["_share_SPEC"].fillna(0.0)

    n_spec_pos = (wide["_share_SPEC"] >= MIN_SNAP_SHARE).sum()
    log.info(
        f"  SPEC positives (st_snaps >= {MIN_SNAP_SHARE:.0%}): {n_spec_pos}/{len(wide)}"
    )

    # ── join car_av for career quality weight ─────────────────────────────
    draft = draft_df.copy()
    draft = draft.rename(
        columns={"season": "draft_year", "pfr_player_name": "player_name"},
        errors="ignore",
    )
    draft["draft_year"] = pd.to_numeric(draft["draft_year"], errors="coerce").astype(
        "Int64"
    )
    draft["car_av"] = pd.to_numeric(draft.get("car_av"), errors="coerce")
    draft = draft.dropna(subset=["player_name", "draft_year", "car_av"])[
        ["player_name", "draft_year", "car_av"]
    ].drop_duplicates(
        subset=["player_name"], keep="last"
    )  # keep highest car_av season

    merged = wide.merge(
        draft[["player_name", "draft_year", "car_av"]], on="player_name", how="inner"
    )
    if merged.empty:
        raise ValueError(
            "snap_counts × draft_picks join is empty — check player_name alignment."
        )

    career_quality = (merged["car_av"] / QUALIFIED_THRESHOLD).clip(upper=1.0)

    # ── build label columns ───────────────────────────────────────────────
    for grp in POSITION_GROUP_ORDER:
        share_col = f"_share_{grp}"
        label_col = f"label_{grp}"
        # SPEC uses a higher minimum threshold — st_snaps includes PAT unit
        # participation so the standard 5% bar is crossed by most players.
        min_share = SPEC_MIN_SNAP_SHARE if grp == "SPEC" else MIN_SNAP_SHARE
        if share_col in merged.columns:
            raw_label = (merged[share_col] / SNAP_SHARE_FULL).clip(upper=1.0)
            below_min = merged[share_col] < min_share
            merged[label_col] = (raw_label * career_quality).where(~below_min, 0.0)
        else:
            merged[label_col] = 0.0

    # Drop intermediate share columns
    share_cols = [c for c in merged.columns if c.startswith("_share_")]
    merged = merged.drop(columns=share_cols)

    log.info(
        f"  snap labels: {len(merged)} players, "
        f"multi-position rate: "
        f"{(merged[LABEL_COLS].gt(0).sum(axis=1) > 1).mean():.1%}"
    )
    return merged


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------


def build_flex_features(
    combine_df: pd.DataFrame,
    athletic_df: pd.DataFrame,
    draft_df: Optional[pd.DataFrame] = None,
    snap_df: Optional[pd.DataFrame] = None,
    include_target: bool = True,
    label_strategy: str = "archetype",
) -> pd.DataFrame:
    """
    Join source tables and return the feature matrix (+ labels).

    Parameters
    ----------
    combine_df     : from get_combine_data()       — raw measurables
    athletic_df    : from get_athletic_profiles()  — gold scores
    draft_df       : from get_draft_picks()        — car_av for career quality
                     Required when include_target=True
    snap_df        : from get_snap_counts()        — career snaps by position
                     Only used when label_strategy="snap_share"
    include_target : if True, append LABEL_COLS
    label_strategy : one of:
                     "archetype"         — athletic-profile affinity labels (default)
                     "snap_share"        — snap-share × career quality (requires snap_df)
                     "declared_position" — declared position × career quality (fallback)

    Returns
    -------
    DataFrame indexed by (player_name, draft_year) with FEATURE_COLS columns
    (and LABEL_COLS when include_target=True).
    """
    # ── normalise combine ──────────────────────────────────────────────────
    combine = combine_df.copy()
    combine["draft_year"] = pd.to_numeric(combine["draft_year"], errors="coerce")
    combine = combine.dropna(subset=["draft_year"])
    combine["draft_year"] = combine["draft_year"].astype(int)
    combine = combine.sort_values(
        ["draft_year", "draft_round", "draft_pick"], na_position="last"
    ).drop_duplicates(subset=["player_name", "draft_year"], keep="first")

    # ── position group ─────────────────────────────────────────────────────
    combine["pos_group"] = combine["position"].map(POSITION_GROUPS).fillna("SPEC")

    # ── missing drill flags (set BEFORE imputation) ────────────────────────
    for col in DRILL_COLS:
        flag = f"missing_{col}"
        combine[flag] = combine[col].isna().astype(int) if col in combine.columns else 1

    combine["n_drills_completed"] = 6 - sum(combine[f"missing_{c}"] for c in DRILL_COLS)

    # ── impute missing drills ──────────────────────────────────────────────
    combine = _impute_drills(combine)

    # ── join athletic gold scores ──────────────────────────────────────────
    if (
        athletic_df is not None
        and not athletic_df.empty
        and "player_id" in combine.columns
        and "player_id" in athletic_df.columns
    ):
        athletic_slim = athletic_df[
            ["player_id"] + [c for c in GOLD_ATHLETIC_COLS if c in athletic_df.columns]
        ].drop_duplicates(subset=["player_id"], keep="first")
        base = pd.merge(combine, athletic_slim, on="player_id", how="left")
    else:
        base = combine.copy()
        for col in GOLD_ATHLETIC_COLS:
            if col not in base.columns:
                base[col] = np.nan

    # ── derived features ───────────────────────────────────────────────────
    h = base["height_in"].replace(0, np.nan)
    w = base["weight_lbs"].replace(0, np.nan)

    base["bmi"] = (w / (h**2)) * 703
    base["speed_to_weight"] = base["speed_score"] / w
    base["relative_size"] = w / h

    # ── labels ─────────────────────────────────────────────────────────────
    if include_target:
        if draft_df is None:
            raise ValueError("draft_df is required when include_target=True")

        if label_strategy == "snap_share":
            if snap_df is None or snap_df.empty:
                raise ValueError(
                    "label_strategy='snap_share' requires snap_df — "
                    "provide snap_df or use label_strategy='archetype'."
                )
            # ── snap-share × career quality ────────────────────────────────
            log.info("Building snap-share labels …")
            snap_labels = build_snap_labels(snap_df, draft_df)

            base = pd.merge(
                base,
                snap_labels[["player_name", "draft_year"] + LABEL_COLS],
                on=["player_name", "draft_year"],
                how="inner",
            )
            if base.empty:
                raise ValueError(
                    "combine × snap_labels join is empty — "
                    "check player_name / draft_year alignment between combine and snap_counts."
                )
            log.info(f"  after snap-label join: {len(base)} rows")

        else:
            # ── both archetype and declared_position need car_av ───────────
            draft = draft_df.copy()
            draft = draft.rename(
                columns={"season": "draft_year", "pfr_player_name": "player_name"},
                errors="ignore",
            )
            draft["draft_year"] = pd.to_numeric(
                draft["draft_year"], errors="coerce"
            ).astype("Int64")
            draft["car_av"] = pd.to_numeric(draft.get("car_av"), errors="coerce")
            draft = draft.dropna(subset=["player_name", "draft_year", "car_av"])[
                ["player_name", "draft_year", "car_av"]
            ].drop_duplicates(subset=["player_name", "draft_year"], keep="first")
            draft["draft_year"] = draft["draft_year"].astype(int)

            base = pd.merge(base, draft, on=["player_name", "draft_year"], how="inner")
            if base.empty:
                raise ValueError(
                    "combine × draft_picks join is empty — "
                    "check player_name / draft_year alignment."
                )

            if label_strategy == "archetype":
                # ── Athletic archetype affinity labels (default) ───────────
                log.info("Building athletic archetype labels …")
                available_feats = [c for c in FEATURE_COLS if c in base.columns]
                base = _add_archetype_labels(base, feature_cols=available_feats)

            else:
                # ── declared_position fallback ─────────────────────────────
                log.info("Building declared-position labels …")
                for grp in POSITION_GROUP_ORDER:
                    label_col = f"label_{grp}"
                    is_this_group = (base["pos_group"] == grp).astype(float)
                    soft = (base["car_av"] / QUALIFIED_THRESHOLD).clip(upper=1.0)
                    base[label_col] = is_this_group * soft

    # ── assemble result ────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in base.columns]
    result = base[available_features].copy()
    result["pos_group"] = base["pos_group"]  # preserve declared position group

    if include_target:
        for col in LABEL_COLS:
            if col in base.columns:
                result[col] = base[col]

    index_arrays = [base["player_name"], base["draft_year"]]
    index_names = ["player_name", "draft_year"]
    if "player_id" in base.columns:
        index_arrays.append(base["player_id"])
        index_names.append("player_id")
    result.index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)

    return result


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------


async def fetch_flex_features(
    client: DataLakeClient,
    year_start: Optional[int] = 2000,
    year_end: Optional[int] = None,
    include_target: bool = True,
    label_strategy: str = "archetype",
) -> tuple[pd.DataFrame, str]:
    """
    Fetch all source tables and return the complete feature matrix.

    Parameters
    ----------
    label_strategy : one of "archetype" (default), "snap_share", "declared_position".
                     "snap_share" additionally fetches snap_counts from the data lake.

    Returns
    -------
    (df, label_strategy_used)
    """
    log.info("Fetching combine data …")
    combine_df = await get_combine_data(
        client, year_start=year_start, year_end=year_end
    )
    log.info(f"  combine: {len(combine_df)} rows")

    log.info("Fetching athletic profiles …")
    athletic_df = await get_athletic_profiles(client)
    log.info(f"  athletic_profiles: {len(athletic_df)} rows")

    draft_df = snap_df = None

    if include_target:
        log.info("Fetching draft picks (for car_av career quality) …")
        draft_df = await get_draft_picks(
            client, year_start=year_start, year_end=year_end
        )
        log.info(f"  draft_picks: {len(draft_df)} rows")

        if label_strategy == "snap_share":
            log.info("Fetching snap counts (for snap-share labels) …")
            snap_df = await get_snap_counts(client)
            if snap_df is None or snap_df.empty:
                raise RuntimeError(
                    "snap_counts returned empty result — "
                    "cannot use label_strategy='snap_share'."
                )
            log.info(f"  snap_counts: {len(snap_df)} rows")

    log.info(f"Label strategy: {label_strategy}")

    df = build_flex_features(
        combine_df=combine_df,
        athletic_df=athletic_df,
        draft_df=draft_df,
        snap_df=snap_df,
        include_target=include_target,
        label_strategy=label_strategy,
    )
    return df, label_strategy
