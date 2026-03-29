"""
Health Analyzer — Feature Engineering
======================================

Builds the survival frame (counting process format) for CoxTimeVaryingFitter.

ACTUAL DATA LAKE SCHEMAS (verified against live tables):

│ Table                   │ Key columns used                                     │

│ injuries                │ player_id, full_name, season, week, game_type,       │
│                         │ report_status                                         │

│ snap_counts             │ player_id, player, season, week, game_type,          │
│                         │ position, offense_pct                                 │

│ combine                 │ player_id, player_name, weight_lbs, height_in        │

│ player_athletic_profiles│ player_id, speed_score, burst_score, strength_score  │

│ draft_picks             │ player_id, player_name, season (=draft yr), age,     │
│                         │ position                                              │


JOIN KEY: player_id across all tables (GSIS ID, verified exact/gsis confidence).

SURVIVAL FRAME FORMAT (counting process / Andersen-Gill):
    Each row = one player-week interval (regular season only)
    player_season_id : "{player_id}_{season}"  (unique per player-season)
    start            : week - 1  (interval open end)
    stop             : week      (interval close end)
    event            : 1 if report_status == 'Out' that week, else 0
    position_group   : strata column — separate baseline hazard per position
    + FEATURE_COLS   : causal covariates (static + time-varying)

FEATURE DESIGN:
    PRIMARY — workload / overuse (causal mechanism):
        acwr                      : Acute:Chronic Workload Ratio. snap_share last
                                    week / snap_share rolling 8wk. ACWR > 1.3 is
                                    the key overuse threshold in sports science.
        snap_share_vs_pos_median  : Player load relative to position peers.
                                    Normalises for position-baseline differences
                                    (OL at 90% ≠ risk as WR at 90%).
        season_snap_acceleration  : Recent snap share minus 8wk trend.  Captures
                                    ramp-up after return from minor injury.
        snap_share_rolling_8wk    : Chronic workload baseline (8wk lagged).

    SECONDARY — injury history enrichment (prior seasons only, no within-season):
        career_durability_rate    : games_played / games_available across prior
                                    seasons. 1.0 for never-injured players —
                                    treated as protective signal, not missing data.
        has_prior_injury          : Binary. Prior injury amplifies risk from
                                    workload features (scar tissue, compensation).
        prior_season_out_weeks    : Out weeks in the immediately prior season.
                                    Captures recent injury burden without any
                                    within-season feedback loop.

    PHYSICAL PROFILE — load capacity:
        weight_lbs, bmi, strength_score, speed_score, burst_score

    BIOLOGICAL WEAR:
        age, seasons_played, games_played_this_season, career_games_played

    STRATA (not a covariate):
        position_group — separate baseline hazard per position so "RBs get hurt
        more" is absorbed into the baseline.  Coefficients capture within-position
        risk differences only.

TEMPORAL INTEGRITY:
    All time-varying features are strictly lagged: values at week W use only
    weeks 1 … W-1 of the current season. Injury history uses only seasons < W's
    season. No future data enters any feature.
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
    get_injury_history,
    get_snap_counts,
)

# ── Position grouping ──────────────────────────────────────────────────────────

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

# ── Feature columns ────────────────────────────────────────────────────────────

# Static within a player-season (constant across all weeks of the same season).
# KNN imputation is applied at player-season level for these columns.
STATIC_FEATURE_COLS = [
    "weight_lbs",              # joint stress — heavier players sustain more load per play
    "bmi",                     # load relative to frame
    "strength_score",          # muscular support / injury resistance
    "speed_score",             # explosive athletes → higher soft-tissue stress
    "burst_score",             # acceleration-deceleration forces
    "age",                     # biological degradation
    "seasons_played",          # accumulated NFL seasons (career wear proxy)
    "career_durability_rate",  # 1.0 = never missed a game; protective enrichment signal
    "has_prior_injury",        # binary: any prior Out week in career (scar tissue modifier)
    "prior_season_out_weeks",  # Out weeks in season-1 only (recent injury burden)
    "injury_free_streak",      # healthy weeks at end of prior season (0 = still recovering)
]

# Time-varying — updated each week, strictly lagged (no current-week leakage).
# Rolling stats use snap_share (position-aware: defense_pct for DL/LB/DB,
# offense_pct for QB/SKILL/OL) and only include played weeks so injury-forced
# zeros do not contaminate the chronic workload baseline.
TIME_VARYING_COLS = [
    "snap_share_rolling_8wk",    # chronic workload baseline: 8-wk mean of played weeks
    "acwr",                      # acute:chronic ratio — last week / chronic baseline
    "snap_share_vs_pos_median",  # normalised load vs position peers (same week)
    "season_snap_acceleration",  # last-week snap vs 8-wk trend (ramp-up detector)
    "games_played_this_season",  # cumulative healthy games before current week
    "career_games_played",       # total career games (accumulated wear)
]

FEATURE_COLS = STATIC_FEATURE_COLS + TIME_VARYING_COLS

SURVIVAL_COLS = ["player_season_id", "start", "stop", "event", "position_group"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _map_position(pos: str) -> str:
    return POSITION_GROUPS.get(str(pos).upper().strip(), "SKILL")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    return df


_DEFENSIVE_GROUPS = {"DL", "LB", "DB"}


def _compute_snap_share(snaps: pd.DataFrame) -> pd.DataFrame:
    """
    Return snaps with a `snap_share` column that is position-appropriate.

    Offensive players (QB / SKILL / OL):
        offense_pct — already normalized by team offensive snaps.

    Defensive players (DL / LB / DB):
        defense_snaps normalized by the 95th-percentile of defense_snaps
        for that position group × season.  The 95th percentile approximates
        the per-game team defensive snap total for a starter, giving a 0-1
        participation rate without requiring a team column.

    Falls back to offense_pct for defensive players if defense_snaps is absent
    (will be ~0 for most, surfaced as a data-quality warning).
    """
    log = logging.getLogger(__name__)
    snaps = snaps.copy()
    snaps["_pos_group"] = snaps["position"].map(_map_position).fillna("SKILL")

    if "defense_snaps" in snaps.columns:
        snaps["defense_snaps"] = pd.to_numeric(snaps["defense_snaps"], errors="coerce").fillna(0.0)

        def_mask = snaps["_pos_group"].isin(_DEFENSIVE_GROUPS)
        def_norm = (
            snaps.loc[def_mask, ["_pos_group", "season", "defense_snaps"]]
            .groupby(["_pos_group", "season"])["defense_snaps"]
            .quantile(0.95)
            .reset_index(name="_def_norm")
        )
        snaps = snaps.merge(def_norm, on=["_pos_group", "season"], how="left")
        snaps["_defense_pct"] = (
            snaps["defense_snaps"] / snaps["_def_norm"].replace(0, np.nan)
        ).clip(0.0, 1.0).fillna(0.0)

        snaps["snap_share"] = np.where(
            snaps["_pos_group"].isin(_DEFENSIVE_GROUPS),
            snaps["_defense_pct"],
            snaps["offense_pct"].fillna(0.0),
        )
        snaps = snaps.drop(columns=["_def_norm", "_defense_pct"], errors="ignore")
        log.info("  snap_share: using defense_snaps for DL/LB/DB positions")
    else:
        snaps["snap_share"] = snaps["offense_pct"].fillna(0.0)
        log.warning(
            "  defense_snaps column not found in snap_counts — "
            "DL/LB/DB workload features will be near-zero. "
            "Verify get_snap_counts() returns defense_snaps."
        )

    snaps = snaps.drop(columns=["_pos_group"], errors="ignore")
    return snaps


# ── Main builder ───────────────────────────────────────────────────────────────


def build_survival_frame(
    injuries_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    combine_df: pd.DataFrame,
    athletic_df: pd.DataFrame,
    draft_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join raw source tables into a counting-process survival frame.

    Returns DataFrame with SURVIVAL_COLS + FEATURE_COLS.
    One row per player-week (regular season only).
    Ready for CoxTimeVaryingFitter.fit().
    """
    log = logging.getLogger(__name__)

    injuries = _norm_cols(injuries_df)
    snaps = _norm_cols(snap_df)
    combine = _norm_cols(combine_df)
    athletic = _norm_cols(athletic_df)
    draft = _norm_cols(draft_df)

    draft = draft.rename(columns={"season": "draft_year"}, errors="ignore")

    # Regular season only
    if "game_type" in injuries.columns:
        injuries = injuries[injuries["game_type"].str.upper() == "REG"]
    if "game_type" in snaps.columns:
        snaps = snaps[snaps["game_type"].str.upper() == "REG"]

    # Coerce numeric types
    for df, cols in [
        (injuries, ["season", "week"]),
        (snaps, ["season", "week", "offense_pct"]),
        (draft, ["draft_year", "age"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    injuries = injuries.dropna(subset=["player_id", "season", "week"])
    snaps = snaps.dropna(subset=["player_id", "season", "week"])

    injuries["season"] = injuries["season"].astype(int)
    injuries["week"] = injuries["week"].astype(int)
    snaps["season"] = snaps["season"].astype(int)
    snaps["week"] = snaps["week"].astype(int)

    # ── Event indicator ────────────────────────────────────────────────────────
    inj_key = "gsis_id" if "gsis_id" in injuries.columns else "player_id"
    inj_out = injuries[injuries["report_status"].str.strip().str.lower() == "out"][
        [inj_key, "season", "week"]
    ].drop_duplicates()
    inj_out = inj_out.rename(columns={inj_key: "gsis_id"})
    inj_out["event"] = 1

    # ── Snap data ──────────────────────────────────────────────────────────────
    # Compute position-aware snap_share before deduplication so defense_snaps
    # normalization has the full population for each position group × season.
    snaps = _compute_snap_share(snaps)
    snaps = (
        snaps[["player_id", "position", "season", "week", "snap_share"]]
        .drop_duplicates(subset=["player_id", "season", "week"])
        .copy()
    )

    # ── Universe: every player × season × week ────────────────────────────────
    MAX_WEEK = 18
    player_seasons = snaps[["player_id", "position", "season"]].drop_duplicates()
    weeks_df = pd.DataFrame({"week": range(1, MAX_WEEK + 1)})
    universe = player_seasons.merge(weeks_df, how="cross")

    universe = universe.merge(
        snaps[["player_id", "season", "week", "snap_share"]],
        on=["player_id", "season", "week"],
        how="left",
    )
    universe["snap_share"] = universe["snap_share"].fillna(0.0)

    # ── ID bridge: injury gsis_id → snap_counts player_id ─────────────────────
    snap_orig = _norm_cols(snap_df)
    inj_orig = injuries.copy()

    snap_bridge = snap_orig[["player_id", "player", "season"]].drop_duplicates()
    snap_bridge["season"] = pd.to_numeric(snap_bridge["season"], errors="coerce").dropna()
    snap_bridge = snap_bridge.dropna(subset=["season"])
    snap_bridge["season"] = snap_bridge["season"].astype(int)
    snap_bridge["_name_lower"] = snap_bridge["player"].str.lower().str.strip()

    # Injuries table may use either `full_name` or `player_name` for the player name.
    if "full_name" in inj_orig.columns:
        inj_name_col = "full_name"
    elif "player_name" in inj_orig.columns:
        inj_name_col = "player_name"
    else:
        raise KeyError(
            "Injury history DataFrame must contain either 'full_name' or 'player_name' column."
        )

    inj_bridge = inj_orig[[inj_key, inj_name_col, "season"]].drop_duplicates()
    inj_bridge["season"] = pd.to_numeric(inj_bridge["season"], errors="coerce").dropna()
    inj_bridge = inj_bridge.dropna(subset=["season"])
    inj_bridge["season"] = inj_bridge["season"].astype(int)
    inj_bridge["_name_lower"] = inj_bridge[inj_name_col].str.lower().str.strip()
    inj_bridge = inj_bridge.rename(columns={inj_key: "gsis_id"})

    id_bridge = snap_bridge.merge(
        inj_bridge[["gsis_id", "_name_lower", "season"]],
        on=["_name_lower", "season"],
        how="inner",
    )[["player_id", "gsis_id"]].drop_duplicates(subset=["player_id"])

    inj_out = inj_out.merge(
        id_bridge[["gsis_id", "player_id"]],
        on="gsis_id",
        how="inner",
    )[["player_id", "season", "week", "event"]].drop_duplicates(
        subset=["player_id", "season", "week"]
    )

    # ── Pre-compute career injury history (prior seasons only) ─────────────────
    # Aggregate Out weeks per player per season from inj_out
    season_out_counts = (
        inj_out.groupby(["player_id", "season"])
        .size()
        .reset_index(name="season_out_weeks")
    )

    # For each (player, season) in the universe, sum Out weeks from all earlier seasons
    universe_seasons = universe[["player_id", "season"]].drop_duplicates()
    career_inj = universe_seasons.merge(
        season_out_counts.rename(columns={"season": "inj_season"}),
        on="player_id",
        how="left",
    )
    career_inj = career_inj[career_inj["inj_season"] < career_inj["season"]]
    career_history = (
        career_inj.groupby(["player_id", "season"])["season_out_weeks"]
        .sum()
        .reset_index(name="career_out_weeks")
    )
    universe = universe.merge(career_history, on=["player_id", "season"], how="left")
    universe["career_out_weeks"] = universe["career_out_weeks"].fillna(0.0)

    # ── Merge injury events ────────────────────────────────────────────────────
    universe = universe.merge(inj_out, on=["player_id", "season", "week"], how="left")
    universe["event"] = universe["event"].fillna(0).astype(int)

    # ── Position group ─────────────────────────────────────────────────────────
    universe["position_group"] = universe["position"].map(_map_position).fillna("SKILL")

    # ── Static: age / seasons from draft data ─────────────────────────────────
    draft_age = (
        draft[["player_id", "draft_year", "age"]]
        .dropna(subset=["player_id", "draft_year", "age"])
        .drop_duplicates(subset=["player_id"], keep="last")
    )
    universe = universe.merge(draft_age, on="player_id", how="left")
    universe["draft_year"] = pd.to_numeric(universe["draft_year"], errors="coerce")
    universe["age"] = (
        universe["age"] + (universe["season"] - universe["draft_year"])
    ).fillna(27)

    universe["seasons_played"] = (
        (universe["season"] - universe["draft_year"] + 1).clip(lower=1, upper=20)
    ).fillna(3)

    # ── Injury history enrichment (static per season, prior seasons only) ────────
    # career_durability_rate: protective signal — 1.0 for never-injured players
    prior_available_weeks = ((universe["seasons_played"] - 1) * 17).clip(lower=1)
    universe["career_durability_rate"] = (
        1.0 - (universe["career_out_weeks"] / prior_available_weeks)
    ).clip(lower=0.0, upper=1.0)

    # has_prior_injury: binary enrichment (scar tissue / movement compensation)
    universe["has_prior_injury"] = (universe["career_out_weeks"] > 0).astype(float)

    # prior_season_out_weeks: Out weeks in season-1 specifically (recent burden)
    prior_season_out = (
        season_out_counts.rename(columns={"season": "inj_season", "season_out_weeks": "prior_season_out_weeks"})
    )
    prior_season_out["season"] = prior_season_out["inj_season"] + 1
    universe = universe.merge(
        prior_season_out[["player_id", "season", "prior_season_out_weeks"]],
        on=["player_id", "season"],
        how="left",
    )
    universe["prior_season_out_weeks"] = universe["prior_season_out_weeks"].fillna(0.0)

    # injury_free_streak: healthy weeks at END of the prior season.
    # MAX_WEEK - last_out_week in season-1.  Higher = more recovery runway.
    # Players with no prior injury get the full MAX_WEEK (best possible streak).
    last_out_per_season = (
        inj_out.groupby(["player_id", "season"])["week"]
        .max()
        .reset_index(name="last_out_week")
    )
    inj_streak = universe_seasons.merge(
        last_out_per_season.rename(columns={"season": "inj_season"}),
        on="player_id",
        how="left",
    )
    inj_streak = inj_streak[inj_streak["inj_season"] == inj_streak["season"] - 1]
    inj_streak["injury_free_streak"] = (MAX_WEEK - inj_streak["last_out_week"]).clip(lower=0)
    universe = universe.merge(
        inj_streak[["player_id", "season", "injury_free_streak"]],
        on=["player_id", "season"],
        how="left",
    )
    # No injury in prior season → full season healthy = MAX_WEEK
    universe["injury_free_streak"] = universe["injury_free_streak"].fillna(float(MAX_WEEK))

    # ── Time-varying: workload features (all strictly lagged) ─────────────────
    universe = universe.sort_values(["player_id", "season", "week"])

    # Use only weeks the player PLAYED for the chronic rolling baseline.
    # Replacing absence-zeros with NaN ensures the chronic average reflects
    # actual workload capacity. Weeks with snap_share = 0 (Out or inactive)
    # are excluded from the rolling window.
    universe["_snap_played"] = universe["snap_share"].where(universe["snap_share"] > 0)

    # Chronic baseline: 8-week rolling mean of PLAYED weeks only (lagged)
    universe["snap_share_rolling_8wk"] = (
        universe.groupby(["player_id", "season"])["_snap_played"].transform(
            lambda x: x.shift(1).rolling(8, min_periods=1).mean()
        )
    ).fillna(0.0)

    # Acute load: last week's raw snap share (0 if player was Out — legitimate)
    universe["_snap_last_week"] = (
        universe.groupby(["player_id", "season"])["snap_share"].transform(
            lambda x: x.shift(1)
        )
    ).fillna(0.0)

    # ACWR: acute / chronic (played-week baseline).
    # Correctly captures workload spikes without being contaminated by injury zeros.
    # fillna(1.0) when chronic=0 (no prior active weeks) = neutral load assumption.
    universe["acwr"] = (
        universe["_snap_last_week"]
        / universe["snap_share_rolling_8wk"].replace(0, np.nan)
    ).fillna(1.0).clip(upper=3.0)

    # Snap share vs position median: normalises across positions
    pos_median_snap = universe.groupby(
        ["position_group", "season", "week"]
    )["snap_share_rolling_8wk"].transform("median")
    universe["snap_share_vs_pos_median"] = (
        universe["snap_share_rolling_8wk"] / pos_median_snap.replace(0, np.nan)
    ).fillna(1.0).clip(upper=3.0)

    # Season snap acceleration: last week vs 8wk played-week trend (ramp-up signal)
    universe["season_snap_acceleration"] = (
        universe["_snap_last_week"] - universe["snap_share_rolling_8wk"]
    )
    universe = universe.drop(columns=["_snap_last_week", "_snap_played"])

    # ── Time-varying: game count features ─────────────────────────────────────
    universe["_played"] = (universe["snap_share"] > 0).astype(int)
    universe["games_played_this_season"] = universe.groupby(["player_id", "season"])[
        "_played"
    ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    universe = universe.drop(columns=["_played"])

    universe["career_games_played"] = (
        (universe["seasons_played"] - 1) * 17 + universe["games_played_this_season"]
    ).clip(lower=0)

    # ── Static: body composition from combine ─────────────────────────────────
    combine_slim = combine[["player_id"]].copy()
    if "weight_lbs" in combine.columns:
        combine_slim["weight_lbs"] = pd.to_numeric(combine["weight_lbs"], errors="coerce")
    if "height_in" in combine.columns:
        combine_slim["height_in"] = pd.to_numeric(combine["height_in"], errors="coerce")
    if "weight_lbs" in combine_slim.columns and "height_in" in combine_slim.columns:
        valid_height = combine_slim["height_in"].gt(0)
        combine_slim["bmi"] = np.where(
            valid_height,
            combine_slim["weight_lbs"] / (combine_slim["height_in"] ** 2) * 703,
            np.nan,
        )
    else:
        combine_slim["bmi"] = np.nan

    keep = ["player_id"] + [c for c in ["weight_lbs", "bmi"] if c in combine_slim.columns]
    combine_slim = combine_slim[keep].drop_duplicates(subset=["player_id"], keep="last")
    universe = universe.merge(combine_slim, on="player_id", how="left")

    # ── Static: athletic profile ───────────────────────────────────────────────
    ath_want = ["speed_score", "burst_score", "strength_score"]
    ath_cols = ["player_id"] + [c for c in ath_want if c in athletic.columns]
    ath_slim = athletic[ath_cols].drop_duplicates(subset=["player_id"], keep="last")
    universe = universe.merge(ath_slim, on="player_id", how="left")

    # Ensure all FEATURE_COLS present
    for col in FEATURE_COLS:
        if col not in universe.columns:
            universe[col] = np.nan

    # ── Counting-process intervals ─────────────────────────────────────────────
    universe["start"] = (universe["week"] - 1).astype(float)
    universe["stop"] = universe["week"].astype(float)
    universe["player_season_id"] = (
        universe["player_id"].astype(str)
        + "_"
        + universe["season"].astype(int).astype(str)
    )

    result = universe[SURVIVAL_COLS + FEATURE_COLS].copy()
    result = result.dropna(subset=["start", "stop"])

    log.info(
        f"Survival frame: {len(result):,} rows | "
        f"{result['player_season_id'].nunique():,} player-seasons | "
        f"event rate: {result['event'].mean():.3f}"
    )
    return result


# ── Data fetch ─────────────────────────────────────────────────────────────────


async def fetch_health_features(
    client: DataLakeClient,
    year_start: int = 2010,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all source tables and build the survival frame.
    Sequential fetches to avoid saturating the data lake over Tailscale.
    """
    log = logging.getLogger(__name__)

    log.info("Fetching injury history …")
    injuries_df = await get_injury_history(client)
    log.info(f"  injuries: {len(injuries_df):,} rows")

    log.info("Fetching snap counts …")
    snap_df = await get_snap_counts(client)
    log.info(f"  snap_counts: {len(snap_df):,} rows")

    log.info("Fetching combine data …")
    combine_df = await get_combine_data(
        client, year_start=year_start, year_end=year_end
    )
    log.info(f"  combine: {len(combine_df):,} rows")

    log.info("Fetching athletic profiles …")
    athletic_df = await get_athletic_profiles(client)
    log.info(f"  athletic_profiles: {len(athletic_df):,} rows")

    log.info("Fetching draft picks …")
    draft_df = await get_draft_picks(client, year_start=year_start, year_end=year_end)
    log.info(f"  draft_picks: {len(draft_df):,} rows")

    return build_survival_frame(injuries_df, snap_df, combine_df, athletic_df, draft_df)
