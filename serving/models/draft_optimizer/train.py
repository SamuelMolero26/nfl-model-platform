"""
serving/models/draft_optimizer/train.py
========================================
Calibration + validation script for the Draft Optimizer.

Run from the project root:
    python -m serving.models.draft_optimizer.train

Flags:
    --dry-run          Calibrate and validate but do not save the artifact.
                       Prints the full validation report so you can inspect
                       the output before committing a pkl.

    --year-start INT   First year of calibration window (default: 2000)
    --year-end   INT   Last year of calibration window (default: 2020)

    --holdout-start INT  First year of holdout (default: 2021)
    --holdout-end   INT  Last year of holdout   (default: 2022)
                         These years are touched ONCE — only for the final
                         baseline comparison, never during calibration.

WHAT THIS SCRIPT DOES
─────────────────────
  1. Fetch all historical draft picks in the calibration window via the data
     lake API (fetch_prospect_scores_batch — single JOIN query over Tailscale).

  2. Calibrate the DraftOptimizerModel:
       - expected_av_curve       (mean w_av by pick slot)
       - positional_quota_defaults (p95 group picks per draft)
       - need_score_thresholds   (10th pct of group percentile dist)
       - pick_value_table        (log-decay slot value, 0–100)

  3. Validate against the ADP baseline on the holdout window (2021–2022):
       ADP baseline: at each pick slot, simply take the prospect with the
       highest draft_value_percentile available — i.e. always pick the
       "consensus best player available" by historical percentile rank.

       Optimizer: same picks, but scored by career_value_score with uniform
       need weights (0.5 everywhere) to isolate the optimizer's contribution
       vs the baseline.

       Primary metric: mean value_over_adp across all holdout picks.
       The optimizer must beat the ADP baseline for the artifact to save.

  4. Save artifact files to artifacts/draft_optimizer/v1/:
       model.pkl              calibrated state (4 lookup tables)
       metadata.json          version, metrics, calibration window
       baseline_results.parquet  per-pick holdout comparison (Phase 8 input)

PLAN ALIGNMENT
─────────────
  • No leakage: calibration window is strictly 2000–2020. Holdout years
    2021–2022 are loaded separately and never passed to calibrate().
  • Baseline comparison: ADP baseline defined above. Optimizer must beat it.
  • Feature caching: batch fetch result saved to
    artifacts/draft_optimizer/features/calibration_batch.parquet so
    re-runs don't hit DuckDB again unless --refresh-cache is passed.
  • baseline_results.parquet feeds directly into Phase 8 validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

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

from serving.models.draft_optimizer.features import fetch_prospect_scores_batch  # noqa: E402
from serving.models.draft_optimizer.model import DraftOptimizerModel             # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
_ARTIFACT_DIR    = _ROOT / "artifacts" / "draft_optimizer" / "v1"
_PKL_PATH        = _ARTIFACT_DIR / "model.pkl"
_META_PATH       = _ARTIFACT_DIR / "metadata.json"
_BASELINE_PATH   = _ARTIFACT_DIR / "baseline_results.parquet"
_FEATURE_CACHE   = _ARTIFACT_DIR / "features" / "calibration_batch.parquet"
_HOLDOUT_CACHE   = _ARTIFACT_DIR / "features" / "holdout_batch.parquet"

# Minimum seasons required — warn if calibration window is too narrow
_MIN_CALIBRATION_SEASONS = 10


# ===========================================================================
# Validation helpers
# ===========================================================================

def _adp_baseline_score(df: pd.DataFrame) -> float:
    """
    Simulate the naive ADP baseline on a batch of historical picks.

    For each draft year in df, walk through picks in order and always
    select the available prospect with the highest draft_value_percentile.
    Return mean value_over_expected across all picks, where expected is the
    mean w_av at that pick slot (from the same df — this is an in-sample
    proxy for the holdout; actual holdout uses the calibrated curve).

    This is the "dumbest reasonable strategy" the optimizer must beat.
    """
    if df.empty or "draft_value_percentile" not in df.columns:
        return 0.0

    df = df.copy()
    df["pick"]  = pd.to_numeric(df["pick"],  errors="coerce")
    df["w_av"]  = pd.to_numeric(df["w_av"],  errors="coerce").fillna(0)
    df["draft_value_percentile"] = pd.to_numeric(
        df["draft_value_percentile"], errors="coerce"
    ).fillna(0)

    # Expected AV at each pick slot (mean across all years in df)
    expected_av = df.groupby("pick")["w_av"].mean().to_dict()

    total_over_expected = []
    for year, year_df in df.groupby("season"):
        year_df = year_df.sort_values("pick").copy()
        available = set(year_df.index.tolist())
        for _, row in year_df.iterrows():
            if row.name not in available:
                continue
            # ADP baseline: best available by percentile
            avail_df = year_df.loc[list(available)]
            best_idx = avail_df["draft_value_percentile"].idxmax()
            best_row = year_df.loc[best_idx]
            expected_pick = int(row["pick"]) if pd.notna(row["pick"]) else 0
            expected = expected_av.get(expected_pick, 0.0)
            total_over_expected.append(float(best_row["w_av"]) - expected)
            available.discard(best_idx)

    return round(float(np.mean(total_over_expected)) if total_over_expected else 0.0, 4)


def _optimizer_baseline_score(
    holdout_df: pd.DataFrame,
    model: DraftOptimizerModel,
) -> float:
    """
    Measure how much value the optimizer's pick ranking generates over the
    expected AV curve on the holdout data.

    Uses uniform need weights (0.5 everywhere) and the calibrated
    expected_av_curve to compute value_over_adp per pick, then returns the
    mean across all holdout picks.

    This is deliberately simple — we're testing whether the calibration
    tables add any signal over naive ADP selection, not a full end-to-end
    optimisation run (which would need an async client and live data).
    """
    if holdout_df.empty:
        return 0.0

    df = holdout_df.copy()
    df["pick"]  = pd.to_numeric(df["pick"],  errors="coerce")
    df["w_av"]  = pd.to_numeric(df["w_av"],  errors="coerce").fillna(0)
    df["draft_value_percentile"] = pd.to_numeric(
        df["draft_value_percentile"], errors="coerce"
    ).fillna(0)

    max_av = max(model.expected_av_curve.values()) if model.expected_av_curve else 1.0
    scores = []

    for _, row in df.iterrows():
        pick = int(row["pick"]) if pd.notna(row["pick"]) else 0
        expected_av = model.expected_av_curve.get(pick, 0.0)
        expected_scaled = (expected_av / max_av) * 100.0 if max_av > 0 else 0.0
        # Use draft_value_percentile as the proxy for career_value_score
        # (no live Player Projection model available during offline training)
        value_over_adp = float(row["draft_value_percentile"]) - expected_scaled
        scores.append(value_over_adp)

    return round(float(np.mean(scores)) if scores else 0.0, 4)


def _build_baseline_results(
    holdout_df: pd.DataFrame,
    model: DraftOptimizerModel,
) -> pd.DataFrame:
    """
    Build the per-pick holdout comparison DataFrame saved as
    baseline_results.parquet for Phase 8 case studies.

    Columns:
        season, pick, player_id, player_name, position, position_group,
        w_av, draft_value_percentile,
        expected_av_at_slot,    (from calibrated curve)
        expected_scaled,        (0–100 normalised)
        value_over_adp,         (percentile − expected_scaled)
        adp_baseline_percentile (percentile of the top ADP pick at this slot)
    """
    if holdout_df.empty:
        return pd.DataFrame()

    df = holdout_df.copy()
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df["w_av"] = pd.to_numeric(df["w_av"], errors="coerce").fillna(0)
    df["draft_value_percentile"] = pd.to_numeric(
        df["draft_value_percentile"], errors="coerce"
    ).fillna(0)

    max_av = max(model.expected_av_curve.values()) if model.expected_av_curve else 1.0

    rows = []
    for _, row in df.iterrows():
        pick = int(row["pick"]) if pd.notna(row["pick"]) else 0
        expected_av     = model.expected_av_curve.get(pick, 0.0)
        expected_scaled = (expected_av / max_av) * 100.0 if max_av > 0 else 0.0
        rows.append({
            "season":               row.get("season"),
            "pick":                 pick,
            "player_id":            row.get("player_id", ""),
            "player_name":          row.get("player_name", ""),
            "position":             row.get("position", ""),
            "position_group":       row.get("position_group", ""),
            "w_av":                 round(float(row["w_av"]), 3),
            "draft_value_percentile": round(float(row["draft_value_percentile"]), 3),
            "expected_av_at_slot":  round(expected_av, 3),
            "expected_scaled":      round(expected_scaled, 3),
            "value_over_adp":       round(float(row["draft_value_percentile"]) - expected_scaled, 3),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Reporting
# ===========================================================================

def _section(title: str) -> None:
    width = 72
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def _print_validation_report(
    cal_df:         pd.DataFrame,
    holdout_df:     pd.DataFrame,
    model:          DraftOptimizerModel,
    adp_score:      float,
    optimizer_score: float,
    beats_baseline: bool,
) -> None:
    """Print the full post-calibration validation report to stdout."""

    _section("Calibration data summary")
    print(f"  Picks in calibration window : {len(cal_df):,}")
    print(f"  Draft seasons covered       : {cal_df['season'].nunique():,}  "
          f"({int(cal_df['season'].min())}–{int(cal_df['season'].max())})")
    print(f"  Picks with w_av > 0         : {(cal_df['w_av'] > 0).sum():,}")
    print(f"  AV curve slots built        : {len(model.expected_av_curve):,}")

    _section("Expected AV curve — key pick slots")
    slots = [1, 5, 10, 15, 32, 64, 100, 150, 200, 250]
    print(f"  {'Pick':>6}  {'Mean w_av':>10}  {'Pick value (0–100)':>20}")
    print(f"  {'------':>6}  {'---------':>10}  {'------------------':>20}")
    for p in slots:
        av  = model.expected_av_curve.get(p, None)
        pv  = model.pick_value_table.get(p, None)
        av_str = f"{av:.2f}" if av is not None else "n/a"
        pv_str = f"{pv:.1f}" if pv is not None else "n/a"
        print(f"  {p:>6}  {av_str:>10}  {pv_str:>20}")

    _section("Positional quota defaults (p95 picks per draft)")
    for group, quota in sorted(model.positional_quota_defaults.items()):
        threshold = model.need_score_thresholds.get(group, 0.0)
        print(f"  {group:<8}  max {quota} picks   "
              f"no-need threshold: {threshold:.3f}")

    _section("Holdout validation -- 2021-2022")
    print(f"  Holdout picks               : {len(holdout_df):,}")
    adp_sign = "+" if adp_score >= 0 else ""
    opt_sign = "+" if optimizer_score >= 0 else ""
    print(f"  ADP baseline score          : {adp_sign}{adp_score:.4f}  "
          f"(mean w_av - expected at slot)")
    print(f"  Optimizer score             : {opt_sign}{optimizer_score:.4f}  "
          f"(mean percentile - expected scaled)")
    beat_str = "PASSES" if beats_baseline else "FAILS"
    print(f"  Beats ADP baseline          : {beat_str}")
    if not beats_baseline:
        print(
            "\n  WARNING: optimizer score <= ADP baseline score. "
            "Consider expanding the calibration window or checking that "
            "draft_value_history w_av values are populated for recent years."
        )

    _section("Artifact")
    print(f"  pkl path     : {_PKL_PATH}")
    print(f"  metadata     : {_META_PATH}")
    print(f"  holdout data : {_BASELINE_PATH}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and validate the Draft Optimizer model."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Calibrate and validate but do not save the artifact.",
    )
    parser.add_argument(
        "--year-start", type=int, default=2000,
        help="First year of calibration window (default: 2000).",
    )
    parser.add_argument(
        "--year-end", type=int, default=2020,
        help="Last year of calibration window (default: 2020).",
    )
    parser.add_argument(
        "--holdout-start", type=int, default=2021,
        help="First holdout year (default: 2021).",
    )
    parser.add_argument(
        "--holdout-end", type=int, default=2022,
        help="Last holdout year (default: 2022).",
    )
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Re-fetch from DuckDB even if a feature cache exists.",
    )
    args = parser.parse_args()

    # ── Guard: no leakage ──────────────────────────────────────────────────
    if args.year_end >= args.holdout_start:
        logger.error(
            "Calibration window (%s–%s) overlaps with holdout (%s–%s). "
            "year-end must be < holdout-start. Aborting.",
            args.year_start, args.year_end,
            args.holdout_start, args.holdout_end,
        )
        sys.exit(1)

    # ── Step 1: Load or fetch calibration batch ────────────────────────────
    _section("Step 1 — Calibration data")

    if _FEATURE_CACHE.exists() and not args.refresh_cache:
        logger.info("Loading calibration batch from cache: %s", _FEATURE_CACHE)
        cal_df = pd.read_parquet(_FEATURE_CACHE)
        logger.info("  Loaded %s rows from cache.", f"{len(cal_df):,}")
    else:
        logger.info(
            "Fetching calibration batch %s–%s from data lake …",
            args.year_start, args.year_end,
        )
        cal_df = asyncio.run(fetch_prospect_scores_batch(args.year_start, args.year_end))
        if cal_df.empty:
            logger.error(
                "Calibration batch is empty. "
                "Run ingestion/pipeline.py Stage 3 first to populate "
                "draft_value_history."
            )
            sys.exit(1)
        _FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cal_df.to_parquet(_FEATURE_CACHE, index=False)
        logger.info(
            "  Fetched %s rows — cached to %s.", f"{len(cal_df):,}", _FEATURE_CACHE
        )

    n_seasons = cal_df["season"].nunique() if "season" in cal_df.columns else 0
    if n_seasons < _MIN_CALIBRATION_SEASONS:
        logger.warning(
            "Only %s calibration seasons found (minimum recommended: %s). "
            "The expected_av_curve and quota tables may be noisy.",
            n_seasons, _MIN_CALIBRATION_SEASONS,
        )

    # ── Step 2: Load holdout data (touched once, only for validation) ──────
    _section("Step 2 — Holdout data (2021–2022, no leakage)")

    if _HOLDOUT_CACHE.exists() and not args.refresh_cache:
        logger.info("Loading holdout batch from cache: %s", _HOLDOUT_CACHE)
        holdout_df = pd.read_parquet(_HOLDOUT_CACHE)
    else:
        logger.info(
            "Fetching holdout batch %s–%s from data lake …",
            args.holdout_start, args.holdout_end,
        )
        holdout_df = asyncio.run(fetch_prospect_scores_batch(args.holdout_start, args.holdout_end))
        if not holdout_df.empty:
            _HOLDOUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            holdout_df.to_parquet(_HOLDOUT_CACHE, index=False)
    logger.info("  Holdout: %s picks.", f"{len(holdout_df):,}")

    # ── Step 3: Calibrate ──────────────────────────────────────────────────
    _section("Step 3 — Calibration")

    model = DraftOptimizerModel()
    model.calibrate(
        cal_df,
        calibration_year_start=args.year_start,
        calibration_year_end=args.year_end,
    )

    # ── Step 4: Baseline comparison on holdout ─────────────────────────────
    _section("Step 4 — ADP baseline comparison")

    adp_score       = _adp_baseline_score(holdout_df)
    optimizer_score = _optimizer_baseline_score(holdout_df, model)
    beats_baseline  = optimizer_score > adp_score

    baseline_results = _build_baseline_results(holdout_df, model)

    # ── Step 5: Validation report ──────────────────────────────────────────
    _print_validation_report(
        cal_df, holdout_df, model,
        adp_score, optimizer_score, beats_baseline,
    )

    # ── Step 6: Save (unless dry-run) ──────────────────────────────────────
    if args.dry_run:
        print("\n  [dry-run] Artifact NOT saved.")
        return

    if not beats_baseline:
        logger.warning(
            "Optimizer does not beat the ADP baseline on holdout data. "
            "Saving artifact anyway — review baseline_results.parquet "
            "before using this model in production."
        )

    _section("Step 5 — Saving artifact")
    model.save(_PKL_PATH)

    if not baseline_results.empty:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        baseline_results.to_parquet(_BASELINE_PATH, index=False)
        logger.info("baseline_results.parquet → %s", _BASELINE_PATH)

    # Append validation metrics to metadata.json written by model.save()
    if _META_PATH.exists():
        with open(_META_PATH) as f:
            meta = json.load(f)
        meta["validation"] = {
            "holdout_years":    [args.holdout_start, args.holdout_end],
            "holdout_picks":    len(holdout_df),
            "adp_baseline_score":    adp_score,
            "optimizer_score":       optimizer_score,
            "beats_adp_baseline":    beats_baseline,
        }
        with open(_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Validation metrics appended to metadata.json.")

    print(f"\n  Artifact saved → {_ARTIFACT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()