"""
Health Analyzer — Training Script
===================================

Trains a stratified Cox PH model (Andersen-Gill recurrent events) to predict
injury risk probability per season per player.

Walk-forward CV folds (season-based):
    Fold 1: train 2010–2016 → val 2017–2018
    Fold 2: train 2010–2018 → val 2019–2020
    Holdout: 2021–2022  (touched once, final evaluation only)

No Optuna needed — Cox PH has one tunable parameter: the L2 penalizer.
We search over a small grid instead.

Evaluation:
    Concordance index (c-index) — measures rank ordering of risk predictions.
    Target: ≥ 0.65 (naive positional baseline is typically ~0.60).

Usage:
    python -m serving.models.health_analyzer.train
    python -m serving.models.health_analyzer.train --version v2
"""

import argparse
import asyncio
import logging
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.impute import KNNImputer

from serving.data_lake_client import DataLakeClient
from serving.models.health_analyzer.features import FEATURE_COLS, fetch_health_features
from serving.models.health_analyzer.model import HealthAnalyzerModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Walk-forward fold definitions ─────────────────────────────────────────────

FOLDS = [
    {"train_end": 2016, "val_start": 2017, "val_end": 2018},
    {"train_end": 2018, "val_start": 2019, "val_end": 2020},
]
TRAIN_START = 2010
HOLDOUT_START = 2021

# L2 penalizer candidates — small grid, Cox PH isn't sensitive to fine tuning
PENALIZER_CANDIDATES = [0.001, 0.01, 0.1, 0.5, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _season_from_id(player_season_id: pd.Series) -> pd.Series:
    """Extract season year from player_season_id = '{name}_{year}'."""
    return player_season_id.str.extract(r"_(\d{4})$")[0].astype(int)


def _split(frame: pd.DataFrame, train_end: int, val_start: int, val_end: int):
    seasons = _season_from_id(frame["player_season_id"])
    return (
        frame[seasons <= train_end],
        frame[(seasons >= val_start) & (seasons <= val_end)],
    )


def _holdout_split(frame: pd.DataFrame):
    seasons = _season_from_id(frame["player_season_id"])
    return frame[seasons < HOLDOUT_START], frame[seasons >= HOLDOUT_START]


def _knn_fit(frame: pd.DataFrame, n_neighbors: int = 10) -> tuple:
    """
    Fit a KNNImputer on static feature columns at player-season level.
    Must be called on TRAINING data only — never the full dataset — to avoid
    temporal leakage where future player measurements inform past imputation.
    Returns (imputer, static_cols).
    """
    from serving.models.health_analyzer.features import STATIC_FEATURE_COLS

    static_cols = [c for c in STATIC_FEATURE_COLS if c in frame.columns]
    player_static = (
        frame[["player_season_id"] + static_cols]
        .drop_duplicates(subset=["player_season_id"])
        .copy()
    )
    player_static[static_cols] = player_static[static_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputer.fit(player_static[static_cols])
    return imputer, static_cols


def _knn_apply(frame: pd.DataFrame, imputer, static_cols: list[str]) -> pd.DataFrame:
    """
    Transform a frame using a pre-fitted KNNImputer.
    Operates at player-season level then merges back, same as fit step.
    """
    frame = frame.copy()
    frame[static_cols] = frame[static_cols].replace([np.inf, -np.inf], np.nan)
    player_static = (
        frame[["player_season_id"] + static_cols]
        .drop_duplicates(subset=["player_season_id"])
        .copy()
    )
    player_static[static_cols] = imputer.transform(player_static[static_cols])
    frame = frame.drop(columns=static_cols)
    frame = frame.merge(
        player_static[["player_season_id"] + static_cols],
        on="player_season_id",
        how="left",
    )
    return frame


def _drop_low_variance(
    frame: pd.DataFrame, feat_cols: list[str], threshold: float = 0.01
) -> list[str]:
    """Drop feature columns with near-zero variance to avoid Cox convergence failures."""
    keep, dropped = [], []
    for col in feat_cols:
        if col not in frame.columns:
            continue
        var = frame[col].var()
        if var < threshold:
            dropped.append((col, var))
        else:
            keep.append(col)
    if dropped:
        log.warning(
            "Dropping %d low-variance feature(s): %s",
            len(dropped),
            ", ".join(f"{c} (var={v:.4f})" for c, v in dropped),
        )
    return keep


def _c_index(
    cox: CoxTimeVaryingFitter, frame: pd.DataFrame, feat_cols: list[str] | None = None
) -> float:
    if feat_cols is None:
        feat_cols = [c for c in FEATURE_COLS if c in frame.columns]
    log_hz = cox.predict_log_partial_hazard(frame[feat_cols])
    tmp = frame[["player_season_id", "event"]].copy()
    tmp["_log_hz"] = log_hz.values
    per_ps = (
        tmp.groupby("player_season_id")
        .agg(
            event=("event", "max"),
            risk=("_log_hz", "max"),
        )
        .reset_index()
    )
    return concordance_index(per_ps["duration"], -per_ps["risk"], per_ps["event"])


def _naive_baseline_cindex(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> float:
    """
    Baseline: predict injury risk using only positional injury rate from training.
    If our model can't beat this, it adds no value beyond "RBs get hurt".
    """
    pos_rates = train_frame.groupby("position_group")["event"].mean()
    val = val_frame.copy()
    val["_baseline"] = val["position_group"].map(pos_rates).fillna(pos_rates.mean())
    per_ps = (
        val.groupby("player_season_id")
        .agg(
            event=("event", "max"),
            risk=("_baseline", "mean"),
        )
        .reset_index()
    )
    return concordance_index(per_ps["event"], per_ps["risk"])


# ── Main training function ─────────────────────────────────────────────────────


def train(version: str = "v1") -> HealthAnalyzerModel:

    async def _fetch():
        async with DataLakeClient() as client:
            return await fetch_health_features(client, year_start=TRAIN_START)

    log.info("Fetching health features from data lake …")
    frame = asyncio.run(_fetch())
    log.info(
        f"Survival frame: {len(frame):,} rows | "
        f"{frame['player_season_id'].nunique():,} player-seasons | "
        f"event rate: {frame['event'].mean():.3f}"
    )

    # ── Temporal split BEFORE imputation to prevent holdout leakage ──────────
    # The KNN imputer must be fit only on pre-holdout seasons so that future
    # player measurements do not inform imputation of past training rows.
    train_full, holdout_frame = _holdout_split(frame)

    feat_cols = [c for c in FEATURE_COLS if c in frame.columns]
    imputer, static_cols = _knn_fit(train_full, n_neighbors=10)
    train_full = _knn_apply(train_full, imputer, static_cols)
    holdout_frame = _knn_apply(holdout_frame, imputer, static_cols)

    # Variance check on training data only
    feat_cols = _drop_low_variance(train_full, feat_cols)

    # Reconstruct full imputed frame for CV splitting
    frame = pd.concat([train_full, holdout_frame], ignore_index=True)

    # ── Penalizer grid search via walk-forward CV ─────────────────────────────
    log.info(f"Penalizer search: {PENALIZER_CANDIDATES}")
    best_penalizer = 0.1
    best_cv_cindex = 0.0

    _fit_cols = [
        "player_season_id",
        "start",
        "stop",
        "event",
        "position_group",
    ] + feat_cols

    for penalizer in PENALIZER_CANDIDATES:
        fold_cindexes = []
        for fold in FOLDS:
            train_fold, val_fold = _split(frame, **fold)
            if len(val_fold) == 0:
                continue
            try:
                cox = CoxTimeVaryingFitter(penalizer=penalizer)
                cox.fit(
                    train_fold[_fit_cols],
                    id_col="player_season_id",
                    event_col="event",
                    start_col="start",
                    stop_col="stop",
                    strata=["position_group"],
                    show_progress=False,
                )
                ci = _c_index(cox, val_fold, feat_cols)
                fold_cindexes.append(ci)
            except Exception as exc:
                log.warning(f"  penalizer={penalizer} fold failed: {exc}")

        if fold_cindexes:
            mean_ci = float(np.mean(fold_cindexes))
            log.info(
                f"  penalizer={penalizer}: c-index={mean_ci:.4f} "
                f"(folds: {[round(c, 4) for c in fold_cindexes]})"
            )
            if mean_ci > best_cv_cindex:
                best_cv_cindex = mean_ci
                best_penalizer = penalizer

    log.info(f"Best penalizer: {best_penalizer}  CV c-index: {best_cv_cindex:.4f}")

    # ── Final model on all pre-holdout data ───────────────────────────────────
    # train_full and holdout_frame were already split and imputed above
    train_frame = train_full
    log.info(
        f"Final train: {train_frame['player_season_id'].nunique():,} player-seasons | "
        f"Holdout: {holdout_frame['player_season_id'].nunique():,} player-seasons"
    )

    log.info(f"Training final Cox PH model (penalizer={best_penalizer}) …")
    final_cox = CoxTimeVaryingFitter(penalizer=best_penalizer)
    final_cox.fit(
        train_frame[_fit_cols],
        id_col="player_season_id",
        event_col="event",
        start_col="start",
        stop_col="stop",
        strata=["position_group"],
        show_progress=True,
    )

    # ── Holdout evaluation ────────────────────────────────────────────────────
    holdout_cindex = _c_index(final_cox, holdout_frame, feat_cols)
    baseline_cindex = _naive_baseline_cindex(train_frame, holdout_frame)
    beat_baseline = bool(holdout_cindex > baseline_cindex)

    log.info(f"Holdout c-index:        {holdout_cindex:.4f}")
    log.info(
        f"Naive baseline c-index: {baseline_cindex:.4f}  Beat baseline: {beat_baseline}"
    )

    # ── Build inference medians ────────────────────────────────────────────────
    feature_medians = {
        col: float(train_frame[col].median())
        for col in feat_cols
        if col in train_frame.columns
    }
    position_feature_medians: dict = {}
    for pos in train_frame["position_group"].unique():
        pos_frame = train_frame[train_frame["position_group"] == pos]
        position_feature_medians[pos] = {
            col: float(pos_frame[col].median())
            for col in feat_cols
            if col in pos_frame.columns
        }

    # ── Log top coefficients ──────────────────────────────────────────────────
    log.info("\nTop model coefficients (hazard ratio drivers):")
    coefs = final_cox.params_.sort_values(key=abs, ascending=False)
    log.info("\n" + coefs.head(10).to_string())

    # ── Package artifact ──────────────────────────────────────────────────────
    wrapper = HealthAnalyzerModel()
    wrapper._metadata = {
        "penalizer": best_penalizer,
        "best_cv_cindex": round(best_cv_cindex, 4),
        "holdout_cindex": round(holdout_cindex, 4),
        "baseline_cindex": round(baseline_cindex, 4),
        "beat_baseline": beat_baseline,
        "train_player_seasons": int(train_frame["player_season_id"].nunique()),
        "holdout_player_seasons": int(holdout_frame["player_season_id"].nunique()),
        "event_rate": round(float(frame["event"].mean()), 4),
        "folds": FOLDS,
        "feature_names": feat_cols,
        "feature_medians": feature_medians,
        "position_feature_medians": position_feature_medians,
    }
    wrapper.MODEL_VERSION = version
    wrapper._model = final_cox
    wrapper._cox = final_cox
    wrapper._imputer = imputer
    wrapper._is_trained = True
    wrapper._feature_cols = feat_cols
    wrapper._build_position_percentiles(train_frame)

    artifact_path = wrapper.save()
    log.info(f"\nArtifact saved → {artifact_path}")

    return wrapper


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HealthAnalyzerModel")
    parser.add_argument(
        "--version", type=str, default="v1", help="Artifact version tag"
    )
    args = parser.parse_args()
    train(version=args.version)
