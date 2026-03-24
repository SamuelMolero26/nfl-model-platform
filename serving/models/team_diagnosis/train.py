"""
Team Diagnosis — Training Script
==================================

Fetches full team_statistics history from the data lake, fits the
expected-wins RidgeCV model, evaluates on a temporal holdout, and saves
the complete artifact bundle to artifacts/team_diagnosis/{version}/:

  model.pkl                    — RidgeCV expected-wins model
  metadata.json                — training metadata + holdout metrics
  core_model.pkl               — full TeamDiagnosticModel (scoring logic)
  shap_values.pkl              — pre-computed SHAP DataFrame for training set
  features/
    train_features.parquet     — cached full feature matrix (speeds up retraining)

Training strategy: temporal holdout (last N seasons held out for evaluation).
  Default: last 2 seasons held out.  All earlier seasons used for training.
  No cross-validation needed — RidgeCV uses built-in LOO-CV for α selection.

Usage:
    python -m serving.models.team_diagnosis.train
    python -m serving.models.team_diagnosis.train --holdout-seasons 3 --version v2
"""

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from serving.data_lake_client import DataLakeClient
from serving.models.team_diagnosis.features import FEATURE_COLS, fetch_feature_matrix
from serving.models.team_diagnosis.model import TeamDiagnosisModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

TRAIN_START = 1999
DEFAULT_HOLDOUT_SEASONS = 2


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


def _temporal_split(df: pd.DataFrame, holdout_seasons: int):
    """
    Split feature matrix into train / holdout by the last N seasons.

    df is indexed by (team, season).  Returns two DataFrames with the same
    schema — train contains all seasons before the cutoff, holdout the rest.
    """
    seasons = sorted(df.index.get_level_values("season").unique())
    if len(seasons) <= holdout_seasons:
        raise ValueError(
            f"Dataset has only {len(seasons)} season(s); "
            f"cannot hold out {holdout_seasons}."
        )

    cutoff_season = seasons[-holdout_seasons]
    season_idx = df.index.get_level_values("season")

    df_train = df[season_idx < cutoff_season]
    df_holdout = df[season_idx >= cutoff_season]

    log.info(
        "Train seasons: %d–%d (%d team-seasons) | "
        "Holdout seasons: %d–%d (%d team-seasons)",
        seasons[0], cutoff_season - 1, len(df_train),
        cutoff_season, seasons[-1], len(df_holdout),
    )
    return df_train, df_holdout


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(
    holdout_seasons: int = DEFAULT_HOLDOUT_SEASONS,
    version: str = "v1",
) -> TeamDiagnosisModel:
    """
    Fit TeamDiagnosisModel on the full data lake history, evaluate on a
    temporal holdout, and save the artifact bundle.

    Parameters
    ----------
    holdout_seasons : int
        Number of most-recent seasons to withhold for evaluation.
    version : str
        Artifact version tag (used in the artifact directory path).

    Returns
    -------
    Trained and saved TeamDiagnosisModel instance.
    """

    async def _load():
        async with DataLakeClient() as client:
            return await fetch_feature_matrix(
                client, year_start=TRAIN_START, include_target=True
            )

    df = asyncio.run(_load())
    log.info("Feature matrix: %d rows × %d cols", len(df), df.shape[1])

    df_train, df_holdout = _temporal_split(df, holdout_seasons)

    # Pass the full feature matrix so model.train() can:
    #   (a) select EW_FEATURE_COLS for RidgeCV
    #   (b) use all FEATURE_COLS for unit z-score scoring
    #   (c) cache train_features.parquet with the complete column set
    feat_cols = [c for c in FEATURE_COLS if c in df_train.columns]
    missing = [c for c in FEATURE_COLS if c not in df_train.columns]
    if missing:
        log.warning("Missing FEATURE_COLS (absent from data lake): %s", missing)

    X_train = df_train[feat_cols]
    y_train = df_train["wins"]

    # --- Fit ---
    wrapper = TeamDiagnosisModel()
    wrapper.MODEL_VERSION = version
    wrapper.train(X_train, y_train)
    log.info("Model fitted on %d team-seasons.", len(X_train))

    # --- Evaluate on holdout ---
    # Pass ALL teams for the holdout seasons so within-season z-scores are valid.
    holdout_season_list = sorted(
        df_holdout.index.get_level_values("season").unique().tolist()
    )
    holdout_full = df.reset_index()
    holdout_full = holdout_full[holdout_full["season"].isin(holdout_season_list)]
    holdout_full = holdout_full.set_index(["team", "season"])

    metrics = wrapper.evaluate(
        holdout_full[feat_cols],
        holdout_full["wins"],
    )
    log.info(
        "Holdout evaluation (%d team-seasons, %d seasons): R²=%.4f  MAE=%.4f  RMSE=%.4f",
        len(holdout_full),
        len(holdout_season_list),
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )

    # --- Baseline: predict league-average wins for every holdout team ---
    baseline_wins = float(y_train.mean())
    baseline_rmse = round(
        float(
            ((holdout_full["wins"] - baseline_wins) ** 2).mean() ** 0.5
        ),
        4,
    )
    beat_baseline = metrics["rmse"] < baseline_rmse
    log.info(
        "Baseline (mean) RMSE: %.4f  Beat baseline: %s",
        baseline_rmse, beat_baseline,
    )

    # --- Metadata ---
    n_train_seasons = df_train.index.get_level_values("season").nunique()
    n_hold_seasons = df_holdout.index.get_level_values("season").nunique()
    train_seasons_range = sorted(df_train.index.get_level_values("season").unique().tolist())
    holdout_seasons_range = sorted(df_holdout.index.get_level_values("season").unique().tolist())

    ew_cols_used = wrapper.feature_names  # EW cols actually present in training data

    wrapper._metadata.update(
        {
            "train_rows": len(X_train),
            "holdout_rows": len(df_holdout),
            "train_seasons": n_train_seasons,
            "holdout_seasons_count": n_hold_seasons,
            "train_season_range": [train_seasons_range[0], train_seasons_range[-1]],
            "holdout_season_range": [holdout_seasons_range[0], holdout_seasons_range[-1]],
            "holdout_r2": metrics["r2"],
            "holdout_mae": metrics["mae"],
            "holdout_rmse": metrics["rmse"],
            "baseline_rmse": baseline_rmse,
            "beat_baseline": beat_baseline,
            "ew_features": ew_cols_used,
            "all_feature_cols": feat_cols,
            "feature_names": ew_cols_used,   # what BaseModel exposes
            "ridge_alpha": (
                float(wrapper._core._expected_wins_model.alpha_)
                if wrapper._core._is_fitted else None
            ),
        }
    )

    artifact_path = wrapper.save()
    log.info("Artifact saved → %s", artifact_path)

    return wrapper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TeamDiagnosisModel")
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=DEFAULT_HOLDOUT_SEASONS,
        help="Number of most-recent seasons to hold out for evaluation (default: 2)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Artifact version tag (default: v1)",
    )
    args = parser.parse_args()
    train(holdout_seasons=args.holdout_seasons, version=args.version)
