"""
Team Diagnosis — Training Script
==================================

Fetches full team_statistics history from the data lake, fits the
expected-wins RidgeCV model, evaluates on a temporal holdout, and saves
the artifact to artifacts/team_diagnosis/{version}/.

Training strategy: temporal holdout (last N seasons held out).
  Default: last 2 seasons held out for evaluation.
  All earlier seasons used for training.

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
from serving.models.team_diagnosis.features import EW_FEATURE_COLS, fetch_feature_matrix
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
    Fit TeamDiagnosisModel, evaluate on holdout, save artifact.

    Parameters
    ----------
    holdout_seasons : int
        Number of most-recent seasons to withhold for evaluation.
    version : str
        Artifact version tag (used in artifact directory path).

    Returns
    -------
    Trained TeamDiagnosisModel instance.
    """

    async def _load():
        async with DataLakeClient() as client:
            return await fetch_feature_matrix(
                client, year_start=TRAIN_START, include_target=True
            )

    df = asyncio.run(_load())
    log.info("Feature matrix: %d rows × %d cols", len(df), df.shape[1])

    df_train, df_holdout = _temporal_split(df, holdout_seasons)

    feat_cols = [c for c in EW_FEATURE_COLS if c in df_train.columns]
    missing = [c for c in EW_FEATURE_COLS if c not in df_train.columns]
    if missing:
        log.warning("Missing EW feature columns (will be skipped): %s", missing)

    X_train = df_train[feat_cols]
    y_train = df_train["wins"]

    # --- Fit ---
    wrapper = TeamDiagnosisModel()
    wrapper.MODEL_VERSION = version
    wrapper.train(X_train, y_train)
    log.info("Model fitted on %d team-seasons.", len(X_train))

    # --- Evaluate on holdout ---
    # Pass the full holdout DataFrame (all teams per season) so that
    # within-season z-scores are computed correctly.
    holdout_seasons_list = sorted(
        df_holdout.index.get_level_values("season").unique().tolist()
    )
    holdout_full = df.reset_index()
    holdout_full = holdout_full[holdout_full["season"].isin(holdout_seasons_list)]
    holdout_full = holdout_full.set_index(["team", "season"])

    metrics = wrapper.evaluate(
        holdout_full[feat_cols],
        holdout_full["wins"],
    )
    log.info(
        "Holdout evaluation (%d team-seasons): R²=%.4f  MAE=%.4f  RMSE=%.4f",
        len(holdout_full), metrics["r2"], metrics["mae"], metrics["rmse"],
    )

    # --- Package metadata ---
    n_train_seasons = df_train.index.get_level_values("season").nunique()
    n_hold_seasons = df_holdout.index.get_level_values("season").nunique()

    wrapper._metadata.update(
        {
            "train_rows": len(X_train),
            "holdout_rows": len(df_holdout),
            "train_seasons": n_train_seasons,
            "holdout_seasons": n_hold_seasons,
            "holdout_r2": metrics["r2"],
            "holdout_mae": metrics["mae"],
            "holdout_rmse": metrics["rmse"],
            "ew_features": feat_cols,
            "feature_names": feat_cols,
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
        help="Number of most-recent seasons to hold out for evaluation",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Artifact version tag (default: v1)",
    )
    args = parser.parse_args()
    train(holdout_seasons=args.holdout_seasons, version=args.version)
