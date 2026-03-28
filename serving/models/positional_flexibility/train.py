"""
Positional Flexibility — Training Script (KNN)
===============================================
Fits a StandardScaler + Mahalanobis whitening + indexes training data for KNN lookup.
No Optuna, no CV loops — training completes in seconds.

Usage
-----
    python -m serving.models.positional_flexibility.train
    python -m serving.models.positional_flexibility.train --version v4 --label-strategy archetype
"""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from serving.data_lake_client import DataLakeClient
from serving.models.positional_flexibility.features import (
    FEATURE_COLS,
    LABEL_COLS,
    POSITION_GROUP_ORDER,
    fetch_flex_features,
)
from serving.models.positional_flexibility.model import PositionalFlexibilityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HOLDOUT_START = 2021


async def run_training(
    version: str = "v1",
    year_start: int = 2000,
    holdout_start: int = HOLDOUT_START,
    label_strategy: str = "archetype",
    k: int = 15,
) -> PositionalFlexibilityModel:

    log.info("Fetching feature matrix …")
    async with DataLakeClient() as client:
        df, label_strategy = await fetch_flex_features(
            client,
            year_start=year_start,
            include_target=True,
            label_strategy=label_strategy,
        )
    df = df.reset_index()

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    label_cols = [c for c in LABEL_COLS if c in df.columns]

    # ── train / holdout split ─────────────────────────────────────────────
    train_mask = df["draft_year"] < holdout_start
    holdout_mask = df["draft_year"] >= holdout_start

    X_train = df[train_mask][feature_cols]
    X_holdout = df[holdout_mask][feature_cols]
    y_train = df[train_mask][label_cols]
    y_holdout = df[holdout_mask][label_cols]

    log.info(
        f"Train rows: {len(X_train)}  Holdout rows: {len(X_holdout)}  "
        f"Features: {len(feature_cols)}  label_strategy: {label_strategy}"
    )

    # ── build player_meta for comparables ────────────────────────────────
    id_cols = ["player_name", "draft_year", "pos_group"]
    if "player_id" in df.columns:
        id_cols.append("player_id")
    player_meta = (
        df[train_mask][id_cols]
        .copy()
        .reset_index(drop=True)
        .rename(columns={"pos_group": "primary_group"})
    )

    # ── fit KNN model ─────────────────────────────────────────────────────
    flex_model = PositionalFlexibilityModel(k=k)
    flex_model.MODEL_VERSION = version

    log.info(f"Training KNN model (k={k}) …")
    flex_model.train(X_train, y_train, player_meta=player_meta)
    log.info("Training complete.")

    # ── holdout evaluation ────────────────────────────────────────────────
    # Binarise archetype labels at each position's training-set median so
    # AUC-ROC has a meaningful positive class ("better athletic fit than average").
    label_medians = {
        grp: float(y_train[f"label_{grp}"].median())
        for grp in POSITION_GROUP_ORDER
        if f"label_{grp}" in label_cols
    }

    log.info("\n── Holdout evaluation ──")
    holdout_metrics: dict = {
        "per_position": {},
        "macro_spearman_r": None,
        "macro_mae": None,
        "macro_auc_roc": None,
    }
    if len(X_holdout) > 0 and len(y_holdout) > 0:
        holdout_metrics = flex_model.evaluate(X_holdout, y_holdout)

        # ── AUC-ROC per position ─────────────────────────────────────────
        from serving.models.positional_flexibility.features import FEATURE_COLS as _FEAT
        X_holdout_scaled = flex_model._scale(X_holdout[feature_cols])
        flex_preds = flex_model._knn_scores(X_holdout_scaled)  # (n × n_pos)

        aucs: list[float] = []
        for i, grp in enumerate(POSITION_GROUP_ORDER):
            label_col = f"label_{grp}"
            if label_col not in label_cols:
                continue
            y_true_cont = y_holdout[label_col].values
            y_pred_col  = flex_preds[:, i]
            threshold   = label_medians.get(grp, float(np.median(y_true_cont)))
            y_true_bin  = (y_true_cont > threshold).astype(int)
            n_pos = int(y_true_bin.sum())

            if n_pos < 5 or n_pos == len(y_true_bin):
                holdout_metrics["per_position"].setdefault(grp, {})["auc_roc"] = None
                continue

            auc = float(roc_auc_score(y_true_bin, y_pred_col))
            holdout_metrics["per_position"].setdefault(grp, {})["auc_roc"] = round(auc, 4)
            aucs.append(auc)

        macro_auc = round(float(np.mean(aucs)), 4) if aucs else None
        holdout_metrics["macro_auc_roc"] = macro_auc

        log.info("Per-position holdout metrics:")
        for pos, m in holdout_metrics["per_position"].items():
            rho  = m.get("spearman_r")
            auc  = m.get("auc_roc")
            npos = m.get("n_pos", "?")
            ntot = m.get("n_total", "?")
            if rho is not None:
                log.info(
                    f"  {pos:<6}  Spearman-ρ={rho:.4f}  "
                    f"AUC-ROC={auc if auc is not None else 'n/a':>6}  "
                    f"MAE={m['mae']:.4f}  (n_pos={npos}/{ntot})"
                )
            else:
                log.info(
                    f"  {pos:<6}  n_pos={npos} — too few positives to evaluate  "
                    f"AUC-ROC={auc if auc is not None else 'n/a'}"
                )
        log.info(
            f"  Macro  Spearman-ρ={holdout_metrics['macro_spearman_r']}  "
            f"AUC-ROC={macro_auc}  MAE={holdout_metrics['macro_mae']}"
        )
    else:
        log.warning("No holdout rows — skipping evaluation.")

    # ── stamp metadata ────────────────────────────────────────────────────
    flex_model._metadata = {
        **flex_model._build_metadata(),
        "holdout_metrics": holdout_metrics,
        "holdout_macro_spearman_r": holdout_metrics.get("macro_spearman_r"),
        "holdout_macro_mae": holdout_metrics.get("macro_mae"),
        "holdout_macro_auc_roc": holdout_metrics.get("macro_auc_roc"),
        "feature_names": feature_cols,
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "year_start": year_start,
        "holdout_start": holdout_start,
        "label_strategy": label_strategy,
    }

    # ── cache features for later inspection ──────────────────────────────
    features_dir = (
        Path(__file__).parents[3]
        / "artifacts"
        / flex_model.MODEL_NAME
        / version
        / "features"
    )
    features_dir.mkdir(parents=True, exist_ok=True)

    # Holdout parquet (features + labels)
    train_medians = X_train[feature_cols].astype(float).median()
    X_holdout_clean = X_holdout[feature_cols].astype(float).fillna(train_medians)
    holdout_cache = X_holdout_clean.copy()
    for col in label_cols:
        holdout_cache[col] = y_holdout[col].values
    holdout_cache.to_parquet(features_dir / "holdout.parquet", index=False)
    log.info(f"Holdout features cached → {features_dir / 'holdout.parquet'}")

    # Player metadata parquet (needed by notebook and comparables)
    player_meta.to_parquet(features_dir / "player_meta.parquet", index=False)
    log.info(f"Player metadata cached → {features_dir / 'player_meta.parquet'}")


    artifact_path = flex_model.save()
    log.info(f"\nArtifact saved → {artifact_path}")

    return flex_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PositionalFlexibilityModel (KNN)"
    )
    parser.add_argument(
        "--version", type=str, default="v4", help="Artifact version tag (e.g. v4)"
    )
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument(
        "--year-start",
        type=int,
        default=2000,
        help="Earliest draft year to include in training",
    )
    parser.add_argument(
        "--holdout-start",
        type=int,
        default=HOLDOUT_START,
        help="Draft year where holdout begins",
    )
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="archetype",
        choices=["archetype", "snap_share", "declared_position"],
        help=(
            "Label strategy: 'archetype' (default) scores athletic profile similarity "
            "to each position's mean combine profile; 'snap_share' uses career snap "
            "fractions (requires snap_counts); 'declared_position' uses primary position only."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        run_training(
            version=args.version,
            k=args.k,
            year_start=args.year_start,
            holdout_start=args.holdout_start,
            label_strategy=args.label_strategy,
        )
    )
