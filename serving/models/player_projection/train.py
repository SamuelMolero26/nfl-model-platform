"""
Player Projection — Training Script
=====================================

Training strategy: 3-fold walk-forward CV + same-fold early stopping.

    Fold 1: train 2000–2014 → val 2015–2016
    Fold 2: train 2000–2016 → val 2017–2018
    Fold 3: train 2000–2018 → val 2019–2020
    Holdout: 2021–2022  (touched once, after all tuning is done)

Two training modes
------------------
  full     (v1) : all features including draft_value_score / draft_value_percentile
  athletic (v2) : drops draft-value-chart features so the model learns from
                  combine / athletic signal only, independent of draft consensus

for testing / comparison:
    python -m serving.models.player_projection.train (full smaller trial size)
    python -m serving.models.player_projection.train --mode athletic --version v2
    python -m serving.models.player_projection.train --trials 200 --version v3
"""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from serving.data_lake_client import DataLakeClient
from serving.models.player_projection.features import FEATURE_COLS, fetch_feature_matrix
from serving.models.player_projection.model import PlayerProjectionModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Walk-forward fold definitions

FOLDS = [
    {"train_end": 2014, "val_start": 2015, "val_end": 2016},
    {"train_end": 2016, "val_start": 2017, "val_end": 2018},
    {"train_end": 2018, "val_start": 2019, "val_end": 2020},
]
TRAIN_START = 2000
HOLDOUT_START = 2021

# Features that encode "historical average career value by pick slot" —
# essentially a pre-computed lookup of the target variable.
# Excluded in 'athletic' mode to force the model to learn combine signal.
DRAFT_VALUE_FEATURES = [
    "draft_value_score",
    "draft_value_percentile",
    "round_x_draft_value",
]


def _feature_cols(mode: str) -> list[str]:
    """Return the active feature list for the given mode."""
    if mode == "athletic":
        return [c for c in FEATURE_COLS if c not in DRAFT_VALUE_FEATURES]
    return list(FEATURE_COLS)  # 'full' — all features


# Target transform

# car_av is right-skewed (most players ~0–10, rare elites up to 120).
# Training on raw values causes RMSE to over-focus on elite outliers and
# pushes the model toward pick-number proxies (draft_value_score).
# log1p compresses the tail so every career tier gets balanced loss weight.
# Metrics stored in metadata are always converted back to original scale.


def _t(y):
    return np.log1p(np.maximum(y, 0))


def _inv(y):
    return np.expm1(y)


# Data loading & splitting


async def _load_all(client: DataLakeClient) -> pd.DataFrame:
    log.info("Fetching feature matrix from data lake …")
    df = await fetch_feature_matrix(client, year_start=TRAIN_START, include_target=True)
    log.info(f"Feature matrix: {len(df)} rows × {df.shape[1]} cols")
    return df


def _split(
    df: pd.DataFrame, feature_cols: list, train_end: int, val_start: int, val_end: int
):
    years = df.index.get_level_values("draft_year").astype(int)
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols]
    y = df["car_av"]
    train_mask = years <= train_end
    val_mask = (years >= val_start) & (years <= val_end)
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def _holdout_split(df: pd.DataFrame, feature_cols: list):
    years = df.index.get_level_values("draft_year").astype(int)
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols]
    y = df["car_av"]
    train_mask = years < HOLDOUT_START
    holdout_mask = years >= HOLDOUT_START
    return X[train_mask], y[train_mask], X[holdout_mask], y[holdout_mask]


# Optuna objective


def _make_objective(df: pd.DataFrame, feature_cols: list):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        fold_rmses = []
        for fold_idx, fold in enumerate(FOLDS):
            X_tr, y_tr, X_val, y_val = _split(df, feature_cols, **fold)

            if len(X_val) < 10:
                log.warning(
                    f"Fold {fold_idx+1} val set too small ({len(X_val)} rows) — skipping"
                )
                continue

            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(X_tr, _t(y_tr), eval_set=[(X_val, _t(y_val))], verbose=False)

            # RMSE on original car_av scale for interpretability
            preds = np.maximum(_inv(model.predict(X_val)), 0)
            rmse = mean_squared_error(y_val, preds) ** 0.5
            fold_rmses.append(rmse)

            trial.report(rmse, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

    return objective


# Main training entry point


def train(
    n_trials: int = 100, version: str = "v1", mode: str = "full"
) -> PlayerProjectionModel:
    if mode not in ("full", "athletic"):
        raise ValueError(f"mode must be 'full' or 'athletic', got {mode!r}")

    async def _run():
        async with DataLakeClient() as client:
            return await _load_all(client)

    df = asyncio.run(_run())

    before = len(df)
    df = df[np.isfinite(df["car_av"]).values]
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with NaN/inf car_av (kept {len(df)})")
    neg = (df["car_av"] < 0).sum()
    if neg:
        log.warning(f"Clipping {neg} negative car_av values to 0")
    df["car_av"] = df["car_av"].clip(lower=0)

    feat_cols = _feature_cols(mode)
    excluded = [c for c in DRAFT_VALUE_FEATURES if c not in feat_cols]
    log.info(
        f"Mode: {mode!r} | features: {len(feat_cols)} | excluded: {excluded or 'none'}"
    )

    # --- Optuna study ---
    log.info(f"Starting Optuna study — {n_trials} trials, 3-fold walk-forward CV")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(
        _make_objective(df, feat_cols), n_trials=n_trials, show_progress_bar=True
    )

    best_params = study.best_params
    best_cv_rmse = round(study.best_value, 4)
    log.info(f"Best CV RMSE: {best_cv_rmse}")
    log.info(f"Best params: {best_params}")

    # --- Final training on everything pre-holdout ---
    log.info("Retraining on full pre-holdout set …")
    X_train, y_train, X_holdout, y_holdout = _holdout_split(df, feat_cols)
    _, _, X_es, y_es = _split(df, feat_cols, **FOLDS[-1])

    final_model = xgb.XGBRegressor(
        **best_params,
        n_estimators=2000,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(X_train, _t(y_train), eval_set=[(X_es, _t(y_es))], verbose=False)
    log.info(
        f"Final model n_estimators (after early stopping): {final_model.best_iteration}"
    )

    # --- Holdout evaluation (original car_av scale) ---
    holdout_preds = np.maximum(_inv(final_model.predict(X_holdout)), 0)
    holdout_rmse = round(mean_squared_error(y_holdout, holdout_preds) ** 0.5, 4)
    holdout_mae = round(float(np.mean(np.abs(holdout_preds - y_holdout))), 4)
    log.info(f"Holdout RMSE: {holdout_rmse}  MAE: {holdout_mae}")

    baseline_rmse = round(
        mean_squared_error(y_holdout, np.full(len(y_holdout), y_train.mean())) ** 0.5, 4
    )
    beat_baseline = holdout_rmse < baseline_rmse
    log.info(f"Baseline (mean) RMSE: {baseline_rmse}  Beat baseline: {beat_baseline}")

    # --- Package artifact ---
    import shap

    wrapper = PlayerProjectionModel()
    wrapper._metadata = {
        "mode": mode,
        "excluded_features": excluded,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "holdout_rmse": holdout_rmse,
        "holdout_mae": holdout_mae,
        "baseline_rmse": baseline_rmse,
        "beat_baseline": beat_baseline,
        "n_estimators": final_model.best_iteration,
        "train_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "folds": FOLDS,
        "feature_names": [c for c in feat_cols if c in X_train.columns],
    }
    wrapper.MODEL_VERSION = version
    wrapper._model = final_model
    wrapper._is_trained = True
    wrapper._train_features = X_train.copy()
    wrapper._train_features["_car_av"] = y_train.values
    wrapper._explainer = shap.Explainer(final_model)

    artifact_path = wrapper.save()
    log.info(f"Artifact saved → {artifact_path}")

    return wrapper


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlayerProjectionModel")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trial count")
    parser.add_argument(
        "--version", type=str, default="v1", help="Artifact version tag"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "athletic"],
        help=(
            "full     : all features including draft_value_score (v1 behaviour)\n"
            "athletic : drops draft-value-chart features — pure combine/athletic signal"
        ),
    )
    args = parser.parse_args()

    train(n_trials=args.trials, version=args.version, mode=args.mode)
