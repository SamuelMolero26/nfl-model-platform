"""
Player Projection Model
========================
XGBoost regressor predicting Career Approximate Value (car_av).

predict() returns:
  - career_value_score   : raw car_av prediction
  - grade                : human-readable tier (Bust / Backup / Starter / Pro Bowl / Elite)
  - confidence           : std dev across trees (lower = more confident)
  - comparables          : top-3 historical players with similar feature vectors + actual car_av
  - shap_values          : feature_name → SHAP contribution (sums to prediction)
  - metadata             : model_name, version, timestamp
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import normalize

from serving.models.base import BaseModel
from .features import FEATURE_COLS, fetch_feature_matrix

# ---------------------------------------------------------------------------
# Career value grade bands
# ---------------------------------------------------------------------------


def _grade(car_av: float) -> str:
    if car_av < 10:
        return "Bust / Career Backup"
    if car_av < 25:
        return "Backup / Rotational"
    if car_av < 45:
        return "Solid Starter"
    if car_av < 65:
        return "Pro Bowl Calibre"
    return "Elite / All-Pro"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PlayerProjectionModel(BaseModel):

    MODEL_NAME = "player_projection"
    MODEL_VERSION = "v1"

    def __init__(self):
        super().__init__()
        self._model: Optional[xgb.XGBRegressor] = None
        self._explainer: Optional[shap.TreeExplainer] = None
        # Training data kept in memory for comparables lookup
        self._train_features: Optional[pd.DataFrame] = None
        self._train_targets: Optional[pd.Series] = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit XGBoost on the provided feature matrix.
        Called by train.py after Optuna finds best_params.

        Stores training data for comparables lookup at inference time.
        """
        self._feature_cols = [c for c in FEATURE_COLS if c in X.columns]
        X_fit = X[self._feature_cols]

        self._model = xgb.XGBRegressor(
            **self._metadata.get("best_params", {}),
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_fit, y)
        self._explainer = shap.Explainer(self._model)

        # Keep for comparables
        self._train_features = X_fit.copy()
        self._train_features["_car_av"] = y.values
        self._is_trained = True

    def predict(self, inputs: dict) -> dict:
        """
        Run inference for a single prospect.

        inputs keys (all optional except player_name):
            player_name   : str
            draft_year    : int  (if None, uses current year)
            feature_row   : pd.Series or dict  — pre-built feature vector
                            (if not provided, fetched from data lake synchronously)

        Returns dict with career_value_score, grade, confidence,
        comparables, shap_values, metadata.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Load an artifact first.")

        # --- resolve feature vector ---
        if "feature_row" in inputs:
            raw = inputs["feature_row"]
            if isinstance(raw, dict):
                raw = pd.Series(raw)
            feature_row = raw.reindex(self._feature_cols).to_frame().T
        else:
            raise ValueError(
                "Provide 'feature_row' in inputs. "
                "Use fetch_feature_matrix() to build it from the data lake."
            )

        X = feature_row[self._feature_cols].astype(float)

        # --- prediction ---
        pred = float(self._model.predict(X)[0])
        pred = max(0.0, pred)  # car_av can't be negative

        # --- confidence: std dev across individual trees ---
        tree_preds = (
            np.array([est.predict(xgb.DMatrix(X)) for est in self._model.get_booster()])
            if hasattr(self._model, "get_booster")
            else np.array([pred])
        )
        confidence = float(np.std(tree_preds)) if len(tree_preds) > 1 else None

        # --- SHAP values ---
        # shap.Explainer returns an Explanation object in SHAP 0.41+;
        # .values[0] gives the 1-D array of contributions for the first (only) row.
        sv = self._explainer(X)
        shap_vals = sv.values[0]
        shap_dict = {
            col: round(float(val), 4) for col, val in zip(self._feature_cols, shap_vals)
        }
        # Sort by absolute contribution descending
        shap_dict = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # --- comparables ---
        comparables = self._find_comparables(X, n=3)

        return {
            "career_value_score": round(pred, 2),
            "grade": _grade(pred),
            "confidence": round(confidence, 4) if confidence else None,
            "comparables": comparables,
            "shap_values": shap_dict,
            "metadata": self._base_response(),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute regression metrics on a held-out set."""
        X_eval = X[[c for c in self._feature_cols if c in X.columns]].astype(float)
        preds = self._model.predict(X_eval)
        preds = np.maximum(preds, 0)
        return {
            "rmse": round(float(mean_squared_error(y, preds) ** 0.5), 4),
            "mae": round(float(mean_absolute_error(y, preds)), 4),
            "r2": round(float(r2_score(y, preds)), 4),
        }

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    # ------------------------------------------------------------------
    # Comparables
    # ------------------------------------------------------------------

    def _find_comparables(self, X: pd.DataFrame, n: int = 3) -> list[dict]:
        """
        Find the n most similar historical players using cosine similarity
        in the feature space. Only compares within ±15 car_av of prediction
        to surface realistic archetypes, not just closest vectors.
        """
        if self._train_features is None or self._train_features.empty:
            return []

        pred = float(self._model.predict(X)[0])
        car_av_col = "_car_av"

        train = self._train_features.copy()
        feat_cols = [c for c in self._feature_cols if c in train.columns]

        # Normalise both query and training set
        query_vec = X[feat_cols].fillna(0).values
        train_mat = train[feat_cols].fillna(0).values

        # cosine similarity
        query_norm = normalize(query_vec)
        train_norm = normalize(train_mat)
        similarities = (train_norm @ query_norm.T).flatten()

        train = train.copy()
        train["_similarity"] = similarities

        # Filter to players within ±20 car_av of prediction
        if car_av_col in train.columns:
            mask = (train[car_av_col] >= pred - 20) & (train[car_av_col] <= pred + 20)
            filtered = train[mask]
            if len(filtered) < n:
                filtered = train  # fall back to full set if window too narrow
        else:
            filtered = train

        top = filtered.nlargest(n, "_similarity")

        comps = []
        for idx, row in top.iterrows():
            player_name, draft_year = (
                idx if isinstance(idx, tuple) else (str(idx), None)
            )
            comps.append(
                {
                    "player_name": player_name,
                    "draft_year": int(draft_year) if draft_year else None,
                    "actual_car_av": (
                        round(float(row[car_av_col]), 1) if car_av_col in row else None
                    ),
                    "similarity": round(float(row["_similarity"]), 4),
                }
            )
        return comps

    # ------------------------------------------------------------------
    # Save / Load (extends BaseModel to also persist training features)
    # ------------------------------------------------------------------

    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        path = super().save(artifact_dir)

        # Save training features for comparables (separate file — can be large)
        if self._train_features is not None:
            self._train_features.to_parquet(path / "train_features.parquet")

        # Save SHAP explainer
        if self._explainer is not None:
            with open(path / "explainer.pkl", "wb") as f:
                pickle.dump(self._explainer, f)

        return path

    def load(self, artifact_dir: Optional[Path] = None) -> "PlayerProjectionModel":
        super().load(artifact_dir)

        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[3]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )

        train_path = artifact_dir / "train_features.parquet"
        if train_path.exists():
            self._train_features = pd.read_parquet(train_path)

        explainer_path = artifact_dir / "explainer.pkl"
        if explainer_path.exists():
            with open(explainer_path, "rb") as f:
                self._explainer = pickle.load(f)
        elif self._model is not None:
            # Rebuild explainer from model if not saved
            self._explainer = shap.TreeExplainer(self._model)

        self._feature_cols = self._metadata.get("feature_names", FEATURE_COLS)
        return self
