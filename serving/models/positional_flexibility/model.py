"""
Positional Flexibility Model — KNN-based athletic fit scorer
============================================================
For a query player, finds the k most athletically similar historical players
and computes per-position affinity as their inverse-distance-weighted average
archetype-affinity label.

Labels (archetype strategy — default)
--------------------------------------
Each training player carries a label_G score for every position group G that
measures how closely their combine/athletic profile matches the G archetype
(mean of career-qualified primary-G players in standardised feature space):

    label_G = exp(−dist_to_archetype_G / sigma_G) × career_quality

This means a player scores high at G if their ATHLETIC PROFILE resembles
typical G players — regardless of whether they ever played a snap there.
Cross-position fits (e.g. an LB with DB speed/size) are surfaced naturally.

KNN scoring
-----------
For query player P with standardised features x:
  1. Find k=15 nearest neighbours by Euclidean distance in feature space.
  2. Per position group G:
         affinity(G) = Σ( label_G_i × w_i ) / Σ( w_i ),  w_i = 1 / (d_i + ε)
  3. Percentile rank of affinity(G) against training population → viable/package flags.

predict() returns
-----------------
  position_scores  : {group → {affinity_score, percentile, viable_backup, package_player}}
  primary_group    : group with highest affinity_score
  flex_candidates  : groups above package threshold excluding primary
  comparables      : top-k nearest neighbours (name, draft_year, primary_group, labels)
  metadata         : model_name, version, timestamp, k, n_train
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from serving.models.base import BaseModel
from .features import (
    FEATURE_COLS,
    LABEL_COLS,
    POSITION_GROUP_ORDER,
    QUALIFIED_THRESHOLD,
)

# Percentile thresholds — position in training populatio---------

VIABLE_BACKUP_PERCENTILE = 70  # score ≥ 70th pct of training distribution
PACKAGE_PLAYER_PERCENTILE = 50  # score ≥ 50th pct of training distribution

class PositionalFlexibilityModel(BaseModel):

    MODEL_NAME = "positional_flexibility"
    MODEL_VERSION = "v12"

    def __init__(self, k: int = 15, use_mahalanobis: bool = True):
        super().__init__()
        self.k = k
        self.use_mahalanobis = use_mahalanobis
        self._scaler: Optional[StandardScaler] = None
        self._whiten: Optional[np.ndarray] = None     # Cholesky whitening matrix W
        self._nn: Optional[NearestNeighbors] = None
        self._train_X: Optional[np.ndarray] = None    # (n_train × n_feat) whitened
        self._train_y: Optional[np.ndarray] = None    # (n_train × n_pos)  labels
        self._train_scores: Optional[np.ndarray] = None  # (n_train × n_pos) KNN predictions on training set
        self._train_meta: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []
        self._thresholds: dict[str, dict] = {}
        self._label_percentiles: dict[str, dict] = {}

    # BaseModel interface

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        player_meta: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fit the StandardScaler (+ optional Mahalanobis whitening) and index
        training data for KNN lookup.

        Parameters
        ----------
        X           : feature matrix (FEATURE_COLS columns)
        y           : label DataFrame (LABEL_COLS) — continuous archetype affinity values
        player_meta : DataFrame with player_name, draft_year, primary_group
                      (same row order as X)
        """
        self._feature_cols = [c for c in FEATURE_COLS if c in X.columns]

        X_arr = X[self._feature_cols].astype(float)
        X_filled = X_arr.fillna(X_arr.median()).fillna(0.0)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_filled.values)

        if self.use_mahalanobis:
            # Cholesky whitening: W = inv(L) where L = cholesky(cov + α·I)
            # Euclidean distance in whitened space == Mahalanobis distance in scaled space,
            # which accounts for correlated features (e.g. forty_yard ↔ speed_score).
            cov = np.cov(X_scaled.T)
            cov_reg = cov + 1e-3 * np.eye(cov.shape[0])
            L = np.linalg.cholesky(cov_reg)
            self._whiten = np.linalg.inv(L)
            self._train_X = X_scaled @ self._whiten.T
        else:
            self._whiten = None
            self._train_X = X_scaled

        # Enforce that all expected label columns are present to keep alignment with
        # POSITION_GROUP_ORDER and avoid silent misalignment of label indices.
        missing_labels = [c for c in LABEL_COLS if c not in y.columns]
        if missing_labels:
            raise ValueError(
                f"Missing label columns in training data: {missing_labels}. "
                "All LABEL_COLS must be present to train PositionalFlexibilityModel."
            )

        self._label_cols = list(LABEL_COLS)
        self._train_y = y[self._label_cols].values.astype(float)
        self._train_meta = (
            player_meta.reset_index(drop=True) if player_meta is not None else None
        )

        self._nn = NearestNeighbors(
            n_neighbors=self.k,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        self._nn.fit(self._train_X)

        # Score all training players via leave-one-out KNN to avoid self-contamination.
        # Without LOO, each point finds itself as a neighbor (distance=0, weight=1e6),
        # making predicted scores ≈ raw labels. That produces near-binary threshold
        # distributions where p50/p70 collapse to ~0, making all flags meaningless.
        self._train_scores = self._knn_scores(self._train_X)  # (n_train × n_pos) — LOO-safe

        # Enforce 1:1 alignment between label columns and POSITION_GROUP_ORDER.
        # If any LABEL_COLS were dropped when constructing y, this will fail fast
        # instead of silently skipping thresholds for some position groups.
        if self._train_y.shape[1] != len(POSITION_GROUP_ORDER):
            raise ValueError(
                "PositionalFlexibilityModel requires a label matrix with one column per "
                "POSITION_GROUP_ORDER entry. Got "
                f"{self._train_y.shape[1]} columns for {len(POSITION_GROUP_ORDER)} groups. "
                "Ensure all LABEL_COLS are present when constructing y."
            )

        for i, grp in enumerate(POSITION_GROUP_ORDER):
            pred_col  = self._train_scores[:, i]   # continuous predicted scores
            label_col = self._train_y[:, i]        # raw labels

            nonzero_labels = label_col[label_col > 0]
            # Thresholds computed on predictions for label-positive players only.
            # Using pred > 0 would include players with near-zero LOO scores from
            # archetype neighbors — restricting to label_col > 0 ensures thresholds
            # separate strong fits from weak ones within the true archetype population.
            archetype_pred = pred_col[label_col > 0]
            if len(archetype_pred) >= 10:
                viable_backup  = float(np.percentile(archetype_pred, VIABLE_BACKUP_PERCENTILE))
                package_player = float(np.percentile(archetype_pred, PACKAGE_PLAYER_PERCENTILE))
            else:
                viable_backup = package_player = float(np.percentile(pred_col, 90))

            self._thresholds[grp] = {
                "viable_backup":  viable_backup,
                "package_player": package_player,
            }
            self._label_percentiles[grp] = {
                "p50":         float(np.percentile(label_col, 50)),
                "p70":         float(np.percentile(label_col, 70)),
                "p90":         float(np.percentile(label_col, 90)),
                "pred_p50":    float(np.percentile(pred_col, 50)),
                "pred_p70":    float(np.percentile(pred_col, 70)),
                "pred_p90":    float(np.percentile(pred_col, 90)),
                # Positive-tier percentiles (what thresholds are now based on)
                "pos_pred_p50": float(np.percentile(archetype_pred, 50)) if len(archetype_pred) else float("nan"),
                "pos_pred_p70": float(np.percentile(archetype_pred, 70)) if len(archetype_pred) else float("nan"),
                "mean":        float(np.mean(label_col)),
                "base_rate":   float((label_col > 0).mean()),
                "n_nonzero":   int(len(nonzero_labels)),
            }

        self._is_trained = True

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        """Align features, impute with training means, apply scaler + optional whitening."""
        X_arr = X[self._feature_cols].astype(float)
        means = pd.Series(self._scaler.mean_, index=self._feature_cols)
        X_filled = X_arr.fillna(means)
        X_scaled = self._scaler.transform(X_filled.values)
        if self._whiten is not None:
            return X_scaled @ self._whiten.T
        return X_scaled

    def _knn_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Core KNN scoring — always LOO-safe.

        Fetches k+1 neighbors and drops any exact self-match (distance < 1e-9)
        before computing weighted scores. Harmless for new players (no self-match
        exists, so we simply trim to k). Prevents training players from finding
        themselves and inflating their own scores.

        Returns (n_query × n_positions) flex probability matrix.
        """
        dists, indices = self._nn.kneighbors(X_scaled, n_neighbors=self.k + 1)
        n_query = X_scaled.shape[0]
        n_pos = self._train_y.shape[1]
        scores = np.zeros((n_query, n_pos))

        for qi in range(n_query):
            d = dists[qi]
            idx = indices[qi]
            mask = d > 1e-9  # drop self-match
            d, idx = d[mask][: self.k], idx[mask][: self.k]
            if len(d) == 0:
                continue
            w = 1.0 / (d + 1e-6)
            w /= w.sum()
            scores[qi] = (self._train_y[idx] * w[:, None]).sum(axis=0)

        return scores

    def predict(self, inputs: dict) -> dict:
        """
        Score a single player across all 7 position groups.

        inputs must contain 'feature_row': pd.Series or dict of feature values.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Load an artifact first.")

        raw = inputs.get("feature_row")
        if raw is None:
            raise ValueError("Provide 'feature_row' in inputs.")

        if isinstance(raw, dict):
            raw = pd.Series(raw)

        X = raw.reindex(self._feature_cols).to_frame().T
        X_scaled = self._scale(X)

        flex_probs = self._knn_scores(X_scaled)[0]  # (n_positions,)
        raw_dists, raw_indices = self._nn.kneighbors(X_scaled, n_neighbors=self.k + 1)
        # Drop self-match for comparables too
        _d, _i = raw_dists[0], raw_indices[0]
        _mask = _d > 1e-9
        dists = _d[_mask][: self.k][np.newaxis, :]
        indices = _i[_mask][: self.k][np.newaxis, :]

        scores: dict[str, dict] = {}
        for i, grp in enumerate(POSITION_GROUP_ORDER):
            affinity = float(flex_probs[i])
            t = self._thresholds.get(grp, {})
            # Rank against training predicted scores (same space as affinity)
            train_col = self._train_scores[:, i] if self._train_scores is not None else self._train_y[:, i]
            pct_rank = float(np.mean(train_col <= affinity) * 100)
            scores[grp] = {
                "affinity_score": round(affinity, 4),
                "percentile": round(pct_rank, 1),
                "viable_backup": affinity >= t.get("viable_backup", 0.0),
                "package_player": affinity >= t.get("package_player", 0.0),
            }

        primary = max(scores, key=lambda g: scores[g]["affinity_score"])
        flex = [g for g, s in scores.items() if s["package_player"] and g != primary]

        return {
            "position_scores": scores,
            "primary_group": primary,
            "flex_candidates": flex,
            "comparables": self._build_comparables(indices[0], dists[0]),
            "thresholds": self._thresholds,
            "metadata": {
                **self._base_response(),
                "k": self.k,
                "n_train": int(len(self._train_X)),
            },
        }

    def _build_comparables(self, indices: np.ndarray, dists: np.ndarray) -> list[dict]:
        comps = []
        for i, d in zip(indices, dists):
            entry: dict = {"distance": round(float(d), 4)}
            if self._train_meta is not None and i < len(self._train_meta):
                row = self._train_meta.iloc[i]
                entry["player_name"] = row.get("player_name", "?")
                entry["draft_year"] = int(row.get("draft_year", 0))
                entry["primary_group"] = row.get("primary_group", "?")
            for j, grp in enumerate(POSITION_GROUP_ORDER):
                if j < self._train_y.shape[1]:
                    entry[f"label_{grp}"] = round(float(self._train_y[i, j]), 3)
            comps.append(entry)
        return comps

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> dict:
        """
        Evaluate on a held-out set using continuous metrics.

        Per position group: Spearman ρ, MAE, n_pos, n_total.
        """
        X_scaled = self._scale(X)
        flex_probs = self._knn_scores(X_scaled)

        label_cols = [c for c in LABEL_COLS if c in y.columns]
        Y_true = y[label_cols].values.astype(float)

        per_position: dict[str, dict] = {}
        rhos, maes = [], []

        for i, grp in enumerate(POSITION_GROUP_ORDER):
            label_col = f"label_{grp}"
            if label_col not in label_cols or i >= Y_true.shape[1]:
                continue

            col_idx = label_cols.index(label_col)
            y_true_col = Y_true[:, col_idx]
            y_pred_col = flex_probs[:, i]
            n_pos = int((y_true_col > 0).sum())
            n_total = len(y_true_col)

            if n_pos < 5:
                per_position[grp] = {
                    "spearman_r": None,
                    "mae": None,
                    "n_pos": n_pos,
                    "n_total": n_total,
                }
                continue

            rho, _ = spearmanr(y_true_col, y_pred_col)
            mae = float(np.mean(np.abs(y_true_col - y_pred_col)))

            per_position[grp] = {
                "spearman_r": round(float(rho), 4),
                "mae": round(mae, 4),
                "n_pos": n_pos,
                "n_total": n_total,
            }
            rhos.append(float(rho))
            maes.append(mae)

        return {
            "per_position": per_position,
            "macro_spearman_r": round(float(np.mean(rhos)), 4) if rhos else None,
            "macro_mae": round(float(np.mean(maes)), 4) if maes else None,
        }

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    # Comparables (public API — filtered to players who played at position)

    def find_comparables(
        self,
        X: pd.DataFrame,
        position: str,
        n: int = 5,
    ) -> list[dict]:
        """
        Find n most athletically similar players who played at position (label > 0).
        Over-fetches k×10 neighbours then filters by position label.
        """
        if self._train_X is None:
            return []

        grp_idx = (
            POSITION_GROUP_ORDER.index(position)
            if position in POSITION_GROUP_ORDER
            else -1
        )

        X_scaled = self._scale(X)
        k_over = min(self.k * 10, len(self._train_X))
        nn_over = NearestNeighbors(
            n_neighbors=k_over, algorithm="ball_tree", metric="euclidean"
        )
        nn_over.fit(self._train_X)
        dists, indices = nn_over.kneighbors(X_scaled)
        dists, indices = dists[0], indices[0]

        comps = []
        for i, d in zip(indices, dists):
            if grp_idx >= 0 and self._train_y[i, grp_idx] == 0:
                continue
            entry: dict = {
                "distance": round(float(d), 4),
                "flex_label": (
                    round(float(self._train_y[i, grp_idx]), 3) if grp_idx >= 0 else None
                ),
                "approx_car_av": (
                    round(float(self._train_y[i, grp_idx]) * QUALIFIED_THRESHOLD, 1)
                    if grp_idx >= 0
                    else None
                ),
            }
            if self._train_meta is not None and i < len(self._train_meta):
                row = self._train_meta.iloc[i]
                entry["player_name"] = row.get("player_name", "?")
                entry["draft_year"] = int(row.get("draft_year", 0))
            comps.append(entry)
            if len(comps) >= n:
                break

        return comps

    # Save / Load

    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        # Pack all KNN state into self._model so BaseModel.save() pickles it correctly.
        self._model = {
            "scaler":           self._scaler,
            "whiten":           self._whiten,
            "nn":               self._nn,
            "train_X":          self._train_X,
            "train_y":          self._train_y,
            "train_scores":     self._train_scores,
            "feature_cols":     self._feature_cols,
            "thresholds":       self._thresholds,
            "label_percentiles": self._label_percentiles,
            "k":                self.k,
            "use_mahalanobis":  self.use_mahalanobis,
        }
        path = super().save(artifact_dir)
        if self._train_meta is not None:
            self._train_meta.to_parquet(path / "train_features.parquet")
        return path

    def load(self, artifact_dir: Optional[Path] = None) -> "PositionalFlexibilityModel":
        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[3]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )

        super().load(artifact_dir)

        state = self._model  # dict packed by save()
        if isinstance(state, dict):
            self._scaler            = state.get("scaler")
            self._whiten            = state.get("whiten")
            self._nn                = state.get("nn")
            self._train_X           = state.get("train_X")
            self._train_y           = state.get("train_y")
            self._train_scores      = state.get("train_scores")
            self._feature_cols      = state.get("feature_cols", FEATURE_COLS)
            self._thresholds        = state.get("thresholds", {})
            self._label_percentiles = state.get("label_percentiles", {})
            self.k                  = state.get("k", self.k)
            self.use_mahalanobis    = state.get("use_mahalanobis", False)
        else:
            # Legacy artifacts saved before this fix — fall back to metadata only
            self._feature_cols      = self._metadata.get("feature_names", FEATURE_COLS)
            self._thresholds        = self._metadata.get("thresholds", {})
            self._label_percentiles = self._metadata.get("label_percentiles", {})
            self.k                  = self._metadata.get("k", self.k)

        train_path = artifact_dir / "train_features.parquet"
        if train_path.exists():
            self._train_meta = pd.read_parquet(train_path)

        return self

    def _build_metadata(self) -> dict:
        return {
            **self._metadata,
            "thresholds": self._thresholds,
            "label_percentiles": self._label_percentiles,
            "k": self.k,
            "qualified_threshold": QUALIFIED_THRESHOLD,
            "viable_backup_percentile": VIABLE_BACKUP_PERCENTILE,
            "package_player_percentile": PACKAGE_PLAYER_PERCENTILE,
        }
