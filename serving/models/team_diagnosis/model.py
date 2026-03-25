"""
Team Diagnosis Model
=====================
Adapts the legacy TeamDiagnosticModel to the serving BaseModel interface.

predict() requires the full season's team_stats_df (all ~32 teams) so that
within-season z-scores remain valid, then filters for the requested team.

Artifact layout
---------------
  model.pkl             — RidgeCV expected-wins model (via BaseModel.save)
  metadata.json         — training metadata (via BaseModel.save)
  core_model.pkl        — full TeamDiagnosticModel for deterministic scoring
  shap_values.pkl       — pre-computed SHAP DataFrame for training set
  features/
    train_features.parquet  — cached full feature matrix (all FEATURE_COLS + wins)

predict() returns:
  - unit_scores    : {pass_offense_z, run_offense_z, pass_defense_z, run_defense_z, turnover_z}
  - composites     : {offense_composite_z, defense_composite_z, team_efficiency_z}
  - expected_wins  : float from RidgeCV
  - win_delta      : actual_wins − expected_wins
  - actual_wins    : raw win total (if present in team_stats_df)
  - rankings       : within-season rank for each unit (1 = best)
  - shap_values    : EW feature contributions to expected_wins prediction
  - cap_efficiency : ROI columns (populated when contracts_df is passed)
  - metadata       : model_name, version, timestamp
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score

from serving.models.base import BaseModel
from .features import EW_FEATURE_COLS, FEATURE_COLS

# Re-use the battle-tested scoring logic from the legacy module.
from serving.models.team_diagnosis.team_diagnostic_model.Team_diagnostic import (
    TeamDiagnosticModel as _CoreModel,
    _PASS_WEIGHT,
    _OFFENSE_WEIGHT,
)

# Helpers
def _safe_val(v):
    """Convert numpy scalars / pandas NA to JSON-safe Python types."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


# Model
class TeamDiagnosisModel(BaseModel):

    MODEL_NAME = "team_diagnosis"
    MODEL_VERSION = "v1"

    def __init__(
        self,
        pass_weight: float = _PASS_WEIGHT,
        offense_weight: float = _OFFENSE_WEIGHT,
    ):
        super().__init__()
        self.pass_weight = pass_weight
        self.offense_weight = offense_weight
        self._core: Optional[_CoreModel] = None
        self._explainer: Optional[shap.LinearExplainer] = None
        self._train_features: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []

    # BaseModel interface

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the expected-wins RidgeCV model.

        X : DataFrame with FEATURE_COLS (team-season rows), indexed by (team, season).
            All columns are used for scoring; only EW_FEATURE_COLS feed the RidgeCV.
        y : wins Series aligned to X.

        The core model requires 'season' and 'team' columns, which are
        restored by resetting the MultiIndex before passing to fit().
        Stores training features for the feature cache and SHAP computation.
        """
        df = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
        df["wins"] = y.values

        self._core = _CoreModel(
            pass_weight=self.pass_weight,
            offense_weight=self.offense_weight,
        )
        self._core.fit(df)

        # Expose the fitted Ridge model via _model so BaseModel.save() pickles it
        if self._core._is_fitted:
            self._model = self._core._expected_wins_model

        # Cache training features for artifact + SHAP background
        self._train_features = X.copy()
        self._train_features["wins"] = y.values

        # Build SHAP LinearExplainer over the EW feature space
        self._explainer = self._build_explainer(X)

        self._feature_cols = list(self._core._ew_features) if self._core._is_fitted else [
            c for c in EW_FEATURE_COLS if c in X.columns
        ]
        self._is_trained = True

    def predict(self, inputs: dict) -> dict:
        """
        Score a single team within its season context.

        Required keys
        -------------
        team           : str               e.g. 'KC'
        team_stats_df  : pd.DataFrame      full-season team stats (all ~32 teams)

        Optional keys
        -------------
        season         : int               filters to that season only
        contracts_df   : pd.DataFrame      staged contracts; enables cap-ROI columns

        Returns
        -------
        Dict with unit_scores, composites, expected_wins, win_delta, shap_values,
        rankings, cap_efficiency, and metadata.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Load an artifact first.")

        team = inputs.get("team", "").upper().strip()
        if not team:
            raise ValueError("'team' key is required in inputs.")

        raw_df: pd.DataFrame = inputs.get("team_stats_df")
        if raw_df is None:
            raise ValueError("'team_stats_df' key is required in inputs.")
        if isinstance(raw_df, dict):
            raw_df = pd.DataFrame([raw_df])

        contracts_df = inputs.get("contracts_df")
        scored = self._core.score_teams(raw_df, contracts_df=contracts_df)

        rows = scored[scored["team"].str.upper() == team]
        if rows.empty:
            raise ValueError(f"Team '{team}' not found in scored results.")

        season_filter = inputs.get("season")
        if season_filter is not None and "season" in rows.columns:
            rows = rows[rows["season"] == int(season_filter)]
        if rows.empty:
            raise ValueError(f"No data for team '{team}' season={season_filter}.")

        row = rows.iloc[0]

        unit_cols = [
            "pass_offense_z",
            "run_offense_z",
            "pass_defense_z",
            "run_defense_z",
            "turnover_z",
        ]
        composite_cols = [
            "offense_composite_z",
            "defense_composite_z",
            "team_efficiency_z",
        ]
        rank_cols = [
            "pass_offense_rank",
            "run_offense_rank",
            "pass_defense_rank",
            "run_defense_rank",
            "team_efficiency_rank",
            "overperformance_rank",
        ]
        cap_cols = [
            "offense_cap_hit",
            "defense_cap_hit",
            "total_cap_hit",
            "offense_cap_epa_roi",
            "defense_cap_epa_roi",
            "cap_efficiency_score",
        ]

        def _pick(cols):
            return {c: _safe_val(row.get(c)) for c in cols if c in scored.columns}

        return {
            "team": team,
            "season": _safe_val(row.get("season")),
            "unit_scores": _pick(unit_cols),
            "composites": _pick(composite_cols),
            "expected_wins": _safe_val(row.get("expected_wins")),
            "win_delta": _safe_val(row.get("win_delta")),
            "actual_wins": _safe_val(row.get("wins")),
            "rankings": _pick(rank_cols),
            "shap_values": self._compute_row_shap(row, scored.columns),
            "cap_efficiency": _pick(cap_cols),
            "metadata": self._base_response(),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate expected-wins accuracy on a held-out set.

        X : DataFrame with FEATURE_COLS, indexed by (team, season).
        y : actual wins Series aligned to X.

        Pass complete seasons (all ~32 teams) so within-season z-scores are valid.
        """
        df = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
        df["wins"] = y.values

        scored = self._core.score_teams(df)
        valid = scored[scored["expected_wins"].notna()].copy()

        if valid.empty:
            return {"r2": None, "mae": None, "rmse": None}

        preds = valid["expected_wins"].values
        actuals = (
            valid["wins"].values if "wins" in valid.columns else y.values[: len(preds)]
        )

        return {
            "r2": round(float(r2_score(actuals, preds)), 4),
            "mae": round(float(mean_absolute_error(actuals, preds)), 4),
            "rmse": round(float(np.mean((actuals - preds) ** 2) ** 0.5), 4),
        }

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    # SHAP helpers

    def _reconstruct_ew_matrix(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Reconstruct the EW feature DataFrame that the scaler/model was fit on.

        Applies _engineer_ew_features to add derived columns, then renames and
        sign-flips source columns via _ew_feature_map, returning a DataFrame
        with columns ordered to match self._core._ew_features.
        """
        ew_features = self._core._ew_features
        feat_map = self._core._ew_feature_map
        if not ew_features or not feat_map:
            return None
        df = self._core._engineer_ew_features(X.copy())
        available = {}
        for feat_name, src_col in feat_map.items():
            if feat_name in ew_features and src_col in df.columns:
                available[feat_name] = -df[src_col] if "def_" in feat_name else df[src_col]
        if not available:
            return None
        feat_df = pd.DataFrame(available, index=df.index)[ew_features]
        return feat_df.dropna()

    def _build_explainer(self, X: pd.DataFrame) -> Optional[shap.LinearExplainer]:
        """
        Build a SHAP LinearExplainer from the fitted RidgeCV + training data.

        Background is the scaled EW feature matrix from training so that SHAP
        values represent deviations from the league-average expected wins.
        """
        if not self._core._is_fitted:
            return None
        ew_features = self._core._ew_features
        if not ew_features or self._core._ew_scaler is None:
            return None
        X_ew = self._reconstruct_ew_matrix(X)
        if X_ew is None or X_ew.empty:
            return None
        X_scaled = self._core._ew_scaler.transform(X_ew.values)
        return shap.LinearExplainer(
            self._core._expected_wins_model,
            masker=shap.maskers.Independent(data=X_scaled),
        )

    def _compute_row_shap(self, row: pd.Series, scored_cols) -> Optional[dict]:
        """
        Compute SHAP values for the expected-wins prediction of a single team-season.

        Returns a dict mapping each EW feature name → its SHAP contribution
        (in wins), sorted by absolute magnitude descending.  Returns None if
        the explainer is not available or any EW feature is missing.
        """
        if self._explainer is None or self._core._ew_scaler is None:
            return None

        # Reconstruct the model's feature values from the row's source columns
        # using the same rename + sign-flip logic as _reconstruct_ew_matrix.
        ew_features = self._core._ew_features
        feat_map = self._core._ew_feature_map
        ew_vals = []
        for feat_name in ew_features:
            src_col = feat_map.get(feat_name)
            if src_col is None:
                return None
            val = _safe_val(row.get(src_col))
            if val is None:
                return None
            ew_vals.append(-val if "def_" in feat_name else val)

        X_ew = np.array([ew_vals])
        X_scaled = self._core._ew_scaler.transform(X_ew)
        sv = self._explainer.shap_values(X_scaled)

        shap_dict = {
            col: round(float(val), 4)
            for col, val in zip(ew_features, sv[0])
        }
        return dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    def _compute_training_shap_df(self) -> Optional[pd.DataFrame]:
        """
        Compute SHAP values for every row in the training feature cache.

        Returns a DataFrame indexed by (team, season) with one column per
        EW feature.  Stored as shap_values.pkl for offline analysis.
        """
        if self._explainer is None or self._train_features is None:
            return None
        X_ew = self._reconstruct_ew_matrix(self._train_features)
        if X_ew is None or X_ew.empty:
            return None
        X_scaled = self._core._ew_scaler.transform(X_ew.values)
        sv = self._explainer.shap_values(X_scaled)
        return pd.DataFrame(sv, index=X_ew.index, columns=X_ew.columns.tolist())

    # Save / Load

    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        """
        Persist all artifact files:
          model.pkl                    — RidgeCV (via BaseModel)
          metadata.json                — training metadata (via BaseModel)
          core_model.pkl               — full TeamDiagnosticModel
          shap_values.pkl              — pre-computed SHAP DataFrame (training set)
          features/train_features.parquet  — cached full feature matrix
        """
        path = super().save(artifact_dir)

        # Core model (full scoring logic)
        if self._core is not None:
            with open(path / "core_model.pkl", "wb") as f:
                pickle.dump(self._core, f)

        # Pre-computed SHAP values for the training set
        shap_df = self._compute_training_shap_df()
        if shap_df is not None:
            with open(path / "shap_values.pkl", "wb") as f:
                pickle.dump(shap_df, f)

        # Feature cache
        if self._train_features is not None:
            features_dir = path / "features"
            features_dir.mkdir(exist_ok=True)
            self._train_features.to_parquet(features_dir / "train_features.parquet")

        return path

    def load(self, artifact_dir: Optional[Path] = None) -> "TeamDiagnosisModel":
        """
        Load all artifact files.  Rebuilds the SHAP explainer from the
        feature cache + core model scaler so shap_values.pkl is not required
        at runtime (it is only for offline inspection).
        """
        super().load(artifact_dir)

        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[3]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )

        # Core model
        core_path = artifact_dir / "core_model.pkl"
        if core_path.exists():
            with open(core_path, "rb") as f:
                self._core = pickle.load(f)

        # Feature cache — also used to rebuild the SHAP explainer
        train_path = artifact_dir / "features" / "train_features.parquet"
        if train_path.exists():
            self._train_features = pd.read_parquet(train_path)
            if self._core is not None and self._core._is_fitted:
                self._explainer = self._build_explainer(self._train_features)

        self._feature_cols = self._metadata.get("feature_names", list(EW_FEATURE_COLS))
        return self
