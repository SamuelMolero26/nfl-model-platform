"""
Team Diagnosis Model
=====================
Adapts the legacy TeamDiagnosticModel to the serving BaseModel interface.

predict() requires the full season's team_stats_df (all ~32 teams) so that
within-season z-scores remain valid, then filters for the requested team.

predict() returns:
  - unit_scores    : {pass_offense_z, run_offense_z, pass_defense_z, run_defense_z, turnover_z}
  - composites     : {offense_composite_z, defense_composite_z, team_efficiency_z}
  - expected_wins  : float from RidgeCV
  - win_delta      : actual_wins − expected_wins
  - actual_wins    : raw win total (if present in team_stats_df)
  - rankings       : within-season rank for each unit (1 = best)
  - cap_efficiency : ROI columns (populated when contracts_df is passed)
  - metadata       : model_name, version, timestamp
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from serving.models.base import BaseModel
from .features import EW_FEATURE_COLS, FEATURE_COLS

# Re-use the battle-tested scoring logic from the legacy module.
from models.team_diagnostic_model.Team_diagnostic import (
    TeamDiagnosticModel as _CoreModel,
    _PASS_WEIGHT,
    _OFFENSE_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


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
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the expected-wins RidgeCV model.

        X : DataFrame with FEATURE_COLS (team-season rows), indexed by (team, season).
        y : wins Series aligned to X.

        The core model also requires 'season' and 'team' columns, which are
        restored by resetting the MultiIndex before passing to fit().
        """
        df = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
        df["wins"] = y.values

        self._core = _CoreModel(
            pass_weight=self.pass_weight,
            offense_weight=self.offense_weight,
        )
        self._core.fit(df)

        self._feature_cols = [c for c in EW_FEATURE_COLS if c in X.columns]
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
        season         : int               if provided, filters to that season only
        contracts_df   : pd.DataFrame      staged contracts; enables cap-ROI columns

        Returns
        -------
        Dict with unit_scores, composites, expected_wins, win_delta, rankings,
        cap_efficiency, and metadata.
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
            "pass_offense_z", "run_offense_z",
            "pass_defense_z", "run_defense_z", "turnover_z",
        ]
        composite_cols = [
            "offense_composite_z", "defense_composite_z", "team_efficiency_z",
        ]
        rank_cols = [
            "pass_offense_rank", "run_offense_rank",
            "pass_defense_rank", "run_defense_rank",
            "team_efficiency_rank", "overperformance_rank",
        ]
        cap_cols = [
            "offense_cap_hit", "defense_cap_hit", "total_cap_hit",
            "offense_cap_epa_roi", "defense_cap_epa_roi", "cap_efficiency_score",
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
            "cap_efficiency": _pick(cap_cols),
            "metadata": self._base_response(),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate expected-wins accuracy on a held-out set.

        X : DataFrame with FEATURE_COLS, indexed by (team, season).
        y : actual wins Series aligned to X.

        Returns R², MAE, and RMSE on the expected_wins prediction.
        Note: z-scoring is within-season, so pass complete seasons for valid metrics.
        """
        df = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
        df["wins"] = y.values

        scored = self._core.score_teams(df)
        valid = scored[scored["expected_wins"].notna()].copy()

        if valid.empty:
            return {"r2": None, "mae": None, "rmse": None}

        preds = valid["expected_wins"].values
        actuals = valid["wins"].values if "wins" in valid.columns else y.values[: len(preds)]

        return {
            "r2": round(float(r2_score(actuals, preds)), 4),
            "mae": round(float(mean_absolute_error(actuals, preds)), 4),
            "rmse": round(float(np.mean((actuals - preds) ** 2) ** 0.5), 4),
        }

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    # ------------------------------------------------------------------
    # Save / Load (extends BaseModel to also persist the core model)
    # ------------------------------------------------------------------

    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        path = super().save(artifact_dir)
        if self._core is not None:
            with open(path / "core_model.pkl", "wb") as f:
                pickle.dump(self._core, f)
        return path

    def load(self, artifact_dir: Optional[Path] = None) -> "TeamDiagnosisModel":
        super().load(artifact_dir)

        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[3]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )

        core_path = artifact_dir / "core_model.pkl"
        if core_path.exists():
            with open(core_path, "rb") as f:
                self._core = pickle.load(f)

        self._feature_cols = self._metadata.get("feature_names", list(EW_FEATURE_COLS))
        return self
