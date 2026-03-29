"""
Health Analyzer Model
======================

Stratified Cox Proportional Hazards (Andersen-Gill recurrent events).

WHY NOT A BINARY CLASSIFIER?
    A binary "injured yes/no" classifier trained on historical data learns
    correlations, not causes: it memorises that RBs get hurt (trivial) and
    that specific players who were historically injured will be injured again
    (circular). With ~25% class imbalance, it defaults to "no injury" and
    is 75% accurate without predicting anything useful.

    Cox PH models the *rate of injury onset* as a function of causal features.
    Stratifying by position group removes positional baseline differences from
    the hazard ratio estimates so coefficients capture WHY one RB is riskier
    than another, not just that RBs are riskier than QBs.

    Output is always a continuous probability (0–1), never a binary threshold.

predict() returns:
    season_injury_probability  : P(≥ 1 game missed this season) = 1 - S(17)
    survival_curve             : [S(1), S(2), …, S(17)] — prob of staying healthy
    expected_games_played      : Σ S(t) for t=1..17  (games expected healthy)
    injury_risk_tier           : Low / Moderate / High / Very High
    position_percentile        : percentile vs same-position peers in training set
    primary_risk_factors       : top-3 features by |coef × value| contribution
    metadata                   : standard block
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.impute import KNNImputer

from serving.models.base import BaseModel
from .features import FEATURE_COLS, STATIC_FEATURE_COLS

SEASON_WEEKS = 17  # standard NFL regular season


def _risk_tier(percentile: float) -> str:
    if percentile < 25:
        return "Low"
    if percentile < 50:
        return "Moderate"
    if percentile < 75:
        return "High"
    return "Very High"


class HealthAnalyzerModel(BaseModel):

    MODEL_NAME = "health_analyzer"
    MODEL_VERSION = "v1"

    def __init__(self):
        super().__init__()
        self._cox: Optional[CoxTimeVaryingFitter] = None
        self._imputer: Optional[KNNImputer] = None
        # Per-position sorted array of training-set season injury probabilities
        # Used to compute position-adjusted percentiles at inference
        self._position_risk_distributions: dict[str, np.ndarray] = {}
        self._feature_cols: list[str] = []

    # BaseModel interface

    def train(self, X: pd.DataFrame, y=None) -> None:
        """
        Fit Cox PH on the survival frame.

        X  : full survival frame (output of build_survival_frame())
             must contain SURVIVAL_COLS + FEATURE_COLS
        y  : ignored — event column is embedded in X
        """
        self._feature_cols = [c for c in FEATURE_COLS if c in X.columns]
        frame = self._impute(X)

        _fit_cols = ["player_season_id", "start", "stop", "event", "position_group"] + self._feature_cols
        penalizer = self._metadata.get("penalizer", 0.1)
        self._cox = CoxTimeVaryingFitter(penalizer=penalizer)
        self._cox.fit(
            frame[_fit_cols],
            id_col="player_season_id",
            event_col="event",
            start_col="start",
            stop_col="stop",
            strata=["position_group"],
            show_progress=False,
        )

        self._build_position_percentiles(frame)
        self._is_trained = True

    def predict(self, inputs: dict) -> dict:
        """
        Predict injury risk for a single player over one season.

        inputs:
            feature_row    : dict or pd.Series with FEATURE_COLS values
            position_group : str — QB / SKILL / OL / DL / LB / DB / SPEC
            mode           : 'season' (default) | 'draft'
                             'draft' mode uses position medians for time-varying
                             features (no career data available at draft time)

        Returns dict with survival curve, probability, tier, and risk factors.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Load an artifact first.")

        position_group = str(inputs.get("position_group", "SKILL"))
        mode = inputs.get("mode", "season")

        raw = inputs.get("feature_row")
        if raw is None:
            raise ValueError("Provide 'feature_row' in inputs.")
        row = pd.Series(raw) if isinstance(raw, dict) else raw.copy()

        # Position medians for fallback / draft-mode imputation
        pos_medians = self._metadata.get("position_feature_medians", {}).get(
            position_group, {}
        )
        global_medians = self._metadata.get("feature_medians", {})

        def _get(col, week_offset=0):
            if col in row.index and pd.notna(row[col]):
                return float(row[col]) + week_offset
            return float(pos_medians.get(col, global_medians.get(col, 0))) + week_offset

        # Build a per-week prediction frame for the full season
        # Time-varying cols handled explicitly below; all others from row/medians
        _time_varying_explicit = {
            "snap_share_rolling_8wk", "acwr", "snap_share_vs_pos_median",
            "season_snap_acceleration", "games_played_this_season", "career_games_played",
        }
        rows = []
        for week in range(1, SEASON_WEEKS + 1):
            r = {
                col: _get(col)
                for col in self._feature_cols
                if col not in _time_varying_explicit
            }

            if mode == "draft":
                # Draft-time: no career history — use position medians
                r["snap_share_rolling_8wk"] = float(
                    pos_medians.get("snap_share_rolling_8wk",
                                    global_medians.get("snap_share_rolling_8wk", 0.3))
                )
                r["acwr"] = 1.0          # no workload spike at draft time
                r["snap_share_vs_pos_median"] = 1.0
                r["season_snap_acceleration"] = 0.0
                r["games_played_this_season"] = float(week - 1)
                r["career_games_played"] = float(week - 1)
            else:
                r["snap_share_rolling_8wk"] = _get("snap_share_rolling_8wk")
                r["acwr"] = _get("acwr")
                r["snap_share_vs_pos_median"] = _get("snap_share_vs_pos_median")
                r["season_snap_acceleration"] = _get("season_snap_acceleration")
                r["games_played_this_season"] = _get(
                    "games_played_this_season", week_offset=week - 1
                )
                r["career_games_played"] = _get(
                    "career_games_played", week_offset=week - 1
                )

            r["start"] = float(week - 1)
            r["stop"] = float(week)
            r["event"] = 0
            r["position_group"] = position_group
            r["player_season_id"] = "_query"
            rows.append(r)

        pred_frame = pd.DataFrame(rows)

        # Predict log partial hazard per week
        log_hz = self._cox.predict_log_partial_hazard(pred_frame[self._feature_cols])
        hz = np.exp(log_hz.values).clip(0, 1)

        # S(t) = Π_{i=1}^{t} (1 - h_i)  — product of conditional survival probs
        survival_curve = list(np.cumprod(1 - hz))

        season_injury_prob = round(float(1 - survival_curve[-1]), 4)
        expected_games = round(float(np.sum(survival_curve)), 2)

        percentile = self._position_percentile(position_group, season_injury_prob)
        tier = _risk_tier(percentile)
        risk_factors = self._top_risk_factors(row, n=3)

        return {
            "season_injury_probability": season_injury_prob,
            "survival_curve": [round(float(s), 4) for s in survival_curve],
            "expected_games_played": expected_games,
            "injury_risk_tier": tier,
            "position_percentile": round(percentile, 1),
            "primary_risk_factors": risk_factors,
            "metadata": self._base_response(),
        }

    def evaluate(self, X: pd.DataFrame, y=None) -> dict:
        """
        Compute concordance index on a held-out survival frame.
        X must be the survival frame (output of build_survival_frame()).
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained.")

        frame = self._impute(X)
        feat_cols = [c for c in self._feature_cols if c in frame.columns]

        log_hz = self._cox.predict_log_partial_hazard(frame[feat_cols])
        frame = frame.copy()
        frame["_log_hz"] = log_hz.values

        per_ps = (
            frame.groupby("player_season_id")
            .agg(
                event=("event", "max"),
                risk=("_log_hz", "max"),
            )
            .reset_index()
        )

        c_idx = concordance_index(per_ps["event"], -per_ps["risk"])

        return {
            "concordance_index": round(float(c_idx), 4),
            "event_rate": round(float(frame["event"].mean()), 4),
            "n_player_seasons": int(per_ps.shape[0]),
        }

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    # Risk percentile helpers

    def _build_position_percentiles(self, frame: pd.DataFrame) -> None:
        """
        Pre-compute per-player-season injury probabilities on training data
        and store sorted distribution per position for percentile lookup.
        """
        feat_cols = [c for c in self._feature_cols if c in frame.columns]
        log_hz = self._cox.predict_log_partial_hazard(frame[feat_cols])
        frame = frame.copy()
        frame["_log_hz"] = log_hz.values

        for pos in frame["position_group"].unique():
            pos_frame = frame[frame["position_group"] == pos]
            # Mean log-hazard per player-season → approximate season injury prob
            per_ps_hz = pos_frame.groupby("player_season_id")["_log_hz"].mean()
            prob = np.clip(1 - np.exp(-np.exp(per_ps_hz.values)), 0, 1)
            self._position_risk_distributions[pos] = np.sort(prob)

    def _position_percentile(self, position_group: str, prob: float) -> float:
        dist = self._position_risk_distributions.get(position_group)
        if dist is None or len(dist) == 0:
            return 50.0
        return float(np.searchsorted(dist, prob) / len(dist) * 100)

    def _top_risk_factors(self, row: pd.Series, n: int = 3) -> list[dict]:
        """Top-n features by |coef × feature_value|."""
        if self._cox is None:
            return []
        coefs = self._cox.params_
        factors = []
        for feat in self._feature_cols:
            if feat in coefs.index and feat in row.index and pd.notna(row[feat]):
                coef = float(coefs[feat])
                val = float(row[feat])
                factors.append(
                    {
                        "feature": feat,
                        "contribution": round(coef * val, 4),
                        "coefficient": round(coef, 4),
                        "value": round(val, 4),
                    }
                )
        factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return factors[:n]

    def _impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        
        frame = frame.copy()
        
        cols = [c for c in self._feature_cols if c in frame.columns]
        frame[cols] = frame[cols].replace([np.inf, -np.inf], np.nan)
        if self._imputer is not None:
            from serving.models.health_analyzer.features import STATIC_FEATURE_COLS
            static_cols = [c for c in STATIC_FEATURE_COLS if c in frame.columns]
            frame[static_cols] = self._imputer.transform(frame[static_cols])
        else:
            for col in cols:
                if frame[col].isna().any():
                    frame[col] = frame[col].fillna(frame[col].median())
                    
        return frame

    # Save / Load (extends BaseModel to also persist risk distributions)
    def save(self, artifact_dir=None):
        path = super().save(artifact_dir)

        with open(path / "position_risk_distributions.pkl", "wb") as f:
            pickle.dump(self._position_risk_distributions, f)
            
        if self._imputer is not None:
            with open(path / "imputer.pkl", "wb") as f:
                pickle.dump(self._imputer, f)

        return path

    def load(self, artifact_dir=None) -> "HealthAnalyzerModel":
        super().load(artifact_dir)

        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[3]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )
            
        artifact_dir = Path(artifact_dir)

        dist_path = Path(artifact_dir) / "position_risk_distributions.pkl"
        if dist_path.exists():
            with open(dist_path, "rb") as f:
                self._position_risk_distributions = pickle.load(f)

        imputer_path = Path(artifact_dir) / "imputer.pkl"
        if imputer_path.exists():
            with open(imputer_path, "rb") as f:
                self._imputer = pickle.load(f)

        self._feature_cols = self._metadata.get("feature_names", FEATURE_COLS)
        self._cox = self._model  # model.pkl stores the CoxTimeVaryingFitter
        return self
