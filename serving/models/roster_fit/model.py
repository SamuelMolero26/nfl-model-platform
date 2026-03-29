"""
roster_fit/model.py
──────────────────────────────────────────────────────────────────────────────
Player-Roster Fit Model.

Answers: "How well does a specific player's athletic and production profile
match what a given team's scheme demands?"

Output: fit_score (0–100) per (player, team) pair, with per-dimension
contribution scores and human-readable fit/misfit explanations.

──────────────────────────────────────────────────────────────────────────────
Algorithm
──────────────────────────────────────────────────────────────────────────────

Stage 1 — Position-Aware Dimension Mapping
    For each position group a fixed list of (player_dim, scheme_dim) pairs
    defines the compatibility axes.  For example, a WR's speed_score is paired
    with the team's air_yards_scheme because a fast receiver is most valuable
    in a scheme that attacks deep.

Stage 2 — Vectorized Cosine Similarity
    For a given (player, team) pair:
        player_vec[k] = player[player_dim_k]   (z-scored by build_player_roster_profile())
        scheme_vec[k] = team[scheme_dim_k]     (z-scored by build_team_scheme_profile())
        cosine = dot(player_vec, scheme_vec) / (||player_vec|| × ||scheme_vec||)

    Both vectors are already on the same scale (z-scores).  Cosine measures
    DIRECTIONAL alignment: a player who excels at speed playing for a team
    that runs a deep-route offense scores high because both vectors point in
    the same direction (positive, positive).

Stage 3 — Ridge-Weighted Composite (trained, optional)
    Element-wise interaction features: interaction[k] = player_vec[k] * scheme_vec[k]
    These are the inputs to a RidgeCV model whose target is a player's
    production z-score in their first two seasons after joining the team.
    Ridge learns which compatibility axes actually predict outcomes, suppressing
    noise dimensions.

Stage 4 — Score Scaling
    Raw Ridge output (or cosine if untrained) is min-max scaled to 0–100
    within each (position_group, season) so scores are directly comparable
    across players at the same position in the same year.

──────────────────────────────────────────────────────────────────────────────
Two scoring paths
──────────────────────────────────────────────────────────────────────────────

score_cosine_only()   No training required.  Works immediately with
                      team_statistics.parquet + curated athletic/production
                      profiles.  Uses cosine similarity as the fit score.

fit() + score()       Ridge-weighted path.  Requires outcomes_df — a DataFrame
                      of historical player-team transitions with a production
                      outcome (built from rosters.parquet after Stage 0).

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────

    from serving.models.roster_fit.features import build_team_scheme_profile, build_player_roster_profile
    from serving.models.roster_fit.model import RosterFitModel
    import pandas as pd

    team_stats = pd.read_parquet("lake/staged/teams/team_statistics.parquet")
    athletic   = pd.read_parquet("lake/curated/player_athletic_profiles.parquet")
    production = pd.read_parquet("lake/curated/player_production_profiles.parquet")

    team_profiles   = build_team_scheme_profile(team_stats)
    player_profiles = build_player_roster_profile(athletic, production)

    model = RosterFitModel()

    # ── Path A: cosine baseline (no training data needed) ────────────────
    results = model.score_cosine_only(player_profiles, team_profiles, season=2024)

    # ── Path B: full trained model (requires rosters.parquet) ────────────
    outcomes = pd.read_parquet("lake/curated/roster_transition_outcomes.parquet")
    model.fit(player_profiles, team_profiles, outcomes)
    results  = model.score(player_profiles, team_profiles, season=2024)

    # ── Inspect a single team ─────────────────────────────────────────────
    sf_fit = results[results["team"] == "SF"].sort_values("fit_score", ascending=False)
    print(sf_fit[["player_name", "position", "fit_score", "top_fit_reasons"]].head(10))

    # ── Persist / reload ─────────────────────────────────────────────────
    model.save()                              # → artifacts/roster_fit/v1/
    model2 = RosterFitModel()
    model2.load()                             # ← artifacts/roster_fit/v1/
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from serving.models.base import BaseModel
from serving.models.roster_fit.features import (
    DIMENSION_PAIRS,
    DEFAULT_PAIRS,
    fetch_team_scheme_profiles,
    fetch_player_roster_profiles,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Ridge regularization candidates: same log-scale sweep as TeamDiagnosticModel.
_RIDGE_ALPHAS: tuple = (0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0)

# Minimum training observations before the Ridge model is considered reliable.
_MIN_TRAINING_ROWS = 50

# Minimum non-null values per vector element before a pair is dropped from
# the cosine calculation for a given player-team combination.
_MIN_VECTOR_DIMS = 2

# ── Dimension pair definitions ────────────────────────────────────────────────
# Imported from features.py — DIMENSION_PAIRS and DEFAULT_PAIRS live there
# so the feature engineering and model stay in sync automatically.

# ── Private module-level helpers ──────────────────────────────────────────────

def _get_pairs(pos_group: str) -> list[tuple[str, str]]:
    """Return dimension pairs for a position group, with fallback."""
    return DIMENSION_PAIRS.get(pos_group, DEFAULT_PAIRS)


def _minmax_scale_within(
    series: pd.Series, group: pd.Series
) -> pd.Series:
    """
    Min-max scale *series* within each unique value of *group*, mapping to [0, 100].
    Groups with zero range (all values identical) are set to 50.0.
    """
    out = series.copy()
    for grp_val in group.unique():
        mask = group == grp_val
        vals = series[mask]
        lo, hi = vals.min(), vals.max()
        if pd.isna(lo) or lo == hi:
            out[mask] = 50.0
        else:
            out[mask] = 100.0 * (vals - lo) / (hi - lo)
    return out


def _readable_dim_name(player_dim: str, scheme_dim: str) -> str:
    """Return a short human-readable label for a dimension pair."""
    labels = {
        ("speed_score",          "air_yards_scheme"):    "Speed vs. vertical scheme",
        ("speed_score",          "def_pass_scheme"):     "Speed vs. pass-D emphasis",
        ("speed_score",          "run_epa_efficiency"):  "Speed vs. run quality",
        ("speed_score",          "pass_rate"):           "Speed vs. pass volume",
        ("agility_score",        "yac_scheme"):          "Agility vs. YAC scheme",
        ("agility_score",        "def_pass_scheme"):     "Agility vs. coverage scheme",
        ("agility_score",        "pass_epa_efficiency"): "Agility vs. pass-pro scheme",
        ("burst_score",          "air_yards_scheme"):    "Burst vs. deep routes",
        ("burst_score",          "def_pass_scheme"):     "Burst vs. pass-rush scheme",
        ("burst_score",          "run_epa_efficiency"):  "Burst vs. run quality",
        ("strength_score",       "run_success_rate"):    "Strength vs. consistent run game",
        ("strength_score",       "def_run_scheme"):      "Strength vs. run-D emphasis",
        ("strength_score",       "run_epa_efficiency"):  "Strength vs. run efficiency",
        ("size_score",           "pass_success_rate"):   "Size vs. consistent passing",
        ("size_score",           "run_success_rate"):    "Size vs. power run game",
        ("size_score",           "def_run_scheme"):      "Size vs. run-D emphasis",
        ("size_score",           "run_epa_efficiency"):  "Size vs. run quality",
        ("nfl_production_score", "pass_rate"):           "Production vs. pass volume",
        ("nfl_production_score", "pass_epa_efficiency"): "Production vs. scheme quality",
        ("target_share",         "pass_rate"):           "Target share vs. pass volume",
        ("snap_share",           "pass_rate"):           "Snap usage vs. pass emphasis",
        ("epa_per_game",         "pass_epa_efficiency"): "EPA quality vs. scheme quality",
        ("passing_cpoe",         "air_yards_scheme"):    "Accuracy vs. depth demands",
    }
    return labels.get((player_dim, scheme_dim), f"{player_dim} × {scheme_dim}")


# ── RosterFitModel ────────────────────────────────────────────────────────────

class RosterFitModel(BaseModel):
    """
    Player-Roster Fit Model.

    Computes a fit score (0–100) for every (player, team) pair in a given
    season using cosine similarity over position-specific dimension pairs,
    optionally weighted by a Ridge regression trained on historical outcomes.

    Parameters
    ----------
    ridge_alphas : tuple
        Regularization candidates for RidgeCV (LOO-CV selects the best).
        Default matches TeamDiagnosticModel.
    """

    MODEL_NAME = "roster_fit"
    MODEL_VERSION = "v1"

    def __init__(self, ridge_alphas: tuple = _RIDGE_ALPHAS) -> None:
        super().__init__()
        self.ridge_alphas = ridge_alphas

        self._ridge:    Optional[RidgeCV]       = None
        self._scaler:   Optional[StandardScaler] = None
        self._dim_pairs_used: list[tuple[str, str]] = []
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────────
    # BaseModel interface
    # ─────────────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Not supported — RosterFitModel uses a domain-specific fit() interface.

        RosterFitModel's training requires separate player_profiles,
        team_profiles, and outcomes_df DataFrames, each with their own schema.
        These cannot be expressed as a flat (X, y) pair.

        Use fit(player_profiles, team_profiles, outcomes_df) instead.  For
        the cosine-only path, no training is needed at all.

        Raises
        ------
        NotImplementedError — always.
        """
        raise NotImplementedError(
            "RosterFitModel does not use the generic train(X, y) interface. "
            "Use fit(player_profiles, team_profiles, outcomes_df) instead."
        )

    def predict(self, inputs: dict) -> dict:
        """
        Synchronous predict using pre-built profile DataFrames.

        Parameters
        ----------
        inputs : dict
            Required keys:
                player_profiles : pd.DataFrame — from build_player_roster_profile()
                team_profiles   : pd.DataFrame — from build_team_scheme_profile()
                season          : int
            Optional keys:
                position_filter : str
                team_filter     : str
                use_ridge       : bool (default False)

        Returns
        -------
        dict with keys: prediction (list of records), shap_values (None), metadata.
        """
        player_profiles = inputs["player_profiles"]
        team_profiles   = inputs["team_profiles"]
        season          = inputs["season"]
        position_filter = inputs.get("position_filter")
        team_filter     = inputs.get("team_filter")
        use_ridge       = inputs.get("use_ridge", False)

        if use_ridge and self._is_fitted:
            result = self.score(
                player_profiles, team_profiles, season,
                position_filter, team_filter,
            )
        else:
            result = self.score_cosine_only(
                player_profiles, team_profiles, season,
                position_filter, team_filter,
            )

        return {
            "prediction": result.to_dict(orient="records"),
            "shap_values": None,
            "metadata": self._base_response(),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Not applicable — returns an empty dict.

        RosterFitModel produces fit scores via cosine similarity (and
        optionally Ridge regression), not a supervised prediction that can
        be evaluated against a held-out y.  Outcome-based quality assessment
        should be done by comparing fit_scores against actual production_z
        values from roster_transition_outcomes.parquet using
        _build_interaction_matrix() + an external correlation analysis.
        """
        return {}

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of interaction feature names used by the Ridge model."""
        return [f"{p}_x_{s}" for p, s in self._dim_pairs_used]

    # ─────────────────────────────────────────────────────────────────────
    # Public API — Scoring
    # ─────────────────────────────────────────────────────────────────────

    def score_cosine_only(
        self,
        player_profiles: pd.DataFrame,
        team_profiles: pd.DataFrame,
        season: int,
        position_filter: Optional[str] = None,
        team_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score all player-team pairs using pure cosine similarity (no training).

        This is the "cold start" path — it produces meaningful fit scores
        immediately using only team_statistics.parquet and the two curated
        gold tables.  No historical outcome data is required.

        Parameters
        ----------
        player_profiles : DataFrame
            Output of build_player_roster_profile().  Use the draft year or the
            most recent available season for prospects; use the actual season
            for current players.

        team_profiles : DataFrame
            Output of build_team_scheme_profile().  Must contain a row for
            *season*.

        season : int
            The scheme season to score against.  Typically the upcoming or
            current draft year.

        position_filter : str, optional
            Restrict output to a single position group (e.g., "WR", "EDGE").

        team_filter : str, optional
            Restrict output to a single team abbreviation (e.g., "SF", "KC").

        Returns
        -------
        DataFrame
            One row per (player, team) pair.  Columns:
                player_id, player_name, position, position_group,
                team, season, fit_score (0–100), cosine_similarity,
                <dim>_fit per dimension pair, top_fit_reasons, bottom_fit_reasons.

        Examples
        --------
        >>> results = model.score_cosine_only(player_prof, team_prof, season=2024,
        ...                                   team_filter="SF")
        >>> results.sort_values("fit_score", ascending=False).head(5)
        """
        players_s = self._filter_players(player_profiles, season, position_filter)
        teams_s   = self._filter_teams(team_profiles, season, team_filter)

        if players_s.empty or teams_s.empty:
            logger.warning(
                "score_cosine_only: no data after filtering "
                "(season=%s, position=%s, team=%s).",
                season, position_filter, team_filter,
            )
            return pd.DataFrame()

        records = self._score_all_pairs(
            players_s, teams_s, use_ridge=False
        )
        result = pd.DataFrame(records)

        if result.empty:
            return result

        # Scale cosine → 0–100 within (position_group, season)
        result["fit_score"] = _minmax_scale_within(
            result["cosine_similarity"],
            result["position_group"],
        )

        result = self._sort_and_clean(result)
        logger.info(
            "score_cosine_only → %s (player, team) pairs scored "
            "(season=%s, position=%s, team=%s).",
            f"{len(result):,}", season, position_filter, team_filter,
        )
        return result

    def score(
        self,
        player_profiles: pd.DataFrame,
        team_profiles: pd.DataFrame,
        season: int,
        position_filter: Optional[str] = None,
        team_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score all player-team pairs using Ridge-weighted fit (requires fit()).

        Identical interface to score_cosine_only() but uses the learned Ridge
        weights to produce a more predictive fit score.  Falls back to cosine
        similarity with a warning if the model has not been fitted.

        Parameters
        ----------
        player_profiles : DataFrame
            Output of build_player_roster_profile().
        team_profiles : DataFrame
            Output of build_team_scheme_profile().
        season : int
            Season to score.
        position_filter : str, optional
            Restrict to one position group.
        team_filter : str, optional
            Restrict to one team.

        Returns
        -------
        DataFrame
            Same schema as score_cosine_only(), with fit_score driven by the
            Ridge model rather than raw cosine similarity.
        """
        if not self._is_fitted:
            logger.warning(
                "RosterFitModel has not been fitted — falling back to cosine "
                "similarity.  Call fit() with outcomes_df to enable Ridge scoring."
            )
            return self.score_cosine_only(
                player_profiles, team_profiles, season,
                position_filter, team_filter,
            )

        players_s = self._filter_players(player_profiles, season, position_filter)
        teams_s   = self._filter_teams(team_profiles, season, team_filter)

        if players_s.empty or teams_s.empty:
            logger.warning(
                "score: no data after filtering "
                "(season=%s, position=%s, team=%s).",
                season, position_filter, team_filter,
            )
            return pd.DataFrame()

        records = self._score_all_pairs(players_s, teams_s, use_ridge=True)
        result  = pd.DataFrame(records)

        if result.empty:
            return result

        result["fit_score"] = _minmax_scale_within(
            result["ridge_raw"],
            result["position_group"],
        )

        result = self._sort_and_clean(result)
        logger.info(
            "score (Ridge) → %s (player, team) pairs scored "
            "(season=%s, position=%s, team=%s).",
            f"{len(result):,}", season, position_filter, team_filter,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Public API — Training
    # ─────────────────────────────────────────────────────────────────────

    def fit(
        self,
        player_profiles: pd.DataFrame,
        team_profiles: pd.DataFrame,
        outcomes_df: pd.DataFrame,
    ) -> "RosterFitModel":
        """
        Fit the Ridge regression using historical player-team transition outcomes.

        Parameters
        ----------
        player_profiles : DataFrame
            Output of build_player_roster_profile() containing all historical
            seasons.

        team_profiles : DataFrame
            Output of build_team_scheme_profile() containing all historical
            seasons.

        outcomes_df : DataFrame
            One row per historical player-team transition.  Required columns:
                player_id      — gsis_id of the player
                team           — 3-letter team abbreviation (new team)
                join_season    — first season on the new team
                production_z   — player's nfl_production_score averaged over
                                 their first two seasons on the new team.
                                 Build this by joining rosters.parquet (Stage 0)
                                 to player_production_profiles.parquet.
            Only rows where join_season <= max_train_season (to prevent leakage)
            should be passed; the caller is responsible for the temporal split.

        Returns
        -------
        self

        Notes
        -----
        Training uses element-wise interaction features: for each dimension pair
        (player_dim, scheme_dim), the feature value is player_val × scheme_val.
        RidgeCV selects the best regularization strength via LOO-CV.

        Temporal integrity: always train on seasons Y-2 and earlier when
        scoring season Y.  The caller constructs the temporal split; this
        method accepts whatever outcomes_df is provided.
        """
        if outcomes_df.empty:
            logger.warning("outcomes_df is empty — Ridge model not fitted.")
            return self

        required_cols = {"player_id", "team", "join_season", "production_z"}
        missing = required_cols - set(outcomes_df.columns)
        if missing:
            raise ValueError(
                f"outcomes_df is missing required columns: {missing}"
            )

        X, y, pairs_used = self._build_interaction_matrix(
            player_profiles, team_profiles, outcomes_df
        )

        if len(X) < _MIN_TRAINING_ROWS:
            logger.warning(
                "Only %d training rows — Ridge model may be unreliable.  "
                "Minimum recommended: %d.",
                len(X), _MIN_TRAINING_ROWS,
            )

        if len(X) == 0:
            logger.error("No training rows — Ridge model not fitted.")
            return self

        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._ridge = RidgeCV(
            alphas=self.ridge_alphas,
            cv=None,           # LOO-CV
            fit_intercept=True,
        )
        self._ridge.fit(X_scaled, y)
        self._dim_pairs_used = pairs_used
        self._is_fitted      = True
        self._is_trained     = True  # keep BaseModel flag in sync

        r2    = self._ridge.score(X_scaled, y)
        alpha = self._ridge.alpha_
        logger.info(
            "RosterFitModel fitted on %d transitions | %d interaction features | "
            "alpha=%.3f | train R²=%.3f",
            len(X), X.shape[1], alpha, r2,
        )
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Public API — Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        """
        Persist the fitted model to disk (BaseModel-compatible).

        Packs all Ridge state into self._model before delegating to
        BaseModel.save(), which writes model.pkl + metadata.json.
        """
        self._model = {
            "ridge":          self._ridge,
            "scaler":         self._scaler,
            "dim_pairs_used": self._dim_pairs_used,
            "is_fitted":      self._is_fitted,
        }
        self._metadata["ridge_alphas"] = list(self.ridge_alphas)
        path = super().save(artifact_dir)
        logger.info("RosterFitModel saved → %s", path)
        return path

    def load(self, artifact_dir: Optional[Path] = None) -> "RosterFitModel":
        """
        Load a previously saved RosterFitModel from disk (BaseModel-compatible).

        Delegates to BaseModel.load(), which reads model.pkl into self._model,
        then unpacks the Ridge state back into the individual instance attributes.
        """
        super().load(artifact_dir)
        if isinstance(self._model, dict):
            self._ridge          = self._model.get("ridge")
            self._scaler         = self._model.get("scaler")
            self._dim_pairs_used = self._model.get("dim_pairs_used", [])
            self._is_fitted      = self._model.get("is_fitted", self._is_trained)
        logger.info("RosterFitModel loaded (fitted=%s)", self._is_fitted)
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Public API — Data lake entry point
    # ─────────────────────────────────────────────────────────────────────

    async def score_from_lake(
        self,
        client,                          # DataLakeClient
        season: int,
        position_filter: Optional[str] = None,
        team_filter: Optional[str] = None,
        use_ridge: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch all required data from the lake and return fit scores.

        This is the primary entry point — equivalent to how train.py calls
        fetch_feature_matrix() in the player projection model.  Sequential
        fetches avoid saturating the remote data lake.

        Parameters
        ----------
        client : DataLakeClient
            Active client pointed at the data lake API.
        season : int
            Season to score against (e.g. 2024 for the upcoming draft).
        position_filter : str, optional
            Restrict to one position group, e.g. "WR".
        team_filter : str, optional
            Restrict to one team abbreviation, e.g. "SF".
        use_ridge : bool
            If True and model is fitted, use Ridge weights. Otherwise cosine only.

        Returns
        -------
        DataFrame — one row per (player, team) pair with columns:
            player_id, player_name, position, position_group,
            team, season, fit_score (0–100), cosine_similarity,
            <dim>_fit per dimension pair,
            top_fit_reasons, bottom_fit_reasons.

        Example
        -------
        >>> import asyncio
        >>> from serving.data_lake_client import DataLakeClient
        >>> from serving.models.roster_fit.model import RosterFitModel
        >>>
        >>> async def main():
        ...     async with DataLakeClient() as client:
        ...         model = RosterFitModel()
        ...         results = await model.score_from_lake(client, season=2024,
        ...                                               position_filter="WR")
        ...         print(results.head(10))
        >>>
        >>> asyncio.run(main())
        """
        logger.info("Fetching team scheme profiles …")
        team_profiles = await fetch_team_scheme_profiles(client, season=season)
        logger.info("  team_profiles: %d rows", len(team_profiles))

        logger.info("Fetching player roster profiles …")
        player_profiles = await fetch_player_roster_profiles(client, season=season)
        logger.info("  player_profiles: %d rows", len(player_profiles))

        if use_ridge and self._is_fitted:
            return self.score(
                player_profiles, team_profiles, season,
                position_filter=position_filter,
                team_filter=team_filter,
            )
        return self.score_cosine_only(
            player_profiles, team_profiles, season,
            position_filter=position_filter,
            team_filter=team_filter,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API — Introspection
    # ─────────────────────────────────────────────────────────────────────

    def feature_weights(self) -> pd.DataFrame:
        """
        Return Ridge coefficient per dimension pair as a DataFrame.

        Only available after fit().  Positive coefficient = this compatibility
        axis is predictive of production; negative = it is not.

        Returns
        -------
        DataFrame with columns: player_dim, scheme_dim, label, ridge_coef.
        Sorted by absolute coefficient descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted.  Call fit() first.")

        coefs = self._ridge.coef_
        rows = []
        for (p_dim, s_dim), coef in zip(self._dim_pairs_used, coefs):
            rows.append({
                "player_dim": p_dim,
                "scheme_dim": s_dim,
                "label":      _readable_dim_name(p_dim, s_dim),
                "ridge_coef": coef,
            })
        return (
            pd.DataFrame(rows)
            .assign(abs_coef=lambda d: d["ridge_coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns=["abs_coef"])
            .reset_index(drop=True)
        )

    def summary(
        self,
        results_df: pd.DataFrame,
        season: Optional[int] = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Print a per-team top-fit summary table.

        Parameters
        ----------
        results_df : DataFrame
            Output of score() or score_cosine_only().
        season : int, optional
            Filter to a single season.
        top_n : int
            Number of top players to show per team.

        Returns
        -------
        DataFrame with columns: team, player_name, position, fit_score, top_fit_reasons.
        """
        df = results_df.copy()
        if season is not None:
            df = df[df["season"] == season]

        return (
            df.sort_values("fit_score", ascending=False)
            .groupby("team", as_index=False)
            .head(top_n)
            [["team", "player_name", "position", "fit_score", "top_fit_reasons"]]
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private: Scoring internals
    # ─────────────────────────────────────────────────────────────────────

    def _filter_players(
        self,
        player_profiles: pd.DataFrame,
        season: int,
        position_filter: Optional[str],
    ) -> pd.DataFrame:
        """
        Select the most relevant player row for the target season.

        For prospects (season=0) or players without a matching season, the
        most recent available season row is used.  This allows scoring draft
        prospects who have no NFL production data yet.
        """
        df = player_profiles.copy()

        # For each player, prefer the row matching *season*.
        # If absent, fall back to the most recent row.
        has_season = df[df["season"] == season]
        no_season  = df[~df["player_id"].isin(has_season["player_id"])]
        if not no_season.empty:
            fallback = (
                no_season.sort_values("season")
                .groupby("player_id", as_index=False)
                .last()
            )
            df = pd.concat([has_season, fallback], ignore_index=True)
        else:
            df = has_season

        if position_filter:
            df = df[df["position_group"] == position_filter]

        return df

    def _filter_teams(
        self,
        team_profiles: pd.DataFrame,
        season: int,
        team_filter: Optional[str],
    ) -> pd.DataFrame:
        """Select team scheme rows for *season*, with optional team filter."""
        df = team_profiles[team_profiles["season"] == season].copy()

        if df.empty:
            # Fall back to the most recent season in the profiles.
            max_season = team_profiles["season"].max()
            logger.warning(
                "No team profiles for season=%s; using latest available: %s.",
                season, max_season,
            )
            df = team_profiles[team_profiles["season"] == max_season].copy()
            df["season"] = season  # Relabel so output is consistent.

        if team_filter:
            df = df[df["team"] == team_filter]

        return df

    def _score_all_pairs(
        self,
        players: pd.DataFrame,
        teams: pd.DataFrame,
        use_ridge: bool,
    ) -> list[dict]:
        """
        Compute cosine similarity (and optionally Ridge predictions) for all
        (player, team) pairs.  Iterates over position groups for vectorization.

        Parameters
        ----------
        players : DataFrame
            Filtered player profiles (one row per player).
        teams : DataFrame
            Filtered team profiles (one row per team).
        use_ridge : bool
            If True and model is fitted, append Ridge raw scores.

        Returns
        -------
        list of dict — one record per (player, team) pair.
        """
        records: list[dict] = []

        for pos_group in players["position_group"].unique():
            if pos_group == "UNK":
                continue

            pos_players = players[players["position_group"] == pos_group]
            pairs       = _get_pairs(pos_group)

            # Keep only pairs where both dimensions exist in the DataFrames.
            valid_pairs = [
                (p, s) for p, s in pairs
                if p in pos_players.columns and s in teams.columns
            ]
            if len(valid_pairs) < _MIN_VECTOR_DIMS:
                logger.debug(
                    "Position group %s: only %d valid dimension pair(s) — "
                    "skipping (need ≥ %d).",
                    pos_group, len(valid_pairs), _MIN_VECTOR_DIMS,
                )
                continue

            p_dims = [p for p, _ in valid_pairs]
            s_dims = [s for _, s in valid_pairs]

            # Build matrices: fill NaN with 0 (neutral contribution).
            P = pos_players[p_dims].fillna(0.0).values.astype(float)  # (n_players, n_dims)
            T = teams[s_dims].fillna(0.0).values.astype(float)         # (n_teams, n_dims)

            # ── Vectorized cosine similarity ──────────────────────────────
            P_norms = np.linalg.norm(P, axis=1, keepdims=True)  # (n_players, 1)
            T_norms = np.linalg.norm(T, axis=1, keepdims=True)  # (n_teams, 1)

            dot     = P @ T.T                   # (n_players, n_teams)
            denom   = P_norms @ T_norms.T       # (n_players, n_teams)
            cosine  = np.divide(
                dot, denom,
                out=np.zeros_like(dot),
                where=denom > 0,
            )  # (n_players, n_teams)

            # ── Element-wise interaction: player_val × scheme_val ─────────
            # Shape: (n_players, n_teams, n_dims)
            interaction = P[:, np.newaxis, :] * T[np.newaxis, :, :]

            # ── Ridge raw score (if fitted) ───────────────────────────────
            ridge_scores: Optional[np.ndarray] = None
            if use_ridge and self._is_fitted:
                # Flatten interactions to (n_players × n_teams, n_all_pairs)
                # then project onto the fitted Ridge feature space.
                ridge_scores = self._predict_ridge_flat(
                    interaction, valid_pairs
                )  # shape: (n_players, n_teams)

            # ── Build output records ───────────────────────────────────────
            for i, (_, p_row) in enumerate(pos_players.iterrows()):
                for j, (_, t_row) in enumerate(teams.iterrows()):
                    dim_scores: dict[str, float] = {
                        f"{p}_{s}_fit": float(interaction[i, j, k])
                        for k, (p, s) in enumerate(valid_pairs)
                    }
                    sorted_dims = sorted(
                        dim_scores.items(), key=lambda x: x[1], reverse=True
                    )
                    record: dict = {
                        "player_id":           p_row.get("player_id"),
                        "player_name":         p_row.get("player_name", ""),
                        "position":            p_row.get("position", ""),
                        "position_group":      pos_group,
                        "team":                t_row["team"],
                        "season":              t_row["season"],
                        "cosine_similarity":   float(cosine[i, j]),
                        "top_fit_reasons":     self._reasons(
                            sorted_dims[:2], valid_pairs, positive=True
                        ),
                        "bottom_fit_reasons":  self._reasons(
                            sorted_dims[-2:], valid_pairs, positive=False
                        ),
                        **dim_scores,
                    }
                    if ridge_scores is not None:
                        record["ridge_raw"] = float(ridge_scores[i, j])
                    records.append(record)

        return records

    def _predict_ridge_flat(
        self,
        interaction: np.ndarray,
        valid_pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """
        Apply the fitted Ridge model to a (n_players, n_teams, n_dims) interaction
        array.  Returns a (n_players, n_teams) score matrix.

        Only the interaction features that correspond to dimension pairs present
        in the current scoring run are used.  Missing pairs (from other position
        groups not present at fit time) default to 0.
        """
        n_players, n_teams, n_dims = interaction.shape

        # Build the full feature vector expected by the Ridge model.
        n_features = len(self._dim_pairs_used)
        pair_index  = {pair: idx for idx, pair in enumerate(self._dim_pairs_used)}

        flat_shape  = (n_players * n_teams, n_features)
        X_flat      = np.zeros(flat_shape, dtype=float)

        for k, pair in enumerate(valid_pairs):
            if pair in pair_index:
                col = pair_index[pair]
                X_flat[:, col] = interaction[:, :, k].ravel()

        X_scaled = self._scaler.transform(X_flat)
        raw      = self._ridge.predict(X_scaled)
        return raw.reshape(n_players, n_teams)

    # ─────────────────────────────────────────────────────────────────────
    # Private: Training internals
    # ─────────────────────────────────────────────────────────────────────

    def _build_interaction_matrix(
        self,
        player_profiles: pd.DataFrame,
        team_profiles: pd.DataFrame,
        outcomes_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
        """
        Construct the training matrix from historical player-team transitions.

        For each row in outcomes_df:
          - Look up the player's profile from join_season - 1 (pre-join stats).
          - Look up the team's scheme profile from join_season.
          - Compute element-wise interaction for each applicable dimension pair.

        The full set of pairs across all position groups is unioned so the
        Ridge model sees a consistent feature vector length.  Pairs absent
        for a given player's position group are set to 0.

        Returns
        -------
        X : np.ndarray of shape (n_transitions, n_all_pairs)
        y : np.ndarray of shape (n_transitions,)
        all_pairs : list of (player_dim, scheme_dim) — the feature column order
        """
        # Collect all unique dimension pairs across all position groups.
        all_pairs: list[tuple[str, str]] = []
        seen: set = set()
        for pairs in list(DIMENSION_PAIRS.values()) + [DEFAULT_PAIRS]:
            for pair in pairs:
                if pair not in seen:
                    all_pairs.append(pair)
                    seen.add(pair)

        pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}
        n_features  = len(all_pairs)

        rows_X: list[np.ndarray] = []
        rows_y: list[float]      = []

        for _, outcome in outcomes_df.iterrows():
            pid        = outcome["player_id"]
            team       = outcome["team"]
            join_season = int(outcome["join_season"])
            target     = float(outcome["production_z"])

            # Player profile: use season before join (pre-join performance)
            p_season = join_season - 1
            p_rows   = player_profiles[
                (player_profiles["player_id"] == pid)
                & (player_profiles["season"] == p_season)
            ]
            if p_rows.empty:
                # Try any available season before join
                p_rows = player_profiles[
                    (player_profiles["player_id"] == pid)
                    & (player_profiles["season"] < join_season)
                ]
                if not p_rows.empty:
                    p_rows = p_rows.sort_values("season").iloc[[-1]]

            if p_rows.empty:
                continue

            # Team profile: use join season scheme
            t_rows = team_profiles[
                (team_profiles["team"] == team)
                & (team_profiles["season"] == join_season)
            ]
            if t_rows.empty:
                continue

            p_row  = p_rows.iloc[0]
            t_row  = t_rows.iloc[0]
            pg     = p_row.get("position_group", "UNK")
            pairs  = _get_pairs(pg)

            # Build interaction vector of length n_features (zeros for absent pairs)
            x_vec = np.zeros(n_features, dtype=float)
            for p_dim, s_dim in pairs:
                pair = (p_dim, s_dim)
                if pair not in pair_to_idx:
                    continue
                p_val = p_row.get(p_dim, np.nan)
                s_val = t_row.get(s_dim, np.nan)
                if pd.notna(p_val) and pd.notna(s_val):
                    x_vec[pair_to_idx[pair]] = float(p_val) * float(s_val)

            rows_X.append(x_vec)
            rows_y.append(target)

        if not rows_X:
            logger.error(
                "_build_interaction_matrix: no valid training rows constructed. "
                "Check that player_profiles and team_profiles have matching "
                "player_id / team / season keys with outcomes_df."
            )
            return np.empty((0, n_features)), np.empty(0), all_pairs

        X = np.vstack(rows_X)
        y = np.array(rows_y, dtype=float)
        logger.info(
            "Interaction matrix built: %d rows × %d features from %d outcomes.",
            X.shape[0], X.shape[1], len(outcomes_df),
        )
        return X, y, all_pairs

    # ─────────────────────────────────────────────────────────────────────
    # Private: Utilities
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reasons(
        sorted_dims: list[tuple[str, float]],
        valid_pairs: list[tuple[str, str]],
        positive: bool,
    ) -> list[str]:
        """
        Build human-readable fit reason strings from sorted dimension scores.

        Parameters
        ----------
        sorted_dims : sorted list of (dim_key, score) — descending by score.
        valid_pairs : the (player_dim, scheme_dim) pairs used for this group.
        positive : True → top fit reasons; False → bottom fit reasons (misfit).
        """
        pair_lookup = {
            f"{p}_{s}_fit": (p, s) for p, s in valid_pairs
        }
        reasons = []
        for dim_key, score in sorted_dims:
            if dim_key not in pair_lookup:
                continue
            p_dim, s_dim = pair_lookup[dim_key]
            label        = _readable_dim_name(p_dim, s_dim)
            direction    = "strong fit" if score > 0 else "misfit"
            if not positive and score >= 0:
                continue   # only include genuine misfits in bottom reasons
            reasons.append(f"{label}: {direction} ({score:+.2f})")
        return reasons

    @staticmethod
    def _sort_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Sort by fit_score descending and reset index."""
        return (
            df.sort_values("fit_score", ascending=False)
            .drop(columns=["ridge_raw"], errors="ignore")
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    import asyncio
    import sys

    async def main() -> None:
        try:
            from serving.data_lake_client import DataLakeClient
        except ImportError:
            print("ERROR: DataLakeClient not found — is the data lake API running?")
            print("Start it with:  uvicorn serving.api.main:app --reload")
            sys.exit(1)

        season = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
        position = sys.argv[2].upper() if len(sys.argv) > 2 else None
        team = sys.argv[3].upper() if len(sys.argv) > 3 else None

        print(f"\nRosterFitModel — cosine scoring")
        print(f"  season={season}  position={position or 'ALL'}  team={team or 'ALL'}")
        print("-" * 60)

        async with DataLakeClient() as client:
            model = RosterFitModel()
            results = await model.score_from_lake(
                client,
                season=season,
                position_filter=position,
                team_filter=team,
            )

        if results.empty:
            print("No results — check that the data lake has data for this season.")
            sys.exit(0)

        display_cols = [
            c for c in
            ["player_name", "position", "team", "fit_score",
             "cosine_similarity", "top_fit_reasons"]
            if c in results.columns
        ]
        print(results[display_cols].sort_values("cosine_similarity", ascending=False).head(20).to_string(index=False))
        print(f"\nTotal pairs scored: {len(results):,}")

    asyncio.run(main())