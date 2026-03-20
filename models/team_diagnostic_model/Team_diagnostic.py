"""
models/team_diagnostic.py
──────────────────────────────────────────────────────────────────────────────
Team Diagnostic & ROI Model
──────────────────────────────────────────────────────────────────────────────
 
PURPOSE
    Evaluate every team-season across four independent units, produce
    expected-win estimates from EPA metrics, and (when contract data is
    available) calculate cap-spend efficiency per unit.
 
ANALYTICAL OUTPUTS
    Per team-season row in score_teams() output:
        Unit scores (z-score within season, higher = better):
            pass_offense_z       — passing efficiency composite
            run_offense_z        — rushing efficiency composite
            pass_defense_z       — pass defense composite (inverted: low allowed = high z)
            run_defense_z        — run defense composite (inverted)
            turnover_z           — net turnovers gained vs. league average
 
        Composites:
            offense_composite_z  — weighted average of pass/run offense z
            defense_composite_z  — weighted average of pass/run defense z
            team_efficiency_z    — overall team quality composite
 
        Expected-wins model (Ridge Regression):
            expected_wins        — wins predicted purely from EPA inputs
            win_delta            — actual_wins − expected_wins
                                   positive  → over-performing (coaching, clutch, luck)
                                   negative  → under-performing relative to talent
 
        Rankings within season (1 = best, 32 = worst):
            pass_offense_rank, run_offense_rank,
            pass_defense_rank, run_defense_rank,
            team_efficiency_rank
 
        Cap ROI (requires contracts_df; NaN otherwise):
            offense_cap_hit      — total dollars allocated to offensive positions
            defense_cap_hit      — total dollars allocated to defensive positions
            offense_cap_epa_roi  — EPA points generated per $1 M of offensive spend
            defense_cap_epa_roi  — EPA points suppressed per $1 M of defensive spend
            cap_efficiency_score — composite z-score of both ROI metrics vs. league
 
INPUT DATA
    Required:
        team_stats_df — nflteamstatistics.csv / staged team_statistics.parquet
            Must contain: season, team, *_epa_*, *_success_rate_*, wins, losses,
            n_interceptions, n_fumbles_lost columns.
 
    Optional (enhances model):
        contracts_df — staged contracts.parquet
            Must contain: season, team, position, cap_hit
            When provided, cap ROI columns are populated.
 
        pbp_df — play-by-play aggregate (team, season, avg_epa_by_situation, ...)
            When provided, situational efficiency splits are added.
 
DESIGN NOTES
    • All z-scores are computed within season so era-to-era comparisons remain
      fair (e.g. 1999 pass EPA is measured against 1999 peers, not 2022 peers).
    • Unit composite z-scores are unweighted averages of individually z-scored
      component metrics.  This is equivalent to the first principal component
      when components are already standardised, but remains interpretable.
    • The expected-wins model uses RidgeCV (automatic α selection via LOO-CV)
      on four EPA totals: pass/run offense and pass/run defense.  A season-
      dummy is NOT included so that the coefficients remain stable across years.
    • Defense EPA is sign-flipped before modelling so that the coefficient is
      always positive (more suppression → more wins).
    • fit() and score_teams() accept the same DataFrame, allowing both
      in-sample diagnostics and out-of-sample scoring.
 
USAGE
    from models.team_diagnostic import TeamDiagnosticModel
    import pandas as pd
 
    team_stats = pd.read_parquet("lake/staged/teams/team_statistics.parquet")
    model = TeamDiagnosticModel()
    model.fit(team_stats)
    results = model.score_teams(team_stats)
 
    # With contracts
    contracts = pd.read_parquet("lake/staged/players/contracts.parquet")
    results = model.score_teams(team_stats, contracts_df=contracts)
 
    # Persist fitted model for pipeline reuse
    model.save("ml/models/team_diagnostic.pkl")
    model2 = TeamDiagnosticModel.load("ml/models/team_diagnostic.pkl")
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# OFfensive positions mapped to "offense" buckets all others -> defense

_OFFENSE_POSITIONS = {"QB", "WR", "TE", "RB", "FB", "HB", "T", "G", 
"C", "OT", "OG", "OL"}

# ── Pass vs. run weights ───────────────────────────────────────────────────
# Source: Pearson correlation of each EPA column against wins,
#         computed on nflteamstatistics.csv (765 rows, 1999–2022).
#
#   offense_total_epa_pass  r = 0.697
#   offense_total_epa_run   r = 0.367
#
# Weights are proportional shares of the two correlations:
#   pass_share = 0.697 / (0.697 + 0.367) = 0.655  → rounded to 0.60
#   run_share  = 0.367 / (0.697 + 0.367) = 0.345  → rounded to 0.40
#
# Rounded to nearest 0.05 to avoid false precision given the sample size.
_PASS_WEIGHT = 0.60
_RUN_WEIGHT  = 1.0 - _PASS_WEIGHT   # = 0.40
 
# ── Offense vs. defense weights ────────────────────────────────────────────
# Source: same correlation check, comparing offense vs. defense EPA columns.
#
#   offense_total_epa_pass  r = +0.697   (offense, positive)
#   offense_total_epa_run   r = +0.367   (offense, positive)
#   defense_total_epa_pass  r = -0.545   (defense, sign-flipped = 0.545)
#   defense_total_epa_run   r = -0.232   (defense, sign-flipped = 0.232)
#
#   avg offense |r| = (0.697 + 0.367) / 2 = 0.532
#   avg defense |r| = (0.545 + 0.232) / 2 = 0.389
#
#   offense_share = 0.532 / (0.532 + 0.389) = 0.578  → rounded to 0.55
#   defense_share = 0.389 / (0.532 + 0.389) = 0.422  → rounded to 0.45
_OFFENSE_WEIGHT = 0.55
_DEFENSE_WEIGHT = 1.0 - _OFFENSE_WEIGHT   # = 0.45
 
# ── RidgeCV alpha candidates ───────────────────────────────────────────────
# Standard log-scale sweep from near-zero to strong regularization.
# RidgeCV selects the best α automatically via leave-one-out cross-validation.
# Range 0.01–500 is wide enough to cover both near-OLS and heavily shrunk fits.
_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0)
 
# ── Minimum group size for z-scoring ──────────────────────────────────────
# Need at least 4 teams in a season group before std dev is meaningful.
# In practice every season has 28–32 teams; this guards against edge cases
# like expansion years or future partial-season data.
_MIN_TEAMS_FOR_ZSCORE = 4


####
####
#### HELPERS #########
####
####

def _zscore_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score *col* within each season group.  Returns NaN where group < 4 teams."""
    def _z(g: pd.Series) -> pd.Series:
        if g.notna().sum() < _MIN_TEAMS_FOR_ZSCORE:
            return pd.Series(np.nan, index=g.index)
        mu, sigma = g.mean(), g.std(ddof=0)
        return (g - mu) / sigma if sigma > 0 else pd.Series(0.0, index=g.index)
    return df.groupby("season")[col].transform(_z)

def _rank_within_season(series: pd.Series, season: pd.Series, ascending: bool = False) -> pd.Series:
    """Rank 1 = best within each season.  ascending=False → highest value = rank 1."""
    df_tmp = pd.DataFrame({"val": series, "season": season})
    return df_tmp.groupby("season")["val"].rank(
        method="min", ascending=ascending, na_option="keep"
    ).astype("Int64")


def _safe_composite(*z_series: pd.Series, weights: Optional[list] = None) -> pd.Series:
    """Weighted mean of z-score Series, ignoring NaN columns per row."""
    mat = pd.concat(z_series, axis=1)
    if weights is None:
        weights = [1.0] * mat.shape[1]
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    numerator = mat.mul(w, axis=1).sum(axis=1, skipna=True)
    denom = mat.notna().mul(w, axis=1).sum(axis=1)
    return numerator.where(denom > 0, other=np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
 
def _compute_unit_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert engineered unit features into season-normalized public z-scores.

    Higher is always better in the returned unit columns. Metrics where lower
    raw values are preferable are sign-flipped before the unit composites are
    calculated.
    """
    out = df.copy()

    feature_directions = {
        "_feat_po_epa": 1.0,
        "_feat_po_sr": 1.0,
        "_feat_po_ypa": 1.0,
        "_feat_po_wpa": 1.0,
        "_feat_po_int_rate": -1.0,
        "_feat_ro_epa": 1.0,
        "_feat_ro_sr": 1.0,
        "_feat_ro_ypc": 1.0,
        "_feat_ro_wpa": 1.0,
        "_feat_ro_fum_rate": -1.0,
        "_feat_pd_epa_raw": -1.0,
        "_feat_pd_sr_raw": -1.0,
        "_feat_pd_ypa_raw": -1.0,
        "_feat_pd_wpa_raw": -1.0,
        "_feat_pd_int_rate": 1.0,
        "_feat_rd_epa_raw": -1.0,
        "_feat_rd_sr_raw": -1.0,
        "_feat_rd_ypc_raw": -1.0,
        "_feat_rd_wpa_raw": -1.0,
        "_feat_net_turnovers": 1.0,
    }

    for feature_col, direction in feature_directions.items():
        if feature_col in out.columns:
            out[f"{feature_col}_z"] = _zscore_within_season(out, feature_col) * direction

    out["pass_offense_z"] = _safe_composite(
        out.get("_feat_po_epa_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_po_sr_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_po_ypa_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_po_wpa_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_po_int_rate_z", pd.Series(np.nan, index=out.index)),
    )
    out["run_offense_z"] = _safe_composite(
        out.get("_feat_ro_epa_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_ro_sr_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_ro_ypc_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_ro_wpa_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_ro_fum_rate_z", pd.Series(np.nan, index=out.index)),
    )
    out["pass_defense_z"] = _safe_composite(
        out.get("_feat_pd_epa_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_pd_sr_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_pd_ypa_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_pd_wpa_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_pd_int_rate_z", pd.Series(np.nan, index=out.index)),
    )
    out["run_defense_z"] = _safe_composite(
        out.get("_feat_rd_epa_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_rd_sr_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_rd_ypc_raw_z", pd.Series(np.nan, index=out.index)),
        out.get("_feat_rd_wpa_raw_z", pd.Series(np.nan, index=out.index)),
    )
    out["turnover_z"] = out.get(
        "_feat_net_turnovers_z", pd.Series(np.nan, index=out.index)
    )

    internal_z_cols = [
        c for c in out.columns if c.startswith("_feat_") and c.endswith("_z")
    ]
    if internal_z_cols:
        out.drop(columns=internal_z_cols, inplace=True)

    return out


def _extract_unit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-unit efficiency metrics derived from raw team stats columns.
 
    Turnover-adjusted EPA:
        Turnovers (INT + fumble lost) represent real negative EPA events but
        are counted separately in the raw data.  We penalise/reward units by
        incorporating turnover counts as an additional efficiency dimension.
 
    All columns added are prefixed with '_feat_' to avoid polluting the
    public result DataFrame.
    """
    out = df.copy()
 
    # ── Pass offense ──────────────────────────────────────────────────────
    out["_feat_po_epa"]     = out.get("offense_ave_epa_pass",      np.nan) #AVERAGE EPA PER PASS PLAY (EFFICIENCY)
    out["_feat_po_sr"]      = out.get("offense_success_rate_pass", np.nan) # SUCCESS RATE ON PASS PLAYS (CONSISTENCY)
    out["_feat_po_ypa"]     = out.get("offense_ave_yards_gained_pass", np.nan) # YARDS PER ATTEMPT (VOLUME/EXPLOSIVENESS)
    out["_feat_po_wpa"]     = out.get("offense_ave_wpa_pass",      np.nan) #WIN PROBABILITY ADDED PER PASS (CLUTCH VALUE)
 
    # Interceptions thrown — penalise (negative direction, so we negate later)
    if "offense_n_interceptions" in out.columns: 
        n_pass = out.get("offense_n_plays_pass", pd.Series(np.nan, index=out.index))
        out["_feat_po_int_rate"] = (
            out["offense_n_interceptions"] / n_pass.replace(0, np.nan)
        )
    else:
        out["_feat_po_int_rate"] = np.nan # INTERCEPTION DIVIDED BY PASS PER PLAY PENALTY METRIC
 
    # ── Run offense ───────────────────────────────────────────────────────
    out["_feat_ro_epa"]     = out.get("offense_ave_epa_run",       np.nan) #AVERAGE EPA PER RUN PLAY (EFFICIENCY)
    out["_feat_ro_sr"]      = out.get("offense_success_rate_run",  np.nan) # SUCCESS RATE ON RUN PLAYS (CONSISTENCY)
    out["_feat_ro_ypc"]     = out.get("offense_ave_yards_gained_run", np.nan) # YARDS PER ATTEMPT (VOLUME/EXPLOSIVENESS)
    out["_feat_ro_wpa"]     = out.get("offense_ave_wpa_run",       np.nan) #WIN PROBABILITY ADDED PER RUN (CLUTCH VALUE)
 
 
    if "offense_n_fumbles_lost_run" in out.columns: 
        n_run = out.get("offense_n_plays_run", pd.Series(np.nan, index=out.index))
        out["_feat_ro_fum_rate"] = (
            out["offense_n_fumbles_lost_run"] / n_run.replace(0, np.nan)
        )
    else:
        out["_feat_ro_fum_rate"] = np.nan # FUMBLE METRIC, PENALTY
 
    # ── Pass defense ──────────────────────────────────────────────────────
    # Raw columns: HIGHER = worse for defense → invert sign when z-scoring
    out["_feat_pd_epa_raw"] = out.get("defense_ave_epa_pass",        np.nan)  # 
    out["_feat_pd_sr_raw"]  = out.get("defense_success_rate_pass",   np.nan)
    out["_feat_pd_ypa_raw"] = out.get("defense_ave_yards_gained_pass", np.nan)
    out["_feat_pd_wpa_raw"] = out.get("defense_ave_wpa_pass",        np.nan)
    # defense INTs are GOOD for defense → don't invert
    if "defense_n_interceptions" in out.columns:
        n_pass_d = out.get("defense_n_plays_pass", pd.Series(np.nan, index=out.index))
        out["_feat_pd_int_rate"] = (
            out["defense_n_interceptions"] / n_pass_d.replace(0, np.nan)
        )
    else:
        out["_feat_pd_int_rate"] = np.nan
 
    # ── Run defense ───────────────────────────────────────────────────────
    out["_feat_rd_epa_raw"] = out.get("defense_ave_epa_run",          np.nan) 
    out["_feat_rd_sr_raw"]  = out.get("defense_success_rate_run",     np.nan) #
    out["_feat_rd_ypc_raw"] = out.get("defense_ave_yards_gained_run", np.nan)
    out["_feat_rd_wpa_raw"] = out.get("defense_ave_wpa_run",          np.nan)
 
    # ── Net turnovers ─────────────────────────────────────────────────────
    if all(c in out.columns for c in [
        "defense_n_interceptions", "defense_n_fumbles_lost_pass", "defense_n_fumbles_lost_run",
        "offense_n_interceptions", "offense_n_fumbles_lost_pass", "offense_n_fumbles_lost_run",
    ]):
        to_gained = (
            out["defense_n_interceptions"]
            + out["defense_n_fumbles_lost_pass"]
            + out["defense_n_fumbles_lost_run"]
        )
        to_given  = (
            out["offense_n_interceptions"]
            + out["offense_n_fumbles_lost_pass"]
            + out["offense_n_fumbles_lost_run"]
        )
        out["_feat_net_turnovers"] = to_gained - to_given
    else:
        out["_feat_net_turnovers"] = np.nan
 
    return out


class TeamDiagnosticModel:
    """
    Team Diagnostic & ROI Model for the NFL Data Lake Platform.
 
    Evaluates team efficiency across four units (pass/run offense, pass/run
    defense), predicts expected wins from EPA metrics, and — when contract
    data is supplied — calculates cap-spend efficiency per unit.
 
    Parameters
    ----------
    pass_weight : float
        Weight of passing metrics vs. running within each side of the ball.
        Default 0.60 (reflects higher EPA correlation with wins for passing).
    offense_weight : float
        Weight of offense vs. defense in the overall team composite.
        Default 0.55.
    ridge_alphas : tuple
        Candidate regularisation strengths for RidgeCV expected-wins model.
    """
    def __init__(
        self,
        pass_weight: float = _PASS_WEIGHT,
        offense_weight: float = _OFFENSE_WEIGHT,
        ridge_alphas: tuple = _RIDGE_ALPHAS,
    ):
        self.pass_weight = pass_weight
        self.offense_weight = offense_weight
        self.ridge_alphas = ridge_alphas
 
        self._expected_wins_model: Optional[RidgeCV] = None
        self._ew_scaler: Optional[StandardScaler] = None
        self._ew_features: list[str] = []
        self._is_fitted = False

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _validate_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input DataFrame and coerce data types.

        Parameters
        ----------
        df : DataFrame
            Input team statistics DataFrame

        Returns
        -------
        DataFrame
            Validated and coerced copy of the input
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        out = df.copy()

        # Ensure numeric columns are coerced to float
        numeric_cols = [
            "offense_total_epa_pass", "offense_total_epa_run",
            "defense_total_epa_pass", "defense_total_epa_run",
            "wins", "losses",
        ]

        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        return out

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
 
    def fit(self, team_stats_df: pd.DataFrame) -> "TeamDiagnosticModel":
        """
        Fit the expected-wins regression model.
 
        Parameters
        ----------
        team_stats_df : DataFrame
            Raw team statistics (one row per team-season).
            Must contain: offense_total_epa_pass, offense_total_epa_run,
            defense_total_epa_pass, defense_total_epa_run, wins.
 
        Returns
        -------
        self
        """
        df = team_stats_df.copy()
        df = self._validate_and_coerce(df)
 
        # Build regression feature matrix.
        # Defense EPA is sign-flipped: positive value = good outcome for model
        ew_feature_map = {
            "ew_off_pass_epa": "offense_total_epa_pass",
            "ew_off_run_epa":  "offense_total_epa_run",
            "ew_def_pass_epa": "defense_total_epa_pass",   # will be negated
            "ew_def_run_epa":  "defense_total_epa_run",    # will be negated
        }
 
        available_feats = {}
        for feat_name, src_col in ew_feature_map.items():
            if src_col in df.columns and df[src_col].notna().sum() > 10:
                if "def_" in feat_name:
                    available_feats[feat_name] = -df[src_col]
                else:
                    available_feats[feat_name] = df[src_col]
 
        if not available_feats:
            logger.warning(
                "No EPA columns found — expected-wins model not fitted. "
                "Check that team_stats_df contains *_total_epa_* columns."
            )
            return self
 
        feat_df = pd.DataFrame(available_feats, index=df.index)
        target  = df["wins"] if "wins" in df.columns else None
 
        if target is None or target.isna().all():
            logger.warning("No wins column — expected-wins model not fitted.")
            return self
 
        # Drop rows where any feature or target is NaN
        mask = feat_df.notna().all(axis=1) & target.notna()
        X = feat_df.loc[mask].values
        y = target.loc[mask].values
 
        if len(X) < 30:
            logger.warning(
                "Only %d complete rows for regression — model may be unreliable.", len(X)
            )
 
        self._ew_features = list(feat_df.columns)
        self._ew_scaler   = StandardScaler()
        X_scaled          = self._ew_scaler.fit_transform(X)
 
        self._expected_wins_model = RidgeCV(
            alphas=self.ridge_alphas,
            cv=None,            # LOO-CV
            fit_intercept=True,
        )
        self._expected_wins_model.fit(X_scaled, y)
        self._is_fitted = True
 
        chosen_alpha = self._expected_wins_model.alpha_
        r2_train     = self._expected_wins_model.score(X_scaled, y)
        logger.info(
            "Expected-wins model fitted on %d team-seasons | "
            "features: %s | alpha=%.2f | train R²=%.3f",
            len(X), self._ew_features, chosen_alpha, r2_train,
        )
        return self
    
    def score_teams(
        self,
        team_stats_df: pd.DataFrame,
        contracts_df: Optional[pd.DataFrame] = None,
        pbp_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Score every team-season in team_stats_df.
 
        Parameters
        ----------
        team_stats_df : DataFrame
            Raw team statistics.
        contracts_df : DataFrame, optional
            Staged contracts (season, team, position, cap_hit).
            When provided, cap ROI columns are populated.
        pbp_df : DataFrame, optional
            Reserved for future situational EPA splits from play-by-play.
 
        Returns
        -------
        DataFrame with one row per (team, season) containing all diagnostic
        columns documented in the module docstring.
        """
        df = team_stats_df.copy()
        df = self._validate_and_coerce(df)
 
        # ── Step 1: Feature engineering ───────────────────────────────────
        df = _extract_unit_features(df)
 
        # ── Step 2: Unit z-scores ─────────────────────────────────────────
        df = _compute_unit_zscores(df)
 
        # ── Step 3: Composite z-scores ────────────────────────────────────
        run_w  = 1.0 - self.pass_weight
        def_w  = 1.0 - self.offense_weight
 
        df["offense_composite_z"] = _safe_composite(
            df["pass_offense_z"], df["run_offense_z"],
            weights=[self.pass_weight, run_w],
        )
        df["defense_composite_z"] = _safe_composite(
            df["pass_defense_z"], df["run_defense_z"],
            weights=[self.pass_weight, run_w],
        )
        df["team_efficiency_z"] = _safe_composite(
            df["offense_composite_z"], df["defense_composite_z"],
            weights=[self.offense_weight, def_w],
        )
 
        # ── Step 4: Expected wins ─────────────────────────────────────────
        df = self._append_expected_wins(df)
 
        # ── Step 5: Season rankings ───────────────────────────────────────
        for col, rank_col in [
            ("pass_offense_z",    "pass_offense_rank"),
            ("run_offense_z",     "run_offense_rank"),
            ("pass_defense_z",    "pass_defense_rank"),
            ("run_defense_z",     "run_defense_rank"),
            ("team_efficiency_z", "team_efficiency_rank"),
            ("win_delta",         "overperformance_rank"),
        ]:
            if col in df.columns:
                # ascending=False → highest z = rank 1
                df[rank_col] = _rank_within_season(
                    df[col], df["season"], ascending=False
                )
 
        # ── Step 6: Cap ROI (optional) ────────────────────────────────────
        df = self._append_cap_roi(df, contracts_df)
 
        # ── Step 7: PBP situational splits (future hook) ─────────────────
        if pbp_df is not None:
            df = self._append_pbp_splits(df, pbp_df)
 
        # ── Step 8: Drop internal _feat_ columns ─────────────────────────
        feat_cols = [c for c in df.columns if c.startswith("_feat_")]
        df.drop(columns=feat_cols, inplace=True)
 
        # ── Step 9: Sort and return ───────────────────────────────────────
        df.sort_values(["season", "team_efficiency_z"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
 
        n_seasons = df["season"].nunique()
        logger.info(
            "TeamDiagnosticModel scored %d team-seasons across %d seasons.",
            len(df), n_seasons,
        )
        return df
 
    def summary(self, results_df: pd.DataFrame, season: Optional[int] = None) -> pd.DataFrame:
        """
        Return a human-readable summary of the most important diagnostic
        columns for a given season (or all seasons if season=None).
 
        Columns returned:
            season, team, pass_offense_rank, run_offense_rank,
            pass_defense_rank, run_defense_rank, team_efficiency_rank,
            expected_wins, actual_wins, win_delta,
            [cap_efficiency_score if available]
        """
        df = results_df.copy()
        if season is not None:
            df = df[df["season"] == season]
 
        cols = [
            "season", "team",
            "pass_offense_z",    "pass_offense_rank",
            "run_offense_z",     "run_offense_rank",
            "pass_defense_z",    "pass_defense_rank",
            "run_defense_z",     "run_defense_rank",
            "team_efficiency_z", "team_efficiency_rank",
            "expected_wins",     "wins", "win_delta",
            "overperformance_rank",
        ]
        if "cap_efficiency_score" in df.columns:
            cols.append("cap_efficiency_score")
 
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)
 
    def top_units(
        self,
        results_df: pd.DataFrame,
        unit: str,
        season: Optional[int] = None,
        n: int = 5,
        bottom: bool = False,
    ) -> pd.DataFrame:
        """
        Return the top-N (or bottom-N) teams for a specific unit.
 
        Parameters
        ----------
        unit : {'pass_offense', 'run_offense', 'pass_defense', 'run_defense',
                'team_efficiency', 'win_delta'}
        season : int, optional
        n : int
        bottom : bool  — if True, return worst N teams
        """
        valid_units = {
            "pass_offense", "run_offense", "pass_defense", "run_defense",
            "team_efficiency", "win_delta"
        }
        if unit not in valid_units:
            raise ValueError(f"unit must be one of {valid_units}")
 
        z_col = f"{unit}_z" if unit != "win_delta" else "win_delta"
        df = results_df.copy()
        if season is not None:
            df = df[df["season"] == season]
        if z_col not in df.columns:
            raise KeyError(f"Column '{z_col}' not found in results DataFrame.")
 
        df = df.sort_values(z_col, ascending=bottom).head(n)
        return df[["season", "team", z_col, f"{unit}_rank" if unit != "win_delta" else "overperformance_rank"]].reset_index(drop=True)
 
    def unit_trends(
        self,
        results_df: pd.DataFrame,
        team: str,
        unit: str = "team_efficiency",
    ) -> pd.DataFrame:
        """
        Return season-by-season trend for one unit for a specific team.
 
        Parameters
        ----------
        team : str  — e.g. 'KC', 'SF'
        unit : str  — same choices as top_units
        """
        z_col = f"{unit}_z" if unit != "win_delta" else "win_delta"
        mask  = results_df["team"].str.upper() == team.upper()
        df    = results_df.loc[mask, ["season", "team", z_col]].copy()
        if f"{unit}_rank" in results_df.columns:
            df[f"{unit}_rank"] = results_df.loc[mask, f"{unit}_rank"].values
        if "wins" in results_df.columns:
            df["wins"] = results_df.loc[mask, "wins"].values
        if "expected_wins" in results_df.columns:
            df["expected_wins"] = results_df.loc[mask, "expected_wins"].values
        return df.sort_values("season").reset_index(drop=True)
 
    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────
 
    def save(self, path: str | Path) -> None:
        """Serialize fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("TeamDiagnosticModel saved to %s", path)
 
    @classmethod
    def load(cls, path: str | Path) -> "TeamDiagnosticModel":
        """Load a previously serialized model."""
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected TeamDiagnosticModel.")
        logger.info("TeamDiagnosticModel loaded from %s", path)
        return obj
 
    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────
 
    def _validate_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic type coercion and column existence checks."""
        if "season" not in df.columns or "team" not in df.columns:
            raise ValueError("team_stats_df must contain 'season' and 'team' columns.")
        df["season"] = df["season"].astype(int)
        df["team"]   = df["team"].str.upper().str.strip()
        numeric_cols = [c for c in df.columns if c not in ("season", "team")]
        for c in numeric_cols:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
 
    def _append_expected_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expected_wins and win_delta columns."""
        if not self._is_fitted:
            logger.debug("Model not fitted — skipping expected-wins calculation.")
            df["expected_wins"] = np.nan
            df["win_delta"]     = np.nan
            return df
 
        feat_map = {
            "ew_off_pass_epa": "offense_total_epa_pass",
            "ew_off_run_epa":  "offense_total_epa_run",
            "ew_def_pass_epa": "defense_total_epa_pass",
            "ew_def_run_epa":  "defense_total_epa_run",
        }
        available = {}
        for feat_name, src_col in feat_map.items():
            if feat_name in self._ew_features and src_col in df.columns:
                if "def_" in feat_name:
                    available[feat_name] = -df[src_col]
                else:
                    available[feat_name] = df[src_col]
 
        if len(available) != len(self._ew_features):
            logger.warning(
                "Expected-wins features mismatch — needed %s, got %s.",
                self._ew_features, list(available.keys()),
            )
            df["expected_wins"] = np.nan
            df["win_delta"]     = np.nan
            return df
 
        feat_df  = pd.DataFrame(available, index=df.index)[self._ew_features]
        mask     = feat_df.notna().all(axis=1)
        X_scaled = self._ew_scaler.transform(feat_df.loc[mask].values)
        preds    = self._expected_wins_model.predict(X_scaled)
 
        df["expected_wins"] = np.nan
        df.loc[mask, "expected_wins"] = np.clip(np.round(preds, 2), 0, 17)
 
        if "wins" in df.columns:
            df["win_delta"] = (df["wins"] - df["expected_wins"]).round(2)
        else:
            df["win_delta"] = np.nan
 
        return df
 
    def _append_cap_roi(
        self,
        df: pd.DataFrame,
        contracts_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Add cap efficiency columns when contracts_df is provided.
 
        Columns added:
            offense_cap_hit      ($)
            defense_cap_hit      ($)
            total_cap_hit        ($)
            offense_cap_epa_roi  — total_offense_EPA / (offense_cap_hit / 1_000_000)
            defense_cap_epa_roi  — |total_defense_EPA_suppressed| / (defense_cap_hit / 1_000_000)
            cap_efficiency_score — composite z-score of both ROI metrics (within season)
        """
        cap_cols = [
            "offense_cap_hit", "defense_cap_hit", "total_cap_hit",
            "offense_cap_epa_roi", "defense_cap_epa_roi", "cap_efficiency_score",
        ]
 
        if contracts_df is None:
            for c in cap_cols:
                df[c] = np.nan
            return df
 
        try:
            cap = _aggregate_cap_by_team_season_unit(contracts_df)
            df  = df.merge(cap, on=["season", "team"], how="left")
        except (ValueError, KeyError) as exc:
            logger.warning("Cap ROI skipped — %s", exc)
            for c in cap_cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df
 
        # Total offensive EPA (pass + run)
        off_epa_total = (
            df.get("offense_total_epa_pass", pd.Series(0, index=df.index)).fillna(0)
            + df.get("offense_total_epa_run",  pd.Series(0, index=df.index)).fillna(0)
        )
        # Total defensive EPA suppressed (negated so higher = better)
        def_epa_suppressed = -(
            df.get("defense_total_epa_pass", pd.Series(0, index=df.index)).fillna(0)
            + df.get("defense_total_epa_run",  pd.Series(0, index=df.index)).fillna(0)
        )
 
        cap_m_off = df["offense_cap_hit"].replace(0, np.nan) / 1_000_000
        cap_m_def = df["defense_cap_hit"].replace(0, np.nan) / 1_000_000
 
        df["offense_cap_epa_roi"] = (off_epa_total / cap_m_off).round(3)
        df["defense_cap_epa_roi"] = (def_epa_suppressed / cap_m_def).round(3)
 
        # Z-score both ROI metrics within season for a normalised composite
        for roi_col in ["offense_cap_epa_roi", "defense_cap_epa_roi"]:
            if df[roi_col].notna().sum() > _MIN_TEAMS_FOR_ZSCORE:
                df[f"_{roi_col}_z"] = _zscore_within_season(df, roi_col)
            else:
                df[f"_{roi_col}_z"] = np.nan
 
        df["cap_efficiency_score"] = _safe_composite(
            df.get("_offense_cap_epa_roi_z", pd.Series(np.nan, index=df.index)),
            df.get("_defense_cap_epa_roi_z", pd.Series(np.nan, index=df.index)),
        ).round(3)
 
        # Clean up internal z cols
        for c in ["_offense_cap_epa_roi_z", "_defense_cap_epa_roi_z"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
 
        return df
 
    def _append_pbp_splits(
        self,
        df: pd.DataFrame,
        pbp_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reserved hook for play-by-play situational EPA splits.
 
        When PBP data (lake/staged/games/play_by_play/) becomes available,
        this method will add:
            redzone_offense_epa_z   — EPA per play inside the 20
            third_down_offense_sr_z — success rate on 3rd down
            fourth_down_go_rate     — aggressiveness metric
            early_down_pass_rate    — tendency vs. optimal
 
        For now, logs a notice and returns df unchanged.
        """
        logger.info(
            "PBP splits requested but not yet implemented — "
            "run ingestion/pipeline.py with PBP enabled first."
        )
        return df
    
    
