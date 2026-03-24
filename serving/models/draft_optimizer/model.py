"""
serving/models/draft_optimizer/model.py
========================================
CVXPY constrained binary integer program for NFL draft pick optimization.

HOW IT FITS INTO THE PLATFORM
  This model sits downstream of the Player Projection model.  It does not
  learn from data the way XGBoost does — instead it solves a combinatorial
  puzzle: given a set of available prospects (each with a projected career
  value score from the Player Projection model) and a team's available pick
  slots, find the assignment of prospects to picks that maximises total
  projected value subject to positional and roster constraints.

  The "learning" step here is calibration — running over historical draft
  data (2000–2020) once to freeze three lookup tables into the pkl:

    expected_av_curve       {pick_number → mean historical w_av}
                            Used at inference to compute value_over_adp:
                            how much better/worse is this prospect than
                            what teams historically got at this slot?

    positional_quota_defaults  {position_group → max picks per draft}
                            Derived from how many players teams actually
                            draft at each group.  Prevents the optimizer
                            from stacking 4 QBs.

    pick_value_table        {pick_number → 0–100 normalised slot cost}
                            From the nflreadpy trade chart.  Encodes that
                            pick 1 is worth ~10× pick 32 in trade value.

OPTIMIZATION FORMULATION
  Variables:
    X[i, j]  binary — assign prospect i to pick slot j (1) or not (0)

  Objective (maximise):
    Σ_{i,j} X[i,j] × career_value_score[i]
                   × need_weight[position_group[i]]
                   × pick_value[j]

    career_value_score  — Player Projection output, 0–100
    need_weight         — 0–1, how much the team needs this position group
                          (1.0 = desperate need, 0.1 = fully stacked)
    pick_value          — 0–100, slot importance from trade chart
                          (early picks amplify value more)

  Hard constraints:
    • One prospect per slot:    Σ_i X[i,j] ≤ 1  ∀ j
    • One slot per prospect:    Σ_j X[i,j] ≤ 1  ∀ i
    • Positional quotas:        Σ_{i∈group g, j} X[i,j] ≤ quota[g]
    • Roster eligibility:       prospects at positions with need < threshold
                                are excluded from X (set column to 0)

  Solver: GLPK_MI via CVXPY (open-source, no license required).
          Typical solve time < 1s for a 250-prospect × 10-pick problem.

PKL CONTENTS  (artifacts/draft_optimizer/v1/model.pkl)
  expected_av_curve        dict[int, float]   pick → mean w_av (2000–2020)
  positional_quota_defaults dict[str, int]    group → max picks
  need_score_thresholds    dict[str, float]   group → no-need cutoff
  pick_value_table         dict[int, float]   pick → 0–100 slot value
  calibration_year_range   tuple[int, int]    (year_start, year_end)

OUTPUT SCHEMA  (predict() return value)
  board           list of dicts, one per pick slot, sorted by pick:
    pick            int
    player_id       str
    player_name     str
    position        str
    position_group  str
    career_value_score   float  0–100
    draft_value_percentile float
    value_over_adp  float   career_value_score − expected_at_slot (0–100 scale)
    need_weight     float   applied weight for this position group
    pick_value      float   slot importance weight
    composite_score float   the full objective term for this assignment
  alternatives    dict[pick → list[dict]]  top-3 next-best prospects per slot
  need_weights    dict[str, float]  the need vector used (for NullClaw to explain)
  solver_status   str  "optimal" | "infeasible" | "unbounded" | "error"
  meta            dict  picks used, prospects considered, model version
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — CWD-first since python -m is run from the project root.
# Falls back to walking upward from __file__ looking for known root markers.
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "config.py").exists() or (cwd / "duckdb_client.py").exists():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.py").exists() or (parent / "duckdb_client.py").exists():
            return parent
    return cwd  # last resort — let the import error surface naturally

_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact path
# ---------------------------------------------------------------------------
_ARTIFACT_DIR = _ROOT / "artifacts" / "draft_optimizer" / "v1"
_PKL_PATH = _ARTIFACT_DIR / "model.pkl"
_META_PATH = _ARTIFACT_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Default positional quotas
# Used as a hard fallback if calibration data is thin for a group.
# ---------------------------------------------------------------------------
_QUOTA_FALLBACK: dict[str, int] = {
    "QB":    2,
    "SKILL": 8,
    "OL":    6,
    "DL":    6,
    "LB":    4,
    "DB":    6,
    "SPEC":  2,
    "UNK":   2,
}

# Minimum need weight — positions below this threshold are excluded from X.
# Calibrated from historical distributions; overridden by pkl at inference.
_NEED_THRESHOLD_FALLBACK = 0.10


# ===========================================================================
# DraftOptimizerModel
# ===========================================================================

class DraftOptimizerModel:
    """
    Constrained draft board optimizer.

    Lifecycle
    ---------
    1. calibrate(batch_df)  — run once offline, saves pkl + metadata.json
    2. load(path)           — ModelRegistry calls this on first predict()
    3. predict(inputs)      — async, called by NullClaw / FastAPI at inference
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        # Calibration state — None until calibrate() or load() is called
        self.expected_av_curve:         dict[int, float]  = {}
        self.positional_quota_defaults: dict[str, int]    = _QUOTA_FALLBACK.copy()
        self.need_score_thresholds:     dict[str, float]  = {
            g: _NEED_THRESHOLD_FALLBACK for g in _QUOTA_FALLBACK
        }
        self.pick_value_table:          dict[int, float]  = {}
        self.calibration_year_range:    tuple[int, int]   = (0, 0)
        self._calibrated: bool = False

    # -----------------------------------------------------------------------
    # Calibration  (offline, runs over historical draft data)
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        batch_df: pd.DataFrame,
        *,
        pick_value_df: Optional[pd.DataFrame] = None,
        calibration_year_start: int = 2000,
        calibration_year_end:   int = 2020,
    ) -> "DraftOptimizerModel":
        """
        Build and freeze calibration state from historical draft data.

        Parameters
        ----------
        batch_df:
            Output of features.fetch_prospect_scores_batch() — one row per
            drafted player across calibration_year_start..calibration_year_end.
            Must contain: pick, w_av, position_group, season.
        pick_value_df:
            Optional DataFrame with columns [pick, pick_value] from the
            nflreadpy trade chart.  If None, a synthetic log-decay curve is
            used (good enough for relative weighting).
        """
        logger.info(
            "Calibrating draft optimizer on %s historical picks (%s–%s).",
            f"{len(batch_df):,}", calibration_year_start, calibration_year_end,
        )

        df = batch_df.copy()
        df["pick"]  = pd.to_numeric(df["pick"],  errors="coerce")
        df["w_av"]  = pd.to_numeric(df["w_av"],  errors="coerce")
        df["season"] = pd.to_numeric(df["season"], errors="coerce")

        # --- 1. expected_av_curve: mean w_av by pick slot -------------------
        # Only score picks where w_av > 0 (players who never played get 0
        # and would drag the mean down unfairly for early slots).
        scored = df[df["w_av"].notna() & (df["w_av"] > 0)].copy()
        curve = (
            scored.groupby("pick")["w_av"]
            .mean()
            .round(3)
            .to_dict()
        )
        # Fill any gaps in the pick range with linear interpolation
        if curve:
            max_pick = max(curve.keys())
            s = pd.Series(curve).reindex(range(1, max_pick + 1))
            s = s.interpolate(method="linear").fillna(method="bfill").fillna(0)
            self.expected_av_curve = {int(k): round(float(v), 3) for k, v in s.items()}
        logger.info("  expected_av_curve: %s pick slots.", len(self.expected_av_curve))

        # --- 2. positional_quota_defaults: p95 of group picks per draft -----
        # "What's the most any team typically drafts of one position group?"
        # Use the 95th percentile across all team-seasons to be permissive
        # (we want hard caps, not typical counts).
        if "position_group" in df.columns and "season" in df.columns:
            group_counts = (
                df.groupby(["season", "position_group"])
                .size()
                .reset_index(name="n")
            )
            quotas = (
                group_counts.groupby("position_group")["n"]
                .quantile(0.95)
                .apply(lambda x: max(int(np.ceil(x)), 1))
                .to_dict()
            )
            # Merge with fallback so any missing group is covered
            self.positional_quota_defaults = {
                **_QUOTA_FALLBACK,
                **quotas,
            }
        logger.info(
            "  positional_quota_defaults: %s", self.positional_quota_defaults
        )

        # --- 3. need_score_thresholds: bottom 10th percentile of group need --
        # The need threshold per group is calibrated as the 10th percentile
        # of the draft_value_percentile distribution for that group.
        # Prospects below this threshold are considered "no roster need" by
        # default.  Inference-time need scores override this per team.
        if "position_group" in df.columns and "draft_value_percentile" in df.columns:
            thresholds = (
                df.groupby("position_group")["draft_value_percentile"]
                .quantile(0.10)
                .round(3)
                .to_dict()
            )
            self.need_score_thresholds = {
                g: thresholds.get(g, _NEED_THRESHOLD_FALLBACK * 100) / 100.0
                for g in _QUOTA_FALLBACK
            }
        logger.info(
            "  need_score_thresholds: %s", self.need_score_thresholds
        )

        # --- 4. pick_value_table: 0–100 normalised slot value ---------------
        if pick_value_df is not None and not pick_value_df.empty:
            pv = pick_value_df.copy()
            pv["pick"] = pd.to_numeric(pv["pick"], errors="coerce")
            pv["pick_value"] = pd.to_numeric(pv["pick_value"], errors="coerce")
            pv = pv.dropna(subset=["pick", "pick_value"])
            raw = dict(zip(pv["pick"].astype(int), pv["pick_value"]))
        else:
            # Synthetic log-decay: pick 1 = 100, pick 32 ≈ 58, pick 256 ≈ 10
            raw = {
                p: max(100.0 * (1.0 / np.log2(p + 1)), 1.0)
                for p in range(1, 263)
            }

        # Normalise to 0–100
        vals = list(raw.values())
        lo, hi = min(vals), max(vals)
        self.pick_value_table = {
            int(k): round((v - lo) / (hi - lo) * 100.0, 3)
            for k, v in raw.items()
        }
        logger.info(
            "  pick_value_table: %s slots (pick 1 = %.1f, pick 32 = %.1f).",
            len(self.pick_value_table),
            self.pick_value_table.get(1, 0),
            self.pick_value_table.get(32, 0),
        )

        self.calibration_year_range = (calibration_year_start, calibration_year_end)
        self._calibrated = True
        logger.info("Calibration complete.")
        return self

    # -----------------------------------------------------------------------
    # Inference  (async — called at serve time via ModelRegistry / NullClaw)
    # -----------------------------------------------------------------------

    async def predict(self, inputs: dict) -> dict:
        """
        Run the draft optimizer for a team's available picks.

        Parameters (keys in inputs dict)
        ----------------------------------
        client          DataLakeClient  required — for fetching prospect scores
        team_abbr       str             e.g. "KC"
        draft_year      int             e.g. 2024
        available_picks list[int]       overall pick numbers, e.g. [23, 54, 87]
        need_weights    dict[str,float] optional — override per position group
                                        e.g. {"QB": 0.9, "SKILL": 0.4}
                                        any missing groups use default 0.5
        model           optional        Player Projection model instance;
                                        None → auto-load from ModelRegistry

        Returns
        -------
        dict  matching the OUTPUT SCHEMA in the module docstring.
        """
        if not self._calibrated:
            logger.warning(
                "DraftOptimizerModel.predict() called before calibration. "
                "Load a calibrated artifact with DraftOptimizerModel.load()."
            )

        client        = inputs["client"]
        team_abbr     = inputs.get("team_abbr", "UNK")
        draft_year    = int(inputs["draft_year"])
        available_picks: list[int] = [int(p) for p in inputs["available_picks"]]
        need_overrides: dict[str, float] = inputs.get("need_weights", {})
        proj_model    = inputs.get("model", None)

        if not available_picks:
            return self._error_result("available_picks is empty.", team_abbr, draft_year)

        # -- 1. Build need weight vector ------------------------------------
        need_weights = self._build_need_weights(need_overrides)

        # -- 2. Fetch prospect scores (async, DataLakeClient) ---------------
        from serving.models.draft_optimizer.features import fetch_prospect_scores  # noqa: PLC0415

        min_pick = min(available_picks)
        max_pick = max(available_picks)

        prospects = await fetch_prospect_scores(
            client,
            draft_year=draft_year,
            model=proj_model,
            min_pick=min_pick,
            max_pick=max_pick,
        )

        if prospects.empty:
            return self._error_result(
                f"No prospects found for draft year {draft_year}.",
                team_abbr, draft_year,
            )

        # -- 3. Filter to eligible prospects (need threshold) ---------------
        prospects = self._filter_by_need(prospects, need_weights)
        if prospects.empty:
            return self._error_result(
                "All prospects filtered out by need threshold. "
                "Try lowering need_weights or adding more positions.",
                team_abbr, draft_year,
            )

        # -- 4. Attach pick_value to prospects ------------------------------
        prospects = prospects.copy()
        prospects["_pick_value"] = prospects["pick"].map(
            lambda p: self.pick_value_table.get(int(p) if pd.notna(p) else 999, 1.0)
        )
        prospects["_need_weight"] = prospects["position_group"].map(
            lambda g: need_weights.get(g, 0.5)
        )

        # -- 5. Build pick slot value vector --------------------------------
        pick_values = [
            self.pick_value_table.get(p, 1.0) for p in available_picks
        ]

        # -- 6. Solve CVXPY binary integer program --------------------------
        board, solver_status = self._solve(prospects, available_picks, pick_values, need_weights)

        # -- 7. Attach value_over_adp to each assignment --------------------
        for rec in board:
            slot = rec["pick"]
            expected = self.expected_av_curve.get(slot, 0.0)
            # Convert expected AV to 0–100 scale to match career_value_score
            # using the same normalisation range (0–max in curve)
            max_av = max(self.expected_av_curve.values()) if self.expected_av_curve else 1.0
            expected_scaled = (expected / max_av) * 100.0 if max_av > 0 else 0.0
            rec["value_over_adp"] = round(rec["career_value_score"] - expected_scaled, 2)

        # -- 8. Top-3 alternatives per slot ---------------------------------
        alternatives = self._build_alternatives(
            prospects, board, available_picks, need_weights
        )

        return {
            "board":        board,
            "alternatives": alternatives,
            "need_weights": need_weights,
            "solver_status": solver_status,
            "meta": {
                "team":              team_abbr,
                "draft_year":        draft_year,
                "picks_used":        len(board),
                "prospects_considered": len(prospects),
                "model_version":     "v1",
                "calibration_years": list(self.calibration_year_range),
            },
        }

    # -----------------------------------------------------------------------
    # CVXPY solver
    # -----------------------------------------------------------------------

    def _solve(
        self,
        prospects:     pd.DataFrame,
        available_picks: list[int],
        pick_values:   list[float],
        need_weights:  dict[str, float],
    ) -> tuple[list[dict], str]:
        """
        Solve the binary integer program.

        Returns (board, solver_status).  board is a list of assignment dicts,
        one per pick slot, sorted by pick.  If the solver fails, returns an
        empty board and a status string describing the failure.
        """
        try:
            import cvxpy as cp  # noqa: PLC0415
        except ImportError:
            logger.error(
                "cvxpy is not installed. Run: pip install cvxpy. "
                "Cannot solve draft optimizer without it."
            )
            return [], "error:cvxpy_not_installed"

        n_prospects = len(prospects)
        n_picks     = len(available_picks)

        if n_prospects == 0 or n_picks == 0:
            return [], "infeasible:empty_inputs"

        # Build objective coefficient matrix C[i, j]
        # C[i,j] = career_value_score[i] × need_weight[group[i]] × pick_value[j]
        cvs   = prospects["career_value_score"].fillna(0).to_numpy()
        nw    = prospects["position_group"].map(lambda g: need_weights.get(g, 0.5)).to_numpy()
        pv    = np.array(pick_values, dtype=float)
        C     = np.outer(cvs * nw, pv)  # shape (n_prospects, n_picks)

        # Decision variable
        X = cp.Variable((n_prospects, n_picks), boolean=True)

        # Objective
        objective = cp.Maximize(cp.sum(cp.multiply(C, X)))

        constraints = []

        # One prospect per slot
        constraints += [cp.sum(X, axis=0) <= 1]

        # One slot per prospect
        constraints += [cp.sum(X, axis=1) <= 1]

        # Positional quotas
        pos_groups = prospects["position_group"].to_numpy()
        for group, quota in self.positional_quota_defaults.items():
            idx = np.where(pos_groups == group)[0]
            if len(idx) > 0:
                constraints += [cp.sum(X[idx, :]) <= quota]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.GLPK_MI, verbose=False)
        except Exception as exc:
            logger.error("CVXPY solver raised: %s", exc)
            return [], f"error:{exc}"

        status = prob.status
        if status not in ("optimal", "optimal_inaccurate"):
            logger.warning("CVXPY solver status: %s", status)
            return [], status

        # Extract assignments
        X_val = X.value
        if X_val is None:
            return [], "error:no_solution_value"

        board: list[dict] = []
        prospect_rows = prospects.reset_index(drop=True)

        for j, pick_num in enumerate(available_picks):
            col = X_val[:, j]
            best_i = int(np.argmax(col))
            if col[best_i] < 0.5:
                # Solver left this slot empty (infeasible or no gain)
                continue
            row = prospect_rows.iloc[best_i]
            board.append({
                "pick":                int(pick_num),
                "player_id":           str(row.get("player_id", "")),
                "player_name":         str(row.get("player_name", "")),
                "position":            str(row.get("position", "")),
                "position_group":      str(row.get("position_group", "")),
                "career_value_score":  round(float(row.get("career_value_score", 0)), 2),
                "draft_value_percentile": round(float(row.get("draft_value_percentile") or 0), 2),
                "value_over_adp":      0.0,   # filled in after solve
                "need_weight":         round(float(need_weights.get(str(row.get("position_group", "")), 0.5)), 3),
                "pick_value":          round(float(pv[j]), 3),
                "composite_score":     round(float(C[best_i, j]), 4),
                "projection_source":   str(row.get("projection_source", "unknown")),
            })

        board.sort(key=lambda r: r["pick"])
        return board, "optimal"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_need_weights(self, overrides: dict[str, float]) -> dict[str, float]:
        """
        Merge caller-supplied need overrides with a uniform default (0.5).
        All groups not in overrides get 0.5 — neutral, neither prioritised
        nor excluded.  Values are clamped to [0.0, 1.0].
        """
        from serving.models.shared.positions import POSITION_GROUP_ORDER  # noqa: PLC0415
        weights: dict[str, float] = {g: 0.5 for g in POSITION_GROUP_ORDER + ["UNK"]}
        for group, val in overrides.items():
            weights[group.upper()] = float(np.clip(val, 0.0, 1.0))
        return weights

    def _filter_by_need(
        self,
        prospects: pd.DataFrame,
        need_weights: dict[str, float],
    ) -> pd.DataFrame:
        """
        Drop prospects at positions where need_weight < threshold.

        This prevents the optimizer from wasting picks on positions the team
        is fully stocked at.  The threshold per group comes from the
        calibrated need_score_thresholds.  Callers can override need_weights
        to include any position explicitly (set weight ≥ threshold).
        """
        def _keep(row) -> bool:
            group = str(row.get("position_group", "UNK"))
            weight = need_weights.get(group, 0.5)
            threshold = self.need_score_thresholds.get(group, _NEED_THRESHOLD_FALLBACK)
            return weight >= threshold

        mask = prospects.apply(_keep, axis=1)
        dropped = (~mask).sum()
        if dropped:
            logger.debug(
                "Need threshold filtered out %s prospects.", f"{dropped:,}"
            )
        return prospects[mask].copy()

    def _build_alternatives(
        self,
        prospects:      pd.DataFrame,
        board:          list[dict],
        available_picks: list[int],
        need_weights:   dict[str, float],
    ) -> dict[int, list[dict]]:
        """
        For each pick slot, return the top-3 prospects NOT selected elsewhere.

        This is a greedy fallback list — useful when NullClaw needs to explain
        "if [recommended player] is gone, who else fits here?".
        """
        assigned_ids = {rec["player_id"] for rec in board}
        alts: dict[int, list[dict]] = {}

        for rec in board:
            pick_num = rec["pick"]
            pv_slot  = self.pick_value_table.get(pick_num, 1.0)

            candidates = prospects[
                ~prospects["player_id"].isin(assigned_ids - {rec["player_id"]})
                & (prospects["player_id"] != rec["player_id"])
            ].copy()

            if candidates.empty:
                alts[pick_num] = []
                continue

            candidates["_score"] = (
                candidates["career_value_score"].fillna(0)
                * candidates["position_group"].map(lambda g: need_weights.get(g, 0.5))
                * pv_slot
            )
            top3 = candidates.nlargest(3, "_score")

            alts[pick_num] = [
                {
                    "player_id":          str(r.get("player_id", "")),
                    "player_name":        str(r.get("player_name", "")),
                    "position":           str(r.get("position", "")),
                    "career_value_score": round(float(r.get("career_value_score", 0)), 2),
                    "need_weight":        round(float(need_weights.get(str(r.get("position_group", "")), 0.5)), 3),
                }
                for _, r in top3.iterrows()
            ]

        return alts

    @staticmethod
    def _error_result(message: str, team_abbr: str, draft_year: int) -> dict:
        logger.error("DraftOptimizer predict error: %s", message)
        return {
            "board":         [],
            "alternatives":  {},
            "need_weights":  {},
            "solver_status": f"error:{message}",
            "meta": {
                "team":       team_abbr,
                "draft_year": draft_year,
                "error":      message,
            },
        }

    # -----------------------------------------------------------------------
    # Persistence  (matches Team Diagnostic's pattern, fits ModelRegistry)
    # -----------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """
        Serialize calibration state to pkl + write metadata.json.

        The pkl contains only the four calibration lookup tables and the year
        range — not the CVXPY problem structure, which is rebuilt each call.
        """
        pkl_path  = Path(path) if path else _PKL_PATH
        meta_path = pkl_path.parent / "metadata.json"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "expected_av_curve":          self.expected_av_curve,
            "positional_quota_defaults":  self.positional_quota_defaults,
            "need_score_thresholds":      self.need_score_thresholds,
            "pick_value_table":           self.pick_value_table,
            "calibration_year_range":     self.calibration_year_range,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("DraftOptimizerModel saved → %s", pkl_path)

        metadata = {
            "model":                  "draft_optimizer",
            "version":                "v1",
            "saved_at":               datetime.now(timezone.utc).isoformat(),
            "calibration_year_range": list(self.calibration_year_range),
            "n_pick_slots":           len(self.pick_value_table),
            "n_av_curve_slots":       len(self.expected_av_curve),
            "positional_quotas":      self.positional_quota_defaults,
            "solver":                 "GLPK_MI (CVXPY)",
            "description": (
                "Calibration state for the CVXPY draft board optimizer. "
                "Contains expected_av_curve, positional_quota_defaults, "
                "need_score_thresholds, and pick_value_table. "
                "No learned weights — deterministic optimizer."
            ),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("metadata.json written → %s", meta_path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DraftOptimizerModel":
        """
        Load a previously calibrated model from pkl.
        Called by ModelRegistry on first predict().
        """
        pkl_path = Path(path) if path else _PKL_PATH
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"DraftOptimizer artifact not found at {pkl_path}. "
                "Run calibration first: python -m serving.models.draft_optimizer.calibrate"
            )
        with open(pkl_path, "rb") as f:
            state: dict = pickle.load(f)

        obj = cls()
        obj.expected_av_curve          = state.get("expected_av_curve",         {})
        obj.positional_quota_defaults  = state.get("positional_quota_defaults", _QUOTA_FALLBACK.copy())
        obj.need_score_thresholds      = state.get("need_score_thresholds",      {})
        obj.pick_value_table           = state.get("pick_value_table",           {})
        obj.calibration_year_range     = tuple(state.get("calibration_year_range", (0, 0)))
        obj._calibrated = True

        logger.info(
            "DraftOptimizerModel loaded from %s "
            "(calibrated %s–%s, %s pick slots).",
            pkl_path,
            obj.calibration_year_range[0],
            obj.calibration_year_range[1],
            len(obj.pick_value_table),
        )
        return obj