"""
serving/models/shared/positions.py
====================================================================================================
PURPOSE
  Single source of truth for NFL position → position group mapping across the entire model platform.

  Every model in this project that works with player positions (Player Projection, Draft Optimizer,
  Team Diagnosis, Roster Fit) needs to bucket the 40+ raw NFL position strings (QB, OLB, EDGE,
  MLB, FS...) into a smaller set of 7 broad groups so they can be compared meaningfully.

  Before this file existed, each model defined its own local copy of that mapping.  Those copies
  drifted out of sync, producing silent bugs — a player listed as OLB would land in group "LB" in
  the Player Projection model but group "DL" in the Draft Optimizer, so the two models were never
  actually talking about the same position groups.  MLB was missing from Player Projection entirely,
  causing every middle linebacker in the training data to be silently misclassified as a specialist.

  This file fixes that permanently.  There is now exactly one place to update the mapping, and all
  models import from here.

====================================================================================================
HOW TO USE IT

  from serving.models.shared.positions import POS_TO_GROUP, POSITION_GROUP_ORDER, pos_group

  # Map a single position string
  pos_group("OLB")          # → "DL"
  pos_group("MLB")          # → "LB"
  pos_group("EDGE")         # → "DL"
  pos_group(None)           # → "UNK"  (never raises)

  # Apply to a DataFrame column
  df["position_group"] = df["position"].map(pos_group)

  # Build one-hot columns for an ML model (e.g. Player Projection / XGBoost)
  for grp in POSITION_GROUP_ORDER:
      df[f"pos_{grp}"] = (df["position_group"] == grp).astype(int)

  # Look up the raw dict directly if needed
  POS_TO_GROUP["QB"]        # → "QB"

====================================================================================================
THE 7 GROUPS

  QB      Quarterbacks only.

  SKILL   Offensive skill positions: RB, HB, FB, WR, TE.
          Grouped together because they share combine drill relevance (speed, burst, agility)
          and are all evaluated on receiving / separation metrics at the next model layer.

  OL      Offensive line: OT, OG, C, G, T.
          Strength score and size score dominate; speed is less predictive here.

  DL      Defensive line + all edge rushers: DE, DT, NT, OLB, EDGE.
          OLB is here — not in LB — because modern 3-4 OLBs are pass rushers, not coverage
          linebackers.  Their combine profile (burst, size, speed) matches DE, not ILB.
          EDGE is an explicit nflverse label introduced around 2019 and maps the same way.

  LB      Interior / classic linebackers: ILB, MLB, LB.
          These players play downhill, defend the run, and drop into coverage — a distinct
          athletic and schematic profile from edge rushers.

  DB      All defensive backs: CB, S, FS, SS, DB.
          DB is the catch-all used when safety vs. corner is ambiguous in the source data.

  SPEC    Specialists: K, P, LS.
          Small sample sizes; rarely drafted in early rounds.  Grouped together to prevent
          individual specialist positions from having too few observations to z-score reliably.

====================================================================================================
POSITION_GROUP_ORDER

  The ordered list ["QB", "SKILL", "OL", "DL", "LB", "DB", "SPEC"] defines the column order
  for one-hot encoding in ML models.  THIS ORDER MUST NOT CHANGE after a model artifact has been
  trained — changing it shifts which column index each group maps to and silently corrupts the
  feature matrix that the saved XGBoost booster expects.  If a new group ever needs to be added,
  append to the end and retrain all affected models.

====================================================================================================
BUGS FIXED BY CENTRALISING HERE  (do not re-introduce these in local copies)

  Position  | Old behaviour (local dicts)        | Correct (this file)
  ──────────┼────────────────────────────────────┼───────────────────────────────
  OLB       | → "LB" in Player Projection        | → "DL"  (edge rusher)
            | → "DL" in Draft Optimizer          |
  MLB       | missing → fell through to "SPEC"   | → "LB"  (interior linebacker)
            | in Player Projection               |
  EDGE      | missing → fell through to "SPEC"   | → "DL"  (nflverse label 2019+)
            | in Player Projection               |
====================================================================================================
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical mapping
# ---------------------------------------------------------------------------

POS_TO_GROUP: dict[str, str] = {
    # Quarterback
    "QB": "QB",
    # Skill
    "RB": "SKILL",
    "HB": "SKILL",
    "FB": "SKILL",
    "WR": "SKILL",
    "TE": "SKILL",
    # Offensive line
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "G": "OL",
    "T": "OL",
    "OL": "OL",
    # Defensive line + edge rushers
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "OLB": "DL",  # edge rusher — NOT grouped with ILB/MLB
    "EDGE": "DL",  # nflverse explicit edge label (2019+)
    # Linebackers (interior / classic)
    "ILB": "LB",
    "MLB": "LB",  # was missing from Player Projection — fixed here
    "LB": "LB",
    # Defensive backs
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    # Specialists
    "K": "SPEC",
    "P": "SPEC",
    "LS": "SPEC",
}

# Stable ordered list used for one-hot encoding in ML models.
# Order must never change between versions — it determines column positions
# in the feature matrix that XGBoost was trained on.
POSITION_GROUP_ORDER: list[str] = ["QB", "SKILL", "OL", "DL", "LB", "DB", "SPEC"]


def pos_group(position: object) -> str:
    """
    Map a raw position string to its canonical group.

    Returns "UNK" for None, non-string, or unrecognised values rather than
    raising — callers can filter on "UNK" rows if needed.

    Examples
    --------
    >>> pos_group("OLB")
    'DL'
    >>> pos_group("MLB")
    'LB'
    >>> pos_group("EDGE")
    'DL'
    >>> pos_group(None)
    'UNK'
    """
    if not isinstance(position, str):
        return "UNK"
    return POS_TO_GROUP.get(position.strip().upper(), "UNK")
