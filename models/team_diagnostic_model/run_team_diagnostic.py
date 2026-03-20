"""
models/run_team_diagnostic.py
─────────────────────────────────────────────────────────────────────────────
Runner script for the Team Diagnostic & ROI Model.

Run from the project root:
    python ml/run_team_diagnostic.py

Optional flags:
    --season 2022          Only print summary for a specific season
    --team KC              Print full trend for a specific team
    --save                 Save fitted model to ml/models/team_diagnostic.pkl
    --contracts            Include cap ROI (requires staged contracts.parquet)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make sure project root is on the path so config imports work ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from ml.Team_diagnostic import TeamDiagnosticModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_team_stats() -> pd.DataFrame:
    """
    Load team statistics. Prefers the staged Parquet (faster, typed) but
    falls back to the raw CSV so the model works before the pipeline runs.
    """
    if config.STAGED_TEAM_STATS.exists():
        logger.info("Loading staged team stats from %s", config.STAGED_TEAM_STATS)
        return pd.read_parquet(config.STAGED_TEAM_STATS)

    logger.warning(
        "Staged Parquet not found — falling back to raw CSV. "
        "Run ingestion/pipeline.py to generate staged files."
    )
    raw_csv = config.RAW_TEAM_STATS
    if not raw_csv.exists():
        # Last resort: check repo root (dev convenience)
        raw_csv = ROOT / "nflteamstatistics.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Cannot find team stats. Expected at:\n"
            f"  {config.STAGED_TEAM_STATS}\n"
            f"  {config.RAW_TEAM_STATS}"
        )
    return pd.read_csv(raw_csv)


def load_contracts() -> pd.DataFrame | None:
    """Load contracts if staged, otherwise return None gracefully."""
    if config.STAGED_CONTRACTS.exists():
        logger.info("Loading contracts from %s", config.STAGED_CONTRACTS)
        return pd.read_parquet(config.STAGED_CONTRACTS)
    logger.info("No staged contracts found — cap ROI columns will be NaN.")
    return None


def print_section(title: str) -> None:
    width = 70
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Team Diagnostic & ROI Model runner")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to summarize (e.g. 2022)")
    parser.add_argument("--team",   type=str, default=None,
                        help="Team abbreviation for trend view (e.g. KC)")
    parser.add_argument("--save",   action="store_true",
                        help="Save fitted model to ml/models/team_diagnostic.pkl")
    parser.add_argument("--contracts", action="store_true",
                        help="Include cap ROI (requires staged contracts.parquet)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    team_stats = load_team_stats()
    contracts  = load_contracts() if args.contracts else None

    logger.info(
        "Loaded %d team-seasons (%d–%d)",
        len(team_stats),
        int(team_stats["season"].min()),
        int(team_stats["season"].max()),
    )

    # ── Fit + score ───────────────────────────────────────────────────────────
    model = TeamDiagnosticModel()
    model.fit(team_stats)
    results = model.score_teams(team_stats, contracts_df=contracts)

    # ── Summary table ─────────────────────────────────────────────────────────
    season = args.season or int(team_stats["season"].max())
    print_section(f"Team Diagnostic Summary — {season}")
    summary = model.summary(results, season=season)
    with pd.option_context("display.max_columns", 20, "display.width", 120,
                           "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))

    # ── Top / bottom units ────────────────────────────────────────────────────
    print_section(f"Top 5 Pass Offenses — {season}")
    print(model.top_units(results, "pass_offense", season=season, n=5).to_string(index=False))

    print_section(f"Bottom 5 Pass Defenses — {season}")
    print(model.top_units(results, "pass_defense", season=season, n=5, bottom=True).to_string(index=False))

    print_section(f"Most Over-Performing Teams — {season}  (actual wins > expected)")
    print(model.top_units(results, "win_delta", season=season, n=5).to_string(index=False))

    print_section(f"Most Under-Performing Teams — {season}  (actual wins < expected)")
    print(model.top_units(results, "win_delta", season=season, n=5, bottom=True).to_string(index=False))

    # ── Team trend (optional) ─────────────────────────────────────────────────
    if args.team:
        team = args.team.upper()
        print_section(f"Team Trend — {team} (all seasons)")
        trend = model.unit_trends(results, team=team, unit="team_efficiency")
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(trend.to_string(index=False))

    # ── Cap ROI (if contracts loaded) ─────────────────────────────────────────
    if contracts is not None and "cap_efficiency_score" in results.columns:
        print_section(f"Top 5 Cap Efficiency — {season}")
        cap_cols = ["season", "team", "offense_cap_epa_roi",
                    "defense_cap_epa_roi", "cap_efficiency_score"]
        cap_summary = (
            results[results["season"] == season][cap_cols]
            .sort_values("cap_efficiency_score", ascending=False)
            .head(5)
        )
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(cap_summary.to_string(index=False))

    # ── Save model ────────────────────────────────────────────────────────────
    if args.save:
        save_path = config.ML_MODELS_DIR / "team_diagnostic.pkl"
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

    # ── Write results parquet ─────────────────────────────────────────────────
    output_path = config.LAKE_CURATED_DIR / "team_diagnostic_results.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)
    logger.info("Full results written to %s", output_path)


if __name__ == "__main__":
    main()
