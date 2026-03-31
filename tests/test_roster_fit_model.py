import numpy as np
import pandas as pd
import pytest
from serving.models.roster_fit.model import RosterFitModel

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def player_profiles():
    # season=2021 so _build_interaction_matrix can find join_season-1 rows
    # when outcomes_df has join_season=2022. season=2022 rows are for scoring.
    return pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Alice",
                "position": "WR",
                "position_group": "WR",
                "season": 2021,
                "speed_score": 1.2,
                "agility_score": 0.8,
                "burst_score": 0.5,
                "nfl_production_score": 0.9,
                "target_share": 0.25,
                "snap_share": 0.7,
                "epa_per_game": 0.3,
            },
            {
                "player_id": "p2",
                "player_name": "Bob",
                "position": "WR",
                "position_group": "WR",
                "season": 2021,
                "speed_score": -0.5,
                "agility_score": 1.1,
                "burst_score": 0.2,
                "nfl_production_score": 0.4,
                "target_share": 0.15,
                "snap_share": 0.5,
                "epa_per_game": 0.1,
            },
            {
                "player_id": "p1",
                "player_name": "Alice",
                "position": "WR",
                "position_group": "WR",
                "season": 2022,
                "speed_score": 1.2,
                "agility_score": 0.8,
                "burst_score": 0.5,
                "nfl_production_score": 0.9,
                "target_share": 0.25,
                "snap_share": 0.7,
                "epa_per_game": 0.3,
            },
            {
                "player_id": "p2",
                "player_name": "Bob",
                "position": "WR",
                "position_group": "WR",
                "season": 2022,
                "speed_score": -0.5,
                "agility_score": 1.1,
                "burst_score": 0.2,
                "nfl_production_score": 0.4,
                "target_share": 0.15,
                "snap_share": 0.5,
                "epa_per_game": 0.1,
            },
        ]
    )


@pytest.fixture
def team_profiles():
    return pd.DataFrame(
        [
            {
                "team": "SF",
                "season": 2022,
                "air_yards_scheme": 1.0,
                "yac_scheme": 0.5,
                "pass_rate": 0.6,
                "pass_epa_efficiency": 0.8,
                "run_epa_efficiency": 0.3,
                "def_pass_scheme": 0.4,
                "def_run_scheme": 0.5,
                "run_success_rate": 0.5,
                "pass_success_rate": 0.6,
            },
            {
                "team": "KC",
                "season": 2022,
                "air_yards_scheme": 0.3,
                "yac_scheme": 1.2,
                "pass_rate": 0.7,
                "pass_epa_efficiency": 1.1,
                "run_epa_efficiency": 0.2,
                "def_pass_scheme": 0.6,
                "def_run_scheme": 0.3,
                "run_success_rate": 0.4,
                "pass_success_rate": 0.7,
            },
        ]
    )


@pytest.fixture
def outcomes_df():
    return pd.DataFrame(
        [
            {"player_id": "p1", "team": "SF", "join_season": 2022, "production_z": 0.9},
            {
                "player_id": "p2",
                "team": "KC",
                "join_season": 2022,
                "production_z": -0.2,
            },
        ]
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_instantiation():
    model = RosterFitModel()
    assert model.MODEL_NAME == "roster_fit"
    assert not model._is_fitted


def test_cosine_only_returns_dataframe(player_profiles, team_profiles):
    model = RosterFitModel()
    results = model.score_cosine_only(player_profiles, team_profiles, season=2022)
    assert not results.empty
    assert "fit_score" in results.columns
    assert "cosine_similarity" in results.columns
    # fit_score should be 0–100
    assert results["fit_score"].between(0, 100).all()
    # Should have 2 players × 2 teams = 4 rows
    assert len(results) == 4


def test_fit_marks_fitted(player_profiles, team_profiles, outcomes_df):
    model = RosterFitModel()
    model.fit(player_profiles, team_profiles, outcomes_df)
    assert model._is_fitted


def test_score_ridge_after_fit(player_profiles, team_profiles, outcomes_df):
    model = RosterFitModel()
    model.fit(player_profiles, team_profiles, outcomes_df)
    results = model.score(player_profiles, team_profiles, season=2022)
    assert not results.empty
    assert results["fit_score"].between(0, 100).all()


def test_score_falls_back_to_cosine_if_unfitted(player_profiles, team_profiles):
    model = RosterFitModel()
    # score() without fit() should fall back gracefully, not crash
    results = model.score(player_profiles, team_profiles, season=2022)
    assert not results.empty


def test_predict_interface(player_profiles, team_profiles):
    """Verify the BaseModel predict(inputs: dict) interface works."""
    model = RosterFitModel()
    result = model.predict(
        {
            "player_profiles": player_profiles,
            "team_profiles": team_profiles,
            "season": 2022,
        }
    )
    assert "prediction" in result
    assert isinstance(result["prediction"], list)


def test_save_load_round_trip(player_profiles, team_profiles, outcomes_df, tmp_path):
    model = RosterFitModel()
    model.fit(player_profiles, team_profiles, outcomes_df)
    model.save(tmp_path / "artifacts/roster_fit/v1")

    model2 = RosterFitModel()
    model2.load(tmp_path / "artifacts/roster_fit/v1")
    assert model2._is_fitted

    # Scores should be identical before and after round-trip
    r1 = model.score(player_profiles, team_profiles, season=2022)
    r2 = model2.score(player_profiles, team_profiles, season=2022)
    pd.testing.assert_frame_equal(r1, r2)


def test_empty_season_returns_empty(player_profiles, team_profiles):
    model = RosterFitModel()
    results = model.score_cosine_only(player_profiles, team_profiles, season=1999)
    # No team data for 1999 — should return empty or fall back gracefully
    assert isinstance(results, pd.DataFrame)
