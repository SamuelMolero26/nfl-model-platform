from typing import Optional
import pandas as pd
from .client import DataLakeClient

_NGS_STAT_TYPES = {"passing", "receiving", "rushing"}


def _safe_str(value: str) -> str:
    """Escape single quotes in a string value before SQL interpolation."""
    return value.replace("'", "''")


async def get_combine_data(
    client: DataLakeClient,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Combine measurables for all draft prospects.
    Columns: player_id, player_name, position, draft_year, height_in, weight,
             forty, vertical, broad_jump, bench, three_cone, shuttle,
             draft_team, draft_round, draft_pick.
    """
    where_clauses = []
    if year_start:
        where_clauses.append(f"draft_year >= {year_start}")
    if year_end:
        where_clauses.append(f"draft_year <= {year_end}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT *
        FROM combine
        {where}
        ORDER BY draft_year, draft_round, draft_pick
    """
    return await client.query(sql)


async def get_draft_picks(
    client: DataLakeClient,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Draft pick history with player associations and pick value.
    Columns: season, round, pick, team, player_name, position, pick_value.
    """

    where_clauses = []
    if year_start:
        where_clauses.append(f"season >= {year_start}")
    if year_end:
        where_clauses.append(f"season <= {year_end}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT *
        FROM draft_picks
        {where}
        ORDER BY season, round, pick
    """
    return await client.query(sql)


async def get_player_career_stats(
    client: DataLakeClient,
    player_name: str,
) -> pd.DataFrame:
    """
    Season-by-season career stats for a single player.
    Returns all available stat columns for that player.
    """
    sql = f"""
        SELECT *
        FROM weekly_stats
        WHERE player_display_name ILIKE '%{_safe_str(player_name)}%'
        ORDER BY season, week
    """
    return await client.query(sql)


async def get_injury_history(
    client: DataLakeClient,
    position: Optional[str] = None,
) -> pd.DataFrame:
    """
    Injury records, optionally filtered by position.
    Columns: player_id, player_name, position, season, week, report_status,
             practice_status, primary_injury.
    """
    where = f"WHERE position ILIKE '%{_safe_str(position)}%'" if position else ""
    sql = f"""
        SELECT *
        FROM injuries
        {where}
        ORDER BY season, week
    """
    return await client.query(sql)


async def get_snap_counts(
    client: DataLakeClient,
    position: Optional[str] = None,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Snap count data, optionally filtered by position and/or season.
    """
    where_clauses = []
    if position:
        where_clauses.append(f"position ILIKE '%{_safe_str(position)}%'")
    if season:
        where_clauses.append(f"season = {season}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT *
        FROM snap_counts
        {where}
        ORDER BY season, week
    """
    return await client.query(sql)


async def get_depth_charts(
    client: DataLakeClient,
    team: Optional[str] = None,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Depth chart data. Useful for Team Diagnosis and Roster Fit models.
    """
    where_clauses = []
    if team:
        where_clauses.append(f"club_code ILIKE '%{_safe_str(team)}%'")
    if season:
        where_clauses.append(f"season = {season}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT *
        FROM depth_charts
        {where}
        ORDER BY season, week, depth_position, depth_team_abbr
    """
    return await client.query(sql)


async def get_ngs_stats(
    client: DataLakeClient,
    stat_type: str = "passing",
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Next Gen Stats: passing (CPOE), receiving (target sep), rushing (RYOE).
    stat_type: 'passing' | 'receiving' | 'rushing'
    """
    if stat_type not in _NGS_STAT_TYPES:
        raise ValueError(
            f"stat_type must be one of {_NGS_STAT_TYPES}, got {stat_type!r}"
        )
    where = f"WHERE season = {int(season)}" if season else ""
    sql = f"""
        SELECT *
        FROM ngs_{stat_type}
        {where}
        ORDER BY season, week
    """
    return await client.query(sql)


async def get_college_stats(
    client: DataLakeClient,
    player_name: Optional[str] = None,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    College football stats from CFBD. Used for Player Projection model.
    """
    where_clauses = []
    if player_name:
        where_clauses.append(f"player_name ILIKE '%{_safe_str(player_name)}%'")
    if year:
        where_clauses.append(f"year = {year}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT *
        FROM college_stats
        {where}
        ORDER BY year
    """
    return await client.query(sql)


# ---------------------------------------------------------------------------
# Gold profile queries (lake/curated/ tables)
# These are pre-computed feature tables — use for training (batch)
# and as feature inputs for all models that need athletic/production context.
# ---------------------------------------------------------------------------


async def get_athletic_profiles(
    client: DataLakeClient,
    player_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gold athletic profiles — pre-computed combine-derived scores.

    Columns include:
        player_id, player_name, position, draft_year,
        speed_score      — weight-adjusted 40-yard dash (Barnwell formula)
        agility_score    — composite of 3-cone + shuttle
        burst_score      — composite of vertical + broad jump
        strength_score   — bench press relative to body weight
        size_score       — height/weight vs positional ideal

    Use for: Player Projection, Positional Flexibility, Roster Fit features.
    """
    where = f"WHERE player_id = '{_safe_str(player_id)}'" if player_id else ""
    sql = f"SELECT * FROM player_athletic_profiles {where}"
    return await client.query(sql)


async def get_production_profiles(
    client: DataLakeClient,
    player_id: Optional[str] = None,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Gold production profiles — season-level NFL production efficiency scores.

    Columns include:
        player_id, player_name, position, season,
        nfl_production_score  — output efficiency relative to snap share
        snap_share            — fraction of team offensive snaps
        target_share          — for receivers

    Use for: Team Diagnosis, Career Simulator, Roster Fit features.
    """
    where_clauses = []
    if player_id:
        where_clauses.append(f"player_id = '{_safe_str(player_id)}'")
    if season:
        where_clauses.append(f"season = {season}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT * FROM player_production_profiles {where} ORDER BY season"
    return await client.query(sql)


async def get_durability_profiles(
    client: DataLakeClient,
    player_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gold durability profiles — career-level injury and availability metrics.

    Columns include:
        player_id, player_name, position,
        durability_score      — composite of games played rate + injury frequency
        games_played_rate     — games played / games available
        injury_frequency      — injury reports per 16 games
        seasons_played

    Use for: Health Analyzer, Career Simulator, Player Projection features.
    """
    where = f"WHERE player_id = '{_safe_str(player_id)}'" if player_id else ""
    sql = f"SELECT * FROM player_durability_profiles {where}"
    return await client.query(sql)


async def get_draft_value_history(
    client: DataLakeClient,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Gold draft value history — pick-level AV performance vs expected.

    Columns include:
        player_id, player_name, season, round, pick,
        draft_value_score       — z-score of car_av within draft round
        draft_value_percentile  — 0–100 percentile within round

    Use for: Player Projection (as a training label supplement),
             Draft Optimizer (pick value curve), Team Diagnosis.
    """
    where_clauses = []
    if year_start:
        where_clauses.append(f"season >= {year_start}")
    if year_end:
        where_clauses.append(f"season <= {year_end}")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT * FROM draft_value_history {where} ORDER BY season, round, pick"
    return await client.query(sql)


async def get_player_full_profile(
    client: DataLakeClient,
    player_id: str,
) -> dict:
    """
    Single-player enriched profile joining all 4 gold tables.
    Uses GET /players/id/{player_id}/profile endpoint.

    Returns a dict with keys:
        player, athletic_profile, production_profile,
        durability_profile, draft_value

    Use for: inference path — when NullClaw asks about a specific player.
    """
    return await client._get_with_retry(f"/players/id/{player_id}/profile")
