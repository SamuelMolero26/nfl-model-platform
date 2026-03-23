"""Integration tests for DataLakeClient against the live data lake API (:8000).

Run with: pytest tests/test_client.py -v
Requires: nfl-data-platform running at http://localhost:8000
"""

import pytest
import pytest_asyncio
import pandas as pd

from serving.data_lake_client import DataLakeClient, DataLakeNotFoundError
from serving.data_lake_client import queries

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    async with DataLakeClient() as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(client):
    assert await client.health_check() is True


# ---------------------------------------------------------------------------
# SQL query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_dataframe(client):
    df = await client.query("SELECT 1 AS n")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


@pytest.mark.asyncio
async def test_list_tables(client):
    tables = await client.list_tables()
    assert isinstance(tables, list)
    assert len(tables) > 0


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_players_returns_dataframe(client):
    df = await client.list_players()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.asyncio
async def test_list_players_filter_by_position(client):
    df = await client.list_players(position="QB")
    assert isinstance(df, pd.DataFrame)


@pytest.mark.asyncio
async def test_get_player_not_found_raises(client):
    with pytest.raises(DataLakeNotFoundError):
        await client.get_player("XYZNONEXISTENTPLAYER9999")


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_team_stats(client):
    df = await client.get_team_stats("KC", year_start=2020, year_end=2023)
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_college_pipeline(client):
    df = await client.get_college_pipeline("Alabama")
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Named query helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_combine_data(client):
    df = await queries.get_combine_data(client, year_start=2015, year_end=2020)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.asyncio
async def test_get_draft_picks(client):
    df = await queries.get_draft_picks(client, year_start=2018, year_end=2022)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.asyncio
async def test_get_injury_history(client):
    df = await queries.get_injury_history(client, position="WR")
    assert isinstance(df, pd.DataFrame)


@pytest.mark.asyncio
async def test_get_snap_counts(client):
    df = await queries.get_snap_counts(client, season=2022)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.asyncio
async def test_get_college_stats(client):
    df = await queries.get_college_stats(client)
    assert isinstance(df, pd.DataFrame)
