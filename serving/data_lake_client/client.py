import asyncio
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import yaml
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


class DataLakeConnectionError(Exception):
    """Raised when the data lake API is unreachable."""


class DataLakeQueryError(Exception):
    """Raised when a SQL query or API call returns an error response."""


class DataLakeNotFoundError(Exception):
    """Raised when a requested resource (player, team) does not exist."""


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config" / "data_lake.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["data_lake"]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(
        exc, (httpx.ConnectError, httpx.TimeoutException, DataLakeConnectionError)
    ):
        return True
    if isinstance(exc, DataLakeQueryError) and getattr(exc, "status_code", 0) >= 500:
        return True
    return False


# Client


class DataLakeClient:
    """
    Async client for the nfl-data-platform REST API.

    Usage (async):
        async with DataLakeClient() as client:
            df = await client.query("SELECT * FROM combine LIMIT 10")

    Usage (sync, e.g. training scripts):
        client = DataLakeClient()
        df = client.run_sync(client.query("SELECT * FROM combine LIMIT 10"))
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)
        self._base_url = cfg["base_url"].rstrip("/")
        self._timeout = httpx.Timeout(
            connect=cfg["timeout"]["connect"],
            read=cfg["timeout"]["read"],
            write=cfg["timeout"]["connect"],
            pool=cfg["timeout"]["connect"],
        )
        retry_cfg = cfg["retry"]
        self._max_attempts = retry_cfg["max_attempts"]
        self._wait_min = retry_cfg["wait_min"]
        self._wait_max = retry_cfg["wait_max"]
        self._client: Optional[httpx.AsyncClient] = None

    # Context manager

    async def __aenter__(self) -> "DataLakeClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "DataLakeClient must be used as an async context manager "
                "or call open()/close() explicitly."
            )
        return self._client

    async def open(self) -> "DataLakeClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # Sync shim for training scripts / notebooks

    def run_sync(self, coro):
        """Run an async coroutine synchronously. Convenience for training scripts."""
        return asyncio.run(coro)

    # Internal helpers

    async def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        client = self._get_client()
        try:
            resp = await client.get(path, params=params)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise DataLakeConnectionError(
                f"Cannot reach data lake at {self._base_url}: {e}"
            ) from e
        return self._parse(resp)

    async def _post(self, path: str, json: dict) -> dict | list:
        client = self._get_client()
        try:
            resp = await client.post(path, json=json)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise DataLakeConnectionError(
                f"Cannot reach data lake at {self._base_url}: {e}"
            ) from e
        return self._parse(resp)

    def _parse(self, resp: httpx.Response) -> dict | list:
        if resp.status_code == 404:
            raise DataLakeNotFoundError(f"Resource not found: {resp.url}")
        if resp.status_code >= 400:
            err = DataLakeQueryError(f"API error {resp.status_code}: {resp.text}")
            err.status_code = resp.status_code  # type: ignore[attr-defined]
            raise err
        return resp.json()

    @staticmethod
    def _to_df(data: dict | list) -> pd.DataFrame:
        # {"rows": [...], "columns": [...], "count": N}  — data lake /query format
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            return pd.DataFrame(data["rows"], columns=data["columns"])
        if isinstance(data, list):
            return pd.DataFrame(data)
        if "results" in data:
            return pd.DataFrame(data["results"])
        if "data" in data:
            return pd.DataFrame(data["data"])
        return pd.DataFrame([data])

    # Public API — with tenacity retry baked in per method

    async def health_check(self) -> bool:
        """Return True if the data lake API is reachable and healthy."""
        try:
            data = await self._get("/health")
            return True
        except (DataLakeConnectionError, DataLakeQueryError):
            return False

    async def query(self, sql: str) -> pd.DataFrame:
        """Execute a read-only SQL query via DuckDB. Returns a DataFrame."""
        data = await self._post_with_retry("/query", {"sql": sql})
        return self._to_df(data)

    async def list_tables(self) -> list[str]:
        """List all virtual tables available for SQL querying."""
        data = await self._get_with_retry("/query/tables")
        if isinstance(data, list):
            return data
        return data.get("tables", [])

    async def get_player(self, name: str) -> dict:
        """Get a player profile by name."""
        return await self._get_with_retry(f"/players/{name}")

    async def list_players(
        self,
        position: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """List/filter players. Returns a DataFrame."""
        params = {}
        if position:
            params["position"] = position
        if year:
            params["year"] = year
        data = await self._get_with_retry("/players", params=params or None)
        return self._to_df(data)

    async def get_team_stats(
        self,
        abbr: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get team season stats. Returns a DataFrame."""
        params = {}
        if year_start:
            params["year_start"] = year_start
        if year_end:
            params["year_end"] = year_end
        data = await self._get_with_retry(f"/teams/{abbr}/stats", params=params or None)
        return self._to_df(data)

    async def get_player_graph_profile(self, name: str) -> dict:
        """Full player graph profile (Neo4j traversal)."""
        return await self._get_with_retry(f"/graph/player/{name}/profile")

    async def get_player_neighbors(self, name: str, hops: int = 2) -> dict:
        """Graph neighbors up to N hops."""
        return await self._get_with_retry(
            f"/graph/player/{name}/neighbors", params={"hops": hops}
        )

    async def get_college_pipeline(self, college: str) -> pd.DataFrame:
        """All players from a college + their draft outcomes."""
        data = await self._get_with_retry(f"/graph/college/{college}/pipeline")
        return self._to_df(data)

    async def get_team_drafted(self, team: str) -> pd.DataFrame:
        """Players drafted by a team."""
        data = await self._get_with_retry(f"/graph/team/{team}/drafted")
        return self._to_df(data)

    async def list_datasets(self) -> list[str]:
        """List all Parquet files across lake zones."""
        data = await self._get_with_retry("/manage/datasets")
        if isinstance(data, list):
            return data
        return data.get("datasets", [])

    async def preview_dataset(self, dataset: str, rows: int = 5) -> pd.DataFrame:
        """Preview a dataset's rows and schema."""
        data = await self._get_with_retry(
            f"/manage/preview/{dataset}", params={"rows": rows}
        )
        return self._to_df(data)

    # Retry wrappers (tenacity applied at call site for flexibility)

    async def _get_with_retry(self, path: str, params: Optional[dict] = None):
        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(min=self._wait_min, max=self._wait_max),
            reraise=True,
        )
        async def _inner():
            return await self._get(path, params)

        return await _inner()

    async def _post_with_retry(self, path: str, json: dict):
        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(min=self._wait_min, max=self._wait_max),
            reraise=True,
        )
        async def _inner():
            return await self._post(path, json)

        return await _inner()
