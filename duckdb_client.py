"""
duckdb_client.py
================
Thin synchronous wrapper around DataLakeClient for the calibration path
of the Draft Optimizer (and any future offline pipeline scripts).

Provides a single `execute(sql, params)` function that mirrors the DuckDB
positional-parameter convention (`?` placeholders) used in batch SQL queries.

Why a separate module instead of calling DataLakeClient directly?
  - Calibration scripts run synchronously (no event loop).
  - The `?` placeholder syntax keeps SQL readable without f-string injection.
  - A single import (`import duckdb_client`) is cleaner than wiring up an
    async client just to run one bulk query at training time.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _interpolate(sql: str, params: list[Any]) -> str:
    """
    Replace positional `?` placeholders with their values.

    Strings are single-quoted and escaped; numbers are inlined as literals.
    Only called for offline calibration queries — never for user-facing input.
    """
    parts = iter(params)

    def _replace(_match: re.Match) -> str:  # noqa: ANN001
        val = next(parts)
        if isinstance(val, str):
            escaped = val.replace("'", "''")
            return f"'{escaped}'"
        return str(val)

    return re.sub(r"\?", _replace, sql)


def execute(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """
    Execute a read-only SQL query against the data lake and return a DataFrame.

    Parameters
    ----------
    sql    : DuckDB-compatible SQL with optional `?` positional placeholders.
    params : Values substituted for each `?` in left-to-right order.

    Returns
    -------
    pd.DataFrame — empty DataFrame on any transport or query error.
    """
    import sys

    # Locate project root so we can import DataLakeClient regardless of CWD.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from serving.data_lake_client import DataLakeClient  # noqa: PLC0415

    final_sql = _interpolate(sql, list(params or []))

    async def _run() -> pd.DataFrame:
        async with DataLakeClient() as client:
            return await client.query(final_sql)

    return asyncio.run(_run())
