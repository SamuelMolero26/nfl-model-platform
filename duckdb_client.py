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

    This implementation:
      - Only treats `?` characters outside single-quoted string literals as
        placeholders.
      - Raises a clear ValueError if the number of placeholders does not
        match the number of provided params.
    """
    parts = iter(params)
    result: list[str] = []
    in_string = False
    placeholder_count = 0
    i = 0
    length = len(sql)

    while i < length:
        ch = sql[i]

        if ch == "'":
            # Always append the quote
            result.append(ch)
            if in_string:
                # Inside a string: handle escaped single quote '' by consuming
                # the next character if it is also a single quote.
                if i + 1 < length and sql[i + 1] == "'":
                    result.append("'")
                    i += 1
                else:
                    in_string = False
            else:
                # Entering a string literal
                in_string = True
        elif ch == "?" and not in_string:
            # Positional parameter placeholder
            try:
                val = next(parts)
            except StopIteration:
                raise ValueError(
                    "Not enough parameters for SQL placeholders: "
                    f"encountered at least {placeholder_count + 1} "
                    f"placeholder(s) but only {len(params)} parameter(s) provided."
                ) from None

            if isinstance(val, str):
                escaped = val.replace("'", "''")
                result.append(f"'{escaped}'")
            else:
                result.append(str(val))
            placeholder_count += 1
        else:
            result.append(ch)

        i += 1

    # Detect extra parameters that were not consumed by placeholders.
    try:
        extra = next(parts)
    except StopIteration:
        extra = None

    if extra is not None:
        raise ValueError(
            "Too many parameters supplied for SQL placeholders: "
            f"{placeholder_count} placeholder(s) but {len(params)} parameter(s) provided."
        )

    return "".join(result)
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
