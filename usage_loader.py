"""Load Hermes token usage from the local state.db.

Reads ~/.hermes/state.db (the session store) and returns per-session usage
records with provider / model / token buckets / timestamps. No external deps.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # running standalone / outside Hermes runtime
    def get_hermes_home() -> str:
        return os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))


@dataclass
class UsageRow:
    session_id: str
    provider: Optional[str]
    model: Optional[str]
    started_at: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    actual_cost_usd: Optional[float]
    estimated_cost_usd: Optional[float]
    cost_status: Optional[str]


def _resolve_db_path(db_path: Optional[str]) -> str:
    if db_path:
        return os.path.expanduser(db_path)
    return os.path.join(get_hermes_home(), "state.db")


def load_usage(db_path: Optional[str] = None) -> list[UsageRow]:
    """Read all sessions from state.db into UsageRow objects."""
    path = _resolve_db_path(db_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"state.db not found at {path}")

    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, billing_provider, model, started_at,
               input_tokens, output_tokens, cache_read_tokens,
               cache_write_tokens, reasoning_tokens,
               actual_cost_usd, estimated_cost_usd, cost_status
        FROM sessions
        WHERE started_at IS NOT NULL AND started_at > 0
        """
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            UsageRow(
                session_id=r["id"],
                provider=(r["billing_provider"] or "unknown"),
                model=(r["model"] or "unknown"),
                started_at=float(r["started_at"]),
                input_tokens=int(r["input_tokens"] or 0),
                output_tokens=int(r["output_tokens"] or 0),
                cache_read_tokens=int(r["cache_read_tokens"] or 0),
                cache_write_tokens=int(r["cache_write_tokens"] or 0),
                reasoning_tokens=int(r["reasoning_tokens"] or 0),
                actual_cost_usd=(r["actual_cost_usd"] if r["actual_cost_usd"] is not None else None),
                estimated_cost_usd=(r["estimated_cost_usd"] if r["estimated_cost_usd"] is not None else None),
                cost_status=r["cost_status"],
            )
        )
    con.close()
    return rows


def period_key(ts: float, period: str) -> str:
    """Return a grouping key (date string) for a timestamp + period."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if period == "daily":
        return dt.strftime("%Y-%m-%d")
    if period == "weekly":
        # ISO week: year-Www
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if period == "monthly":
        return dt.strftime("%Y-%m")
    return "all"


if __name__ == "__main__":
    data = load_usage()
    print(f"loaded {len(data)} sessions")
    print("sample:", asdict(data[0]))
