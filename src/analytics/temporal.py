"""DataLens AI — Temporal Analytics Tools"""

from __future__ import annotations

from typing import Optional

from src.data_loader import load_processed_duckdb_view


def _safe_query(sql: str, params: list | None = None) -> list[dict]:
    con = load_processed_duckdb_view()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def analyze_weekday_vs_weekend(user_id: Optional[str] = None) -> dict:
    """Compares total & average spend on weekdays vs weekends."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT is_weekend, ROUND(SUM(amount), 2) AS total,
                   COUNT(*) AS txn_count, ROUND(AVG(amount), 2) AS avg_amount
            FROM transactions_final
            {where}
            GROUP BY is_weekend
        """, params)
        weekend = next((r for r in rows if r["is_weekend"]), {"total": 0, "txn_count": 0, "avg_amount": 0})
        weekday = next((r for r in rows if not r["is_weekend"]), {"total": 0, "txn_count": 0, "avg_amount": 0})
        return {
            "tool": "analyze_weekday_vs_weekend",
            "result": {"weekday": weekday, "weekend": weekend},
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_weekday_vs_weekend", "error": str(e)}


def analyze_peak_hours(user_id: Optional[str] = None, top_n: int = 5) -> dict:
    """Hours of day with the highest transaction volume / spend."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT hour, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY hour
            ORDER BY txn_count DESC
            LIMIT ?
        """, params + [top_n])
        return {"tool": "analyze_peak_hours", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_peak_hours", "error": str(e)}


def analyze_time_of_day(user_id: Optional[str] = None) -> dict:
    """Spend broken down by Morning / Afternoon / Evening / Night buckets."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT time_of_day, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY time_of_day
            ORDER BY total DESC
        """, params)
        return {"tool": "analyze_time_of_day", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_time_of_day", "error": str(e)}


def analyze_monthly_trend(user_id: Optional[str] = None) -> dict:
    """Month-over-month spend trend with rolling percentage change."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            WITH monthly AS (
                SELECT year, month, SUM(amount) AS total
                FROM transactions_final
                {where}
                GROUP BY year, month
                ORDER BY year, month
            )
            SELECT year, month, ROUND(total, 2) AS total,
                   ROUND(100.0 * (total - LAG(total) OVER (ORDER BY year, month))
                         / NULLIF(LAG(total) OVER (ORDER BY year, month), 0), 2) AS pct_change_mom
            FROM monthly
        """, params)
        return {"tool": "analyze_monthly_trend", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_monthly_trend", "error": str(e)}
