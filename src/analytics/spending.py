"""
DataLens AI — Spending Analytics Tools

Every function here is a self-contained "tool" with:
  - a clear name & docstring (used as the tool description for the LLM)
  - typed parameters
  - a structured dict return value (never raw text)
  - defensive error handling (never raises to the caller; returns {"error": ...})

These are the building blocks the agent's planner selects from.
"""

from __future__ import annotations

from typing import Optional

from src.data_loader import load_processed_duckdb_view, DataLoadError


def _safe_query(sql: str, params: list | None = None) -> list[dict]:
    con = load_processed_duckdb_view()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def get_overview(user_id: Optional[str] = None) -> dict:
    """
    High-level snapshot of the dataset (or a single user): total transactions,
    total spend, average transaction, date range, active merchants.
    """
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT
                COUNT(*)                       AS total_transactions,
                ROUND(SUM(amount), 2)           AS total_spend,
                ROUND(AVG(amount), 2)           AS avg_transaction,
                MIN(txn_date)                   AS start_date,
                MAX(txn_date)                   AS end_date,
                COUNT(DISTINCT user_id)         AS active_users,
                COUNT(DISTINCT merchant_name)   AS active_merchants
            FROM transactions_final
            {where}
        """, params)
        return {"tool": "get_overview", "result": rows[0] if rows else {}}
    except DataLoadError as e:
        return {"tool": "get_overview", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"tool": "get_overview", "error": f"Unexpected error: {e}"}


def get_date_range(user_id: Optional[str] = None) -> dict:
    """Returns the earliest and latest transaction date available."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT MIN(txn_date) AS start_date, MAX(txn_date) AS end_date
            FROM transactions_final {where}
        """, params)
        return {"tool": "get_date_range", "result": rows[0] if rows else {}}
    except Exception as e:  # noqa: BLE001
        return {"tool": "get_date_range", "error": str(e)}


def compare_periods(
    start_date_1: str,
    end_date_1: str,
    start_date_2: str,
    end_date_2: str,
    user_id: Optional[str] = None,
) -> dict:
    """
    Compare total spending between two date ranges (e.g. previous month vs
    current month). Dates as 'YYYY-MM-DD'. Returns totals, absolute and
    percentage change.
    """
    try:
        user_filter = "AND user_id = ?" if user_id else ""
        base_params = [start_date_1, end_date_1]
        if user_id:
            base_params.append(user_id)
        rows1 = _safe_query(f"""
            SELECT ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            WHERE txn_date BETWEEN ? AND ? {user_filter}
        """, base_params)

        base_params2 = [start_date_2, end_date_2]
        if user_id:
            base_params2.append(user_id)
        rows2 = _safe_query(f"""
            SELECT ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            WHERE txn_date BETWEEN ? AND ? {user_filter}
        """, base_params2)

        p1_total = (rows1[0]["total"] or 0) if rows1 else 0
        p2_total = (rows2[0]["total"] or 0) if rows2 else 0
        abs_change = round(p2_total - p1_total, 2)
        pct_change = round((abs_change / p1_total) * 100, 2) if p1_total else None

        return {
            "tool": "compare_periods",
            "result": {
                "period_1": {"start": start_date_1, "end": end_date_1,
                             "total": p1_total, "txn_count": rows1[0]["txn_count"] if rows1 else 0},
                "period_2": {"start": start_date_2, "end": end_date_2,
                             "total": p2_total, "txn_count": rows2[0]["txn_count"] if rows2 else 0},
                "absolute_change": abs_change,
                "percentage_change": pct_change,
            },
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "compare_periods", "error": str(e)}


def get_monthly_spending(user_id: Optional[str] = None, year: Optional[int] = None) -> dict:
    """Total spend grouped by year-month."""
    try:
        filters = []
        params: list = []
        if user_id:
            filters.append("user_id = ?")
            params.append(user_id)
        if year:
            filters.append("year = ?")
            params.append(year)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = _safe_query(f"""
            SELECT year, month, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY year, month
            ORDER BY year, month
        """, params)
        return {"tool": "get_monthly_spending", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "get_monthly_spending", "error": str(e)}


def get_daily_spending(user_id: Optional[str] = None, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> dict:
    """Total spend grouped by calendar date, optionally bounded by a date range."""
    try:
        filters = []
        params: list = []
        if user_id:
            filters.append("user_id = ?")
            params.append(user_id)
        if start_date and end_date:
            filters.append("txn_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = _safe_query(f"""
            SELECT txn_date, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY txn_date
            ORDER BY txn_date
        """, params)
        return {"tool": "get_daily_spending", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "get_daily_spending", "error": str(e)}


def find_spending_changes(
    user_id: Optional[str] = None,
    top_n: int = 5,
) -> dict:
    """
    Identify the months with the largest month-over-month spending swings
    (useful as a starting point for 'why did spending change' investigations
    when no explicit period is given).
    """
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            WITH monthly AS (
                SELECT year, month, SUM(amount) AS total
                FROM transactions_final
                {where}
                GROUP BY year, month
            ),
            with_prev AS (
                SELECT *,
                    LAG(total) OVER (ORDER BY year, month) AS prev_total
                FROM monthly
            )
            SELECT year, month, ROUND(total, 2) AS total, ROUND(prev_total, 2) AS prev_total,
                   ROUND(total - prev_total, 2) AS change,
                   ROUND(100.0 * (total - prev_total) / NULLIF(prev_total, 0), 2) AS pct_change
            FROM with_prev
            WHERE prev_total IS NOT NULL
            ORDER BY ABS(total - prev_total) DESC
            LIMIT ?
        """, params + [top_n])
        return {"tool": "find_spending_changes", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "find_spending_changes", "error": str(e)}
