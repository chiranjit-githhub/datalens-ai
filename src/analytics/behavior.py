"""
DataLens AI — Behavioral Analytics Tools

Frequency, transaction size distribution, merchant diversity, recurring
expenses. Reads from the pre-aggregated `user_baselines` table where
possible (see src/feature_engineering.py) to avoid repeated full scans.
"""

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


def analyze_transaction_frequency(
    user_id: Optional[str] = None,
    start_date_1: Optional[str] = None, end_date_1: Optional[str] = None,
    start_date_2: Optional[str] = None, end_date_2: Optional[str] = None,
) -> dict:
    """
    Transaction count. If two periods are given, compares frequency between
    them (used to test whether a spending change was driven by frequency
    rather than transaction size).
    """
    try:
        if start_date_1 and end_date_1 and start_date_2 and end_date_2:
            user_filter = "AND user_id = ?" if user_id else ""

            def _count(start, end):
                params = [start, end]
                if user_id:
                    params.append(user_id)
                r = _safe_query(f"""
                    SELECT COUNT(*) AS txn_count
                    FROM transactions_final
                    WHERE txn_date BETWEEN ? AND ? {user_filter}
                """, params)
                return r[0]["txn_count"] if r else 0

            c1, c2 = _count(start_date_1, end_date_1), _count(start_date_2, end_date_2)
            pct = round(100.0 * (c2 - c1) / c1, 2) if c1 else None
            return {
                "tool": "analyze_transaction_frequency",
                "result": {
                    "period_1_count": c1, "period_2_count": c2,
                    "absolute_change": c2 - c1, "percentage_change": pct,
                },
            }
        else:
            where = "WHERE user_id = ?" if user_id else ""
            params = [user_id] if user_id else []
            rows = _safe_query(f"""
                SELECT COUNT(*) AS total_txn_count,
                       COUNT(DISTINCT txn_date) AS active_days,
                       ROUND(CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT txn_date), 0), 2) AS avg_per_day
                FROM transactions_final {where}
            """, params)
            return {"tool": "analyze_transaction_frequency", "result": rows[0] if rows else {}}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_transaction_frequency", "error": str(e)}


def analyze_average_transaction(
    user_id: Optional[str] = None,
    start_date_1: Optional[str] = None, end_date_1: Optional[str] = None,
    start_date_2: Optional[str] = None, end_date_2: Optional[str] = None,
) -> dict:
    """Average transaction size, optionally compared across two periods."""
    try:
        if start_date_1 and end_date_1 and start_date_2 and end_date_2:
            user_filter = "AND user_id = ?" if user_id else ""

            def _avg(start, end):
                params = [start, end]
                if user_id:
                    params.append(user_id)
                r = _safe_query(f"""
                    SELECT ROUND(AVG(amount), 2) AS avg_amount
                    FROM transactions_final
                    WHERE txn_date BETWEEN ? AND ? {user_filter}
                """, params)
                return (r[0]["avg_amount"] or 0) if r else 0

            a1, a2 = _avg(start_date_1, end_date_1), _avg(start_date_2, end_date_2)
            pct = round(100.0 * (a2 - a1) / a1, 2) if a1 else None
            return {
                "tool": "analyze_average_transaction",
                "result": {"period_1_avg": a1, "period_2_avg": a2,
                            "absolute_change": round(a2 - a1, 2), "percentage_change": pct},
            }
        else:
            where = "WHERE user_id = ?" if user_id else ""
            params = [user_id] if user_id else []
            rows = _safe_query(f"""
                SELECT ROUND(AVG(amount), 2) AS avg_amount,
                       ROUND(STDDEV_SAMP(amount), 2) AS std_amount,
                       ROUND(MIN(amount), 2) AS min_amount,
                       ROUND(MAX(amount), 2) AS max_amount
                FROM transactions_final {where}
            """, params)
            return {"tool": "analyze_average_transaction", "result": rows[0] if rows else {}}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_average_transaction", "error": str(e)}


def analyze_merchant_diversity(user_id: Optional[str] = None) -> dict:
    """How many distinct merchants a user (or the population) transacts with."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT COUNT(DISTINCT merchant_name) AS unique_merchants,
                   COUNT(*) AS total_txns
            FROM transactions_final {where}
        """, params)
        return {"tool": "analyze_merchant_diversity", "result": rows[0] if rows else {}}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_merchant_diversity", "error": str(e)}


def analyze_recurring_expenses(user_id: str, min_occurrences: int = 3) -> dict:
    """
    Identifies merchants a specific user transacts with repeatedly — a proxy
    for recurring/subscription-like expenses. Requires a user_id since
    'recurring' is inherently a per-user concept.
    """
    try:
        rows = _safe_query("""
            SELECT merchant_name, COUNT(*) AS occurrences,
                   ROUND(AVG(amount), 2) AS avg_amount,
                   ROUND(SUM(amount), 2) AS total_amount
            FROM transactions_final
            WHERE user_id = ?
            GROUP BY merchant_name
            HAVING COUNT(*) >= ?
            ORDER BY occurrences DESC
        """, [user_id, min_occurrences])
        return {"tool": "analyze_recurring_expenses", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_recurring_expenses", "error": str(e)}
