"""DataLens AI — Merchant Analytics Tools"""

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


def analyze_merchants(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 10,
) -> dict:
    """Top merchants by total spend, with transaction count and average amount."""
    try:
        filters, params = [], []
        if user_id:
            filters.append("user_id = ?")
            params.append(user_id)
        if start_date and end_date:
            filters.append("txn_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = _safe_query(f"""
            SELECT merchant_name, ROUND(SUM(amount), 2) AS total_spend,
                   COUNT(*) AS txn_count, ROUND(AVG(amount), 2) AS avg_amount
            FROM transactions_final
            {where}
            GROUP BY merchant_name
            ORDER BY total_spend DESC
            LIMIT ?
        """, params + [top_n])
        return {"tool": "analyze_merchants", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_merchants", "error": str(e)}


def compare_merchant_periods(
    start_date_1: str, end_date_1: str,
    start_date_2: str, end_date_2: str,
    user_id: Optional[str] = None,
    top_n: int = 10,
) -> dict:
    """Which merchants contributed most to a change in spend between two periods."""
    try:
        user_filter = "AND user_id = ?" if user_id else ""

        def _period(start, end):
            params = [start, end]
            if user_id:
                params.append(user_id)
            return _safe_query(f"""
                SELECT merchant_name, SUM(amount) AS total
                FROM transactions_final
                WHERE txn_date BETWEEN ? AND ? {user_filter}
                GROUP BY merchant_name
            """, params)

        p1 = {r["merchant_name"]: r["total"] for r in _period(start_date_1, end_date_1)}
        p2 = {r["merchant_name"]: r["total"] for r in _period(start_date_2, end_date_2)}
        merchants = set(p1) | set(p2)
        diffs = [{
            "merchant_name": m,
            "period_1_total": round(p1.get(m, 0) or 0, 2),
            "period_2_total": round(p2.get(m, 0) or 0, 2),
            "change": round((p2.get(m, 0) or 0) - (p1.get(m, 0) or 0), 2),
        } for m in merchants]
        diffs.sort(key=lambda d: abs(d["change"]), reverse=True)
        return {"tool": "compare_merchant_periods", "result": diffs[:top_n]}
    except Exception as e:  # noqa: BLE001
        return {"tool": "compare_merchant_periods", "error": str(e)}


def analyze_locations(
    user_id: Optional[str] = None,
    top_n: int = 10,
) -> dict:
    """Spend broken down by merchant city/state."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT merchant_state, merchant_city, ROUND(SUM(amount), 2) AS total,
                   COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY merchant_state, merchant_city
            ORDER BY total DESC
            LIMIT ?
        """, params + [top_n])
        return {"tool": "analyze_locations", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_locations", "error": str(e)}


def analyze_payment_methods(user_id: Optional[str] = None) -> dict:
    """Spend broken down by 'Use Chip' (payment channel)."""
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT use_chip, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count
            FROM transactions_final
            {where}
            GROUP BY use_chip
            ORDER BY total DESC
        """, params)
        return {"tool": "analyze_payment_methods", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_payment_methods", "error": str(e)}
