"""
DataLens AI — Category Analytics Tools

The dataset does not have an explicit "category" column; MCC (Merchant
Category Code) is the categorical proxy used throughout. Where a human
label is useful, a small MCC -> category-name lookup is applied, but the
underlying grouping is always the real MCC value — never fabricated.
"""

from __future__ import annotations

from typing import Optional

from src.data_loader import load_processed_duckdb_view

# Small, non-exhaustive MCC -> friendly label map for common codes.
# Unmapped codes are shown as "MCC {code}" rather than guessed.
MCC_LABELS = {
    5411: "Grocery Stores",
    5812: "Restaurants",
    5814: "Fast Food",
    5541: "Gas Stations",
    5311: "Department Stores",
    5732: "Electronics",
    4900: "Utilities",
    5912: "Pharmacies",
    5999: "Retail (Misc.)",
    4899: "Cable / Streaming",
    5651: "Clothing Stores",
    7996: "Entertainment",
    4111: "Transit",
    5211: "Home Improvement",
}


def _label(mcc) -> str:
    try:
        return MCC_LABELS.get(int(mcc), f"MCC {mcc}")
    except (TypeError, ValueError):
        return "Unknown"


def _safe_query(sql: str, params: list | None = None) -> list[dict]:
    con = load_processed_duckdb_view()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def analyze_categories(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 15,
) -> dict:
    """Spend broken down by MCC category over an optional date range/user."""
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
            SELECT mcc, ROUND(SUM(amount), 2) AS total, COUNT(*) AS txn_count,
                   ROUND(AVG(amount), 2) AS avg_amount
            FROM transactions_final
            {where}
            GROUP BY mcc
            ORDER BY total DESC
            LIMIT ?
        """, params + [top_n])
        for r in rows:
            r["category_label"] = _label(r["mcc"])
        return {"tool": "analyze_categories", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "analyze_categories", "error": str(e)}


def compare_category_periods(
    start_date_1: str, end_date_1: str,
    start_date_2: str, end_date_2: str,
    user_id: Optional[str] = None,
    top_n: int = 10,
) -> dict:
    """
    Compare per-category spend between two periods — used to identify which
    category contributed most to an overall spending change.
    """
    try:
        user_filter = "AND user_id = ?" if user_id else ""

        def _period(start, end):
            params = [start, end]
            if user_id:
                params.append(user_id)
            return _safe_query(f"""
                SELECT mcc, SUM(amount) AS total
                FROM transactions_final
                WHERE txn_date BETWEEN ? AND ? {user_filter}
                GROUP BY mcc
            """, params)

        p1 = {r["mcc"]: r["total"] for r in _period(start_date_1, end_date_1)}
        p2 = {r["mcc"]: r["total"] for r in _period(start_date_2, end_date_2)}

        all_mccs = set(p1) | set(p2)
        diffs = []
        for mcc in all_mccs:
            t1, t2 = p1.get(mcc, 0) or 0, p2.get(mcc, 0) or 0
            diffs.append({
                "mcc": mcc,
                "category_label": _label(mcc),
                "period_1_total": round(t1, 2),
                "period_2_total": round(t2, 2),
                "change": round(t2 - t1, 2),
            })
        diffs.sort(key=lambda d: abs(d["change"]), reverse=True)
        return {"tool": "compare_category_periods", "result": diffs[:top_n]}
    except Exception as e:  # noqa: BLE001
        return {"tool": "compare_category_periods", "error": str(e)}
