"""DataLens AI — Single-Transaction Analytics Tools

Used when a user asks about a specific transaction ("why is this suspicious?")
and the agent needs to place it against the user's and merchant's baselines.
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


def get_transaction_detail(user_id: str, txn_datetime: str, amount: float) -> dict:
    """
    Fetch a specific transaction (matched by user, timestamp, amount — the
    dataset has no unique transaction ID) plus that user's baseline stats.
    """
    try:
        txn = _safe_query("""
            SELECT * FROM transactions_final
            WHERE user_id = ? AND txn_datetime = ? AND amount = ?
            LIMIT 1
        """, [user_id, txn_datetime, amount])

        baseline = _safe_query("""
            SELECT avg_amount, std_amount, avg_daily_frequency, merchant_diversity
            FROM user_baselines WHERE user_id = ?
        """, [user_id])

        return {
            "tool": "get_transaction_detail",
            "result": {
                "transaction": txn[0] if txn else None,
                "user_baseline": baseline[0] if baseline else None,
            },
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "get_transaction_detail", "error": str(e)}


def compare_user_behavior(user_id: str, amount: float) -> dict:
    """How a given amount compares to a user's own historical average/std."""
    try:
        baseline = _safe_query("""
            SELECT avg_amount, std_amount FROM user_baselines WHERE user_id = ?
        """, [user_id])
        if not baseline or baseline[0]["avg_amount"] in (None, 0):
            return {"tool": "compare_user_behavior", "error": "No baseline available for this user."}
        avg = baseline[0]["avg_amount"]
        std = baseline[0]["std_amount"] or 0
        deviation_multiple = round(amount / avg, 2) if avg else None
        z_score = round((amount - avg) / std, 2) if std else None
        return {
            "tool": "compare_user_behavior",
            "result": {
                "amount": amount, "user_avg_amount": round(avg, 2),
                "deviation_multiple": deviation_multiple, "z_score": z_score,
            },
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "compare_user_behavior", "error": str(e)}


def compare_merchant_behavior(merchant_name: str, amount: float) -> dict:
    """How a given amount compares to a merchant's typical transaction size."""
    try:
        baseline = _safe_query("""
            SELECT avg_amount, std_amount FROM merchant_baselines WHERE merchant_name = ?
        """, [merchant_name])
        if not baseline or baseline[0]["avg_amount"] in (None, 0):
            return {"tool": "compare_merchant_behavior", "error": "No baseline available for this merchant."}
        avg = baseline[0]["avg_amount"]
        std = baseline[0]["std_amount"] or 0
        z_score = round((amount - avg) / std, 2) if std else None
        return {
            "tool": "compare_merchant_behavior",
            "result": {"amount": amount, "merchant_avg_amount": round(avg, 2), "z_score": z_score},
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "compare_merchant_behavior", "error": str(e)}
