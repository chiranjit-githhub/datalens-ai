"""
DataLens AI — Anomaly Explanations & Fraud-Label Evaluation

Turns raw anomaly-detector output into the evidence-backed narrative
format described in MASTER PROMPT #13, and (separately, per #43)
evaluates detected anomalies against the `Is Fraud?` label WITHOUT
turning this into a supervised fraud classifier — it is purely an
evaluation utility for the anomaly-detection component.
"""

from __future__ import annotations

from typing import Optional

from src.data_loader import load_processed_duckdb_view


def explain_transaction_anomaly(txn: dict) -> str:
    """
    Build a short, evidence-grounded explanation string for a single
    flagged transaction. Never fabricates numbers — only uses fields
    already present on the transaction dict produced by the detector.
    """
    lines = []
    amount = txn.get("amount")
    if amount is not None:
        lines.append(f"Transaction amount: {amount}")
    if txn.get("user_avg_amount") is not None:
        lines.append(f"User's average transaction: {txn['user_avg_amount']}")
    if txn.get("deviation_multiple") is not None:
        lines.append(f"Deviation: {txn['deviation_multiple']}x user's average")
    if txn.get("z_score") is not None:
        lines.append(f"Z-score: {txn['z_score']}")
    reasons = txn.get("anomaly_reasons") or []
    if reasons:
        lines.append("Anomaly reasons: " + "; ".join(reasons))
    return "\n".join(lines) if lines else "No supporting evidence available for this transaction."


def _safe_query(sql: str, params: list | None = None) -> list[dict]:
    con = load_processed_duckdb_view()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def evaluate_against_fraud_label(
    flagged_pairs: list[tuple],
    method_name: str = "combined",
) -> dict:
    """
    Compare a set of flagged (user_id, txn_datetime, amount) tuples against
    the ground-truth `is_fraud` column and compute precision/recall/F1.

    This is an EVALUATION utility for the anomaly detector, not a
    classifier itself (MASTER PROMPT #43): DataLens's core product is
    financial investigation, and fraud-label agreement is just one
    quality signal for the anomaly-detection component.
    """
    try:
        con = load_processed_duckdb_view()
        try:
            total_fraud = con.execute(
                "SELECT COUNT(*) FROM transactions_final WHERE is_fraud = TRUE"
            ).fetchone()[0]
        finally:
            con.close()

        if not flagged_pairs:
            return {"tool": "evaluate_against_fraud_label", "result": {
                "method": method_name, "flagged_count": 0, "total_fraud_in_data": total_fraud,
                "precision": None, "recall": None, "f1": None,
            }}

        true_positives = 0
        for user_id, txn_datetime, amount in flagged_pairs:
            rows = _safe_query("""
                SELECT is_fraud FROM transactions_final
                WHERE user_id = ? AND txn_datetime = ? AND amount = ?
                LIMIT 1
            """, [user_id, txn_datetime, amount])
            if rows and rows[0]["is_fraud"]:
                true_positives += 1

        flagged_count = len(flagged_pairs)
        precision = round(true_positives / flagged_count, 4) if flagged_count else None
        recall = round(true_positives / total_fraud, 4) if total_fraud else None
        f1 = (round(2 * precision * recall / (precision + recall), 4)
              if precision and recall and (precision + recall) else None)

        return {
            "tool": "evaluate_against_fraud_label",
            "result": {
                "method": method_name,
                "flagged_count": flagged_count,
                "true_positives": true_positives,
                "total_fraud_in_data": total_fraud,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "evaluate_against_fraud_label", "error": str(e)}
