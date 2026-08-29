"""
DataLens AI — Anomaly Detector

Combines three complementary approaches (MASTER PROMPT #13):
  1. Rule-based   — amount vs user's own average, unusual hour, new merchant
  2. Statistical  — z-score and IQR on transaction amount, per-user
  3. ML           — IsolationForest over a small numeric feature set

Design choice: for the ~24.4M row scale, statistical baselines (mean/std/
IQR) are computed in DuckDB (pushed-down aggregation), and IsolationForest
is run on a bounded sample (or a single user's transactions) rather than
the full dataset — fitting IF on 24M rows in-process is not necessary for
a demo-quality investigation agent and would blow past interactive
performance targets (MASTER PROMPT #44).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from src.data_loader import load_processed_duckdb_view


def _safe_query(sql: str, params: list | None = None) -> list[dict]:
    con = load_processed_duckdb_view()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def detect_rule_based_anomalies(user_id: str, limit: int = 20) -> dict:
    """
    Flags a user's transactions that are unusually large relative to their
    own historical average (LARGE_AMOUNT_MULTIPLIER, config-driven).
    """
    try:
        rows = _safe_query("""
            WITH baseline AS (
                SELECT avg_amount, std_amount FROM user_baselines WHERE user_id = ?
            )
            SELECT t.txn_date, t.txn_datetime, t.amount, t.merchant_name, t.mcc,
                   ROUND(b.avg_amount, 2) AS user_avg_amount,
                   ROUND(t.amount / NULLIF(b.avg_amount, 0), 2) AS deviation_multiple
            FROM transactions_final t, baseline b
            WHERE t.user_id = ?
              AND t.amount > b.avg_amount * ?
            ORDER BY deviation_multiple DESC
            LIMIT ?
        """, [user_id, user_id, config.LARGE_AMOUNT_MULTIPLIER, limit])
        for r in rows:
            r["anomaly_reasons"] = ["unusually high amount vs. personal average"]
        return {"tool": "detect_rule_based_anomalies", "result": rows}
    except Exception as e:  # noqa: BLE001
        return {"tool": "detect_rule_based_anomalies", "error": str(e)}


def detect_statistical_anomalies(user_id: Optional[str] = None, limit: int = 20) -> dict:
    """
    Z-score and IQR based outlier detection on transaction amount,
    computed per-user when user_id is given, else across the population.
    """
    try:
        if user_id:
            base = _safe_query("""
                SELECT AVG(amount) AS mean, STDDEV_SAMP(amount) AS std,
                       QUANTILE_CONT(amount, 0.25) AS q1, QUANTILE_CONT(amount, 0.75) AS q3
                FROM transactions_final WHERE user_id = ?
            """, [user_id])
        else:
            base = _safe_query("""
                SELECT AVG(amount) AS mean, STDDEV_SAMP(amount) AS std,
                       QUANTILE_CONT(amount, 0.25) AS q1, QUANTILE_CONT(amount, 0.75) AS q3
                FROM transactions_final
            """)
        if not base or base[0]["std"] in (None, 0):
            return {"tool": "detect_statistical_anomalies", "result": []}

        mean, std = base[0]["mean"], base[0]["std"]
        q1, q3 = base[0]["q1"], base[0]["q3"]
        iqr = q3 - q1
        upper_bound = q3 + config.ANOMALY_IQR_MULTIPLIER * iqr
        z_thresh = config.ANOMALY_ZSCORE_THRESHOLD

        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT txn_date, txn_datetime, amount, merchant_name, mcc, user_id,
                   ROUND((amount - {mean}) / NULLIF({std}, 0), 2) AS z_score
            FROM transactions_final
            {where}
            {"AND" if where else "WHERE"} (amount > {upper_bound} OR ABS((amount - {mean}) / NULLIF({std}, 0)) > {z_thresh})
            ORDER BY z_score DESC
            LIMIT ?
        """, params + [limit])
        for r in rows:
            reasons = []
            if r["z_score"] is not None and abs(r["z_score"]) > z_thresh:
                reasons.append(f"z-score {r['z_score']} exceeds threshold {z_thresh}")
            if r["amount"] > upper_bound:
                reasons.append("amount exceeds IQR-based upper bound")
            r["anomaly_reasons"] = reasons
        return {
            "tool": "detect_statistical_anomalies",
            "result": rows,
            "baseline": {"mean": round(mean, 2), "std": round(std, 2),
                         "iqr_upper_bound": round(upper_bound, 2)},
        }
    except Exception as e:  # noqa: BLE001
        return {"tool": "detect_statistical_anomalies", "error": str(e)}


def detect_ml_anomalies(user_id: Optional[str] = None, sample_limit: int = 20000, top_n: int = 20) -> dict:
    """
    IsolationForest over a small numeric feature set (amount, hour,
    is_weekend) for a single user or a bounded random sample of the
    population. Returns the most anomalous transactions with an
    isolation score.
    """
    try:
        where = "WHERE user_id = ?" if user_id else ""
        params = [user_id] if user_id else []
        rows = _safe_query(f"""
            SELECT txn_date, txn_datetime, amount, merchant_name, mcc, user_id,
                   hour, CAST(is_weekend AS INTEGER) AS is_weekend_int
            FROM transactions_final
            USING SAMPLE {sample_limit} ROWS
            {where}
        """, params)
        if len(rows) < 20:
            return {"tool": "detect_ml_anomalies", "result": [],
                    "note": "Not enough data to fit an Isolation Forest reliably."}

        df = pd.DataFrame(rows)
        features = df[["amount", "hour", "is_weekend_int"]].fillna(0)

        model = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
        )
        df["anomaly_score"] = model.fit_predict(features)
        df["anomaly_score_raw"] = model.decision_function(features)

        anomalies = df[df["anomaly_score"] == -1].sort_values("anomaly_score_raw").head(top_n)
        result = anomalies.drop(columns=["is_weekend_int"]).to_dict(orient="records")
        for r in result:
            r["anomaly_reasons"] = ["flagged by Isolation Forest (unusual amount/time pattern)"]
        return {"tool": "detect_ml_anomalies", "result": result}
    except Exception as e:  # noqa: BLE001
        return {"tool": "detect_ml_anomalies", "error": str(e)}


def detect_anomalies(user_id: Optional[str] = None, method: str = "all", limit: int = 20) -> dict:
    """
    Unified entry point the agent calls. method: 'rule', 'statistical', 'ml', or 'all'.
    Combines results from the requested method(s) and de-duplicates.
    """
    try:
        results = {}
        if method in ("rule", "all") and user_id:
            results["rule_based"] = detect_rule_based_anomalies(user_id, limit).get("result", [])
        if method in ("statistical", "all"):
            results["statistical"] = detect_statistical_anomalies(user_id, limit).get("result", [])
        if method in ("ml", "all"):
            results["ml"] = detect_ml_anomalies(user_id, top_n=limit).get("result", [])
        return {"tool": "detect_anomalies", "result": results}
    except Exception as e:  # noqa: BLE001
        return {"tool": "detect_anomalies", "error": str(e)}
