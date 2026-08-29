"""
DataLens AI — Feature Engineering

Builds derived, pre-aggregated tables that power fast analytics and
anomaly detection without repeated full scans of the transaction table
(MASTER PROMPT #28: cache expensive aggregations).

Per section 6, identifiers (User, Card, Merchant Name) are never fed
directly into models — they are only used to *derive* behavioral
statistics (counts, averages, diversity).
"""

from __future__ import annotations

import logging

from src.data_loader import get_duckdb_connection

logger = logging.getLogger(__name__)


def build_user_baselines() -> int:
    """
    Per-user behavioral baseline: average/std transaction amount, typical
    daily frequency, merchant diversity. Used as the reference point for
    anomaly detection ("this transaction is 5.7x the user's average").
    """
    con = get_duckdb_connection()
    try:
        con.execute("""
            CREATE OR REPLACE TABLE user_baselines AS
            SELECT
                user_id,
                COUNT(*)                                   AS txn_count,
                AVG(amount)                                 AS avg_amount,
                STDDEV_SAMP(amount)                          AS std_amount,
                MIN(amount)                                  AS min_amount,
                MAX(amount)                                  AS max_amount,
                COUNT(DISTINCT merchant_name)                AS merchant_diversity,
                COUNT(DISTINCT txn_date)                     AS active_days,
                CAST(COUNT(*) AS DOUBLE) /
                    NULLIF(COUNT(DISTINCT txn_date), 0)      AS avg_daily_frequency
            FROM transactions_final
            GROUP BY user_id
        """)
        n = con.execute("SELECT COUNT(*) FROM user_baselines").fetchone()[0]
        logger.info("Built user_baselines for %s users", n)
        return n
    finally:
        con.close()


def build_merchant_baselines() -> int:
    con = get_duckdb_connection()
    try:
        con.execute("""
            CREATE OR REPLACE TABLE merchant_baselines AS
            SELECT
                merchant_name,
                COUNT(*)              AS txn_count,
                AVG(amount)            AS avg_amount,
                STDDEV_SAMP(amount)     AS std_amount,
                COUNT(DISTINCT user_id) AS unique_customers
            FROM transactions_final
            GROUP BY merchant_name
        """)
        n = con.execute("SELECT COUNT(*) FROM merchant_baselines").fetchone()[0]
        logger.info("Built merchant_baselines for %s merchants", n)
        return n
    finally:
        con.close()


def build_daily_aggregates() -> int:
    """Pre-aggregated daily spend — powers trend charts without a full scan each time."""
    con = get_duckdb_connection()
    try:
        con.execute("""
            CREATE OR REPLACE TABLE daily_spending AS
            SELECT
                txn_date,
                user_id,
                SUM(amount)     AS total_amount,
                COUNT(*)        AS txn_count
            FROM transactions_final
            GROUP BY txn_date, user_id
        """)
        n = con.execute("SELECT COUNT(*) FROM daily_spending").fetchone()[0]
        logger.info("Built daily_spending aggregate: %s rows", n)
        return n
    finally:
        con.close()


def build_all_features() -> dict:
    """Run all feature engineering steps. Called once after cleaning."""
    return {
        "user_baselines": build_user_baselines(),
        "merchant_baselines": build_merchant_baselines(),
        "daily_spending": build_daily_aggregates(),
    }
