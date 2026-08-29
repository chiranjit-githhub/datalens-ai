"""
DataLens AI — Data Cleaner

Implements the cleaning rules from the MASTER PROMPT (section 9):
  - Amount: "$134.09" -> 134.09 (numeric)
  - Date: Year/Month/Day -> Date
  - DateTime: Date + Time -> DateTime
  - Errors?: extremely sparse -> Has_Error flag, original text preserved
  - Zip / Merchant State: missingness preserved, "Unknown" only for display/grouping

All transformations are pushed into DuckDB SQL so they scale to the full
~24.4M row dataset without ever materializing it in pandas.
"""

from __future__ import annotations

import logging

import duckdb

import config
from src.data_loader import get_duckdb_connection, load_raw_csv_to_duckdb

logger = logging.getLogger(__name__)


def _build_cleaning_sql(raw_table: str = "raw_transactions") -> str:
    """
    Returns the SQL that transforms the raw table into the cleaned,
    feature-engineered table. Written defensively: TRY_CAST / regex
    stripping rather than assuming a fixed currency format, per the
    MASTER PROMPT instruction not to hard-code formatting assumptions
    without inspecting the data.
    """
    return f"""
        SELECT
            "User"                                                          AS user_id,
            "Card"                                                          AS card_id,
            "Year"                                                          AS year,
            "Month"                                                         AS month,
            "Day"                                                           AS day,
            "Time"                                                          AS time_raw,

            -- Amount: strip any non-numeric currency formatting, then cast
            TRY_CAST(
                regexp_replace(CAST("Amount" AS VARCHAR), '[^0-9.\\-]', '', 'g')
                AS DOUBLE
            )                                                                AS amount,

            "Use Chip"                                                      AS use_chip,
            "Merchant Name"                                                 AS merchant_name,
            "Merchant City"                                                 AS merchant_city,
            "Merchant State"                                                AS merchant_state_raw,
            COALESCE(CAST("Merchant State" AS VARCHAR), 'Unknown')          AS merchant_state,

            "Zip"                                                           AS zip_raw,
            COALESCE(CAST("Zip" AS VARCHAR), 'Unknown')                     AS zip_code,

            "MCC"                                                           AS mcc,

            "Errors?"                                                       AS errors_raw,
            CASE WHEN "Errors?" IS NULL OR CAST("Errors?" AS VARCHAR) = ''
                 THEN FALSE ELSE TRUE END                                   AS has_error,

            CASE WHEN CAST("Is Fraud?" AS VARCHAR) IN ('Yes', 'yes', '1', 'True', 'true')
                 THEN TRUE ELSE FALSE END                                   AS is_fraud,

            -- Date: combine Year/Month/Day
            TRY_CAST(
                make_date(TRY_CAST("Year" AS INTEGER), TRY_CAST("Month" AS INTEGER), TRY_CAST("Day" AS INTEGER))
                AS DATE
            )                                                                AS txn_date,

            -- DateTime: Date + Time (Time expected as HH:MM); tolerate HH:MM:SS too
            TRY_CAST(
                (make_date(TRY_CAST("Year" AS INTEGER), TRY_CAST("Month" AS INTEGER), TRY_CAST("Day" AS INTEGER))
                 || ' ' || CAST("Time" AS VARCHAR))
                AS TIMESTAMP
            )                                                                AS txn_datetime,

            TRY_CAST(split_part(CAST("Time" AS VARCHAR), ':', 1) AS INTEGER) AS hour

        FROM {raw_table}
    """


def build_processed_dataset(
    csv_path=None,
    load_raw: bool = True,
) -> int:
    """
    Full pipeline: raw CSV -> cleaned/feature-engineered table -> Parquet.
    Returns the number of rows written.
    """
    if load_raw:
        load_raw_csv_to_duckdb(csv_path=csv_path)

    con = get_duckdb_connection()
    try:
        cleaning_sql = _build_cleaning_sql()
        con.execute(f"CREATE OR REPLACE TABLE cleaned_transactions AS {cleaning_sql}")

        # Derived temporal features (section 9: Hour, DayOfWeek, IsWeekend, TimeOfDay)
        con.execute("""
            CREATE OR REPLACE TABLE transactions_final AS
            SELECT
                *,
                dayname(txn_date)                                            AS day_of_week,
                CASE WHEN dayofweek(txn_date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend,
                CASE
                    WHEN hour BETWEEN 5 AND 11  THEN 'Morning'
                    WHEN hour BETWEEN 12 AND 16 THEN 'Afternoon'
                    WHEN hour BETWEEN 17 AND 20 THEN 'Evening'
                    ELSE 'Night'
                END                                                          AS time_of_day
            FROM cleaned_transactions
            WHERE amount IS NOT NULL AND txn_date IS NOT NULL
        """)

        row_count = con.execute("SELECT COUNT(*) FROM transactions_final").fetchone()[0]
        dropped = con.execute("SELECT COUNT(*) FROM cleaned_transactions").fetchone()[0] - row_count
        if dropped:
            logger.warning(
                "Dropped %s rows with unparseable amount/date during cleaning "
                "(this count should be reported to the user, not hidden).",
                dropped,
            )

        # Export to Parquet — the canonical artifact all analytics tools query
        con.execute(f"""
            COPY transactions_final TO '{config.PROCESSED_PARQUET_PATH}' (FORMAT PARQUET)
        """)
        logger.info("Wrote %s cleaned rows to %s", row_count, config.PROCESSED_PARQUET_PATH)
        return row_count
    finally:
        con.close()


def get_missingness_report() -> dict:
    """
    Calculate ACTUAL missingness percentages from the processed dataset.
    Per MASTER PROMPT section 5: the numbers in the spec are reference-only;
    always compute the real values.
    """
    con = get_duckdb_connection(read_only=True)
    try:
        total = con.execute("SELECT COUNT(*) FROM transactions_final").fetchone()[0]
        if total == 0:
            return {}
        cols = ["errors_raw", "zip_raw", "merchant_state_raw"]
        report = {}
        for col in cols:
            missing = con.execute(
                f"SELECT COUNT(*) FROM transactions_final WHERE {col} IS NULL"
            ).fetchone()[0]
            report[col] = {
                "missing_count": missing,
                "missing_pct": round(100.0 * missing / total, 2),
            }
        return report
    finally:
        con.close()
