"""
DataLens AI — Data Loader

Responsible ONLY for getting raw data off disk and into a queryable form.
Cleaning and feature engineering live in data_cleaner.py / feature_engineering.py.

Design principle (see MASTER PROMPT #27, #41):
    Raw CSV -> DuckDB / Parquet -> Analytical Query -> Small Structured Result -> LLM
We never load the full multi-million-row dataset into the LLM's context, and we
avoid repeated full-table pandas scans by pushing aggregation down into DuckDB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "User", "Card", "Year", "Month", "Day", "Time", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
    "Zip", "MCC", "Errors?", "Is Fraud?",
]


class DataLoadError(Exception):
    """Raised when the raw dataset cannot be loaded or is structurally invalid."""


def get_duckdb_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection backed by the on-disk database file."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)


def validate_raw_schema(csv_path: Path) -> list[str]:
    """
    Peek at the header of the raw CSV and report any expected columns that
    are missing. Does NOT load the full file.
    """
    if not csv_path.exists():
        raise DataLoadError(f"Raw dataset not found at: {csv_path}")

    header_df = pd.read_csv(csv_path, nrows=0)
    actual_columns = list(header_df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
    if missing:
        logger.warning("Raw CSV is missing expected columns: %s", missing)
    return missing


def load_raw_csv_to_duckdb(
    csv_path: Optional[Path] = None,
    table_name: str = "raw_transactions",
) -> int:
    """
    Stream the raw CSV directly into a DuckDB table using DuckDB's native
    CSV reader (fast, out-of-core — never materializes the whole file in
    Python/pandas memory). Returns the row count loaded.
    """
    csv_path = Path(csv_path or config.RAW_CSV_PATH)
    validate_raw_schema(csv_path)

    con = get_duckdb_connection()
    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto(?, header=True, sample_size=-1)
        """, [str(csv_path)])
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("Loaded %s rows into DuckDB table '%s'", count, table_name)
        return count
    finally:
        con.close()


def load_sample_pandas(csv_path: Optional[Path] = None, n_rows: int = 50_000) -> pd.DataFrame:
    """
    Load a bounded sample of the raw CSV into pandas — useful for quick
    interactive inspection (e.g. in the notebook) without paying the cost
    of a full 24M-row read.
    """
    csv_path = Path(csv_path or config.RAW_CSV_PATH)
    if not csv_path.exists():
        raise DataLoadError(f"Raw dataset not found at: {csv_path}")
    return pd.read_csv(csv_path, nrows=n_rows)


def processed_parquet_exists() -> bool:
    return config.PROCESSED_PARQUET_PATH.exists()


def processed_dataset_ready() -> bool:
    """True once the pipeline has produced a queryable transactions_final table."""
    if not config.DUCKDB_PATH.exists():
        return False
    try:
        con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
        try:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            return "transactions_final" in tables
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return False


def load_processed_duckdb_view() -> duckdb.DuckDBPyConnection:
    """
    Return a read-only DuckDB connection to the persistent processed
    database — the same database the pipeline wrote `transactions_final`
    and the baseline tables (`user_baselines`, `merchant_baselines`,
    `daily_spending`) into. This is the single entry point every analytics
    and anomaly tool uses, so all tools share one consistent source of
    truth without repeated full scans of the raw data.

    Query `transactions_final` (not a generic "transactions" alias) so the
    table name is unambiguous across the codebase.
    """
    if not config.DUCKDB_PATH.exists():
        raise DataLoadError(
            "Processed dataset not found. Run the pipeline (scripts/run_pipeline.py) "
            "or call src.data_cleaner.build_processed_dataset() first."
        )
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "transactions_final" not in existing_tables:
        con.close()
        raise DataLoadError(
            "Processed dataset exists but has not been cleaned yet. "
            "Run the pipeline (scripts/run_pipeline.py) first."
        )
    return con


def get_row_count() -> int:
    con = load_processed_duckdb_view()
    try:
        return con.execute("SELECT COUNT(*) FROM transactions_final").fetchone()[0]
    finally:
        con.close()
