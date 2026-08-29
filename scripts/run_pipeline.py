"""
DataLens AI — Pipeline Runner

Runs the full data pipeline: raw CSV -> DuckDB -> cleaned/feature-engineered
Parquet -> behavioral baseline tables. Run this once after placing a CSV at
data/raw/transactions.csv (or DATALENS_RAW_CSV), and again whenever the raw
data changes.

Usage:
    python scripts/run_pipeline.py
"""

import logging
import time

from src.data_cleaner import build_processed_dataset, get_missingness_report
from src.feature_engineering import build_all_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    t0 = time.time()
    logger.info("Phase 1/3: Cleaning raw data and writing processed Parquet...")
    row_count = build_processed_dataset()
    logger.info("Cleaned dataset: %s rows (%.1fs)", row_count, time.time() - t0)

    logger.info("Phase 2/3: Computing actual missingness report...")
    report = get_missingness_report()
    for col, stats in report.items():
        logger.info("  %s: %.2f%% missing (%s rows)", col, stats["missing_pct"], stats["missing_count"])

    logger.info("Phase 3/3: Building behavioral baseline tables (user/merchant/daily)...")
    feature_counts = build_all_features()
    logger.info("Feature tables built: %s", feature_counts)

    logger.info("Pipeline complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
