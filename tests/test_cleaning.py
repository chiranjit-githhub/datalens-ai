"""
Tests for the cleaning SQL logic. Uses a small in-memory DuckDB table so
these run fast and don't depend on any real dataset being present.
"""

import duckdb
import pandas as pd
import pytest

from src.data_cleaner import _build_cleaning_sql


@pytest.fixture
def raw_con():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE raw_transactions (
            "User" VARCHAR, "Card" INTEGER, "Year" INTEGER, "Month" INTEGER, "Day" INTEGER,
            "Time" VARCHAR, "Amount" VARCHAR, "Use Chip" VARCHAR, "Merchant Name" VARCHAR,
            "Merchant City" VARCHAR, "Merchant State" VARCHAR, "Zip" VARCHAR, "MCC" INTEGER,
            "Errors?" VARCHAR, "Is Fraud?" VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO raw_transactions VALUES
        ('U001', 0, 2023, 5, 14, '13:45', '$134.09', 'Chip Transaction', 'Whole Foods',
         'New York', 'NY', '10001', 5411, NULL, 'No'),
        ('U001', 0, 2023, 5, 15, '09:10', '$38.48', 'Swipe Transaction', 'Shell Gas',
         'New York', 'NY', '10001', 5541, 'Bad PIN', 'No'),
        ('U002', 1, 2023, 5, 16, '22:05', '$1,204.34', 'Online Transaction', 'Amazon',
         NULL, NULL, NULL, 5999, NULL, 'Yes')
    """)
    yield con
    con.close()


def test_amount_parsing_strips_currency_symbols(raw_con):
    sql = _build_cleaning_sql()
    df = raw_con.execute(sql).fetchdf()
    amounts = sorted(df["amount"].tolist())
    assert amounts == pytest.approx([38.48, 134.09, 1204.34])


def test_date_combines_year_month_day(raw_con):
    sql = _build_cleaning_sql()
    df = raw_con.execute(sql).fetchdf()
    row = df[df["user_id"] == "U001"].iloc[0]
    assert row["txn_date"].strftime("%Y-%m-%d") == "2023-05-14"


def test_has_error_flag_set_correctly(raw_con):
    sql = _build_cleaning_sql()
    df = raw_con.execute(sql).fetchdf()
    row_with_error = df[df["errors_raw"] == "Bad PIN"].iloc[0]
    row_without_error = df[df["user_id"] == "U001"].iloc[0]
    assert row_with_error["has_error"] == True  # noqa: E712
    assert (df[(df["user_id"] == "U001") & (df["errors_raw"].isnull())]["has_error"] == False).all()  # noqa: E712


def test_missing_state_and_zip_default_to_unknown(raw_con):
    sql = _build_cleaning_sql()
    df = raw_con.execute(sql).fetchdf()
    row = df[df["user_id"] == "U002"].iloc[0]
    assert row["merchant_state"] == "Unknown"
    assert row["zip_code"] == "Unknown"
    # original missingness preserved separately
    assert pd.isna(row["merchant_state_raw"])


def test_is_fraud_boolean_conversion(raw_con):
    sql = _build_cleaning_sql()
    df = raw_con.execute(sql).fetchdf()
    fraud_row = df[df["user_id"] == "U002"].iloc[0]
    not_fraud_row = df[df["user_id"] == "U001"].iloc[0]
    assert fraud_row["is_fraud"] == True  # noqa: E712
    assert not_fraud_row["is_fraud"] == False  # noqa: E712
