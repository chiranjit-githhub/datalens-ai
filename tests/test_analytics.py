"""
Tests for the analytics tool layer. These run against whatever processed
dataset currently exists at config.DUCKDB_PATH — run
`python scripts/generate_sample_data.py` and `python scripts/run_pipeline.py`
before running this file (see README "Running Tests").

Tests assert on structure/invariants rather than exact business values,
since the underlying data may be the synthetic sample or the real dataset.
"""

import pytest

from src.data_loader import processed_dataset_ready
from src.analytics import spending, categories, merchants, temporal, behavior

pytestmark = pytest.mark.skipif(
    not processed_dataset_ready(),
    reason="No processed dataset found — run scripts/generate_sample_data.py "
           "and scripts/run_pipeline.py first.",
)


def test_get_overview_returns_expected_keys():
    r = spending.get_overview()
    assert "error" not in r
    result = r["result"]
    for key in ("total_transactions", "total_spend", "avg_transaction", "start_date", "end_date"):
        assert key in result
    assert result["total_transactions"] > 0


def test_compare_periods_totals_are_consistent():
    overview = spending.get_overview()["result"]
    start, end = str(overview["start_date"]), str(overview["end_date"])
    r = spending.compare_periods(start, end, start, end)
    assert "error" not in r
    p1 = r["result"]["period_1"]["total"]
    p2 = r["result"]["period_2"]["total"]
    # Same period compared to itself: totals must match and change must be zero.
    assert p1 == p2
    assert r["result"]["absolute_change"] == 0


def test_compare_periods_percentage_change_is_none_when_period_1_is_zero():
    r = spending.compare_periods("1900-01-01", "1900-01-02", "2023-01-01", "2023-01-31")
    assert r["result"]["period_1"]["total"] in (0, None)
    assert r["result"]["percentage_change"] is None


def test_get_monthly_spending_returns_rows_sorted_by_period():
    r = spending.get_monthly_spending()
    rows = r["result"]
    assert len(rows) > 0
    periods = [(row["year"], row["month"]) for row in rows]
    assert periods == sorted(periods)


def test_analyze_categories_sorted_descending_by_total():
    r = categories.analyze_categories(top_n=10)
    rows = r["result"]
    assert len(rows) > 0
    totals = [row["total"] for row in rows]
    assert totals == sorted(totals, reverse=True)
    assert all("category_label" in row for row in rows)


def test_analyze_merchants_respects_top_n():
    r = merchants.analyze_merchants(top_n=3)
    assert len(r["result"]) <= 3


def test_analyze_weekday_vs_weekend_covers_all_transactions():
    r = temporal.analyze_weekday_vs_weekend()
    result = r["result"]
    overview = spending.get_overview()["result"]
    total_txn_count = result["weekday"]["txn_count"] + result["weekend"]["txn_count"]
    assert total_txn_count == overview["total_transactions"]


def test_analyze_transaction_frequency_basic_mode():
    r = behavior.analyze_transaction_frequency()
    result = r["result"]
    assert result["total_txn_count"] > 0
    assert result["active_days"] > 0


def test_tool_error_handling_on_bad_user_id_returns_empty_not_crash():
    r = merchants.analyze_merchants(user_id="THIS_USER_DOES_NOT_EXIST")
    assert "error" not in r
    assert r["result"] == []


def test_find_spending_changes_change_matches_diff():
    r = spending.find_spending_changes(top_n=5)
    for row in r["result"]:
        assert row["change"] == pytest.approx(round(row["total"] - row["prev_total"], 2), abs=0.01)
