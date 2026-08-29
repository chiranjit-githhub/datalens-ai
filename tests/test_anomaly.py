"""
Tests for the anomaly detection layer (rule-based, statistical, ML) and the
fraud-label evaluation utility. Requires a processed sample dataset (see
test_analytics.py docstring).
"""

import pytest

from src.data_loader import processed_dataset_ready, load_processed_duckdb_view
from src.anomaly import detector, explanations
from src.analytics import spending

pytestmark = pytest.mark.skipif(
    not processed_dataset_ready(),
    reason="No processed dataset found — run scripts/generate_sample_data.py "
           "and scripts/run_pipeline.py first.",
)


def _any_user_id() -> str:
    con = load_processed_duckdb_view()
    try:
        return con.execute("SELECT user_id FROM transactions_final LIMIT 1").fetchone()[0]
    finally:
        con.close()


def test_detect_rule_based_anomalies_only_flags_amounts_above_multiple_of_average():
    import config
    user_id = _any_user_id()
    r = detector.detect_rule_based_anomalies(user_id, limit=50)
    assert "error" not in r
    for row in r["result"]:
        assert row["deviation_multiple"] >= config.LARGE_AMOUNT_MULTIPLIER
        assert "anomaly_reasons" in row and len(row["anomaly_reasons"]) > 0


def test_detect_statistical_anomalies_returns_baseline_and_flagged_rows():
    r = detector.detect_statistical_anomalies(limit=20)
    assert "error" not in r
    assert "baseline" in r
    assert r["baseline"]["mean"] is not None
    for row in r["result"]:
        assert len(row["anomaly_reasons"]) > 0


def test_detect_statistical_anomalies_scoped_to_missing_user_is_empty():
    r = detector.detect_statistical_anomalies(user_id="NO_SUCH_USER", limit=10)
    assert r["result"] == []


def test_detect_ml_anomalies_returns_bounded_result():
    r = detector.detect_ml_anomalies(sample_limit=5000, top_n=10)
    assert "error" not in r
    assert len(r["result"]) <= 10
    for row in r["result"]:
        assert "anomaly_reasons" in row


def test_detect_anomalies_unified_all_methods_no_crash():
    user_id = _any_user_id()
    r = detector.detect_anomalies(user_id=user_id, method="all", limit=5)
    assert "error" not in r
    assert set(r["result"].keys()) == {"rule_based", "statistical", "ml"}


def test_detect_anomalies_single_method_only_populates_that_key():
    r = detector.detect_anomalies(method="statistical", limit=5)
    assert "error" not in r
    assert list(r["result"].keys()) == ["statistical"]


def test_explain_transaction_anomaly_never_fabricates_missing_fields():
    txn_with_data = {"amount": 500, "z_score": 4.2, "anomaly_reasons": ["unusually high amount"]}
    explanation = explanations.explain_transaction_anomaly(txn_with_data)
    assert "500" in explanation
    assert "4.2" in explanation

    empty_txn = {}
    explanation_empty = explanations.explain_transaction_anomaly(empty_txn)
    assert "No supporting evidence" in explanation_empty


def test_evaluate_against_fraud_label_empty_flags_returns_none_metrics():
    r = explanations.evaluate_against_fraud_label([], method_name="test")
    assert r["result"]["flagged_count"] == 0
    assert r["result"]["precision"] is None


def test_evaluate_against_fraud_label_metrics_are_between_0_and_1():
    r = detector.detect_statistical_anomalies(limit=20)
    rows = r["result"]
    pairs = [(row["user_id"], str(row["txn_datetime"]), row["amount"]) for row in rows]
    result = explanations.evaluate_against_fraud_label(pairs, method_name="statistical")["result"]
    if result["precision"] is not None:
        assert 0 <= result["precision"] <= 1
    if result["recall"] is not None:
        assert 0 <= result["recall"] <= 1


def test_overview_total_spend_is_positive_sanity_check():
    # Sanity cross-check that analytics and anomaly modules see the same data.
    overview = spending.get_overview()["result"]
    assert overview["total_spend"] > 0
