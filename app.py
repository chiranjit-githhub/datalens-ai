"""
DataLens AI — Streamlit Application

An AI-powered financial data investigation agent. See README.md for setup.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.data_loader import processed_dataset_ready, load_processed_duckdb_view, DataLoadError
from src.data_cleaner import build_processed_dataset, get_missingness_report
from src.feature_engineering import build_all_features
from src.agent.agent import investigate
from src.visualization import charts

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title=config.APP_NAME, page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": ..., "result": InvestigationResult}


# ---------------------------------------------------------------------------
# Sidebar — dataset setup
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 " + config.APP_NAME)
    st.caption(config.APP_TAGLINE)

    st.divider()
    st.subheader("Dataset")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        save_path = config.RAW_DATA_DIR / "transactions.csv"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to {save_path}")

    if st.button("⚙️ Run / Refresh Pipeline", use_container_width=True):
        if not config.RAW_CSV_PATH.exists():
            st.error(f"No raw CSV found at {config.RAW_CSV_PATH}. Upload one above first.")
        else:
            with st.spinner("Cleaning data, building features... this may take a while for large files."):
                try:
                    row_count = build_processed_dataset()
                    build_all_features()
                    st.success(f"Pipeline complete: {row_count:,} rows processed.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Pipeline failed: {e}")

    st.divider()
    llm_status = "🟢 Connected" if (
        (config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY)
        or (config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY)
        or (config.LLM_PROVIDER == "ollama")
    ) else "🟡 Fallback mode (no LLM configured)"
    st.caption(f"Agent LLM: **{config.LLM_PROVIDER}** — {llm_status}")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Guard: dataset must be processed before anything else works
# ---------------------------------------------------------------------------
if not processed_dataset_ready():
    st.title(config.APP_NAME)
    st.caption(config.APP_TAGLINE)
    st.info(
        "👋 No processed dataset found yet. Upload a transactions CSV in the sidebar "
        "and click **Run / Refresh Pipeline** to get started.\n\n"
        "No dataset handy? Generate a synthetic sample with:\n\n"
        "`python scripts/generate_sample_data.py --rows 200000`\n\n"
        "then place it at `data/raw/transactions.csv` and run the pipeline."
    )
    st.stop()

try:
    _con = load_processed_duckdb_view()
    overview_row = _con.execute("""
        SELECT COUNT(*) AS total_transactions, ROUND(SUM(amount),2) AS total_spend,
               MIN(txn_date) AS start_date, MAX(txn_date) AS end_date,
               COUNT(DISTINCT user_id) AS active_users
        FROM transactions_final
    """).fetchdf().iloc[0]
    _con.close()
except DataLoadError as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title(config.APP_NAME)
st.caption(config.APP_TAGLINE)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", f"{int(overview_row['total_transactions']):,}")
k2.metric("Total Spend", f"${overview_row['total_spend']:,.0f}")
k3.metric("Active Users", f"{int(overview_row['active_users']):,}")
k4.metric("Date Range", f"{overview_row['start_date']} → {overview_row['end_date']}")

tab_chat, tab_dashboard = st.tabs(["💬 Investigate", "📊 Dashboard"])

# ---------------------------------------------------------------------------
# Investigate tab — the chat / agent interface
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask DataLens")
    example_qs = [
        "Why did my spending increase last month?",
        "Show me unusual transactions.",
        "Which merchant did I spend the most on?",
        "Was my spending higher on weekends?",
        "What are my recurring expenses?",
    ]
    cols = st.columns(len(example_qs))
    clicked_example = None
    for c, q in zip(cols, example_qs):
        if c.button(q, use_container_width=True):
            clicked_example = q

    user_question = st.chat_input("Ask a question about your financial data...") or clicked_example

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["result"].answer)
            if turn["result"].trace:
                with st.expander("🔍 Investigation trace"):
                    for step in turn["result"].trace:
                        st.write(f"✓ {step}")
            if turn["result"].mode == "fallback":
                st.caption("⚠️ Answered in deterministic fallback mode (no LLM connected).")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("🔎 Investigating..."):
                result = investigate(user_question)
            st.markdown(result.answer)
            if result.trace:
                with st.expander("🔍 Investigation trace", expanded=True):
                    for step in result.trace:
                        st.write(f"✓ {step}")

            # Render a chart for the most chart-worthy evidence item, if any
            for ev in result.evidence:
                tool = ev.get("source_tool")
                value = ev.get("value")
                try:
                    if tool in ("get_monthly_spending",) and value:
                        st.plotly_chart(charts.spending_trend_chart(value), use_container_width=True)
                        break
                    if tool in ("analyze_categories", "compare_category_periods") and value:
                        st.plotly_chart(charts.category_contribution_chart(value), use_container_width=True)
                        break
                    if tool in ("analyze_merchants", "compare_merchant_periods") and value:
                        rows = [{"merchant_name": r.get("merchant_name"),
                                 "total_spend": r.get("total_spend", r.get("period_2_total", 0))}
                                for r in value]
                        st.plotly_chart(charts.merchant_ranking_chart(rows), use_container_width=True)
                        break
                    if tool == "analyze_weekday_vs_weekend" and value:
                        st.plotly_chart(charts.weekday_weekend_chart(value), use_container_width=True)
                        break
                except Exception:  # noqa: BLE001
                    pass  # chart is best-effort; never break the answer over a viz failure

            if result.error:
                st.caption(f"⚠️ {result.error}")

            st.session_state.chat_history.append({"question": user_question, "result": result})


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Overview Dashboard")

    con = load_processed_duckdb_view()
    try:
        monthly = con.execute("""
            SELECT year, month, ROUND(SUM(amount),2) AS total, COUNT(*) AS txn_count
            FROM transactions_final GROUP BY year, month ORDER BY year, month
        """).fetchdf()
        top_merchants = con.execute("""
            SELECT merchant_name, ROUND(SUM(amount),2) AS total_spend, COUNT(*) AS txn_count
            FROM transactions_final GROUP BY merchant_name ORDER BY total_spend DESC LIMIT 10
        """).fetchdf()
        by_category = con.execute("""
            SELECT mcc, ROUND(SUM(amount),2) AS total FROM transactions_final
            GROUP BY mcc ORDER BY total DESC LIMIT 12
        """).fetchdf()
        fraud_rate = con.execute("""
            SELECT ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS pct
            FROM transactions_final
        """).fetchdf().iloc[0]["pct"]
    finally:
        con.close()

    c1, c2 = st.columns(2)
    with c1:
        if not monthly.empty:
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            st.plotly_chart(px.line(monthly, x="period", y="total", markers=True,
                                     title="Spending Over Time"), use_container_width=True)
    with c2:
        if not by_category.empty:
            st.plotly_chart(px.bar(by_category, x="mcc", y="total", title="Spending by MCC"),
                             use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not top_merchants.empty:
            st.plotly_chart(px.bar(top_merchants.sort_values("total_spend"), x="total_spend",
                                    y="merchant_name", orientation="h", title="Top Merchants"),
                             use_container_width=True)
    with c4:
        st.metric("Fraud Rate (labeled)", f"{fraud_rate}%" if fraud_rate is not None else "N/A")
        st.caption("Based on the `Is Fraud?` label in the dataset, for reference only — "
                   "DataLens's anomaly detection is independent of this label.")

    with st.expander("📋 Data quality report"):
        report = get_missingness_report()
        if report:
            st.json(report)
        else:
            st.write("Run the pipeline to generate a missingness report.")
