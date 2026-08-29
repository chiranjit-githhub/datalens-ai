"""
DataLens AI — Visualization

Every function takes a tool's structured `result` (a list of dicts or a
dict) and returns a Plotly figure. Charts are only ever built from real
computed evidence — never from placeholder/fabricated data.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def spending_trend_chart(monthly_rows: list[dict]) -> go.Figure:
    """Line chart of spend over year-month. Expects rows from get_monthly_spending."""
    if not monthly_rows:
        return go.Figure()
    df = pd.DataFrame(monthly_rows)
    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    fig = px.line(df, x="period", y="total", markers=True, title="Spending Trend")
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Spend")
    return fig


def category_contribution_chart(category_rows: list[dict]) -> go.Figure:
    """Bar chart of spend by category. Expects rows from analyze_categories."""
    if not category_rows:
        return go.Figure()
    df = pd.DataFrame(category_rows)
    label_col = "category_label" if "category_label" in df.columns else "mcc"
    fig = px.bar(df, x=label_col, y="total", title="Spending by Category")
    fig.update_layout(xaxis_title="Category", yaxis_title="Total Spend")
    return fig


def merchant_ranking_chart(merchant_rows: list[dict]) -> go.Figure:
    """Horizontal bar chart of top merchants by spend."""
    if not merchant_rows:
        return go.Figure()
    df = pd.DataFrame(merchant_rows).sort_values("total_spend")
    fig = px.bar(df, x="total_spend", y="merchant_name", orientation="h",
                 title="Top Merchants by Spend")
    fig.update_layout(xaxis_title="Total Spend", yaxis_title="Merchant")
    return fig


def weekday_weekend_chart(comparison: dict) -> go.Figure:
    """Bar chart comparing weekday vs weekend spend."""
    weekday = comparison.get("weekday", {})
    weekend = comparison.get("weekend", {})
    df = pd.DataFrame([
        {"period": "Weekday", "total": weekday.get("total", 0)},
        {"period": "Weekend", "total": weekend.get("total", 0)},
    ])
    fig = px.bar(df, x="period", y="total", title="Weekday vs Weekend Spending")
    return fig


def transaction_distribution_chart(amounts: list[float], flagged_amount: float | None = None) -> go.Figure:
    """Histogram of a user's transaction amounts, optionally marking one transaction."""
    if not amounts:
        return go.Figure()
    fig = px.histogram(x=amounts, nbins=40, title="Transaction Amount Distribution")
    fig.update_layout(xaxis_title="Amount", yaxis_title="Count")
    if flagged_amount is not None:
        fig.add_vline(x=flagged_amount, line_color="red", line_dash="dash",
                       annotation_text="This transaction")
    return fig


def anomaly_scatter_chart(rows: list[dict]) -> go.Figure:
    """Scatter of anomalous transactions: time vs amount."""
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    y_col = "amount" if "amount" in df.columns else None
    x_col = "txn_datetime" if "txn_datetime" in df.columns else "txn_date"
    if not y_col or x_col not in df.columns:
        return go.Figure()
    fig = px.scatter(df, x=x_col, y=y_col, title="Flagged Anomalous Transactions",
                      hover_data=[c for c in ["merchant_name", "mcc"] if c in df.columns])
    return fig
