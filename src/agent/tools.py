"""
DataLens AI — Agent Tool Registry

Wires every analytics/anomaly function into a single registry with:
  - the callable
  - a JSON-schema tool definition (Anthropic tool-use format; trivially
    convertible to OpenAI's function-calling format — see agent.py)

This is the ONLY place that needs to change when a new analytics
capability is added (MASTER PROMPT #32: Notebook Analysis -> Reusable
Function -> Agent Tool).
"""

from __future__ import annotations

from typing import Any, Callable

from src.analytics import spending, categories, merchants, temporal, behavior, transactions
from src.anomaly import detector


def _schema(description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


# name -> (callable, tool schema)
TOOL_REGISTRY: dict[str, tuple[Callable[..., Any], dict]] = {

    "get_overview": (
        spending.get_overview,
        _schema(
            "High-level snapshot: total transactions, total spend, average transaction, "
            "date range, active users/merchants. Optionally scoped to one user.",
            {"user_id": {"type": "string", "description": "Optional user ID to scope to"}},
        ),
    ),
    "compare_periods": (
        spending.compare_periods,
        _schema(
            "Compare total spending between two date ranges. Use this FIRST for any "
            "'why did spending change' question to confirm the change actually happened.",
            {
                "start_date_1": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date_1": {"type": "string", "description": "YYYY-MM-DD"},
                "start_date_2": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date_2": {"type": "string", "description": "YYYY-MM-DD"},
                "user_id": {"type": "string"},
            },
            ["start_date_1", "end_date_1", "start_date_2", "end_date_2"],
        ),
    ),
    "get_monthly_spending": (
        spending.get_monthly_spending,
        _schema("Total spend grouped by year-month. Good for trend charts.",
                {"user_id": {"type": "string"}, "year": {"type": "integer"}}),
    ),
    "get_daily_spending": (
        spending.get_daily_spending,
        _schema("Total spend grouped by calendar date, optionally bounded by a date range.",
                {"user_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}),
    ),
    "find_spending_changes": (
        spending.find_spending_changes,
        _schema("Find the months with the largest month-over-month spending swings. "
                "Use when the user asks about a change but gives no explicit period.",
                {"user_id": {"type": "string"}, "top_n": {"type": "integer"}}),
    ),

    "analyze_categories": (
        categories.analyze_categories,
        _schema("Spend broken down by merchant category (MCC).",
                {"user_id": {"type": "string"}, "start_date": {"type": "string"},
                 "end_date": {"type": "string"}, "top_n": {"type": "integer"}}),
    ),
    "compare_category_periods": (
        categories.compare_category_periods,
        _schema("Which category contributed most to a spending change between two periods. "
                "Use as the SECOND step after compare_periods confirms a change.",
                {"start_date_1": {"type": "string"}, "end_date_1": {"type": "string"},
                 "start_date_2": {"type": "string"}, "end_date_2": {"type": "string"},
                 "user_id": {"type": "string"}, "top_n": {"type": "integer"}},
                ["start_date_1", "end_date_1", "start_date_2", "end_date_2"]),
    ),

    "analyze_merchants": (
        merchants.analyze_merchants,
        _schema("Top merchants by total spend.",
                {"user_id": {"type": "string"}, "start_date": {"type": "string"},
                 "end_date": {"type": "string"}, "top_n": {"type": "integer"}}),
    ),
    "compare_merchant_periods": (
        merchants.compare_merchant_periods,
        _schema("Which merchants contributed most to a spending change between two periods.",
                {"start_date_1": {"type": "string"}, "end_date_1": {"type": "string"},
                 "start_date_2": {"type": "string"}, "end_date_2": {"type": "string"},
                 "user_id": {"type": "string"}, "top_n": {"type": "integer"}},
                ["start_date_1", "end_date_1", "start_date_2", "end_date_2"]),
    ),
    "analyze_locations": (
        merchants.analyze_locations,
        _schema("Spend broken down by merchant city/state.",
                {"user_id": {"type": "string"}, "top_n": {"type": "integer"}}),
    ),
    "analyze_payment_methods": (
        merchants.analyze_payment_methods,
        _schema("Spend broken down by payment channel (chip/swipe/online).",
                {"user_id": {"type": "string"}}),
    ),

    "analyze_weekday_vs_weekend": (
        temporal.analyze_weekday_vs_weekend,
        _schema("Compares total & average spend on weekdays vs weekends.",
                {"user_id": {"type": "string"}}),
    ),
    "analyze_peak_hours": (
        temporal.analyze_peak_hours,
        _schema("Hours of day with the highest transaction volume/spend.",
                {"user_id": {"type": "string"}, "top_n": {"type": "integer"}}),
    ),
    "analyze_time_of_day": (
        temporal.analyze_time_of_day,
        _schema("Spend broken down by Morning/Afternoon/Evening/Night.",
                {"user_id": {"type": "string"}}),
    ),
    "analyze_monthly_trend": (
        temporal.analyze_monthly_trend,
        _schema("Month-over-month spend trend with percentage change.",
                {"user_id": {"type": "string"}}),
    ),

    "analyze_transaction_frequency": (
        behavior.analyze_transaction_frequency,
        _schema("Transaction count/frequency, optionally compared between two periods. "
                "Use to test whether a spending change was driven by frequency vs transaction size.",
                {"user_id": {"type": "string"},
                 "start_date_1": {"type": "string"}, "end_date_1": {"type": "string"},
                 "start_date_2": {"type": "string"}, "end_date_2": {"type": "string"}}),
    ),
    "analyze_average_transaction": (
        behavior.analyze_average_transaction,
        _schema("Average transaction size, optionally compared between two periods.",
                {"user_id": {"type": "string"},
                 "start_date_1": {"type": "string"}, "end_date_1": {"type": "string"},
                 "start_date_2": {"type": "string"}, "end_date_2": {"type": "string"}}),
    ),
    "analyze_merchant_diversity": (
        behavior.analyze_merchant_diversity,
        _schema("Number of distinct merchants transacted with.",
                {"user_id": {"type": "string"}}),
    ),
    "analyze_recurring_expenses": (
        behavior.analyze_recurring_expenses,
        _schema("Identifies merchants a user transacts with repeatedly (recurring expenses).",
                {"user_id": {"type": "string"}, "min_occurrences": {"type": "integer"}},
                ["user_id"]),
    ),

    "get_transaction_detail": (
        transactions.get_transaction_detail,
        _schema("Fetch a specific transaction plus the user's baseline stats.",
                {"user_id": {"type": "string"}, "txn_datetime": {"type": "string"},
                 "amount": {"type": "number"}},
                ["user_id", "txn_datetime", "amount"]),
    ),
    "compare_user_behavior": (
        transactions.compare_user_behavior,
        _schema("Compares a given amount against a user's own historical average/std.",
                {"user_id": {"type": "string"}, "amount": {"type": "number"}},
                ["user_id", "amount"]),
    ),
    "compare_merchant_behavior": (
        transactions.compare_merchant_behavior,
        _schema("Compares a given amount against a merchant's typical transaction size.",
                {"merchant_name": {"type": "string"}, "amount": {"type": "number"}},
                ["merchant_name", "amount"]),
    ),

    "detect_anomalies": (
        detector.detect_anomalies,
        _schema("Detects anomalous transactions using rule-based, statistical (z-score/IQR), "
                "and/or Isolation Forest methods. method: 'rule' | 'statistical' | 'ml' | 'all'.",
                {"user_id": {"type": "string"}, "method": {"type": "string"}, "limit": {"type": "integer"}}),
    ),
}


def get_tool_schemas() -> list[dict]:
    """Anthropic tool-use format: [{name, description, input_schema}, ...]"""
    return [
        {"name": name, **schema}
        for name, (_, schema) in TOOL_REGISTRY.items()
    ]


def get_openai_tool_schemas() -> list[dict]:
    """OpenAI function-calling format wrapper."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for name, (_, schema) in TOOL_REGISTRY.items()
    ]


def call_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call by name. Never raises — errors come back as {'error': ...}."""
    if name not in TOOL_REGISTRY:
        return {"tool": name, "error": f"Unknown tool: {name}"}
    func, _ = TOOL_REGISTRY[name]
    try:
        return func(**arguments)
    except TypeError as e:
        return {"tool": name, "error": f"Invalid arguments: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"tool": name, "error": f"Tool execution failed: {e}"}
