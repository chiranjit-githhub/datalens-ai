"""
DataLens AI — Investigation Planner

Maps a natural-language question to an intent (MASTER PROMPT #17) and an
initial tool-selection plan (MASTER PROMPT #18). This is a lightweight,
deterministic first pass — the LLM orchestrator (agent.py) still drives
the actual multi-step investigation via tool-calling, but the planner
gives it a strong starting point and lets the app work even in fallback
(no-LLM) mode.
"""

from __future__ import annotations
import re

INTENTS = [
"spending_summary",
"period_comparison",
"spending_change",
"savings_recommendation",
"category_analysis",
"merchant_analysis",
"transaction_analysis",
"frequency_analysis",
"recurring_expenses",
"anomaly_detection",
"fraud_analysis",
"trend_analysis",
"location_analysis",
"payment_method_analysis",
"general_data_question",
]

#Ordered: more specific patterns first.

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
(
"anomaly_detection",
[
r"\bunusual\b",
r"\banomal",
r"\bsuspicious\b",
r"\bstrange\b",
r"\bweird\b",
],
),

(
    "fraud_analysis",
    [
        r"\bfraud\b",
    ],
),

(
    "spending_change",
    [
        r"why.*(increase|decrease|change|went up|went down)",
        r"\bincreased?\b",
        r"\bdecreased?\b",
    ],
),

(
    "period_comparison",
    [
        r"\bcompare\b",
        r"\bvs\b",
        r"\bversus\b",
        r"this month.*last month",
    ],
),

# Savings/recommendation questions
(
    "savings_recommendation",
    [
        r"\bsav(e|ing|ings)\b",
        r"\bsave more\b",
        r"\bsavings?\s+(tips?|recommendations?|strategy|opportunities?)\b",
        r"\bhow can i save\b",
        r"\bwhere can i save\b",
        r"\breduce.*spending\b",
        r"\breduce.*expenses?\b",
        r"\bcut.*spending\b",
        r"\bcut.*expenses?\b",
        r"\bsave money\b",
        r"\bspending.*opportunities\b",
        r"\bopportunities.*save\b",
    ],
),

(
    "recurring_expenses",
    [
        r"\brecurring\b",
        r"\bsubscription",
        r"\brepeat",
    ],
),

(
    "frequency_analysis",
    [
        r"\bfrequency\b",
        r"how often",
        r"\bhow many transactions\b",
    ],
),

(
    "category_analysis",
    [
        r"\bcategory\b",
        r"\bcategories\b",
        r"what did i spend on",
    ],
),

(
    "merchant_analysis",
    [
        r"\bmerchant\b",
        r"\bstore\b",
        r"\bvendor\b",
        r"where did i spend",
    ],
),

(
    "location_analysis",
    [
        r"\bcity\b",
        r"\bstate\b",
        r"\blocation\b",
    ],
),

(
    "payment_method_analysis",
    [
        r"\bchip\b",
        r"\bcard\b.*\b(swipe|online)\b",
        r"payment method",
    ],
),

(
    "transaction_analysis",
    [
        r"\bthis transaction\b",
        r"\bspecific transaction\b",
    ],
),

(
    "trend_analysis",
    [
        r"\btrend\b",
        r"\bover time\b",
        r"\bpattern\b",
    ],
),

(
    "spending_summary",
    [
        r"\btotal\b",
        r"\bsummary\b",
        r"how much (have i|did i) spen",
    ],
),

]

_INTENT_TO_PLAN = {
"spending_change": [
"compare_periods",
"compare_category_periods",
"compare_merchant_periods",
"analyze_transaction_frequency",
"analyze_average_transaction",
],

"period_comparison": [
    "compare_periods",
],

"savings_recommendation": [
    "analyze_categories",
    "analyze_merchants",
    "analyze_recurring_expenses",
    "analyze_transaction_frequency",
    "analyze_weekday_vs_weekend",
],

"category_analysis": [
    "analyze_categories",
],

"merchant_analysis": [
    "analyze_merchants",
],

"location_analysis": [
    "analyze_locations",
],

"payment_method_analysis": [
    "analyze_payment_methods",
],

"frequency_analysis": [
    "analyze_transaction_frequency",
],

"recurring_expenses": [
    "analyze_recurring_expenses",
],

"anomaly_detection": [
    "detect_anomalies",
],

"fraud_analysis": [
    "detect_anomalies",
],

"trend_analysis": [
    "analyze_monthly_trend",
    "get_monthly_spending",
],

"transaction_analysis": [
    "get_transaction_detail",
    "compare_user_behavior",
    "compare_merchant_behavior",
],

"spending_summary": [
    "get_overview",
],

"general_data_question": [
    "get_overview",
],

}

def classify_intent(question: str) -> str:
    q = question.lower()
    for intent, patterns in _INTENT_PATTERNS:
        if any(re.search(pattern, q) for pattern in patterns):
            return intent
    return "general_data_question"

def build_initial_plan(question: str) -> dict:
    """
        Returns {"intent": ..., "suggested_tools": ...}.

        This is a suggestion handed to the LLM orchestrator as a starting point
        and is also used by deterministic fallback mode. The LLM may deviate
        from it based on actual evidence.
        """
    intent = classify_intent(question)

    return {
            "intent": intent,
            "suggested_tools": _INTENT_TO_PLAN.get(
                intent,
                ["get_overview"],
            ),
        }
