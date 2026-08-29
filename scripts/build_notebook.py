"""
Builds notebooks/DataLens_Analytics.ipynb programmatically. Run once:
    PYTHONPATH=. python scripts/build_notebook.py
Kept as a script (rather than hand-edited JSON) so the notebook can be
regenerated cleanly if the analytics API changes.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
# 1. Project Introduction
# ---------------------------------------------------------------------------
md("""# DataLens AI — Analytics Foundation

**An AI-powered financial data investigation agent that doesn't just answer questions — it investigates them.**

This notebook is not a generic EDA notebook. It documents and validates the exact
analytical capabilities that later power the DataLens AI agent as tools
(see `src/analytics/`, `src/anomaly/`). Every function demonstrated here has a
1:1 counterpart registered in `src/agent/tools.py` and callable by the LLM
orchestrator in `src/agent/agent.py`.

Pipeline used throughout: **raw CSV → DuckDB → cleaned/feature-engineered Parquet
→ analytical queries → small structured results**. We never load the full
multi-million-row dataset into pandas or into an LLM's context.
""")

code("""import sys
sys.path.insert(0, "..")

import pandas as pd
import plotly.express as px

import config
from src.data_loader import load_raw_csv_to_duckdb, load_sample_pandas, validate_raw_schema
from src.data_cleaner import build_processed_dataset, get_missingness_report
from src.feature_engineering import build_all_features
from src.data_loader import load_processed_duckdb_view

pd.set_option("display.max_columns", None)
print("Raw CSV configured at:", config.RAW_CSV_PATH)
""")

# ---------------------------------------------------------------------------
# 2. Dataset Understanding
# ---------------------------------------------------------------------------
md("""## 2. Dataset Understanding

We first inspect the **actual** structure of whatever CSV is present — we never
assume the reference schema/percentages from the project brief are correct for
the data actually loaded here.
""")

code("""missing_expected = validate_raw_schema(config.RAW_CSV_PATH)
print("Missing expected columns (should be empty for the standard schema):", missing_expected)

sample = load_sample_pandas(n_rows=20000)
print("Sample shape:", sample.shape)
sample.head()
""")

code("""sample.dtypes
""")

code("""print("Sample date range (Year/Month/Day):")
print("Year:", sample['Year'].min(), "-", sample['Year'].max())
print()
print("Distinct users in sample:", sample['User'].nunique())
print("Distinct cards in sample:", sample['Card'].nunique())
print("Distinct merchants in sample:", sample['Merchant Name'].nunique())
""")

# ---------------------------------------------------------------------------
# 3. Data Quality
# ---------------------------------------------------------------------------
md("""## 3. Data Quality

Actual missingness on the sample, computed directly — not assumed from the brief.
""")

code("""missingness_sample = (sample.isna().mean() * 100).round(2).sort_values(ascending=False)
missingness_sample.to_frame("pct_missing")
""")

md("""Note `Errors?` is expected to be extremely sparse (most transactions have no
error). Per the cleaning rules, we do **not** drop this column — we preserve the
raw error text and derive a `Has_Error` boolean flag instead.""")

# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------
md("""## 4. Cleaning

Cleaning is implemented as a single DuckDB SQL transformation
(`src/data_cleaner._build_cleaning_sql`) so it scales to the full dataset without
ever materializing it in pandas. Key rules:

- `Amount` (`"$134.09"`) → numeric via regex-stripped `TRY_CAST`, not a hard-coded format assumption
- `Year`/`Month`/`Day` → `Date`; `Date` + `Time` → `DateTime`
- `Errors?` → preserved as `errors_raw` + derived `has_error` boolean
- `Zip` / `Merchant State` → missingness preserved (`*_raw`), `"Unknown"` only in the display/grouping column

Running the full pipeline below builds `data/processed/datalens.duckdb` and
`data/processed/transactions.parquet` — the canonical artifacts every
analytics tool and the agent query.
""")

code("""row_count = build_processed_dataset()
print(f"Cleaned dataset: {row_count:,} rows")

report = get_missingness_report()
for col, stats in report.items():
    print(f"  {col}: {stats['missing_pct']}% missing ({stats['missing_count']:,} rows)")
""")

code("""feature_counts = build_all_features()
print("Feature/baseline tables built:", feature_counts)
""")

# ---------------------------------------------------------------------------
# 5. Feature Engineering
# ---------------------------------------------------------------------------
md("""## 5. Feature Engineering

Derived temporal features (`Hour`, `DayOfWeek`, `IsWeekend`, `TimeOfDay`) and
pre-aggregated behavioral baselines (`user_baselines`, `merchant_baselines`,
`daily_spending`) are built once here and reused by every downstream tool —
avoiding repeated full scans of the transaction table.
""")

code("""con = load_processed_duckdb_view()
preview = con.execute('''
    SELECT txn_date, hour, day_of_week, is_weekend, time_of_day, amount, has_error, is_fraud
    FROM transactions_final
    LIMIT 10
''').fetchdf()
con.close()
preview
""")

code("""con = load_processed_duckdb_view()
baselines_preview = con.execute("SELECT * FROM user_baselines LIMIT 10").fetchdf()
con.close()
baselines_preview
""")

# ---------------------------------------------------------------------------
# 6. Spending Analytics
# ---------------------------------------------------------------------------
md("""## 6. Spending Analytics

These functions (`src/analytics/spending.py`) become the agent tools
`get_overview`, `compare_periods`, `get_monthly_spending`, `get_daily_spending`,
and `find_spending_changes`.
""")

code("""from src.analytics import spending

overview = spending.get_overview()
overview["result"]
""")

code("""monthly = spending.get_monthly_spending()["result"]
monthly_df = pd.DataFrame(monthly)
monthly_df["period"] = monthly_df["year"].astype(str) + "-" + monthly_df["month"].astype(str).str.zfill(2)
fig = px.line(monthly_df, x="period", y="total", markers=True, title="Monthly Spending Trend")
fig.show()
""")

code("""changes = spending.find_spending_changes(top_n=5)["result"]
pd.DataFrame(changes)
""")

# ---------------------------------------------------------------------------
# 7. Behavioral Analytics
# ---------------------------------------------------------------------------
md("""## 7. Behavioral Analytics

Frequency, average transaction size, merchant diversity, recurring expenses
(`src/analytics/behavior.py`).
""")

code("""from src.analytics import behavior

freq = behavior.analyze_transaction_frequency()
avg_txn = behavior.analyze_average_transaction()
diversity = behavior.analyze_merchant_diversity()

print("Frequency:", freq["result"])
print("Average transaction:", avg_txn["result"])
print("Merchant diversity:", diversity["result"])
""")

code("""con = load_processed_duckdb_view()
example_user = con.execute("SELECT user_id FROM transactions_final LIMIT 1").fetchone()[0]
con.close()

recurring = behavior.analyze_recurring_expenses(example_user, min_occurrences=2)["result"]
pd.DataFrame(recurring).head(10)
""")

# ---------------------------------------------------------------------------
# 8. Transaction Analytics
# ---------------------------------------------------------------------------
md("""## 8. Transaction Analytics

Category, merchant, location, and payment-method breakdowns, plus
weekday/weekend and time-of-day patterns.
""")

code("""from src.analytics import categories, merchants, temporal

cat_result = categories.analyze_categories(top_n=10)["result"]
cat_df = pd.DataFrame(cat_result)
fig = px.bar(cat_df, x="category_label", y="total", title="Spending by Category (MCC)")
fig.show()
""")

code("""merchant_result = merchants.analyze_merchants(top_n=10)["result"]
merch_df = pd.DataFrame(merchant_result).sort_values("total_spend")
fig = px.bar(merch_df, x="total_spend", y="merchant_name", orientation="h", title="Top Merchants")
fig.show()
""")

code("""weekday_weekend = temporal.analyze_weekday_vs_weekend()["result"]
pd.DataFrame(weekday_weekend).T
""")

# ---------------------------------------------------------------------------
# 9. Anomaly Detection
# ---------------------------------------------------------------------------
md("""## 9. Anomaly Detection

Three complementary approaches, all in `src/anomaly/detector.py`:

1. **Rule-based** — amount vs. the user's own historical average (configurable multiplier)
2. **Statistical** — z-score and IQR outliers, computed per-user or population-wide
3. **ML** — IsolationForest over a small numeric feature set (amount, hour, weekend flag)

Every flagged transaction carries a plain-language `anomaly_reasons` list — the
detector never just returns "this is weird" without evidence.
""")

code("""from src.anomaly import detector, explanations

statistical = detector.detect_statistical_anomalies(limit=10)
print("Baseline used:", statistical["baseline"])
pd.DataFrame(statistical["result"])
""")

code("""rule_based = detector.detect_rule_based_anomalies(example_user, limit=10)["result"]
pd.DataFrame(rule_based)
""")

code("""ml_flagged = detector.detect_ml_anomalies(sample_limit=20000, top_n=10)["result"]
pd.DataFrame(ml_flagged)
""")

code("""if statistical["result"]:
    print(explanations.explain_transaction_anomaly(statistical["result"][0]))
else:
    print("No statistical anomalies flagged in this sample at the current threshold.")
""")

# ---------------------------------------------------------------------------
# 10. Fraud Analysis
# ---------------------------------------------------------------------------
md("""## 10. Fraud Analysis

`Is Fraud?` is used **only** to evaluate the anomaly detector's agreement with
labeled fraud — it is not the basis of a supervised classifier. DataLens's core
product is financial investigation broadly; fraud-label agreement is one
quality signal among several.
""")

code("""flagged_pairs = [
    (row["user_id"], str(row["txn_datetime"]), row["amount"])
    for row in statistical["result"]
]
eval_result = explanations.evaluate_against_fraud_label(flagged_pairs, method_name="statistical_z_iqr")
eval_result["result"]
""")

# ---------------------------------------------------------------------------
# 11. Key Insights
# ---------------------------------------------------------------------------
md("""## 11. Key Insights

*(Populate this section with observations specific to whichever dataset —
synthetic sample or the real 24.4M-row file — was processed above. Re-run this
notebook against the real dataset before a live demo so these insights reflect
actual data rather than the synthetic sample.)*

- Total transactions / total spend / date range: see the `get_overview()` output above.
- Spending concentration: see the category and merchant breakdowns above for
  which MCCs / merchants dominate total spend.
- Weekday vs. weekend: see section 8 for whether spend skews toward weekends.
- Anomaly detector coverage: see the fraud-label evaluation in section 10 for
  precision/recall against the `Is Fraud?` label at the current thresholds.
""")

# ---------------------------------------------------------------------------
# 12. Analytics Tool Development
# ---------------------------------------------------------------------------
md("""## 12. Analytics Tool Development — Notebook → Agent Tool

Every function demonstrated above is already registered as an agent tool in
`src/agent/tools.py::TOOL_REGISTRY`, with a JSON-schema description the LLM
uses for tool selection. The mapping is direct:

| Notebook section | Function | Agent tool name |
|---|---|---|
| 6. Spending | `spending.compare_periods` | `compare_periods` |
| 6. Spending | `spending.get_monthly_spending` | `get_monthly_spending` |
| 7. Behavioral | `behavior.analyze_transaction_frequency` | `analyze_transaction_frequency` |
| 8. Transaction | `categories.analyze_categories` | `analyze_categories` |
| 8. Transaction | `merchants.analyze_merchants` | `analyze_merchants` |
| 9. Anomaly | `detector.detect_anomalies` | `detect_anomalies` |

No analytical capability exists in this notebook that isn't callable by the
agent, and no agent tool exists that wasn't first validated here.
""")

code("""from src.agent.tools import get_tool_schemas

schemas = get_tool_schemas()
print(f"{len(schemas)} tools registered for the agent:")
for s in schemas:
    print(f"  - {s['name']}: {s['description'][:80]}...")
""")

# ---------------------------------------------------------------------------
# 13. Conclusion
# ---------------------------------------------------------------------------
md("""## 13. Conclusion

This notebook validated the full analytical foundation of DataLens AI end to
end: raw CSV → cleaned/feature-engineered DuckDB+Parquet → spending, category,
merchant, temporal, and behavioral analytics → rule-based/statistical/ML
anomaly detection → fraud-label evaluation. Every capability shown here is
live inside the Streamlit app (`app.py`) via the agent's investigation loop
(`src/agent/agent.py`), which calls these exact functions as tools rather than
letting the LLM reason over raw or hallucinated numbers.

**Next step for a real demo:** replace the sample/synthetic CSV at
`data/raw/transactions.csv` with the real ~24.4M-row dataset, re-run
`scripts/run_pipeline.py`, and re-run this notebook top to bottom so the "Key
Insights" section reflects real findings.
""")

nb["cells"] = cells

with open("notebooks/DataLens_Analytics.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/DataLens_Analytics.ipynb")
