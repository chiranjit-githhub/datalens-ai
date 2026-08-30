
# DataLens AI

**An AI-powered financial data investigation agent that doesn't just answer questions — it investigates them.**

Built for the Razorpay Buildathon — Open Track.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b">
  <img alt="DuckDB" src="https://img.shields.io/badge/data-DuckDB%20%2B%20Parquet-yellow">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-25%20passing-brightgreen">
</p>

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Running the Notebook](#running-the-notebook)
- [Running Tests](#running-tests)
- [Example Questions](#example-questions)
- [Agent Workflow](#agent-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Problem

Ask a typical financial chatbot *"why did my spending increase last month?"*
and it will confidently generate an answer from the LLM's own reasoning —
often plausible-sounding, frequently unsupported, sometimes hallucinated.
The LLM was never actually shown the data.

## Solution

DataLens AI is an **agent that investigates before it answers**. Given a
question, it plans an investigation, calls real analytics tools against the
actual transaction data, inspects the results, decides whether more evidence
is needed, calls more tools, and only then synthesizes an evidence-backed
answer with a root cause and a recommendation.

## Key Innovation

> The agent should investigate progressively instead of generating an answer immediately.

Concretely: `compare_periods` → `compare_category_periods` →
`analyze_transaction_frequency` → root-cause synthesis, each step chosen
by the LLM based on what the previous step's evidence actually showed —
not a hard-coded script.

---

## Architecture

```
User (Streamlit)
      │
      ▼
 AI Agent Orchestrator (src/agent/agent.py)
      │
      ├── Intent classification + initial plan (src/agent/planner.py)
      ├── LLM tool-calling loop (Anthropic / OpenAI, model-agnostic)
      ├── Evidence chain (src/agent/evidence.py)
      └── Deterministic fallback if no LLM is configured
      │
      ▼
 Analytics / Anomaly Tools (src/analytics/*, src/anomaly/*)
      │
      ▼
 Data Layer — DuckDB over Parquet (src/data_loader.py)
      │
      ▼
 Processed transaction data (data/processed/)
```

The LLM never sees raw transaction rows — only small, structured tool
results (evidence). This keeps the agent grounded and keeps the dataset,
however large, out of the LLM's context window.

---

## Features

- **Progressive, multi-step investigation** — not a single LLM call
- **23 analytics/anomaly tools** across spending, category, merchant,
  temporal, behavioral, and anomaly analysis
- **Three-method anomaly detection**: rule-based, statistical (z-score/IQR),
  and Isolation Forest — each with a plain-language explanation
- **Evidence-backed answers** with an explicit Fact / Inference /
  Recommendation structure
- **Visible investigation trace** (tool calls, not hidden chain-of-thought)
- **Fraud-label evaluation** (precision/recall/F1) as a quality check on the
  anomaly detector — DataLens is not a supervised fraud classifier
- **DuckDB + Parquet** data layer, scaling to tens of millions of rows
  without loading the full dataset into pandas or the LLM
- **Deterministic fallback mode** if the LLM API is unavailable — the
  dashboard and standard analytics keep working
- **Streamlit UI** with a chat-style investigation interface and a
  dashboard tab

---

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | Streamlit |
| Data processing | Python, Pandas, NumPy |
| Large-dataset querying | DuckDB (Parquet-backed) |
| Visualization | Plotly |
| AI agent | Anthropic (`claude-sonnet-5`) or OpenAI, tool/function-calling, model-agnostic |
| ML / anomaly detection | scikit-learn `IsolationForest`, z-score, IQR, rule-based baselines |
| Storage | Parquet + DuckDB (local, no external infra) |

---

## Dataset

Expected columns (standard credit-card transaction schema):

```
User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
```

The real dataset is ~24.4M rows. **Missingness percentages are always
computed from the actual loaded data** (`src/data_cleaner.get_missingness_report`)
— reference figures in project docs are not trusted blindly.

`Is Fraud?` is used only to *evaluate* the anomaly detector
(`src/anomaly/explanations.evaluate_against_fraud_label`), not as the basis
of a supervised classifier — DataLens's core product is financial
investigation, not fraud classification.

### No dataset handy?

Generate a small synthetic sample matching the schema:

```bash
python scripts/generate_sample_data.py --rows 200000 --users 50
```

This writes to `data/raw/transactions.csv` with realistic amount
distributions, injected outliers (for anomaly detection to have something to
find), and the same missingness patterns described in the schema. It is
clearly synthetic — swap in the real file for production use.

---

## Project Structure

```
datalens-ai/
├── app.py                       # Streamlit app (chat + dashboard)
├── config.py                    # Central config (paths, LLM provider, tunables)
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                     # Place transactions.csv here
│   └── processed/                # Pipeline output (DuckDB + Parquet)
│
├── notebooks/
│   └── DataLens_Analytics.ipynb # Analytics foundation, executed with sample data
│
├── src/
│   ├── data_loader.py           # CSV → DuckDB, dataset validation
│   ├── data_cleaner.py          # Cleaning rules (amount, date, missingness)
│   ├── feature_engineering.py   # Behavioral baseline tables
│   │
│   ├── analytics/                # Spending / category / merchant / temporal /
│   │                              # behavior / transaction analytics tools
│   ├── anomaly/                  # Rule-based, statistical, ML anomaly detection
│   ├── agent/                    # Planner, evidence chain, tool registry,
│   │                              # LLM orchestrator (Anthropic/OpenAI/Ollama)
│   └── visualization/            # Plotly chart builders
│
├── scripts/
│   ├── generate_sample_data.py  # Synthetic dataset generator
│   ├── run_pipeline.py          # Raw CSV → cleaned/feature-engineered data
│   └── build_notebook.py        # Regenerates the notebook programmatically
│
└── tests/                       # 25 tests: cleaning, analytics, anomaly detection
```

---

## Setup

```bash
git clone <repo-url>
cd datalens-ai
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### Environment Variables

See `.env.example` for the full list. Key ones:

| Variable | Purpose | Default |
|---|---|---|
| `DATALENS_LLM_PROVIDER` | `anthropic` \| `openai` \| `ollama` \| `none` | `anthropic` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `ANTHROPIC_MODEL` | Anthropic model string | `claude-sonnet-5` |
| `OPENAI_API_KEY` | OpenAI API key (alternative provider) | — |
| `OLLAMA_BASE_URL` | Local Ollama OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Ollama model (must support tool calling) | `llama3.1` |
| `DATALENS_RAW_CSV` | Path to raw transactions CSV | `data/raw/transactions.csv` |
| `DATALENS_MAX_STEPS` | Max investigation loop depth | `5` |

Never commit `.env` — it's git-ignored. `.env.example` contains placeholders only.

---

## Running the App

```bash
# 1. Place (or generate) your raw CSV
python scripts/generate_sample_data.py --rows 200000
# 2. Run the data pipeline (clean + feature-engineer)
python scripts/run_pipeline.py
# 3. Launch the app
streamlit run app.py
```

Or upload a CSV and click **Run / Refresh Pipeline** directly from the
sidebar in the app.

## Running the Notebook

```bash
jupyter notebook notebooks/DataLens_Analytics.ipynb
```

Or regenerate it programmatically after changing the analytics API:

```bash
python scripts/build_notebook.py
```

The notebook documents and validates every analytical capability that later
becomes an agent tool — see its final section for the direct
notebook-function → agent-tool mapping.

## Running Tests

```bash
# Cleaning-logic tests (no dataset required)
pytest tests/test_cleaning.py -v

# Analytics + anomaly tests (require a processed dataset)
python scripts/generate_sample_data.py --rows 50000
python scripts/run_pipeline.py
pytest tests/ -v
```

---

## Example Questions

- "Why did my spending increase last month?"
- "Show me unusual transactions."
- "Which merchant did I spend the most on?"
- "Was my spending higher on weekends?"
- "What are my recurring expenses?"
- "Why does this transaction look suspicious?"

## Agent Workflow

```
Question → Intent classification → Initial tool plan
    → Tool call → Evidence → Evaluate evidence
    → More evidence needed? → Yes: another tool / No: synthesize
    → Final answer (Fact / Inference / Recommendation)
```

Max loop depth is configurable (`DATALENS_MAX_STEPS`, default 5) to prevent
runaway investigations. If the LLM API is unavailable or unconfigured, the
agent falls back to running the planner's suggested tools directly and
formatting the raw evidence — the app never fully breaks.

---

## Limitations

- Anomaly detection thresholds (z-score, IQR multiplier, IsolationForest
  contamination) are configurable heuristics, not a tuned production model.
- `IsolationForest` and the ML anomaly path run on a bounded sample, not the
  full 24.4M-row dataset, to stay interactive.
- The dataset has no unique transaction ID; transaction lookups match on
  `(user, timestamp, amount)`, which is usually but not always unique.
- MCC → category-name labels cover common codes only; unmapped codes display
  as `MCC {code}` rather than a guessed label.
- The deterministic fallback mode cannot infer parameters (dates, user IDs)
  an LLM would normally extract from the question, so it answers with
  whatever tools need no arguments.

## Future Improvements

- Persist per-conversation evidence chains for longer multi-turn context
  ("what caused *that*?" across sessions)
- User-facing threshold tuning for anomaly sensitivity
- Streaming tool-call progress in the UI as the investigation runs
- Support for additional LLM providers via the same tool-schema abstraction
- Richer MCC → category mapping (or a lookup table shipped with the dataset)

<p align="center">Built for the Razorpay Buildathon — Open Track.</p>
