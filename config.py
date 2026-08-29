"""
DataLens AI — Central configuration.

All paths, tunables, and environment-derived settings live here so the
rest of the codebase never hard-codes a path or a magic number.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_CSV_PATH = Path(os.getenv("DATALENS_RAW_CSV", RAW_DATA_DIR / "transactions.csv"))
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "transactions.parquet"
DUCKDB_PATH = PROCESSED_DATA_DIR / "datalens.duckdb"

OUTPUTS_DIR = BASE_DIR / "outputs"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / Agent configuration (model-agnostic)
# ---------------------------------------------------------------------------
# Supported: "anthropic", "openai", "ollama", "none" (deterministic fallback mode)
LLM_PROVIDER = os.getenv("DATALENS_LLM_PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama runs locally and exposes an OpenAI-compatible API — no API key needed.
# Model must support tool/function calling (e.g. llama3.1, qwen2.5, mistral-nemo).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Max tool-calling / investigation loop depth (guards against infinite loops)
MAX_INVESTIGATION_STEPS = int(os.getenv("DATALENS_MAX_STEPS", "5"))

# ---------------------------------------------------------------------------
# Analytics tunables
# ---------------------------------------------------------------------------
# Anomaly detection
ANOMALY_ZSCORE_THRESHOLD = float(os.getenv("DATALENS_Z_THRESHOLD", "3.0"))
ANOMALY_IQR_MULTIPLIER = float(os.getenv("DATALENS_IQR_MULT", "1.5"))
ISOLATION_FOREST_CONTAMINATION = float(os.getenv("DATALENS_IF_CONTAMINATION", "0.01"))

# "Large amount" heuristic relative to a user's own historical average
LARGE_AMOUNT_MULTIPLIER = float(os.getenv("DATALENS_LARGE_AMOUNT_MULT", "3.0"))

# Rows above which we prefer DuckDB SQL over pandas in-memory ops
DUCKDB_ROW_THRESHOLD = int(os.getenv("DATALENS_DUCKDB_THRESHOLD", "200000"))

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME = "DataLens AI"
APP_TAGLINE = "An AI-powered financial data investigation agent that doesn't just answer questions — it investigates them."
