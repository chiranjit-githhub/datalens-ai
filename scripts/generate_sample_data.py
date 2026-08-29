"""
DataLens AI — Sample Data Generator

Generates a small, schema-matching synthetic transaction CSV so the full
pipeline/agent can be exercised end-to-end WITHOUT the real ~24.4M row
dataset. This is clearly synthetic data for development/demo purposes —
replace data/raw/transactions.csv with the real dataset for production use.

Usage:
    python scripts/generate_sample_data.py --rows 200000 --users 50
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

MERCHANTS = [
    ("Whole Foods Market", 5411), ("Trader Joe's", 5411), ("Chipotle", 5814),
    ("Olive Garden", 5812), ("Shell Gas", 5541), ("Chevron", 5541),
    ("Target", 5311), ("Walmart", 5311), ("Best Buy", 5732),
    ("Con Edison", 4900), ("CVS Pharmacy", 5912), ("Amazon.com", 5999),
    ("Netflix", 4899), ("Spotify", 4899), ("Zara", 5651), ("H&M", 5651),
    ("AMC Theatres", 7996), ("MTA Subway", 4111), ("Home Depot", 5211),
    ("Starbucks", 5814),
]

CITIES_STATES = [
    ("New York", "NY"), ("Brooklyn", "NY"), ("Jersey City", "NJ"),
    ("Los Angeles", "CA"), ("San Francisco", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Miami", "FL"), ("Seattle", "WA"), (None, None),
]

USE_CHIP_OPTIONS = ["Chip Transaction", "Swipe Transaction", "Online Transaction"]


def generate(n_rows: int, n_users: int, n_cards_per_user: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    start = date(2023, 1, 1)
    end = date(2023, 12, 31)
    span_days = (end - start).days

    rows = []
    for i in range(n_rows):
        user = f"U{rng.randint(1, n_users):04d}"
        card = rng.randint(0, n_cards_per_user - 1)

        d = start + timedelta(days=rng.randint(0, span_days))
        hour = int(np.clip(np_rng.normal(14, 5), 0, 23))
        minute = rng.randint(0, 59)
        time_str = f"{hour:02d}:{minute:02d}"

        merchant_name, mcc = rng.choice(MERCHANTS)
        city, state = rng.choice(CITIES_STATES)
        zip_code = rng.randint(10000, 99999) if state else None

        base_amount = abs(np_rng.normal(45, 30))
        # Inject occasional large outliers to make anomaly detection meaningful
        if rng.random() < 0.01:
            base_amount *= rng.uniform(5, 15)
        amount = f"${base_amount:.2f}"

        use_chip = rng.choice(USE_CHIP_OPTIONS)
        has_error = rng.random() < 0.016  # ~ matches the described ~98.4% missing/no-error rate
        errors = "Bad PIN" if has_error else None
        is_fraud = "Yes" if (has_error and rng.random() < 0.3) or rng.random() < 0.001 else "No"

        rows.append({
            "User": user, "Card": card, "Year": d.year, "Month": d.month, "Day": d.day,
            "Time": time_str, "Amount": amount, "Use Chip": use_chip,
            "Merchant Name": merchant_name, "Merchant City": city, "Merchant State": state,
            "Zip": zip_code, "MCC": mcc, "Errors?": errors, "Is Fraud?": is_fraud,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DataLens sample data")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--cards-per-user", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/raw/transactions.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.users, args.cards_per_user)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {args.out}")
