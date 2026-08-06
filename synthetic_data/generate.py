"""
generate.py
-----------
Generates a synthetic dataset that mirrors the All of Us Fitbit schema.
Use this for local development and pipeline testing — no real participant
data is needed or used here.

Run: python synthetic_data/generate.py
Outputs: synthetic_data/synthetic_sleep.csv
         synthetic_data/synthetic_steps.csv
         synthetic_data/synthetic_demographics.csv
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_PARTICIPANTS   = 500          # number of synthetic participants
N_NIGHTS_MIN     = 30           # min nights per participant
N_NIGHTS_MAX     = 365          # max nights per participant
START_DATE       = date(2019, 1, 1)
END_DATE         = date(2023, 9, 30)
OUT_DIR          = os.path.dirname(__file__)

# ── Demographics ──────────────────────────────────────────────────────────────
def make_demographics(n):
    person_ids = np.arange(1000, 1000 + n)

    ages = np.random.normal(loc=52, scale=15, size=n).clip(18, 90).astype(int)

    # gender: 1=male, 2=female, 3=other (concept IDs simplified)
    gender = np.random.choice([1, 2, 3], size=n, p=[0.40, 0.57, 0.03])

    # race (simplified concept IDs)
    race = np.random.choice(
        ["White", "Black or African American", "Asian",
         "Hispanic or Latino", "More than one race", "Other/Unknown"],
        size=n, p=[0.52, 0.21, 0.06, 0.12, 0.05, 0.04]
    )

    bmi = np.random.normal(loc=29.5, scale=6.5, size=n).clip(15, 65).round(1)

    # SES proxy: income bracket (1=low, 2=middle, 3=high)
    income = np.random.choice([1, 2, 3], size=n, p=[0.30, 0.45, 0.25])

    # Geographic region
    region = np.random.choice(
        ["Northeast", "Midwest", "South", "West"],
        size=n, p=[0.18, 0.21, 0.38, 0.23]
    )

    return pd.DataFrame({
        "person_id":   person_ids,
        "age":         ages,
        "gender":      gender,
        "race":        race,
        "bmi":         bmi,
        "income_tier": income,
        "region":      region,
    })


# ── Sleep ─────────────────────────────────────────────────────────────────────
def make_sleep(demographics):
    """
    Mirrors the All of Us `sleep_daily_summary` table schema:
      person_id, sleep_date, is_main_sleep, minute_asleep
    """
    rows = []
    date_range = (END_DATE - START_DATE).days

    for _, person in demographics.iterrows():
        pid   = person["person_id"]
        age   = person["age"]
        bmi   = person["bmi"]

        # Participant-level baseline sleep (with realistic variation)
        # Older age → slightly less sleep; higher BMI → slightly less sleep
        base_sleep_hrs = (
            7.5
            - 0.01 * max(0, age - 40)
            - 0.02 * max(0, bmi - 25)
            + np.random.normal(0, 0.5)            # between-person variation
        )
        base_sleep_hrs = np.clip(base_sleep_hrs, 4.0, 10.0)

        n_nights = np.random.randint(N_NIGHTS_MIN, N_NIGHTS_MAX)
        offsets  = sorted(np.random.choice(date_range, size=n_nights, replace=False))

        for offset in offsets:
            sleep_date = START_DATE + timedelta(days=int(offset))

            # Within-person night-to-night variation
            mins = base_sleep_hrs * 60 + np.random.normal(0, 40)
            mins = max(0, mins)

            # ~5% chance of nap record (is_main_sleep=False)
            is_main = np.random.random() > 0.05

            rows.append({
                "person_id":     pid,
                "sleep_date":    sleep_date.isoformat(),
                "is_main_sleep": is_main,
                "minute_asleep": round(mins, 1),
            })

    return pd.DataFrame(rows)


# ── Steps ─────────────────────────────────────────────────────────────────────
def make_steps(demographics):
    """
    Mirrors the All of Us `activity_summary` table schema:
      person_id, date, steps
    """
    rows = []
    date_range = (END_DATE - START_DATE).days

    for _, person in demographics.iterrows():
        pid  = person["person_id"]
        age  = person["age"]
        bmi  = person["bmi"]

        # Baseline steps — younger, lower BMI → more steps
        base_steps = (
            8000
            - 40 * max(0, age - 40)
            - 60 * max(0, bmi - 25)
            + np.random.normal(0, 1500)
        )
        base_steps = np.clip(base_steps, 500, 20000)

        n_days  = np.random.randint(30, 365)
        offsets = sorted(np.random.choice(date_range, size=n_days, replace=False))

        for offset in offsets:
            activity_date = START_DATE + timedelta(days=int(offset))
            steps = base_steps + np.random.normal(0, 2000)
            steps = int(np.clip(steps, 100, 50000))

            rows.append({
                "person_id": pid,
                "date":      activity_date.isoformat(),
                "steps":     steps,
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic demographics...")
    demo_df  = make_demographics(N_PARTICIPANTS)

    print("Generating synthetic sleep data...")
    sleep_df = make_sleep(demo_df)

    print("Generating synthetic step data...")
    steps_df = make_steps(demo_df)

    demo_df.to_csv(os.path.join(OUT_DIR, "synthetic_demographics.csv"), index=False)
    sleep_df.to_csv(os.path.join(OUT_DIR, "synthetic_sleep.csv"), index=False)
    steps_df.to_csv(os.path.join(OUT_DIR, "synthetic_steps.csv"), index=False)

    print(f"\n✓ synthetic_demographics.csv  — {len(demo_df):,} participants")
    print(f"✓ synthetic_sleep.csv         — {len(sleep_df):,} night records")
    print(f"✓ synthetic_steps.csv         — {len(steps_df):,} day records")
    print(f"\nParticipants: {demo_df.person_id.nunique()}")
    print(f"Sleep: {sleep_df.sleep_date.min()} → {sleep_df.sleep_date.max()}")
    print(f"Steps: {steps_df.date.min()} → {steps_df.date.max()}")
