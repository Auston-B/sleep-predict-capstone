"""
features.py
-----------
Cleaning and feature engineering for the *All of Us* sleep analysis.

Takes the night and day-level extracts through the inclusion rules and
aggregates them to one row per participant.

make_model_features(), at the bottom, derives the modelling columns from the aggregated frame.
"""

import numpy as np
import pandas as pd


# ── Inclusion thresholds ──────────────────────────────────────────────────────
# A night outside these bounds is more likely a partial wear or a tracker artifact.
MIN_SLEEP_HRS  = 4
MAX_SLEEP_HRS  = 12
MIN_VALID_RATE = 0.70      # share of a participant's nights that must be in range

# Two night floors. The first admits a night to the cleaned frame; 
# the second decides whose targets are measured precisely enough to model.
MIN_EXTRACT_NIGHTS = 4     # applied by clean_sleep()
MIN_MODEL_NIGHTS   = 30    # applied by restrict_to_measured()

MIN_STEPS = 100            # below this is non-wear
MAX_STEPS = 100_000        # above this is impossible
MIN_DAYS  = 4              # valid days required per participant

ACTIVE_STEPS = 7_500       # CDC's "somewhat active" threshold

# Fitbit moved to PPG-based sleep staging in 2017, and nights either side of that
# are not the same measurement. Applied before the duration and validity rules so
# those judge a participant on the window being analysed.
MIN_SLEEP_DATE = "2017-01-01"


# ── Demographic recoding maps ─────────────────────────────────────────────────
# Survey answers arrive as raw concept strings. Anything absent from a map becomes NaN.

GENDER_MAP = {
    "Female": "Female",
    "Male":   "Male",
    "Not man only, not woman only, prefer not to answer, or skipped": "Other",
    "No matching concept": "Other",
}

RACE_MAP = {
    "White":                     "White",
    "Black or African American": "Black or African American",
    "Asian":                     "Asian",
    "More than one population":  "More than one population",
    "Another single population": "Other",
    "None of these":             "Other",
    "None Indicated":            "Unknown",
    "PMI: Skip":                 "Unknown",
    "I prefer not to answer":    "Unknown",
}

INCOME_TIER_MAP = {
    "Annual Income: less 10k":  "<$35k",
    "Annual Income: 10k 25k":   "<$35k",
    "Annual Income: 25k 35k":   "<$35k",
    "Annual Income: 35k 50k":   "$35-100k",
    "Annual Income: 50k 75k":   "$35-100k",
    "Annual Income: 75k 100k":  "$35-100k",
    "Annual Income: 100k 150k": ">$100k",
    "Annual Income: 150k 200k": ">$100k",
    "Annual Income: more 200k": ">$100k",
}
INCOME_TIER_ORDER = ["<$35k", "$35-100k", ">$100k"]

EDUCATION_MAP = {
    "Less than a high school degree or equivalent": "Less than high school",
    "Highest Grade: Twelve Or GED":                 "High school or GED",
    "Highest Grade: College One to Three":          "Some college",
    "College graduate or advanced degree":          "College graduate or higher",
}
EDUCATION_ORDER = ["Less than high school", "High school or GED",
                   "Some college", "College graduate or higher"]

EMPLOYMENT_MAP = {
    "Employed for wages or self-employed": "Employed",
    "Not currently employed for wages":    "Not employed",
}

HEALTH_MAP = {
    "General Health: Poor":      "Poor",
    "General Health: Fair":      "Fair",
    "General Health: Good":      "Good",
    "General Health: Very Good": "Very good",
    "General Health: Excellent": "Excellent",
}
HEALTH_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]

# The three ordered categoricals again, as (source, numeric column). Linear
# models see the ordering rather than a set of unordered dummies; the categorical
# originals stay for grouping and description. Applied by make_model_features().
ORDINALS = [("education",         "education_num"),
            ("self_rated_health", "health_num"),
            ("income_tier",       "income_num")]

# Fairness stratum, banded from age by make_model_features().
AGE_BANDS = [17, 40, 60, 80, 200]
AGE_BAND_LABELS = ["18-40", "41-60", "61-80", "81+"]


# ── Sleep cleaning ────────────────────────────────────────────────────────────

def _keep_people(df: pd.DataFrame, stat: pd.Series, minimum) -> pd.DataFrame:
    """Keep rows whose participant meets `minimum` on `stat`, a per-person Series."""
    return df[df["person_id"].isin(stat[stat >= minimum].index)]


def _stage_count(label: str, df: pd.DataFrame) -> dict:
    """One funnel row: the rule's label, and what is left after it."""
    return {"stage": label, "nights": len(df), "people": df["person_id"].nunique()}


def _apply_sleep_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Apply the inclusion rules in order, counting what each one costs.

    Returns the surviving nights and the funnel rows. clean_sleep() takes the
    frame and sleep_funnel() takes the rows.
    """
    df = df.copy()
    # astype rather than pd.to_datetime: on the dbdate column BigQuery returns,
    # to_datetime converts element-wise.
    df["sleep_date"] = df["sleep_date"].astype("datetime64[ns]")
    df["hours_asleep"] = df["minute_asleep"] / 60
    df["valid_night"] = df["hours_asleep"].between(MIN_SLEEP_HRS, MAX_SLEEP_HRS)

    rows = [_stage_count("nights extracted", df)]

    df = df[df["sleep_date"] >= pd.Timestamp(MIN_SLEEP_DATE)]
    rows.append(_stage_count(f"on/after {MIN_SLEEP_DATE}", df))

    df = df[df["is_main_sleep"] == True]
    rows.append(_stage_count("main sleep only", df))

    # Ahead of the duration bound below, so the rate is judged on every night the
    # participant recorded, in range or not.
    df = _keep_people(df, df.groupby("person_id")["valid_night"].mean(), MIN_VALID_RATE)
    rows.append(_stage_count(f"participants >={MIN_VALID_RATE:.0%} valid", df))

    df = df[df["valid_night"]]
    rows.append(_stage_count(f"nights within {MIN_SLEEP_HRS}-{MAX_SLEEP_HRS} h", df))

    df = _keep_people(df, df.groupby("person_id").size(), MIN_EXTRACT_NIGHTS)
    rows.append(_stage_count(f">={MIN_EXTRACT_NIGHTS} valid nights", df))

    return df, rows


def clean_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the inclusion rules and return the nights that survive them.

    The duration bounds are applied here rather than at extraction time so the
    validity rule can see the out-of-range nights.

    Takes person_id, sleep_date, is_main_sleep, minute_asleep; returns person_id,
    sleep_date, hours_asleep; one row per retained night.
    """
    clean, _ = _apply_sleep_rules(df)
    return clean[["person_id", "sleep_date", "hours_asleep"]]


def sleep_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report what each clean_sleep() rule costs.

    Returns
    -------
    pd.DataFrame
        stage, nights, people, people_lost
    """
    _, rows = _apply_sleep_rules(df)

    funnel = pd.DataFrame(rows)
    funnel["people_lost"] = funnel["people"].diff().fillna(0).astype(int)

    return funnel


# ── Steps cleaning ────────────────────────────────────────────────────────────

def clean_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the activity inclusion rules to a person_id / date / steps frame: the
    analysis window, step bounds, and a minimum of MIN_DAYS valid days per
    participant. The window is shared with clean_sleep() rather than being
    activity-specific.
    """
    df = df.copy()
    df["date"] = df["date"].astype("datetime64[ns]")   # see _apply_sleep_rules on the cast

    df = df[df["date"] >= pd.Timestamp(MIN_SLEEP_DATE)]
    df = df[(df["steps"] >= MIN_STEPS) & (df["steps"] < MAX_STEPS)]

    return _keep_people(df, df.groupby("person_id").size(), MIN_DAYS)


# ── Demographic cleaning ──────────────────────────────────────────────────────

def _ordered(series: pd.Series, mapping: dict, order: list) -> pd.Categorical:
    """Map through `mapping` as an ordered Categorical, so plots and sorts come
    out in survey order rather than alphabetically."""
    return pd.Categorical(series.map(mapping), categories=order, ordered=True)


def _binary(series: pd.Series, true_value) -> pd.Series:
    """0/1 float indicator that keeps missing missing.

    `.eq()` alone returns False for NaN, which would file someone who skipped the
    question alongside those who answered no.
    """
    return series.eq(true_value).astype("float64").mask(series.isna())


def _ordinal(series: pd.Series) -> pd.Series:
    """Ordered categorical to its integer codes, with NaN preserved.

    `.cat.codes` returns -1 for missing, which is a real value one rank below the
    lowest level. Masking it back to NA leaves it for the pipeline's imputer.
    """
    codes = series.cat.codes.astype("float64")
    return codes.mask(codes < 0)


def clean_demographics(demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode the raw survey demographics into model and plot-ready categories.

    Same rows out. Collapses `race`/`ethnicity` into `race_ethnicity` and buckets
    `income_bracket` into `income_tier`.
    """
    df = demo_df.copy()

    df["gender"]     = df["gender"].map(GENDER_MAP)
    df["employment"] = df["employment"].map(EMPLOYMENT_MAP)

    df["education"]         = _ordered(df["education"], EDUCATION_MAP, EDUCATION_ORDER)
    df["self_rated_health"] = _ordered(df["self_rated_health"], HEALTH_MAP, HEALTH_ORDER)

    # Race and ethnicity are largely redundant in All of Us. Participants who
    # report Hispanic/Latino ethnicity frequently skip the race question. Mapping
    # race first and letting ethnicity overwrite it keeps them in a real category
    # instead of "Unknown".
    hispanic = df["ethnicity"].eq("Hispanic or Latino")
    df["race_ethnicity"] = df["race"].map(RACE_MAP).where(~hispanic, "Hispanic or Latino")

    df["income_tier"] = _ordered(df["income_bracket"], INCOME_TIER_MAP, INCOME_TIER_ORDER)

    return df.drop(columns=["race", "ethnicity", "income_bracket"])


# ── Participant-level feature engineering ─────────────────────────────────────

def make_participant_features(
    sleep_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the night and day-level frames to one row per participant.

    The inner join on `demo_df` is the cohort gate: the person-level extract is
    not cohort-scoped, so the row count out of here is the analytic N.
    """
    sleep_feats = sleep_df.groupby("person_id").agg(
        mean_sleep_hrs   = ("hours_asleep", "mean"),
        std_sleep_hrs    = ("hours_asleep", "std"),                          # consistency
        n_valid_nights   = ("hours_asleep", "count"),
        pct_short_sleep  = ("hours_asleep", lambda x: (x < 6).mean()),
        pct_long_sleep   = ("hours_asleep", lambda x: (x > 9).mean()),
        iqr_sleep_hrs    = ("hours_asleep", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).reset_index()

    steps_feats = steps_df.groupby("person_id").agg(
        mean_daily_steps   = ("steps", "mean"),
        median_daily_steps = ("steps", "median"),
        std_daily_steps    = ("steps", "std"),
        n_valid_days       = ("steps", "count"),
        pct_active_days    = ("steps", lambda x: (x >= ACTIVE_STEPS).mean()),
    ).reset_index()

    return (
        demo_df
        .merge(sleep_feats, on="person_id", how="inner")
        .merge(steps_feats, on="person_id", how="left")
    )


def add_targets(features: pd.DataFrame) -> pd.DataFrame:
    """
    Name the two prediction targets.

    - target_sleep_duration    : mean nightly sleep hours
    - target_sleep_consistency : within-person SD of nightly hours (lower = better)
    """
    df = features.copy()
    df["target_sleep_duration"]    = df["mean_sleep_hrs"]
    df["target_sleep_consistency"] = df["std_sleep_hrs"]
    return df


# ── Model-ready derivations ───────────────────────────────────────────────────
# Everything below operates on the participant-level frame, not on nights or days.
# These are candidates; models.FEATURE_COLS decides what a model actually sees.

def restrict_to_measured(df: pd.DataFrame, min_nights: int = MIN_MODEL_NIGHTS) -> pd.DataFrame:
    """
    Keep participants whose targets are measured well enough to model.
    """
    n = df["n_valid_nights"].astype("float64")
    return df[n >= min_nights]


def make_model_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the modelling columns from a participant-level frame.

    Adds everything in models.FEATURE_COLS not already in the pickle, plus
    `age_band` for models.GROUP_COLS.
    """
    df = features_df.copy()

    steps = df["mean_daily_steps"].astype("float64")

    # The activity-sleep relationship is curved, so the log form predicts better.
    df["log_steps"] = np.log1p(steps)

    # The ratio isolates step irregularity.
    df["steps_cv"] = df["std_daily_steps"].astype("float64") / steps

    # gender is left to the dummies in models.ENCODE_COLS
    # employment is binary, and _binary keeps its missing values missing.
    df["employed"] = _binary(df["employment"], "Employed")

    for src, dst in ORDINALS:
        if src in df.columns:
            df[dst] = _ordinal(df[src])

    # Fairness stratum
    df["age_band"] = pd.cut(df["age"].astype("float64"), bins=AGE_BANDS,
                            labels=AGE_BAND_LABELS, ordered=True)

    return df
