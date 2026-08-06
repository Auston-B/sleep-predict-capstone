"""
features.py
-----------
Cleaning and feature engineering for the *All of Us* sleep analysis.

Takes the night and day-level extracts through the inclusion rules and
aggregates them to one row per participant.

make_model_features(), at the bottom, derives the modelling columns from that
already-aggregated frame — it never sees a night or a day.
"""

import numpy as np
import pandas as pd


# ── Inclusion thresholds ──────────────────────────────────────────────────────
# A night outside these bounds is more likely a partial wear or a tracker artifact
# than a real main-sleep record.
MIN_SLEEP_HRS  = 4
MAX_SLEEP_HRS  = 12
MIN_VALID_RATE = 0.70      # share of a participant's nights that must be in range

# Two night floors, both applied during extraction. The first admits a night to the
# cleaned frame; the second decides whose targets are measured precisely enough to
# model.
MIN_EXTRACT_NIGHTS = 4     # applied by clean_sleep()
MIN_MODEL_NIGHTS   = 30    # applied by restrict_to_measured()

MIN_STEPS = 100            # below this is non-wear
MAX_STEPS = 100_000        # above this is impossible
MIN_DAYS  = 4              # valid days required per participant

ACTIVE_STEPS = 7_500       # CDC's "somewhat active" threshold

# Fitbit moved to PPG-based sleep staging in 2017, and nights either side of that
# are not the same measurement. Applied before the duration and validity rules so
# those judge a participant on the window being analysed. Pass start_date=None to
# keep the full history.
MIN_SLEEP_DATE = "2017-01-01"


# ── Demographic recoding maps ─────────────────────────────────────────────────
# Survey answers arrive as raw concept strings. Anything absent from a map becomes
# NaN, so a new answer surfaces as missing rather than as a silent new category.

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

# Recoded in place: (column, map, ordered categories or None). Ordered categoricals
# keep plots and sorts in real order.
RECODES = [
    ("gender",            GENDER_MAP,     None),
    ("education",         EDUCATION_MAP,  EDUCATION_ORDER),
    ("employment",        EMPLOYMENT_MAP, None),
    ("self_rated_health", HEALTH_MAP,     HEALTH_ORDER),
]

# The same three ordered categoricals again, as (source, numeric column). Linear
# models see the ordering rather than a set of unordered dummies; the categorical
# originals stay for grouping and description. Applied by make_model_features().
ORDINALS = [("education",         "education_num"),
            ("self_rated_health", "health_num"),
            ("income_tier",       "income_num")]

# Fairness stratum, banded from age by make_model_features().
AGE_BANDS = [17, 40, 60, 80, 200]
AGE_BAND_LABELS = ["18-40", "41-60", "61-80", "81+"]


# ── Sleep cleaning ────────────────────────────────────────────────────────────

def _prep_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the inclusion rules test."""
    df = df.copy()
    # astype rather than pd.to_datetime: on the dbdate column BigQuery returns,
    # to_datetime converts element-wise.
    df["sleep_date"] = df["sleep_date"].astype("datetime64[ns]")
    df["hours_asleep"] = df["minute_asleep"] / 60
    df["valid_night"] = df["hours_asleep"].between(MIN_SLEEP_HRS, MAX_SLEEP_HRS)
    return df


def _keep_people(df: pd.DataFrame, stat: pd.Series, minimum) -> pd.DataFrame:
    """Keep rows whose participant meets `minimum` on `stat`, a per-person Series."""
    return df[df["person_id"].isin(stat[stat >= minimum].index)]


def _sleep_stages(start_date: str) -> list:
    """The inclusion rules, in order, as (label, filter) pairs.

    clean_sleep() applies them and returns the frame; sleep_funnel() applies the
    same list and reports what each one costs.
    """
    stages = []
    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        stages.append((f"on/after {start_date}", lambda d: d[d["sleep_date"] >= cutoff]))

    return stages + [
        ("main sleep only",
         lambda d: d[d["is_main_sleep"] == True]),
        (f"participants >={MIN_VALID_RATE:.0%} valid",
         lambda d: _keep_people(d, d.groupby("person_id")["valid_night"].mean(),
                                MIN_VALID_RATE)),
        (f"nights within {MIN_SLEEP_HRS}-{MAX_SLEEP_HRS} h",
         lambda d: d[d["valid_night"]]),
        (f">={MIN_EXTRACT_NIGHTS} valid nights",
         lambda d: _keep_people(d, d.groupby("person_id").size(), MIN_EXTRACT_NIGHTS)),
    ]


def clean_sleep(df: pd.DataFrame, start_date: str = MIN_SLEEP_DATE) -> pd.DataFrame:
    """
    Apply the inclusion rules listed in _sleep_stages().

    The duration bounds are applied here rather than at extraction time so the
    validity rule can see the out-of-range nights. Pass start_date=None to keep the
    full history.

    Takes person_id, sleep_date, is_main_sleep, minute_asleep; returns person_id,
    sleep_date, hours_asleep — one row per retained night.
    """
    df = _prep_sleep(df)
    for _, rule in _sleep_stages(start_date):
        df = rule(df)
    return df[["person_id", "sleep_date", "hours_asleep"]]


def sleep_funnel(df: pd.DataFrame, start_date: str = MIN_SLEEP_DATE) -> pd.DataFrame:
    """
    Report what each clean_sleep() rule costs.

    Returns
    -------
    pd.DataFrame
        stage, nights, people, people_lost
    """
    df = _prep_sleep(df)
    stages = [("nights extracted", df)]

    for label, rule in _sleep_stages(start_date):
        df = rule(df)
        stages.append((label, df))

    funnel = pd.DataFrame([
        {"stage": label, "nights": len(frame), "people": frame["person_id"].nunique()}
        for label, frame in stages
    ])
    funnel["people_lost"] = funnel["people"].diff().fillna(0).astype(int)

    return funnel


# ── Steps cleaning ────────────────────────────────────────────────────────────

def clean_steps(df: pd.DataFrame, start_date: str = MIN_SLEEP_DATE) -> pd.DataFrame:
    """
    Apply the activity inclusion rules to a person_id / date / steps frame: the
    analysis window, step bounds, and a minimum of MIN_DAYS valid days per
    participant. The window is shared with clean_sleep() rather than being
    activity-specific; pass start_date=None to disable it.
    """
    df = df.copy()
    df["date"] = df["date"].astype("datetime64[ns]")   # see _prep_sleep on the cast

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]

    df = df[(df["steps"] >= MIN_STEPS) & (df["steps"] < MAX_STEPS)]

    return _keep_people(df, df.groupby("person_id").size(), MIN_DAYS)


# ── Demographic cleaning ──────────────────────────────────────────────────────

def _recode(series: pd.Series, mapping: dict, order: list = None):
    """Map through `mapping`, optionally as an ordered Categorical.
    """

    already_recoded = {value: value for value in mapping.values()}

    mapped = series.map({**mapping, **already_recoded})
    return pd.Categorical(mapped, categories=order, ordered=True) if order else mapped


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

    Same rows out. Collapses `race`/`ethnicity` into `race_ethnicity`, buckets
    `income_bracket` into `income_tier`, and normalizes the rest through RECODES.
    Absent columns are skipped.
    """
    df = demo_df.copy()

    for col, mapping, order in RECODES:
        if col in df.columns:
            df[col] = _recode(df[col], mapping, order)

    # Race and ethnicity are largely redundant in All of Us, and participants who
    # report Hispanic/Latino ethnicity frequently skip the race question. Mapping
    # race first and letting ethnicity overwrite it keeps them in a real category
    # instead of "Unknown".
    if "race" in df.columns:
        race_ethnicity = _recode(df["race"], RACE_MAP)

        if "ethnicity" in df.columns:
            hispanic = df["ethnicity"].eq("Hispanic or Latino")
            race_ethnicity = race_ethnicity.where(~hispanic, "Hispanic or Latino")

        df["race_ethnicity"] = race_ethnicity
        df = df.drop(columns=[c for c in ("race", "ethnicity") if c in df.columns])

    if "income_bracket" in df.columns:
        df["income_tier"] = _recode(df["income_bracket"], INCOME_TIER_MAP, INCOME_TIER_ORDER)
        df = df.drop(columns=["income_bracket"])

    return df


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

    Pure and idempotent: returns a copy, adds columns only, and re-running it on
    its own output is a no-op.

    Adds everything in models.FEATURE_COLS that is not already in the pickle, plus
    `age_band` for models.GROUP_COLS.
    """
    df = features_df.copy()

    steps = df["mean_daily_steps"].astype("float64")

    # The activity-sleep relationship is curved, so the log form predicts better
    # than raw steps.
    df["log_steps"] = np.log1p(steps)

    # Raw std_daily_steps tracks activity volume while this
    # ratio isolates irregularity.
    df["steps_cv"] = df["std_daily_steps"].astype("float64") / steps

    # gender is left to the dummies in models.ENCODE_COLS
    # employment is binary, and _binary keeps its missing values missing.
    df["employed"] = _binary(df["employment"], "Employed")

    for src, dst in ORDINALS:
        if src in df.columns:
            df[dst] = _ordinal(df[src])

    # Fairness stratum. race_ethnicity is already a column and needs no derivation.
    df["age_band"] = pd.cut(df["age"].astype("float64"), bins=AGE_BANDS,
                            labels=AGE_BAND_LABELS, ordered=True)

    return df
