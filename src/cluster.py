"""
cluster.py
----------
KMeans over a participant's sleep behaviour, on the participant-level frame
produced by features.make_model_features().

Matrix assembly, the scaler-KMeans pipeline, a scan over k, and the naming rules
of the four phenotypes.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── The column contract ───────────────────────────────────────────────────────
# CLUSTER_COLS  what the distance metric sees: duration, night-to-night spread,
#               and the two tails. Complete on the >=30-night cohort, so nothing
#               is imputed.
# PROFILE_COLS  described afterwards, never fitted.

CLUSTER_COLS = ["mean_sleep_hrs", "std_sleep_hrs", "pct_short_sleep", "pct_long_sleep"]

PROFILE_COLS = ["iqr_sleep_hrs", "mean_daily_steps", "age", "bmi", "n_valid_nights"]

# The four archetypes, in the order every table and legend presents them. viz.py
# imports this list.
PHENOTYPES = [
    "Consistent Good Sleepers",
    "Chronic Short & Variable",
    "Short but Regular",
    "Variable Long Sleepers",
]

# How a centroid is recognized, applied in order. Least ambiguous distinction first; the
# leftover cluster takes the remaining name. See name_clusters().
NAMING_RULES = [
    ("Variable Long Sleepers",   lambda c: c["mean_sleep_hrs"].idxmax()),
    ("Consistent Good Sleepers", lambda c: c["std_sleep_hrs"].idxmin()),
    ("Short but Regular",        lambda c: c["mean_sleep_hrs"].idxmin()),
]
LEFTOVER_PHENOTYPE = "Chronic Short & Variable"

RANDOM_STATE = 42
N_INIT = 20          # restarts per fit; KMeans keeps the lowest-inertia one


# ── Matrix assembly ───────────────────────────────────────────────────────────

def prepare_cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select CLUSTER_COLS as float64, keeping the caller's index so labels realign.

    Rows with a missing value are dropped rather than imputed.

    The All of Us frame carries nullable Float64/Int64, which numpy turns into
    object arrays once pd.NA is present, hence the cast.
    """
    missing = [c for c in CLUSTER_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing {missing} — pass the output of "
            f"features.make_participant_features()"
        )

    X = df[CLUSTER_COLS].astype("float64")
    dropped = X.isna().any(axis=1).sum()
    if dropped:
        print(f"  dropped {dropped:,} participants missing a cluster feature")

    return X.dropna()


def fit_clusters(X: pd.DataFrame, k: int = 4) -> Pipeline:
    """
    Standardize, then KMeans, fitted on a prepared matrix. The final step is always
    named "kmeans", which is how centroids() and assign_clusters() reach the estimator.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)),
    ]).fit(X)


def centroids(pipe: Pipeline) -> pd.DataFrame:
    """
    Cluster centroids back in original units, one row per cluster label.
    KMeans fits on the standardized matrix, so its `cluster_centers_` are in SDs.
    """
    km = pipe.named_steps["kmeans"]
    raw = pipe.named_steps["scaler"].inverse_transform(km.cluster_centers_)

    return pd.DataFrame(raw, columns=CLUSTER_COLS,
                        index=pd.RangeIndex(len(raw), name="cluster"))


# ── Naming ────────────────────────────────────────────────────────────────────

def name_clusters(centroids_df: pd.DataFrame) -> dict[int, str]:
    """
    Map KMeans label integers to the phenotype names.
    The integers themselves carry no meaning.
    """
    if len(centroids_df) != 4:
        raise ValueError(
            f"name_clusters is defined for k=4 only, got k={len(centroids_df)}; "
            f"use the integer labels directly"
        )

    remaining = centroids_df.copy()
    names = {}

    for phenotype, pick in NAMING_RULES:
        label = pick(remaining)
        names[label] = phenotype
        remaining = remaining.drop(label)

    names[remaining.index[0]] = LEFTOVER_PHENOTYPE
    return names


def assign_clusters(df: pd.DataFrame, k: int = 4) -> pd.Series:
    """
    Prepare, fit, and label clusters.

    At k=4 this returns the phenotype names as an ordered Categorical; at any
    other k the naming rules do not apply and it returns the raw label integers.
    """
    X = prepare_cluster_matrix(df)
    pipe = fit_clusters(X, k=k)
    labels = pd.Series(pipe.named_steps["kmeans"].labels_, index=X.index)

    if k != 4:
        return labels.reindex(df.index)

    named = labels.map(name_clusters(centroids(pipe)))
    return pd.Series(
        pd.Categorical(named.reindex(df.index), categories=PHENOTYPES, ordered=True),
        index=df.index, name="phenotype",
    )


# ── Choosing k ────────────────────────────────────────────────────────────────

def scan_k(X: pd.DataFrame, ks=range(2, 9), sample_n: int = 10_000,
           subsample_seed: int = 0) -> pd.DataFrame:
    """
    Fit each k and score it two ways.

    Returns
    -------
    pd.DataFrame
        indexed by k: inertia, silhouette

    Inertia is the quantity KMeans minimizes. Silhouette weighs each point's
    own cluster against the nearest rival.

    Silhouette is computed on a fixed random subsample of `sample_n` rows.
    `subsample_seed` draws that subsample and is deliberately not RANDOM_STATE.
    """
    Z = StandardScaler().fit_transform(X)

    idx = np.random.default_rng(subsample_seed).choice(
        len(Z), min(sample_n, len(Z)), replace=False)

    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(Z)

        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Z[idx], km.labels_[idx]),
        })

    return pd.DataFrame(rows).set_index("k")


# ── Describing the clusters ───────────────────────────────────────────────────

def cluster_profile(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Describe the clusters in terms of their size, share, and the mean of each
    feature. The clustering columns first, then the profile-only ones.
    """
    cols = CLUSTER_COLS + PROFILE_COLS

    grouped = df[cols].astype("float64").groupby(labels, observed=True)
    profile = grouped.mean()
    profile.insert(0, "N", grouped.size())
    profile.insert(1, "pct", (profile["N"] / len(labels.dropna()) * 100).round(1))

    profile["pct_female"] = (
        df["gender"].eq("Female").groupby(labels, observed=True).mean()
    )

    return profile
