"""
models.py
---------
Modeling pipeline for the sleep health analysis.

Feature matrix assembly, four models, 5-fold cross-validation, and out-of-fold
fairness evaluation by subgroup.

Every model is scored on the same matrix, so differences across the results table
are attributable to the estimator.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    KFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── The column contract ───────────────────────────────────────────────────────
# Two views of the model matrix and three lists of what stays out of it. All are
# built by features.make_model_features().
#
#   FEATURE_COLS      numeric inputs
#   ENCODE_COLS       categoricals one-hot encoded beside them
#   GROUP_COLS        fairness strata for auditing
#   LEAKY_COLS        functions of the same nightly distribution as the targets
#   CATEGORICAL_COLS  not part of the contract: the recoded demographics for
#                     describing the cohort. data_extraction.ipynb imports it.

# Derived from analysis.ipynb: the features that survive the correlation and condition-number rules.
FEATURE_COLS = [
    "age", "bmi",
    "log_steps",        # activity level
    "steps_cv",         # activity irregularity: std / mean
    "n_valid_nights",   # wear time
    "employed",
    "education_num", "health_num", "income_num",
]

ENCODE_COLS = ["gender", "race_ethnicity"]

GROUP_COLS = ["age_band", "race_ethnicity"]

LEAKY_COLS = ["pct_short_sleep", "pct_long_sleep", "iqr_sleep_hrs"]

CATEGORICAL_COLS = [
    "gender", "race_ethnicity", "income_tier",
    "education", "employment", "self_rated_health",
]


def prepare_X_y(features_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix and target vector.

    Missing values are left as NaN for the pipeline's imputer to fill inside each
    fold.

    X comes back float64 throughout: the All of Us frame carries nullable
    Int64/Float64, which numpy turns into object arrays once pd.NA is present.
    """
    missing = [c for c in FEATURE_COLS + ENCODE_COLS if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"features_df is missing {missing} — pass the output of "
            f"features.make_model_features()"
        )

    df = pd.get_dummies(features_df.dropna(subset=[target]), columns=ENCODE_COLS,
                        drop_first=True)

    cols = [c for c in df.columns
            if c in FEATURE_COLS or any(c.startswith(e + "_") for e in ENCODE_COLS)]

    return df[cols].astype("float64"), df[target].astype("float64")


def max_correlations(X: pd.DataFrame) -> pd.Series:
    """
    Each column's largest absolute correlation with any other.
    """
    corr = X.astype("float64").corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)

    return pd.Series(corr.max(axis=0), index=X.columns).sort_values(ascending=False)


def condition_number(X: pd.DataFrame) -> float:
    """
    Condition number of the median-filled, standardized design matrix.
    """
    Z = X.astype("float64")
    Z = Z.fillna(Z.median())
    Z = (Z - Z.mean()) / Z.std()

    return float(np.linalg.cond(Z.to_numpy()))


# ── Models ────────────────────────────────────────────────────────────────────

def _pipe(model, scale: bool = False, impute: bool = True) -> Pipeline:
    """Wrap an estimator with median imputation, and scaling where necessary.

    Imputation here rather than in prepare_X_y() so the fill value is learned
    from the training fold only. The final step is always named "model", which is
    how coefficients() reaches the estimator inside a fitted pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))] if impute else []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models() -> dict[str, Pipeline]:
    """
    Named model pipelines. Every call builds the pipelines fresh, 
    so a returned model is always unfitted.
    """
    return {
        "Baseline (mean)": _pipe(DummyRegressor(strategy="mean")),

        # RidgeCV over alphas 1e-3 to 1e3 ties alpha=1.0, so it is left untuned.
        "Ridge":           _pipe(Ridge(alpha=1.0), scale=True),

        # Trees are scale-invariant, so no scaler. min_samples_leaf must stay set:
        # at the sklearn default of 1 the forest grows to pure leaves and loses
        # about 0.02 R² on duration.
        "Random Forest":   _pipe(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1)),

        "HistGBM":         _pipe(HistGradientBoostingRegressor(random_state=42),
                                 impute=False),   # handles NaN natively
    }


def get_model(name: str) -> Pipeline:
    """One unfitted pipeline by name, for the case of wanting a single model."""
    models = get_models()

    if name not in models:
        raise KeyError(f"unknown model {name!r}; available: {list(models)}")

    return models[name]


def coefficients(fitted_model: Pipeline, feature_names) -> pd.Series:
    """
    A fitted linear model's coefficients, labelled by feature.

    Takes a pipeline built by _pipe(), whose estimator is the step named "model".
    """
    estimator = fitted_model.named_steps["model"]

    return pd.Series(np.ravel(estimator.coef_), index=feature_names)


# ── Cross-validation ──────────────────────────────────────────────────────────

# The one fold configuration every scorer here uses. It is
# participant-level because the frame is already one row per person.
CV_FOLDS = KFold(n_splits=5, shuffle=True, random_state=42)


def oof_predictions(X: pd.DataFrame, y: pd.Series, model) -> pd.Series:
    """
    Out-of-fold predictions, indexed like `y`, every row predicted by a model that
    did not train on it.
    """
    return pd.Series(cross_val_predict(model, X, y, cv=CV_FOLDS), index=y.index)


def participant_cv(X: pd.DataFrame, y: pd.Series, model) -> dict:
    """5-fold CV, returning mean RMSE, RMSE SD, MAE and R² across folds."""
    scores = cross_validate(
        model, X, y, cv=CV_FOLDS,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
    )
    rmse = -scores["test_neg_root_mean_squared_error"]

    return {
        "rmse_mean": rmse.mean(),
        "rmse_std":  rmse.std(),
        "mae_mean":  -scores["test_neg_mean_absolute_error"].mean(),
        "r2_mean":   scores["test_r2"].mean(),
    }


def run_all_models(features_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Cross-validate every model and return a results table sorted by RMSE.

    Every model is scored on the same X and y, so the only thing varying across the
    table is the estimator.
    """
    X, y = prepare_X_y(features_df, target)

    results = []
    for name, model in get_models().items():
        print(f"  Running {name}...")
        results.append({"Model": name, **participant_cv(X, y, model)})

    return pd.DataFrame(results).sort_values("rmse_mean")


# ── Fairness evaluation ───────────────────────────────────────────────────────

def fairness_cv(
    features_df: pd.DataFrame,
    target: str,
    model,
    subgroup_col: str,
    min_n: int = 100,
) -> pd.DataFrame:
    """
    Out-of-fold R² by subgroup. Every participant scored by a model that did not
    train on them. Subgroups smaller than `min_n` are skipped.

    Each subgroup's R² uses that subgroup's own mean as the denominator. Because R² = 1 - MSE/Var(group),
    a group whose outcome varies less must be predicted more precisely to score the
    same. Compare groups to each other rather than to the overall figure.

    Returns
    -------
    pd.DataFrame
        subgroup, n, r2, overall out-of-fold R² in .attrs["overall_r2"].
    """
    X, y = prepare_X_y(features_df, target)
    oof = oof_predictions(X, y, model)

    # prepare_X_y drops NaN-target rows but keeps the index, so subgroup labels
    # realign by index. groupby drops unlabelled participants.
    groups = features_df.loc[y.index, subgroup_col]

    rows = [
        {"subgroup": name, "n": len(idx), "r2": round(r2_score(y[idx], oof[idx]), 4)}
        for name, idx in y.groupby(groups, observed=True).groups.items()
        if len(idx) >= min_n
    ]

    out = pd.DataFrame(rows).sort_values("r2").reset_index(drop=True)
    out.attrs["overall_r2"] = round(r2_score(y, oof), 4)
    return out


# ── Held-out diagnostics ──────────────────────────────────────────────────────

def holdout_score(X: pd.DataFrame, y: pd.Series, model,
                  test_size: float = 0.2, random_state: int = 7) -> float:
    """
    R² on a split the model never saw, scored against the training mean.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    model.fit(X_tr, y_tr)
    resid = y_te - model.predict(X_te)

    return 1 - (resid ** 2).sum() / ((y_te - y_tr.mean()) ** 2).sum()


def permutation_scores(X: pd.DataFrame, y: pd.Series, model, n_repeats: int = 5,
                       test_size: float = 0.25, random_state: int = 7) -> pd.Series:
    """
    How much R² each feature is worth, highest first.

    Fits on a training split, then shuffles one column of the held-out split at a
    time and measures the drop in R². Does not inflate columns for having many
    distinct values, and it works for HistGBM.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    model.fit(X_tr, y_tr)
    result = permutation_importance(
        model, X_te, y_te, scoring="r2",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )

    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
