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
# Two views of the model matrix, plus three lists of what stays out of it. All are
# built by features.make_model_features().
#
#   FEATURE_COLS      numeric inputs
#   ENCODE_COLS       categoricals one-hot encoded beside them
#   GROUP_COLS        fairness strata — audited, never fitted
#   LEAKY_COLS        functions of the same nightly distribution as the targets
#   CATEGORICAL_COLS  not part of the contract: the recoded demographics, for
#                     describing the cohort. data_extraction.ipynb imports it.

# Activity is two columns and wear time is one; the rest of both blocks was dropped
# for collinearity, taking the largest pairwise |r| from 0.99 to 0.53 and the
# condition number from 24.5 to 5.2. notebooks/analysis.ipynb §4 has the working.
FEATURE_COLS = [
    "age", "bmi",
    "log_steps",        # activity level
    "steps_cv",         # activity irregularity: std / mean
    "n_valid_nights",   # wear time
    "employed",
    "education_num", "health_num", "income_num",
]

# Unordered only. The ordered survey variables are already numeric above, and
# encoding one both ways makes the design matrix singular.
ENCODE_COLS = ["gender", "race_ethnicity"]

# No overlap with FEATURE_COLS: a model trained on a group flag would be auditing
# its own input.
GROUP_COLS = ["age_band", "race_ethnicity"]

# A model given these predicts a target largely from itself. prepare_X_y()
# intersects FEATURE_COLS with the columns present, so dropping them is enough.
LEAKY_COLS = ["pct_short_sleep", "pct_long_sleep", "iqr_sleep_hrs"]

CATEGORICAL_COLS = [
    "gender", "race_ethnicity", "income_tier",
    "education", "employment", "self_rated_health",
]


def prepare_X_y(features_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix and target vector.

    Missing values are left as NaN for the pipeline's imputer to fill inside each
    fold — filling them here would compute the median over train and test together
    and leak it across every split.

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
    Each column's largest absolute correlation with any other, highest first.

    The collinearity rule: a column whose strongest partner is above 0.9 is
    measuring something another column already measures.
    """
    corr = X.astype("float64").corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)

    return pd.Series(corr.max(axis=0), index=X.columns).sort_values(ascending=False)


def condition_number(X: pd.DataFrame) -> float:
    """
    Condition number of the median-filled, standardized design matrix.

    Reported alongside the pairwise rule as corroboration rather than as a second
    criterion: on this feature set it never crossed its own alarm at 30, so every
    drop was made by the correlation rule.

    Missing values are median-filled here as a diagnostic convenience. Do not copy
    that into the modelling path, where imputation belongs inside the CV fold.
    """
    Z = X.astype("float64")
    Z = Z.fillna(Z.median())
    Z = (Z - Z.mean()) / Z.std()

    return float(np.linalg.cond(Z.to_numpy()))


# ── Models ────────────────────────────────────────────────────────────────────

def _pipe(model, scale: bool = False, impute: bool = True) -> Pipeline:
    """Wrap an estimator with median imputation, and scaling where it matters.

    Imputation lives here rather than in prepare_X_y() so the fill value is learned
    from the training fold only. The final step is always named "model", which is
    how coefficients() reaches the estimator inside a fitted pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))] if impute else []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models(names: list = None) -> dict[str, Pipeline]:
    """
    Named model pipelines: a reference point, a regularized linear model, bagged
    trees, boosted trees. Pass `names` to restrict the result, in that order.

    Every call builds the pipelines fresh, so a returned model is always unfitted.

    Lasso and ElasticNet are not here — they score worse than Ridge on every metric
    and both targets, which is what you would expect given there is nothing in this
    feature set for an L1 penalty to select away.
    """
    models = {
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

    if names is None:
        return models

    unknown = [n for n in names if n not in models]
    if unknown:
        raise KeyError(f"unknown model name(s) {unknown}; available: {list(models)}")

    return {n: models[n] for n in names}


def get_model(name: str) -> Pipeline:
    """One unfitted pipeline by name, for the common case of wanting a single model."""
    return get_models([name])[name]


def coefficients(fitted_model, feature_names) -> pd.Series:
    """
    A fitted linear model's coefficients, labelled by feature.

    Accepts either a pipeline built by _pipe() — whose estimator is the step named
    "model" — or a bare estimator.
    """
    estimator = getattr(fitted_model, "named_steps", {}).get("model", fitted_model)

    return pd.Series(np.ravel(estimator.coef_), index=feature_names)


# ── Cross-validation ──────────────────────────────────────────────────────────

def _folds(n_splits: int = 5) -> KFold:
    """The one fold configuration every scorer here uses.

    A plain KFold: it is participant-level because the frame is already one row per
    person, not because the split is grouped.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def oof_predictions(X: pd.DataFrame, y: pd.Series, model, n_splits: int = 5) -> pd.Series:
    """
    Out-of-fold predictions, indexed like `y` — every row predicted by a model that
    did not train on it.

    cross_val_predict clones per fold, so the caller's model is left unfitted.
    """
    return pd.Series(cross_val_predict(model, X, y, cv=_folds(n_splits)), index=y.index)


def participant_cv(X: pd.DataFrame, y: pd.Series, model, n_splits: int = 5) -> dict:
    """5-fold CV, returning mean RMSE, RMSE SD, MAE and R² across folds."""
    scores = cross_validate(
        model, X, y, cv=_folds(n_splits),
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
    )
    rmse = -scores["test_neg_root_mean_squared_error"]

    return {
        "rmse_mean": rmse.mean(),
        "rmse_std":  rmse.std(),
        "mae_mean":  -scores["test_neg_mean_absolute_error"].mean(),
        "r2_mean":   scores["test_r2"].mean(),
    }


def run_all_models(features_df: pd.DataFrame, target: str,
                   estimators: dict = None) -> pd.DataFrame:
    """
    Cross-validate every model and return a results table sorted by RMSE.

    Defaults to all four from get_models(), so the only thing varying across the
    table is the model.
    """
    X, y = prepare_X_y(features_df, target)
    estimators = get_models() if estimators is None else estimators

    results = []
    for name, model in estimators.items():
        print(f"  Running {name}...")
        results.append({"Model": name, **participant_cv(X, y, model)})

    return pd.DataFrame(results).sort_values("rmse_mean")


# ── Fairness evaluation ───────────────────────────────────────────────────────

def fairness_cv(
    features_df: pd.DataFrame,
    target: str,
    model,
    subgroup_col: str,
    n_splits: int = 5,
    min_n: int = 100,
) -> pd.DataFrame:
    """
    Out-of-fold R² by subgroup — every participant scored by a model that did not
    train on them. Subgroups smaller than `min_n` are skipped, since R² on a
    handful of people is noise.

    Each subgroup's R² uses that subgroup's own mean as the denominator, since
    r2_score() derives it from the y it is handed. Because R² = 1 - MSE/Var(group),
    a group whose outcome varies less must be predicted more precisely to score the
    same — so compare groups to each other rather than to the overall figure.

    Returns
    -------
    pd.DataFrame
        subgroup, n, r2 — worst first, with the overall out-of-fold R² in
        .attrs["overall_r2"].
    """
    X, y = prepare_X_y(features_df, target)
    oof = oof_predictions(X, y, model, n_splits)

    # prepare_X_y drops NaN-target rows but keeps the index, so subgroup labels
    # realign by index. groupby drops unlabelled participants for us.
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
    R² on a split the model never saw, scored against the *training* mean — the
    test set's own mean is information the model did not have.

    A sanity check on the CV figures, not a replacement for them: a single 20%
    split varies by more than the selection optimism it is looking for. Settling
    that properly would need nested CV.
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
    time and measures the drop in R². Unlike a tree's built-in
    `feature_importances_`, this does not inflate columns simply for having many
    distinct values, and it works for any estimator — including HistGBM, which
    exposes no importances at all.

    It answers "what is lost if this feature goes", which is not the same question
    as "what does this feature explain on its own". A column can score low here and
    still carry signal, if another column covers for it.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    model.fit(X_tr, y_tr)
    result = permutation_importance(
        model, X_te, y_te, scoring="r2",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )

    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
