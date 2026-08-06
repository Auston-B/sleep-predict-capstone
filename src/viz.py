"""
viz.py
------
Plotting for the sleep health analysis.

The house style is matplotlib's stock `fivethirtyeight` stylesheet and seaborn
does the aggregating and faceting, so all that is left here is a few helpers —
`use_style()`, `label()`, `refline()`, `finish()`, `save_figure()` — and plot
functions that are optional prep, one seaborn call, and `finish()`.

Every function returns a `matplotlib.figure.Figure`, so a notebook renders one by
making the call its last expression.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .cluster import PHENOTYPES

# ── Constants ─────────────────────────────────────────────────────────────────

SOURCE = "Source: All of Us Research Program CDR v9 · SIADS 699"

SLEEP_GUIDELINE = 7.0    # hours; the reference every duration axis is read against
FIGSIZE = (11, 5.5)      # the default single-panel figure

_MUTED, _FAINT, _INK = "#6e6e6e", "#b0b0b0", "#3c3c3c"


# ── Labels ────────────────────────────────────────────────────────────────────
# Kept short: these land on tick labels and heatmap axes, where a long name either
# overlaps its neighbour or forces the figure wider. Anything unlisted falls
# through the dummy-prefix rule and then to its raw column name.

LABELS = {
    "target_sleep_duration":    "Sleep duration (hrs)",
    "target_sleep_consistency": "Sleep SD (hrs)",
    "mean_sleep_hrs":     "Sleep (hrs)",
    "std_sleep_hrs":      "Sleep SD",
    "iqr_sleep_hrs":      "Sleep IQR",
    "pct_short_sleep":    "Short nights",
    "pct_long_sleep":     "Long nights",
    "n_valid_nights":     "Nights",
    "n_valid_days":       "Days",
    "mean_daily_steps":   "Steps",
    "median_daily_steps": "Median steps",
    "std_daily_steps":    "Steps SD",
    "pct_active_days":    "Active days",
    "pct_female":         "% female",
    "log_steps":          "log(steps)",
    "steps_cv":           "Steps CV",
    "age":                "Age",
    "bmi":                "BMI",
    "employed":           "Employed",
    "education_num":      "Education",
    "health_num":         "Health",
    "income_num":         "Income",
    "gender":             "Gender",
    "education":          "Education",
    "employment":         "Employment",
    "income_tier":        "Income tier",
    "self_rated_health":  "Self-rated health",
    "race_ethnicity":     "Race/Ethnicity",
    "age_band":           "Age band",
    # One-hot columns; the prefix is dropped where the level says it on its own.
    "gender_Male":                              "Male",
    "gender_Other":                             "Gender: other",
    "race_ethnicity_Black or African American": "Race/Ethnicity: Black",
    "race_ethnicity_Hispanic or Latino":        "Race/Ethnicity: Hispanic",
    "race_ethnicity_More than one population":  "Multiracial",
    "race_ethnicity_Other":                     "Race/Ethnicity: other",
    "race_ethnicity_Unknown":                   "Race/Ethnicity: unknown",
    # race_ethnicity_White is absent on purpose: the rule below already renders it
    # exactly as it should read, so listing it would only be a second place to
    # keep in step.
}

_DUMMY_PREFIXES = {"gender": "Gender", "race_ethnicity": "Race/Ethnicity"}


def label(col: str) -> str:
    """A column name as a reader should see it."""
    if col in LABELS:
        return LABELS[col]

    for prefix, shown in _DUMMY_PREFIXES.items():
        if col.startswith(prefix + "_"):
            return f"{shown}: {col[len(prefix) + 1:]}"

    return col


# ── Style and helpers ─────────────────────────────────────────────────────────

def use_style() -> None:
    """Apply the stock 538 stylesheet and hand its cycler to seaborn.

    Re-callable: anything that changes rcParams after import — a dark notebook
    theme, another stylesheet — is undone by calling this again.
    """
    # "default" first, so no earlier theme leaks through. The 538 sheet only
    # overrides what it names, and it names no text colour at all.
    plt.style.use(["default", "fivethirtyeight"])
    sns.set_palette(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    plt.rcParams.update({
        # The stylesheet sets a light background but leaves every text colour to
        # whatever was already active, so under a dark theme it renders white on
        # #f0f0f0. Pin them.
        "text.color": _INK, "axes.labelcolor": _INK, "axes.titlecolor": _INK,
        "xtick.color": _MUTED, "ytick.color": _MUTED,
        # Tuned for one full-width panel; too heavy once a figure carries facets.
        "lines.linewidth": 2.5, "font.size": 11, "axes.titlesize": "medium",
        "figure.dpi": 110, "savefig.dpi": 200,
    })


use_style()

# The four cycler slots the plots name directly — taken after use_style() has run,
# so no colour is defined in this file. PHENOTYPE_PALETTE only says which slot each
# phenotype takes, which is what keeps green/red/blue/amber on the same four names
# dashboard/app.py uses. The stylesheet's cycler is fixed, so re-calling use_style()
# would reproduce these exact colours.
BLUE, RED, AMBER, GREEN = sns.color_palette()[:4]
PHENOTYPE_PALETTE = dict(zip(PHENOTYPES, [GREEN, RED, BLUE, AMBER]))


def finish(fig, title: str, subtitle: str = None, source: str = SOURCE):
    """Add the title / subtitle / source stack, then close the figure.

    The last line of every plot function here. Closing is what stops the inline
    backend drawing the figure twice, once on cell flush and once from the
    returned object's repr.
    """
    h = fig.get_figheight()

    fig.text(0.01, 1 - 0.30 / h, title, fontsize=16, fontweight="bold",
             color=_INK, va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.66 / h, subtitle, fontsize=11, color=_MUTED, va="top")
    if source:
        fig.text(0.01, 0.14 / h, source, fontsize=9, color=_FAINT, va="bottom")

    fig.tight_layout(rect=(0.01, 0.52 / h, 0.99,
                           1 - (1.00 if subtitle else 0.60) / h))
    plt.close(fig)
    return fig


def refline(ax, x: float = None, y: float = None, text: str = None) -> None:
    """A dashed reference line and its label, placed by rule rather than by
    argument: vertical lines label at the top, horizontal ones at the right."""
    if x is not None:
        ax.axvline(x, ls="--", lw=1.4, color=_INK)
        if text:
            ax.text(x, 1.0, f" {text}", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=10, color=_INK)
    if y is not None:
        ax.axhline(y, ls="--", lw=1.4, color=_INK)
        if text:
            ax.text(1.0, y, f"{text} ", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=10, color=_INK)


def save_figure(fig, name: str, directory: str = "reports/figures") -> Path:
    """Write a figure out as a PNG, creating the directory if needed.

    Used when publishing the figures to reports/figures/; the notebooks render
    inline and never call it.
    """
    path = Path(directory) / (name if name.endswith(".png") else f"{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")

    return path


# ══════════════════════════════════════════════════════════════════════════════
# The plots
#
# Each takes its title and subtitle as arguments so the caller can override them.
# They are literal defaults, except in the three functions whose title names one
# of their own arguments — those default to None and build it in the body.
# ══════════════════════════════════════════════════════════════════════════════

# ── Extraction — night-level frames, so these run in the Workbench only ───────

def plot_funnel(funnel_df: pd.DataFrame,
                title: str = "Nights retained at each cleaning stage",
                subtitle: str = "Rules applied in order, on the raw sleep extract"):
    """Nights surviving each rule, for `features.sleep_funnel()` output."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(funnel_df, x="nights", y="stage", orient="h", ax=ax)
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=4, fontsize=10)
    ax.margins(x=0.16)
    ax.set(xlabel="Nights retained", ylabel="")

    return finish(fig, title, subtitle)


def plot_yearly_sleep(by_year: pd.DataFrame, cutoff_year: int = 2017,
                      title: str = "Mean nightly sleep by calendar year",
                      subtitle: str = "The analysis window excludes everything before the line"):
    """Mean nightly sleep per year, from the aggregate the notebook builds."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.lineplot(by_year, x="year", y="mean_hours", marker="o", ax=ax)
    refline(ax, x=cutoff_year - 0.5, text=f"window starts {cutoff_year}")
    ax.set(xlabel="", ylabel="Mean hours asleep")

    return finish(fig, title, subtitle)


def plot_sleep_distribution(sleep_df: pd.DataFrame, bins: int = 60,
                            title: str = "Distribution of nightly sleep duration",
                            subtitle: str = "Every retained night in the cleaned extract"):
    """Histogram of nightly sleep duration across every retained night."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.histplot(sleep_df, x="hours_asleep", bins=bins, ax=ax)
    refline(ax, x=SLEEP_GUIDELINE, text="7-hour guideline")
    ax.set_xlabel("Hours asleep")

    return finish(fig, title, subtitle)


# ── Exploration ───────────────────────────────────────────────────────────────

def plot_targets(features_df: pd.DataFrame,
                 targets=("target_sleep_duration", "target_sleep_consistency"),
                 title: str = "Distribution of the two prediction targets",
                 subtitle: str = ("Duration is mean nightly hours; consistency is the "
                                  "within-person SD of those hours")):
    """The two targets side by side, one panel each."""
    fig, axes = plt.subplots(1, len(targets), figsize=(12.5, 5))

    for ax, col, colour in zip(axes, targets, sns.color_palette()):
        sns.histplot(features_df, x=col, bins=50, color=colour, ax=ax)
        ax.set_xlabel(label(col))

    refline(axes[0], x=SLEEP_GUIDELINE, text="7-hour guideline")

    return finish(fig, title, subtitle)


def plot_group_means(features_df: pd.DataFrame, group_cols, target: str,
                     title: str = None,
                     subtitle: str = "Dot is the group mean, bar spans ±1 SD"):
    """Group mean ± 1 SD for every level of every column in `group_cols`.

    Levels sort alphabetically: the melt drops the category orderings and
    restoring them per facet costs more than the ordering is worth.
    """
    long = (features_df[[target, *group_cols]]
            .astype({c: "string" for c in group_cols})
            .melt(id_vars=target, var_name="variable", value_name="level")
            .fillna({"level": "Missing"})
            .dropna(subset=[target])
            .assign(variable=lambda f: f["variable"].map(label)))

    grid = sns.catplot(long, x=target, y="level", col="variable", col_wrap=3,
                       hue="variable", kind="point", errorbar="sd", orient="h",
                       linestyle="none", sharey=False, sharex=True,
                       height=3.2, aspect=1.3, legend=False)
    grid.set_titles("{col_name}").set(ylabel="", xlabel=label(target))

    for i, ax in enumerate(grid.axes.flat):     # label the line once, not six times
        refline(ax, x=features_df[target].mean(),
                text="cohort mean" if i == 0 else None)

    return finish(grid.figure,
                 title or f"Mean {label(target).lower()} by demographic group",
                 subtitle)


def plot_activity_gradient(features_df: pd.DataFrame, n_bins: int = 10,
                           steps_col: str = "mean_daily_steps",
                           targets=("target_sleep_duration", "target_sleep_consistency"),
                           title: str = "Sleep by daily step count",
                           subtitle: str = ("Participants split into ten equal-sized step "
                                            "bins · band is the 95% interval")):
    """Mean outcome by decile of daily steps, one panel per target."""
    long = (features_df
            .assign(step_bin=pd.qcut(features_df[steps_col].astype("float64"),
                                     n_bins, labels=False, duplicates="drop"))
            .melt(id_vars="step_bin", value_vars=list(targets),
                  var_name="target", value_name="value")
            .assign(target=lambda f: f["target"].map(label)))

    # seed makes the band reproducible: seaborn bootstraps the CI, and unseeded it
    # draws from the global RNG, so the figure came out different on every run.
    grid = sns.relplot(long, x="step_bin", y="value", col="target", hue="target",
                       kind="line", marker="o", errorbar=("ci", 95), seed=0,
                       height=4.4, aspect=1.15, legend=False,
                       facet_kws={"sharey": False})
    grid.set_titles("{col_name}").set(xlabel=f"{label(steps_col)} decile", ylabel="")

    return finish(grid.figure, title, subtitle)


# ── Model results ─────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict, baseline: str = "Baseline (mean)",
                          title: str = "Cross-validated R² by model",
                          subtitle: str = ("Five-fold CV, one feature matrix per target · "
                                           "R² is measured against a mean baseline of 0")):
    """Cross-validated R² by model, one panel per target, for a
    {target: run_all_models() frame} mapping.

    `baseline` scores 0 by construction, so it is dropped rather than drawn.
    R² rather than RMSE, which cannot be read across panels in its own units.
    """
    long = (pd.concat([r.assign(target=t) for t, r in results.items()])
              .query("Model != @baseline")
              .assign(target=lambda f: f["target"].map(label)))
    long["target"] = pd.Categorical(long["target"], [label(t) for t in results])

    grid = sns.catplot(long, x="r2_mean", y="Model", col="target", hue="Model",
                       kind="bar", orient="h", height=4.2, aspect=1.5,
                       sharex=False, legend=False)
    grid.set_titles("{col_name}").set(ylabel="", xlabel="Cross-validated R²")

    for ax in grid.axes.flat:
        for bars in ax.containers:      # hue gives each model its own container
            ax.bar_label(bars, fmt="{:.3f}", padding=4, fontsize=10)
        ax.margins(x=0.18)
        ax.locator_params(axis="x", nbins=4)    # the bars carry the values

    return finish(grid.figure, title, subtitle)


def plot_interpretation(coefficients: pd.Series, importances: pd.Series, target_name: str,
                        title: str = None,
                        subtitle: str = ("Ridge coefficients on standardized features "
                                         "(left) and HistGBM permutation importance (right)"),
                        top_n: int = 12):
    """Direction and reliance for one target, side by side.

    Both arguments are feature-indexed Series: `models.coefficients()` on a fitted
    Ridge pipeline, and `models.permutation_scores()` on HistGBM.
    """
    # Magnitude picks which features appear; sign orders them, so the panel reads
    # most positive to most negative rather than folding the two directions.
    coefs = (coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
             .head(top_n).sort_values(ascending=False).rename(index=label))
    imps = importances.nlargest(top_n).rename(index=label)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 0.42 * top_n + 2.4))

    sns.barplot(x=coefs.to_numpy(), y=coefs.index, orient="h", width=0.6,
                hue=coefs.gt(0), palette={True: BLUE, False: RED},
                legend=False, ax=left)      # the bar's direction says the sign
    sns.barplot(x=imps.to_numpy(), y=imps.index, orient="h", width=0.6, ax=right)

    left.set(title="Ridge coefficient", xlabel="Change per SD of the feature", ylabel="")
    right.set(title="Permutation importance", xlabel="Drop in R²", ylabel="")

    for ax in (left, right):    # a horizontal bar is read against vertical rules
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)

    return finish(fig, title or f"What predicts {target_name}", subtitle)


def plot_calibration(predictions: dict, n_bins: int = 10,
                     title: str = "Predicted against actual, by decile of the prediction",
                     subtitle: str = "Out-of-fold predictions · the dashed diagonal is unbiased"):
    """Mean actual against mean predicted per decile, for a
    {name: (y_true, y_pred)} mapping. Points on the diagonal are unbiased."""
    binned = []
    for name, (y_true, y_pred) in predictions.items():
        d = pd.DataFrame({"pred": np.asarray(y_pred, dtype=float),
                          "actual": np.asarray(y_true, dtype=float)})
        d["bin"] = pd.qcut(d["pred"], n_bins, labels=False, duplicates="drop")
        binned.append(d.groupby("bin").mean().assign(target=name))

    grid = sns.relplot(pd.concat(binned), x="pred", y="actual", col="target",
                       hue="target", kind="line", marker="o", height=4.6,
                       aspect=1.0, legend=False,
                       facet_kws={"sharex": False, "sharey": False})
    grid.set_titles("{col_name}")

    # Both axes are the same quantity and need the same scale, or the identity
    # line is not at 45° and a panel looks better calibrated than it is.
    for ax in grid.axes.flat:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.axline((lo, lo), slope=1, ls="--", lw=1.4, color=_MUTED)

    return finish(grid.figure, title, subtitle)


def _stacked_label_ys(values, gap: float) -> list:
    """Where to put each label so none sits on the one before it.

    `values` arrive highest-first, so a label only ever needs pushing *down*: each
    one takes its own value, or `gap` below its predecessor, whichever is lower.
    Most end up at their line's actual endpoint.
    """
    ys, previous = [], None

    for value in values:
        previous = value if previous is None else min(value, previous - gap)
        ys.append(previous)

    return ys


def plot_fairness_slope(fairness: dict, target_name: str, exclude=(), title: str = None,
                        subtitle: str = ("Out-of-fold R², one panel per stratum · every line "
                                         "starts at the overall score and ends at a subgroup's")):
    """Out-of-fold R² per subgroup as a slope chart, one panel per stratum, for a
    {stratum: fairness_cv() frame} mapping, worst group in red.

    Each subgroup's R² uses its own mean as the denominator, so compare groups
    within a panel rather than against the point they start from.
    """
    blocks = {name: block[~block["subgroup"].astype(str).isin(exclude)]
                        .sort_values("r2", ascending=False)
              for name, block in fairness.items()}
    overall = next(b.attrs["overall_r2"] for b in fairness.values()
                   if "overall_r2" in b.attrs)

    # Just enough separation that one label never sits on the next.
    spread = pd.concat(blocks.values())["r2"]
    gap = (spread.max() - spread.min()) * 0.032

    fig, axes = plt.subplots(1, len(blocks), figsize=(6.6 * len(blocks), 7.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (stratum, block) in zip(axes, blocks.items()):
        worst = block["r2"].idxmin()
        label_ys = _stacked_label_ys(block["r2"], gap)

        for (i, row), label_y in zip(block.iterrows(), label_ys):
            colour = RED if i == worst else BLUE
            ax.plot([0, 1], [overall, row["r2"]], color=colour, marker="o",
                    lw=3 if i == worst else 1.8)
            ax.text(1.04, label_y,
                    f"{row['subgroup']}  {row['r2']:.3f}  n={row['n']:,}",
                    va="center", fontsize=9.5, color=colour,
                    fontweight="bold" if i == worst else "normal")

        ax.text(-0.04, overall, f"Overall\n{overall:.3f}", ha="right", va="center",
                fontsize=11, fontweight="bold", color=_INK)
        # The right of each panel is label gutter; trimming it closes the gap
        # between the panels.
        ax.set(title=label(stratum), xlim=(-0.4, 2.35), xticks=[0, 1],
               xticklabels=["overall", "by subgroup"])
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("R²  (higher is better)")

    return finish(fig, title or f"Model accuracy by subgroup — {target_name}", subtitle)


# ── Phenotypes ────────────────────────────────────────────────────────────────

def plot_cluster_scan(scan_df: pd.DataFrame, chosen_k: int = None,
                      title: str = "Choosing k — inertia and silhouette",
                      subtitle: str = ("Inertia falls at every k by construction, so the "
                                       "silhouette is the sharper of the two signals")):
    """Inertia and silhouette against k, for `cluster.scan_k()` output."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    twin = ax.twinx()
    twin.grid(False)

    sns.lineplot(scan_df, x=scan_df.index, y="inertia", marker="o", color=BLUE, ax=ax)
    sns.lineplot(scan_df, x=scan_df.index, y="silhouette", marker="s", color=RED, ax=twin)

    ax.set(xlabel="k", xticks=list(scan_df.index))
    ax.tick_params(axis="y", colors=BLUE)
    twin.tick_params(axis="y", colors=RED)

    if chosen_k is not None:
        refline(ax, x=chosen_k)

    return finish(fig, title, subtitle)


def plot_cluster_heatmap(profile_df: pd.DataFrame, drop_cols=("N", "pct"),
                         title: str = "Sleep phenotype profiles",
                         subtitle: str = ("Cell is the group mean · colour is the z-score "
                                          "across the four phenotypes")):
    """Standardized cluster means, for `cluster.cluster_profile()` output.

    Colour is the column z-scored across clusters, which is what puts hours,
    proportions and step counts on one scale; the printed number is the raw mean.
    """
    means = (profile_df
             .drop(columns=[c for c in drop_cols if c in profile_df.columns])
             .astype("float64"))
    z = (means - means.mean()) / means.std()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(z.rename(columns=label), annot=means.round(2), fmt="g",
                cmap="RdBu_r", center=0, linewidths=1,
                cbar_kws={"label": "SD from the average phenotype"}, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set(xlabel="", ylabel="")

    return finish(fig, title, subtitle)


def plot_cluster_scatter(features_df: pd.DataFrame, labels: pd.Series,
                         sample_n: int = 8000,
                         title: str = "Sleep phenotypes by duration and variability",
                         subtitle: str = "Two of the four clustering features"):
    """Mean nightly sleep against within-person SD, coloured by phenotype.

    Two of the four clustering features are the axes, so this shows what the
    profile table cannot: the groups have no gaps between them.
    """
    d = pd.DataFrame({
        "mean_sleep_hrs": pd.to_numeric(features_df["mean_sleep_hrs"], errors="coerce"),
        "std_sleep_hrs":  pd.to_numeric(features_df["std_sleep_hrs"], errors="coerce"),
        "phenotype":      labels,
    }).dropna()

    if sample_n and len(d) > sample_n:
        d = d.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.scatterplot(d, x="mean_sleep_hrs", y="std_sleep_hrs", hue="phenotype",
                    hue_order=PHENOTYPES, palette=PHENOTYPE_PALETTE,
                    s=9, alpha=0.4, linewidth=0, ax=ax)
    refline(ax, x=SLEEP_GUIDELINE, text="7-hour guideline")

    ax.legend(loc="lower right", markerscale=2)     # the cloud's one empty corner
    ax.set(xlabel=label("mean_sleep_hrs"), ylabel=label("std_sleep_hrs"))

    return finish(fig, title, subtitle)
