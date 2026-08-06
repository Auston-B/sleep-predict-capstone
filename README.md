# 😴 Sleep Health Prediction & Phenotyping in the *All of Us* Research Program

[![Live dashboard](https://img.shields.io/badge/Streamlit-Live%20dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://sleep-predict-capstone.streamlit.app)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![All of Us CDR v9](https://img.shields.io/badge/All%20of%20Us-CDR%20v9-purple?logo=nih)
![Fitbit Wearables](https://img.shields.io/badge/Fitbit-Wearables-teal)

> **SIADS 699 Capstone Project** · University of Michigan School of Information · July 2026  
> *Sophia Boettcher, Auston Balwinski, Hunter Belous, Jared Fox*

---

## 📖 Overview

This project applies machine learning to Fitbit-derived sleep data from **45,259 participants** in the NIH *All of Us* Research Program (CDR v9) to (1) predict individual sleep duration and consistency using sociodemographic, health, and physical activity features, (2) identify distinct sleep behavioral phenotypes via unsupervised clustering, and (3) evaluate predictive model fairness across racial/ethnic and age subgroups. Every model is scored on one feature matrix, so differences across the results table are attributable to the estimator rather than the inputs.

---

## 🔑 Key Findings

- **Model class matters for one target, not the other:** HistGBM reaches R² = 0.277 on sleep consistency, an **11% gain** over Ridge (0.250), but ties Ridge on duration (0.104 vs 0.103). What limits duration prediction is missing information, not model form
- **Top predictors for duration:** activity level (permutation importance 0.045), male gender (0.043), White race/ethnicity (0.039), age (0.031), BMI (0.030)
- **Top predictor for consistency:** activity *irregularity* — the step coefficient of variation (0.133), roughly 3× the next feature. The activity–consistency relationship is curved rather than age-moderated, and no interaction term survived screening
- **Wear time is a control, not a determinant.** Nights tracked scores 0.007 on duration under permutation importance on held-out data
- **4 sleep phenotypes identified** (KMeans on four sleep-behaviour columns):
  - 🟢 **Consistent Good Sleepers** (39%, n=17,796) — adequate duration, lowest variability, highest activity, lowest BMI
  - 🟡 **Short but Regular** (27%, n=12,186) — 47% of nights under 6 h, but predictable schedules
  - 🔴 **Chronic Short & Variable** (24%, n=10,703) — at the cohort's mean duration but with the highest variability, the most nights at both extremes, highest BMI and youngest mean age.
  - 🟣 **Variable Long Sleepers** (10%, n=4,574) — least active by a wide margin
- **Two distinct fairness gaps, on two different targets.** The racial gaps differ in *pattern* more than in size. On **consistency** the ordering is graded against White participants, who score highest (0.285), with Black (0.170) and Asian (0.169) participants roughly 40% below. On **duration** that ordering disappears — White (0.072) sits level with Black (0.072) and well below Hispanic or Latino (0.107). The largest single disparity is by age on duration, where accuracy falls **77%** from 0.139 (18–40) to 0.031 (81+)
- **Cohort:** 45,259 participants; mean age 56.7, 66.6% female, 70.1% White; mean sleep 6.89 hrs/night, 25.4% of nights under 6 h

---

## 📁 Repository Structure

```
sleep-predict-capstone/
├── README.md
├── LICENSE
├── requirements.txt
├── .github/workflows/           # Publishes presentation/ to GitHub Pages
├── notebooks/
│   ├── data_extraction.ipynb   # BigQuery cohort build (documented queries) — writes
│   │                           # the analytic cohort, both night floors applied
│   └── analysis.ipynb          # The supervised analysis and phenotyping
├── src/
│   ├── features.py             # Cleaning + participant-level feature engineering
│   ├── models.py               # Feature matrix, CV, fairness evaluation
│   ├── cluster.py              # KMeans phenotyping and the naming rules
│   └── viz.py                  # Every figure
├── synthetic_data/
│   └── generate.py             # Generates All of Us-shaped CSVs locally; output not
│                               # committed, and not read by the pipeline
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── requirements.txt
│   └── *.csv                   # Aggregate result files
├── presentation/
│   ├── index.html              # Slide deck, published to auston-b.github.io/sleep-predict-capstone
│   └── fig_*.png               # The seven figures the deck embeds
└── reports/
    ├── final_report.md         # The report, plus Appendices A–C (methodological
    │                           # asides, data flow, AI use)
    ├── final_report.pdf        # The same report, typeset
    └── figures/                # All publication figures
```


### 📓 `notebooks/data_extraction.ipynb`

The query record behind the analytic cohort, and the only writer of the participant-level extract. It
runs inside the *All of Us* Researcher Workbench and is structured in three layers: **profiling**
(queries that establish what the data looks like, each one justifying a specific inclusion
threshold), **extraction** (five queries collapsing 39M night-level rows to one row per participant),
and **restriction & validation** (the ≥30-night floor with the sampling-variance argument behind it,
then the cohort funnel, the missingness audit and the demographic recode audit).

Both night floors are applied here, so the pickle it writes *is* the analytic cohort — nothing
downstream reshapes it.

It is readable without Workbench access — every finding is written into the notebook as a table — but
it cannot be executed without Controlled Tier credentials, and it emits no individual-level output.

### 📓 `notebooks/analysis.ipynb`

The analysis itself: exploration, feature selection under explicit leakage and collinearity rules,
four models on one matrix, interpretation, calibration, subgroup fairness, and phenotyping. **It runs
locally** — the sole input is the participant-level extract, nothing is written to disk, and every
output is an aggregate.

It starts at feature construction. The cohort arrives already restricted, so nothing in this notebook
changes who is in it.

### ♻️ How the artifacts are produced

Every data figure in `reports/figures/` and all eight CSVs in `dashboard/` are generated
programmatically by the `src/` modules — regenerated rather than transcribed, so the numbers in the
report, the deck and the dashboard cannot drift apart.

Most are written by `src/viz.py` from the participant-level extract. The three night- and day-level
figures (`fig_yearly_sleep`, `fig_funnel`, `fig_sleep_distribution`) come from the extraction run
instead, because they need a grain that never leaves the Workbench. The two schematic diagrams
(`fig_data_flow`, `fig_ai_workflow`) are hand-authored illustrations rather than plots of data, drawn
by local tooling that is not part of this repository.

---

## 🚀 The Dashboard

**→ [sleep-predict-capstone.streamlit.app](https://sleep-predict-capstone.streamlit.app)**

The interactive Streamlit dashboard visualizes model results, cluster profiles, and fairness metrics.
Nothing to install and no authentication required — everything it displays is aggregate only.

To run the same app locally instead:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. The app reads the eight CSVs in `dashboard/` relative to its own
file, so it runs from any working directory.

---

## 🔬 Methods Overview

| Component | Details |
|---|---|
| **Data source** | *All of Us* CDR v9, `fitbit_sleep_daily_summary` |
| **Data extraction** | BigQuery — see [`notebooks/data_extraction.ipynb`](notebooks/data_extraction.ipynb) for the profiling and extraction queries |
| **Cohort** | 45,259 adults with ≥30 valid Fitbit nights + Basics survey, restricted during extraction from the 48,688 who cleared ≥4 nights |
| **Outcomes** | Mean nightly sleep hours (duration); within-person SD of those hours (consistency) |
| **Feature matrix** | 9 numeric columns + 8 one-hot dummies = 17, after leakage and collinearity pruning (max \|r\| 0.53, condition number 5.2) |
| **Models** | Mean baseline, Ridge, Random Forest, HistGBM — all on the same matrix, 5-fold `KFold` CV |
| **Interpretation** | Ridge coefficients (direction) + HistGBM permutation importance (reliance); no impurity importance |
| **Clustering** | KMeans k=4 on four sleep-behaviour columns; inertia + silhouette selection |
| **Fairness evaluation** | Out-of-fold R² stratified by race/ethnicity and age band |
| **Environment** | *All of Us* Researcher Workbench (Terra) for extraction; local for analysis. Python 3.11 |

---

## 🗃️ Data Notice

> **⚠️ No individual-level data are included in this repository.**

All analyses were conducted within the secure [*All of Us* Researcher Workbench](https://workbench.researchallofus.org/) (Terra cloud environment). Access to the Controlled Tier dataset requires registration and approval through the *All of Us* Research Program.

This public repository contains **aggregate results only** (summary statistics, model coefficients, cluster centroids, and figures). CSV files in `dashboard/` contain only population-level summaries with no records that could identify individual participants.

---

## 📊 Publication Figures

All figures live in `reports/figures/`. The "In report" column gives the figure number the report
assigns each one, so a figure in the report can be traced back to the file that produced it.

| Filename | Description | In report |
|---|---|---|
| `fig_targets.png` | Distribution of the two prediction targets | Figure 1 |
| `fig_group_means.png` | Sleep duration by demographic group | Figure 2 |
| `fig_model_comparison.png` | Cross-validated R² by model | Figure 3 |
| `fig_interpretation_duration.png` | Ridge coefficients + permutation importance — duration | Figure 4 |
| `fig_interpretation_consistency.png` | Ridge coefficients + permutation importance — consistency | Figure 5 |
| `fig_cluster_heatmap.png` | Cluster profile heatmap | Figure 6 |
| `fig_fairness_duration.png` | Out-of-fold R² by subgroup — duration | Figure 7 |
| `fig_fairness_consistency.png` | Out-of-fold R² by subgroup — consistency | Figure 8 |
| `fig_yearly_sleep.png` | Mean nightly sleep by calendar year — the evidence for the 2017 window | Figure A1 |
| `fig_calibration.png` | Predicted vs. actual by decile of the prediction | Figure A2 |
| `fig_cluster_scan.png` | Choosing k — inertia and silhouette | Figure A3 |
| `fig_cluster_scatter.png` | Phenotypes by duration and variability | Figure A4 |
| `fig_data_flow.png` | BigQuery to published artifacts, with the trust boundary | Figure A5 |
| `fig_ai_workflow.png` | Who ran what, and the Controlled Tier boundary | Figure A6 |
| `fig_funnel.png` | Nights retained at each cleaning stage | — |
| `fig_sleep_distribution.png` | Distribution of nightly sleep across every retained night | — |
| `fig_activity_gradient.png` | Sleep by daily step decile | — |

The last three were produced during the analysis but cut from the final report for length. They are
kept here because they document the night-level cleaning and the activity gradient, which the report
asserts in prose.

---

## 📝 Citation

If you use or build on this work, please cite:

```bibtex
@misc{boettcher2026sleep,
  author       = {Boettcher, Sophia and Balwinski, Auston and Belous, Hunter and Fox, Jared},
  title        = {Predicting and Phenotyping Sleep Health in a Diverse National Cohort:
                  Evidence from the All of Us Research Program},
  year         = {2026},
  howpublished = {SIADS 699 Capstone, University of Michigan},
  url          = {https://github.com/Auston-B/sleep-predict-capstone}
}
```

---

## 🙏 Acknowledgments

This work rests on the *All of Us* Research Program participants, who contributed years of their own
wearable data to research. Thanks also to the University of Michigan School of Information SIADS 699
teaching team for their guidance throughout this capstone.

The *All of Us* Research Program requires that publications using its data carry the funding
acknowledgment below. It is reproduced verbatim, and collapsed only so it does not crowd the page.

<details>
<summary><b>Required <i>All of Us</i> funding acknowledgment</b></summary>

The *All of Us* Research Program is supported by the National Institutes of Health, Office of the
Director: Regional Medical Centers: 1 OT2 OD026549; 1 OT2 OD026554; 1 OT2 OD026557; 1 OT2 OD026556;
1 OT2 OD026550; 1 OT2 OD 026552; 1 OT2 OD026553; 1 OT2 OD026548; 1 OT2 OD026551; 1 OT2 OD026555;
IAA #: AOD 16037; Federally Qualified Health Centers: HHSN 263201600085U; Data and Research Center:
5 U2C OD023196; Biobank: 1 U24 OD023121; The Participant Center: U24 OD023176; Participant Technology
Systems Center: 1 U24 OD023163; Communications and Engagement: 3 OT2 OD023205; 3 OT2 OD023206; and
Community Partners: 1 OT2 OD025277; 3 OT2 OD025315; 1 OT2 OD025337; 1 OT2 OD025276.

</details>

---

*Licensed under the [MIT License](LICENSE). See LICENSE for details.*
