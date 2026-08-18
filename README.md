# Predicting and Phenotyping Sleep Health in a Diverse National Cohort

Evidence from the *All of Us* Research Program

Sophia Boettcher, Auston Balwinski, Hunter Belous, Jared Fox  
School of Information, University of Michigan  
SIADS 699 Capstone — August 2026

[Final report](reports/final_report.pdf) · [Slide deck](https://auston-b.github.io/sleep-predict-capstone/) · [Dashboard](https://sleep-predict-capstone.streamlit.app) · [Poster](reports/poster.pdf)

## Overview

This capstone applies supervised and unsupervised machine learning to Fitbit-derived sleep data from
45,259 participants in the NIH *All of Us* Research Program (Controlled Tier, CDR v9). It asks how
well sociodemographic, health, and physical activity characteristics predict sleep duration and
night-to-night consistency, what sleep behavioral phenotypes emerge from unsupervised clustering, and
how evenly predictive accuracy holds across racial/ethnic and age subgroups. All four estimators are
scored on a single feature matrix, so any spread across the results belongs to the estimator alone.

## Data and access

All analysis was conducted inside the [*All of Us* Researcher
Workbench](https://workbench.researchallofus.org/). Access to the Controlled Tier dataset requires
registration and approval through the program.

**No individual-level data are included in this repository.** What is published here is aggregate:
summary statistics, cross-validation scores, model coefficients, cluster centroids, and figures. The
eight CSVs in `dashboard/` are population-level summaries from which no participant can be
identified.

## Methods

| Component | Details |
|---|---|
| Data source | *All of Us* CDR v9, `fitbit_sleep_daily_summary` |
| Data extraction | BigQuery; see [`notebooks/data_extraction.ipynb`](notebooks/data_extraction.ipynb) |
| Cohort | 45,259 adults with at least 30 valid Fitbit nights and a completed Basics survey, restricted during extraction from the 48,688 who cleared the 4-night floor |
| Outcomes | Mean nightly sleep hours (duration) and the within-person SD of those hours (consistency) |
| Feature matrix | 9 numeric columns plus 8 one-hot dummies, after leakage and collinearity pruning (max \|r\| 0.53, condition number 5.2) |
| Models | Mean baseline, Ridge, random forest, histogram gradient boosting, all on the same matrix under 5-fold `KFold` cross-validation |
| Interpretation | Ridge coefficients for direction, HistGBM permutation importance for reliance; impurity importance is not used |
| Clustering | KMeans, k = 4, on four sleep-behavior columns, selected by inertia and silhouette |
| Fairness evaluation | Out-of-fold R² stratified by race/ethnicity and age band |
| Environment | *All of Us* Researcher Workbench (Terra) for extraction, local for analysis; Python 3.11 |



## Results

### Prediction

| Model | Duration R² | Duration RMSE ± SD | Consistency R² | Consistency RMSE ± SD |
|---|---|---|---|---|
| Baseline (mean) | −0.000 | 0.646 ± 0.009 | −0.000 | 0.296 ± 0.002 |
| Ridge | 0.103 | 0.612 ± 0.005 | 0.250 | 0.257 ± 0.001 |
| Random forest | 0.095 | 0.615 ± 0.007 | 0.269 | 0.253 ± 0.002 |
| HistGBM | 0.104 | 0.612 ± 0.006 | 0.277 | 0.252 ± 0.002 |

Both targets clear the baseline. On consistency, histogram gradient boosting reaches R² = 0.277
against Ridge's 0.250, an 11% relative gain and about three times either model's fold-to-fold
variation. On duration the two are indistinguishable at 0.104 and 0.103. Held-out scores land
slightly above the cross-validated figures and out-of-fold predictions are unbiased to within 0.016
hours, so the modest duration R² reflects real unexplained variance.

The targets rest on different features. Duration is carried by person-level attributes: activity
level (permutation importance 0.045), male gender (0.043), White race/ethnicity (0.039), age (0.031),
BMI (0.030). Consistency is dominated by the step coefficient of variation at 0.133, about three
times age (0.045) and activity level (0.030) behind it. That relationship is curved, its slope holds
flat across age quartiles, and no interaction term earned a column. Nights tracked scores 0.007 on
duration and enters the matrix as a wear-time control.

### Phenotypes

KMeans with k = 4 was fitted to four standardized sleep-behavior columns: mean nightly sleep, the
within-person SD, and the shares of nights under six and over nine hours. Demographics and activity
were described after fitting, so the partition reflects sleep behavior alone.

| Phenotype | N | % | Mean sleep (hrs) | SD | Nights < 6h | Mean steps | Mean BMI |
|---|---|---|---|---|---|---|---|
| Consistent Good Sleepers | 17,796 | 39.3% | 7.13 | 0.95 | 12% | 7,788 | 27.6 |
| Short but Regular | 12,186 | 26.9% | 6.13 | 1.07 | 47% | 7,472 | 30.4 |
| Chronic Short & Variable | 10,703 | 23.6% | 6.89 | 1.44 | 29% | 6,181 | 30.8 |
| Variable Long Sleepers | 4,574 | 10.1% | 7.97 | 1.48 | 11% | 5,386 | 30.0 |

Chronic Short & Variable sits at the cohort's mean duration with the highest variability, the most
nights at both extremes, the highest BMI, and the youngest mean age, which marks it as the group of
most clinical interest. Participants form one continuous cloud, so these
names describe cuts through a distribution and are not clinical categories.

### Fairness

Subgroups were scored with out-of-fold R², so every participant is evaluated by a model that did not
train on them. Two disparities appear, on two targets, affecting largely different people.

The largest is by age on duration: accuracy declines monotonically from 0.139 in the 18–40 band to
0.031 for participants 81 and over, a fall of 77%. Consistency shows an age effect in a different
shape, peaking at 41–60 (0.268) and falling to 0.180 in the oldest band.

The racial disparities differ between the two targets in shape more than in size. On consistency,
White participants score highest at 0.285, with Black participants at 0.170 and Asian participants at
0.169, lower by 40% and 41%. On duration that ordering does not hold: White participants (0.072) sit
level with Black participants (0.072), both below Hispanic or Latino participants at 0.107.
Training-data composition, unmeasured confounders, and wearable staging validated primarily in
younger and lighter-skinned populations are all consistent with these results, and this analysis
cannot separate them.

## Repository contents

```
.
├── notebooks/
│   ├── data_extraction.ipynb   # BigQuery cohort build; writes the participant-level extract
│   └── analysis.ipynb          # Modeling, interpretation, fairness, phenotyping
├── src/
│   ├── features.py             # Cleaning and participant-level feature engineering
│   ├── models.py               # Feature matrix, cross-validation, fairness evaluation
│   ├── cluster.py              # KMeans phenotyping and cluster naming
│   └── viz.py                  # Figure functions
├── dashboard/
│   ├── app.py                  # Streamlit application
│   ├── requirements.txt
│   └── *.csv                   # Eight aggregate result tables
├── presentation/
│   ├── index.html              # Slide deck, published to GitHub Pages
│   └── fig_*.png               # The seven figures the deck embeds
├── reports/
│   ├── final_report.md         # Report source
│   ├── final_report.pdf        # The same report, typeset
│   ├── poster.pdf              # Conference poster, 36 × 48 in
│   └── figures/                # All 17 publication figures
├── .github/workflows/          # Publishes presentation/ to GitHub Pages
├── requirements.txt
└── LICENSE
```

**`notebooks/data_extraction.ipynb`** holds every query behind the analytic cohort and is the only
writer of the participant-level extract. It runs in the Researcher Workbench in three layers:
profiling queries that justify each inclusion threshold, five extraction queries collapsing 39
million night-level rows to one row per participant, and restriction and validation covering the
30-night floor, the cohort funnel, the missingness audit, and the demographic recode audit. Both
night floors are applied here, so the extract is already the analytic cohort. Every finding is
written in as a table, so the notebook reads without Workbench access, but running it requires
Controlled Tier credentials.

**`notebooks/analysis.ipynb`** covers feature selection under explicit leakage and collinearity
rules, four models on one matrix, interpretation, calibration, subgroup fairness, and phenotyping. It
runs locally on the participant-level extract, writes nothing to disk, and produces only aggregates.
It begins at feature construction, since the cohort arrives already restricted.

The figures in `reports/figures/` and the eight CSVs in `dashboard/` are generated by code, so the
numbers in the report, the deck, and the dashboard cannot drift apart. The plotting and profiling
functions are published in `src/`; the driver that calls them is kept local, as with the poster
tooling. Three figures come from the extraction run instead, because they need a grain that does not
leave the Workbench (`fig_yearly_sleep`, `fig_funnel`, `fig_sleep_distribution`), and two are
hand-authored schematics (`fig_data_flow`, `fig_ai_workflow`).

## Published artifacts

**Slide deck.** A 21-slide deck at
[auston-b.github.io/sleep-predict-capstone](https://auston-b.github.io/sleep-predict-capstone/)
covering the question, the cohort, the models, the phenotypes, and the fairness results. It is served
from `presentation/` and redeployed by GitHub Actions on each push to `main`.

**Dashboard.** An interactive Streamlit application at
[sleep-predict-capstone.streamlit.app](https://sleep-predict-capstone.streamlit.app) presenting model
results, cluster profiles, and fairness metrics. Everything it displays is aggregate. To run it
locally:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Poster.** A 36 × 48 in conference poster, [`reports/poster.pdf`](reports/poster.pdf), presenting
the cohort, the model comparison, the interpretation panels, and both fairness results on one sheet.
Its layout is produced by a script; the editable sources stay local and the exported print file ships
here.

## Figures

All figures are in `reports/figures/`. The final column gives the number the report assigns each one.

| Filename | Description | In report |
|---|---|---|
| `fig_targets.png` | Distribution of the two prediction targets | Figure 1 |
| `fig_group_means.png` | Sleep duration by demographic group | Figure 2 |
| `fig_model_comparison.png` | Cross-validated R² by model | Figure 3 |
| `fig_interpretation_duration.png` | Ridge coefficients and permutation importance, duration | Figure 4 |
| `fig_interpretation_consistency.png` | Ridge coefficients and permutation importance, consistency | Figure 5 |
| `fig_cluster_heatmap.png` | Cluster profile heatmap | Figure 6 |
| `fig_fairness_duration.png` | Out-of-fold R² by subgroup, duration | Figure 7 |
| `fig_fairness_consistency.png` | Out-of-fold R² by subgroup, consistency | Figure 8 |
| `fig_yearly_sleep.png` | Mean nightly sleep by calendar year, the evidence for the 2017 window | Figure A1 |
| `fig_calibration.png` | Predicted against actual by decile of the prediction | Figure A2 |
| `fig_cluster_scan.png` | Choosing k: inertia and silhouette | Figure A3 |
| `fig_cluster_scatter.png` | Phenotypes by duration and variability | Figure A4 |
| `fig_data_flow.png` | BigQuery to published artifacts, with the trust boundary | Figure A5 |
| `fig_ai_workflow.png` | Who ran what, and the Controlled Tier boundary | Figure A6 |
| `fig_funnel.png` | Nights retained at each cleaning stage | — |
| `fig_sleep_distribution.png` | Distribution of nightly sleep across every retained night | — |
| `fig_activity_gradient.png` | Sleep by daily step decile | — |

The last three document the night-level cleaning and the
activity gradient, which the report states in prose.

## Citation

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

## Acknowledgments

This work rests on the *All of Us* Research Program participants, who contributed years of their own
wearable data to research. We also thank the SIADS 699 teaching team at the University of Michigan
School of Information. The *All of Us* Research Program requires that publications using its data
carry the funding acknowledgment below, reproduced verbatim.

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

## License

Released under the [MIT License](LICENSE).
