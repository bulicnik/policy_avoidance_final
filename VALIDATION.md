# Reproducibility checks

Checked on November 6, 2026, using only the repository's public inputs for the
survey EFA, factor regressions, condition interactions, and political-orientation
models.

## Results

- The survey script completed twice with seed 42. All 10 EFA output files,
  including the scree plot, were byte-identical between runs.
- Factor loadings, explained variance, and all three factor-regression tables
  matched the previously committed numeric results within an absolute tolerance
  of 1e-10. The final five-factor solution is unchanged; parallel analysis
  recommends seven factors.
- The political-orientation script completed with 119 participants and reproduced
  the manuscript's reported slopes, tests, and p-values.
- All seven automated tests passed. They check documentation links, participant links, eligibility,
  sample sizes, factor results, interaction labels, and political-model results.
- All 11 input CSV files were unchanged by the execution checks.

The EFA uses 124 records; the factor regressions and interactions use 118
complete cases. All five interaction tests are now exported, not just the
significant subset. `fit_summary.csv` separately records the parallel-analysis
recommendation, fitted factor count, seed, and regression sample size.

## Environment and limits

The execution checks used the existing research environment: Python 3.8.3,
NumPy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, and rpy2 3.6.1; R 4.5.2 with
psych 2.6.1, nFactors 2.4.1.2, GPArotation 2025.3.1, lme4 1.1.38, lmerTest
3.2.0, and tidyverse 2.0.0.

The pinned packages in `requirements.txt` target Python 3.12 or later. Their
release availability and Python requirements were checked against PyPI; a clean
installation of those pins was not tested. Exact repeatability across other
package versions or platforms is not guaranteed.

The Stan fits, CMDT behavioral models, cross-task GLMMs, and figure-generation
scripts were not rerun in this check. The automated tests inspect saved outputs;
they do not execute the analyses themselves.

After rerunning both survey-model scripts, check the inputs and results with:

```powershell
python -m unittest discover -s tests -v
```
