# Policy Avoidance During COVID-19

Data, code, materials, and results for *Policy Avoidance During COVID-19 Reflects
Context-General Maladaptive Decision-Making*.

## Repository contents

- `analysis/`: scripts used for the final CMDT, cross-task, EFA, and plotting analyses.
- `data/cmdt_trials.csv`: 20,412 CMDT trials (243 participants × 84 trials).
- `data/behavioral_task_trials.csv`: mixed-gamble and uncertainty-preference trial data with final CMDT parameters.
- `data/survey_responses_deidentified.csv`: 124 survey records, including political orientation; see [data notes](data/README.md).
- `data/model_comparison_bic/`: participant-level AIC/BIC tables for the eight final discounting models.
- `models/`: the eight Stan model definitions and a model manifest.
- `materials/`: the CMDT Qualtrics guide and COVID-19 attitude/behavior survey instrument.
- `results/`: final statistical tables and the six figures embedded in the final manuscript.


## Software setup

The pinned packages require Python 3.12 or later. R is also required. From the
repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows and install the Python dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the required R packages once:

```r
install.packages(c(
  "rstan", "bayesplot", "loo", "lme4", "lmerTest",
  "dplyr", "psych", "nFactors", "GPArotation", "tidyverse"
))
```

Keep `Rscript` on your system path. On Windows, if Python cannot locate R, set
`R_HOME` to your R installation folder before running Python.

Python package versions are pinned in `requirements.txt`; tested versions and
validation scope are recorded in [VALIDATION.md](VALIDATION.md).

## Reproducing the analyses

Run commands from the repository root. The survey analyses use only the public
files; the original survey export containing identifiers is not needed.

Downstream analyses use the supplied fitted parameters and BIC tables. Step 1
is optional and saves new fits separately; it does not replace those inputs.

### 1. Refit the CMDT discounting models (optional)

The command accepts one model key from `models/model_manifest.csv`:

```powershell
Rscript analysis/fit_discounting_models.R Rachlin_CW_k
```

The eight valid keys are `Mazur_c`, `Rachlin_c`, `Meyerson_Green_c`,
`Mazur_CW_k`, `Rachlin_CW_k`, `Meyerson_Green_CW_k`, `Rachlin_CW_k+s`, and
`Meyerson_Green_CW_k+s`. Each run uses 4 chains, 5,000 iterations, 500 warmup
iterations, and seed 42. It writes participant information criteria and a
posterior summary to `results/model_fits/`.

### 2. Reproduce Supplement Table S4.1

```powershell
python analysis/compare_discounting_models.py
```

This performs random-effects variational Bayesian model selection on all eight
final participant-level BIC tables and writes
`results/cmdt/table_s4_model_comparison.csv`.

### 3. Reproduce CMDT behavioral results

```powershell
python analysis/cmdt_behavioral_analysis.py
```

This produces overall policy acceptance, the social-support × benefit-frame
mixed ANOVAs for policy acceptance and policy avoidance, CMDT parameter
descriptives, and the reported parameter correlations in `results/cmdt/`.

### 4. Reproduce the mixed-gamble and uncertainty-preference GLMMs

```powershell
python analysis/cross_task_mixed_models.py
```

The Python script applies the original recoding and scaling, then calls
`analysis/glmer_binomial.R`. It fits:

- mixed-gamble acceptance from gain, loss, and net value with social and baseline policy avoidance;
- parallel gain, loss, and net-value models with cost sensitivity; and
- uncertainty-preference choices from task block with both policy-avoidance parameters or cost sensitivity.

The six mixed-gamble tables are written to `results/mixed_gamble/`; the two
uncertainty-preference tables are written to
`results/uncertainty_preference/`.

### 5. Reproduce the survey EFA and factor models

The deidentified survey responses are included in
`data/survey_responses_deidentified.csv`. Run:

```powershell
python analysis/survey_efa_and_factor_models.py
```

The script recodes the 124 survey records and fits a five-factor EFA using
maximum likelihood and oblimin rotation. Parallel analysis recommends seven
factors; the manuscript uses five based on the scree elbow and interpretability.
Seed 42 fixes the parallel-analysis simulations.

The factor regressions use 118 complete cases after applying the original
`analysis_included` flag. Outputs in `results/efa/` include loadings, fit
statistics, three factor regressions, and all five factor × social-support
tests in `condition_interactions.csv`. The significant subset is also saved.

### 6. Reproduce political-orientation models

```powershell
Rscript analysis/political_orientation_models.R
```

The script matches `political_orientation` (original item `Q207`) to behavioral
parameters by `ID` and applies `analysis_included`, yielding 119 complete cases.
Political orientation runs from 1 (very liberal) to 5 (very conservative) and
is centered at the neutral midpoint, 3.

Policy avoidance uses a participant random intercept and a political
orientation × support-condition interaction; no support is the reference.
Cost sensitivity uses a participant-level linear regression. Outcomes are
standardized, so slopes represent outcome standard deviations per one-point
increase toward conservatism. Tests are two-sided and unadjusted.

`results/political_orientation/parameter_associations.csv` contains coefficients,
95% confidence intervals, tests, and sample sizes; `model_summary.txt` includes
model summaries and R package versions. This script runs independently of EFA.

### 7. Reproduce manuscript figures

```powershell
python analysis/plot_mixed_gamble.py
python analysis/plot_cmdt_up_and_factors.py
```

The first command recreates the nine Figure 4 panels; the second recreates
Figures 3, 5, and 6. Original manuscript images remain in `results/figures/`;
regenerated plots use subfolders. Figure 1 uses the task materials. The EFA
script regenerates Figure 2's scree plot as `results/efa/scree_plot.png`.

### Check the public data and results

After running the survey and political-orientation scripts:

```powershell
python -m unittest discover -s tests -v
```

These checks validate the shared inputs and key manuscript statistics. They do
not refit models; see [VALIDATION.md](VALIDATION.md) for the execution checks.

## Paper-to-code map

| Paper section | Script | Primary input | Output |
|---|---|---|---|
| CMDT behavioral results | `analysis/cmdt_behavioral_analysis.py` | `data/cmdt_trials.csv`, `data/behavioral_task_trials.csv` | `results/cmdt/` |
| CMDT discounting models | `analysis/fit_discounting_models.R` | `data/cmdt_trials.csv`, `models/*.stan` | `results/model_fits/` |
| Supplement S4 model selection | `analysis/compare_discounting_models.py` | `data/model_comparison_bic/` | `results/cmdt/table_s4_model_comparison.csv` |
| Mixed-gamble GLMMs | `analysis/cross_task_mixed_models.py` | `data/behavioral_task_trials.csv` | `results/mixed_gamble/` |
| Uncertainty-preference GLMMs | `analysis/cross_task_mixed_models.py` | `data/behavioral_task_trials.csv` | `results/uncertainty_preference/` |
| Survey EFA and factor regressions | `analysis/survey_efa_and_factor_models.py` | public survey responses, behavioral parameters | `results/efa/` |
| Political-orientation models | `analysis/political_orientation_models.R` | public survey responses, behavioral parameters | `results/political_orientation/` |
| Figures 3, 5, and 6 | `analysis/plot_cmdt_up_and_factors.py` | behavioral/CMDT data and EFA tables | `results/figures/regenerated/` |
| Figure 4 | `analysis/plot_mixed_gamble.py` | `data/behavioral_task_trials.csv` | `results/figures/figure4_panels/` |

