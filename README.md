# Policy Avoidance During COVID-19

This repository contains the analyses and final-draft artifacts for **“Policy
Avoidance During COVID-19 Reflects Context-General Maladaptive
Decision-Making.”**.



## Repository contents

- `analysis/`: scripts used for the final CMDT, cross-task, EFA, and plotting analyses.
- `data/cmdt_trials.csv`: 20,412 CMDT trials (243 participants × 84 trials).
- `data/behavioral_task_trials.csv`: mixed-gamble and uncertainty-preference trial data with final CMDT parameters.
- `data/model_comparison_bic/`: participant-level AIC/BIC tables for the eight final discounting models.
- `models/`: the eight Stan model definitions and a model manifest.
- `materials/`: the CMDT Qualtrics guide and COVID-19 attitude/behavior survey instrument.
- `results/`: final statistical tables and the six figures embedded in the final manuscript.


## Software setup

Python 3.11 or later and R are required. From the repository root:

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
  "dplyr", "psych", "tidyverse"
))
```

`requirements.txt` records the Python versions used for the validation run in
this repository. 

## Reproducing the analyses

Run commands from the repository root in the order below. The repository
already includes the final outputs used in the paper, so the expensive Stan
fits do not have to be rerun unless a full computational reproduction is
desired.

### 1. Fit the CMDT discounting models 

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

The original survey file contains MTurk IDs and other potentially identifying
fields, so it is not included in this public-ready copy. Place the original
`DemographicData.csv` at
`data/private/covid_survey_responses.csv`, as described in
`data/private/README.md`, then run:

```powershell
python analysis/survey_efa_and_factor_models.py
```

The script retains the first 124 survey participants, applies the original item
recoding, performs oblimin-rotated EFA, retains five factors for the final
solution, and fits the factor regressions for social policy avoidance, baseline
policy avoidance, cost sensitivity, and factor × social-support interactions.
Outputs are written to `results/efa/`.

### 6. Reproduce manuscript figures

```powershell
python analysis/plot_mixed_gamble.py
python analysis/plot_cmdt_up_and_factors.py
```

The first command recreates the nine Figure 4 panels. The second recreates
Figures 3, 5, and 6. The exact raster images embedded in the final Word draft
are retained in `results/figures/` for comparison; regenerated output goes to
descriptively named subfolders under the same directory. Figure 1 is based on
the task materials, and Figure 2 is also retained as the EFA scree-plot output.

## Paper-to-code map

| Paper section | Script | Primary input | Output |
|---|---|---|---|
| CMDT behavioral results | `analysis/cmdt_behavioral_analysis.py` | `data/cmdt_trials.csv`, `data/behavioral_task_trials.csv` | `results/cmdt/` |
| CMDT discounting models | `analysis/fit_discounting_models.R` | `data/cmdt_trials.csv`, `models/*.stan` | `results/model_fits/` |
| Supplement S4 model selection | `analysis/compare_discounting_models.py` | `data/model_comparison_bic/` | `results/cmdt/table_s4_model_comparison.csv` |
| Mixed-gamble GLMMs | `analysis/cross_task_mixed_models.py` | `data/behavioral_task_trials.csv` | `results/mixed_gamble/` |
| Uncertainty-preference GLMMs | `analysis/cross_task_mixed_models.py` | `data/behavioral_task_trials.csv` | `results/uncertainty_preference/` |
| Survey EFA and factor regressions | `analysis/survey_efa_and_factor_models.py` | private survey input, behavioral parameters | `results/efa/` |
| Figures 3, 5, and 6 | `analysis/plot_cmdt_up_and_factors.py` | behavioral/CMDT data and EFA tables | `results/figures/regenerated/` |
| Figure 4 | `analysis/plot_mixed_gamble.py` | `data/behavioral_task_trials.csv` | `results/figures/figure4_panels/` |

