# Private survey input

Place the original `DemographicData.csv` here as `covid_survey_responses.csv`
before running `analysis/survey_efa_and_factor_models.py`.

The source file contains `MTurkID` and other potentially identifying fields, so
it is intentionally not copied into this repository. The script uses the first
124 retained survey participants, matching the final manuscript. The final,
non-identifying EFA loadings, fit summaries, and regression tables are included
under `results/efa/`.
