# Survey data notes

`survey_responses_deidentified.csv` contains 124 survey records. The original
export containing MTurk IDs and other identifying fields is not required by
the public analysis scripts.

- `ID` is the numeric participant key used to join the survey and behavioral
  data. It is not an MTurk ID.
- `analysis_included` preserves the original complete-case eligibility flag
  without publishing the excluded demographic fields: 119 records have value
  1 and five have value 0. Do not recompute this flag from the reduced release.
- `political_orientation` is original item `Q207`, coded from 1 (very liberal)
  to 5 (very conservative), with 3 = neither liberal nor conservative.
- Survey item codes retain the original response coding. The EFA script renames
  items and recodes the response scales before analysis; the questionnaire is
  in `materials/covid_attitude_behavior_survey.docx`.

The EFA uses all 124 records with pairwise-complete correlations. Factor
regressions and interactions use the eligibility flag and complete factor
scores, leaving 118 cases. Political-orientation models use 119 complete cases
and do not depend on factor scores.
