from pathlib import Path

import pandas as pd
import pingouin as pg
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
CMDT_INPUT = ROOT / "data" / "cmdt_trials.csv"
BEHAVIORAL_INPUT = ROOT / "data" / "behavioral_task_trials.csv"
OUTPUT_FOLDER = ROOT / "results" / "cmdt"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


cmdt = pd.read_csv(CMDT_INPUT).rename(columns={"pid": "ID"})
participants = (
    pd.read_csv(BEHAVIORAL_INPUT)
    .drop_duplicates(subset="ID", keep="first")
    .reset_index(drop=True)
)


# Overall policy acceptance and the social-support manipulation.
acceptance_by_participant = cmdt.groupby("ID", as_index=False)["Resp"].mean()
overall_test = stats.ttest_1samp(acceptance_by_participant["Resp"], popmean=0.5)
n = len(acceptance_by_participant)
mean_acceptance = acceptance_by_participant["Resp"].mean()
sd_acceptance = acceptance_by_participant["Resp"].std(ddof=1)
sem_acceptance = stats.sem(acceptance_by_participant["Resp"])
ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_acceptance, scale=sem_acceptance)

overall_results = pd.DataFrame([{
    "n": n,
    "mean_acceptance": mean_acceptance,
    "sd_acceptance": sd_acceptance,
    "t": overall_test.statistic,
    "df": n - 1,
    "p": overall_test.pvalue,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "cohens_d": (mean_acceptance - 0.5) / sd_acceptance,
}])
overall_results.to_csv(OUTPUT_FOLDER / "overall_policy_acceptance.csv", index=False)

id_to_frame = participants.set_index("ID")["Between"]
acceptance_mixed_anova_data = (
    cmdt.groupby(["ID", "Social"], as_index=False)["Resp"].mean()
)
acceptance_mixed_anova_data["Between"] = acceptance_mixed_anova_data["ID"].map(id_to_frame)
acceptance_mixed_anova_data["Between"] = acceptance_mixed_anova_data["Between"].astype(str).astype("category")

acceptance_anova = pg.mixed_anova(
    data=acceptance_mixed_anova_data,
    dv="Resp",
    within="Social",
    subject="ID",
    between="Between",
    correction="auto",
    effsize="np2",
)
acceptance_anova.to_csv(OUTPUT_FOLDER / "policy_acceptance_mixed_anova.csv", index=False)


# Policy avoidance (k) by social-support and between-subject framing conditions.
no_social = participants["k_no_social"]
social = participants["k_social"]
policy_avoidance_mixed_anova_data = (
    pd.concat([no_social, social], keys=["No Social Support", "Social Support"])
    .reset_index(level=0)
    .rename(columns={"level_0": "Social", 0: "k"})
)
policy_avoidance_mixed_anova_data["ID"] = policy_avoidance_mixed_anova_data.index + 1
policy_avoidance_mixed_anova_data["Between"] = policy_avoidance_mixed_anova_data["ID"].map(id_to_frame)
policy_avoidance_mixed_anova_data["Between"] = policy_avoidance_mixed_anova_data["Between"].astype(str).astype("category")

policy_avoidance_anova = pg.mixed_anova(
    data=policy_avoidance_mixed_anova_data,
    dv="k",
    within="Social",
    subject="ID",
    between="Between",
    correction="auto",
    effsize="np2",
)
policy_avoidance_anova.to_csv(OUTPUT_FOLDER / "policy_avoidance_mixed_anova.csv", index=False)


# Descriptives and correlations reported with the model-derived parameters.
parameter_descriptives = participants[["k_social", "k_no_social", "s"]].describe().T
parameter_descriptives["median"] = participants[["k_social", "k_no_social", "s"]].median()
parameter_descriptives["q1"] = participants[["k_social", "k_no_social", "s"]].quantile(0.25)
parameter_descriptives["q3"] = participants[["k_social", "k_no_social", "s"]].quantile(0.75)
parameter_descriptives["proportion_below_1"] = pd.NA
parameter_descriptives.loc["s", "proportion_below_1"] = (participants["s"] < 1).mean()
parameter_descriptives.to_csv(OUTPUT_FOLDER / "cmdt_parameter_descriptives.csv")

correlation_rows = []
for variable_1, variable_2 in [("k_social", "k_no_social"), ("k_social", "s"), ("k_no_social", "s")]:
    r, p = stats.pearsonr(participants[variable_1], participants[variable_2])
    correlation_rows.append({"variable_1": variable_1, "variable_2": variable_2, "r": r, "p": p})
pd.DataFrame(correlation_rows).to_csv(OUTPUT_FOLDER / "cmdt_parameter_correlations.csv", index=False)

print(f"Saved CMDT results to {OUTPUT_FOLDER}")
