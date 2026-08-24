from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_INPUT = ROOT / "data" / "behavioral_task_trials.csv"
CMDT_INPUT = ROOT / "data" / "cmdt_trials.csv"
EFA_RESULTS = ROOT / "results" / "efa"
OUTPUT_FOLDER = ROOT / "results" / "figures" / "regenerated"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TaskData = pd.read_csv(BEHAVIORAL_INPUT)
TaskData[["MGBinChoice", "UniBinChoice"]] = (
    TaskData[["MGBinChoice", "UniBinChoice"]]
    .replace({"a": 1, "b": 0})
    .apply(pd.to_numeric)
)

cmdt = pd.read_csv(CMDT_INPUT).rename(columns={"pid": "ID"})


sns.set_theme(style="ticks")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_BLUE = "#3b528b"
COLOR_GREEN = "#bddf26"
PANEL_COLOR = "red"


def add_panel_label(ax, label):
    ax.text(-0.20, 1.03, label, transform=ax.transAxes, color=PANEL_COLOR, fontsize=16)


def bar_mean_and_se(ax, data, group, value, order, colors, title, ylabel, ylim):
    summary = (
        data.groupby(group, sort=False)[value]
        .agg(mean="mean", se=lambda values: values.sem())
        .reindex(order)
        .reset_index()
    )
    positions = np.arange(len(summary))
    ax.bar(
        positions,
        summary["mean"],
        yerr=summary["se"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.65,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_title(title, pad=8, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    sns.despine(ax=ax)


# Figure 3: CMDT acceptance, policy avoidance, and cost sensitivity.
acceptance = cmdt.groupby(["ID", "Social"], as_index=False)["Resp"].mean()
acceptance["Resp"] = acceptance["Resp"] * 100
acceptance["Condition"] = acceptance["Social"].map({1: "Social Support", 0: "No Social Support"})

participant_parameters = (
    TaskData.groupby("ID", as_index=False)[["k_social", "k_no_social", "s"]].first()
)
policy_avoidance = participant_parameters.melt(
    id_vars="ID",
    value_vars=["k_social", "k_no_social"],
    var_name="Condition",
    value_name="k",
)
policy_avoidance["Condition"] = policy_avoidance["Condition"].map({
    "k_social": "Social Support",
    "k_no_social": "No Social Support",
})
cost_sensitivity = participant_parameters.assign(Condition="Both Conditions")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
bar_mean_and_se(
    axes[0], acceptance, "Condition", "Resp",
    ["Social Support", "No Social Support"], [COLOR_BLUE, COLOR_GREEN],
    "% Accept by Condition", "% Acceptance", (0, 100),
)
bar_mean_and_se(
    axes[1], policy_avoidance, "Condition", "k",
    ["Social Support", "No Social Support"], [COLOR_BLUE, COLOR_GREEN],
    "Policy Avoidance by Condition", "Policy Avoidance (k)", (0, 600),
)
bar_mean_and_se(
    axes[2], cost_sensitivity, "Condition", "s",
    ["Both Conditions"], [COLOR_BLUE],
    "Cost Sensitivity", "Sensitivity (γ)", (0, 2),
)
for ax, label in zip(axes, ["a)", "b)", "c)"]):
    add_panel_label(ax, label)
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "figure3_cmdt_behavior.svg", bbox_inches="tight")
fig.savefig(OUTPUT_FOLDER / "figure3_cmdt_behavior.png", bbox_inches="tight")
plt.close(fig)


def uncertainty_preference_summary(data, factor, factor_short):
    participant_values = data.groupby("ID")[factor].first()
    z = (participant_values - participant_values.mean()) / participant_values.std(ddof=1)
    work = data.copy()
    work[f"{factor}_z"] = work["ID"].map(z)

    low_mask = work[f"{factor}_z"] <= -1
    high_mask = work[f"{factor}_z"] >= 1
    if low_mask.sum() == 0 and high_mask.sum() > 0:
        high_proportion = float(high_mask.mean())
        low_quantile = float(work[f"{factor}_z"].dropna().quantile(high_proportion))
        low_mask = work[f"{factor}_z"] <= low_quantile
        low_label = f"Low {factor_short} (matched tail)"
    else:
        low_label = f"Low {factor_short} (-1 SD)"
    high_label = f"High {factor_short} (+1 SD)"

    work["Group"] = pd.NA
    work.loc[low_mask, "Group"] = low_label
    work.loc[high_mask, "Group"] = high_label
    work = work.dropna(subset=["Group"])

    participant_block = (
        work.groupby(["ID", "Block", "Group"], as_index=False)
        .agg(p=("UniBinChoice", "mean"))
    )
    summary = (
        participant_block.groupby(["Block", "Group"], as_index=False)
        .agg(mean=("p", "mean"), sd=("p", "std"), n=("p", "count"))
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    return summary, low_label, high_label


# Figure 5: uncertainty-preference task by CMDT parameters.
up_factors = [
    ("k_no_social", "Baseline Policy Avoidance (BPA)", "BPA"),
    ("k_social", "Social Policy Avoidance (SPA)", "SPA"),
    ("s", "Cost Sensitivity (CS)", "CS"),
]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for ax, (factor, title, short), panel_label in zip(axes, up_factors, ["a)", "b)", "c)"]):
    summary, low_label, high_label = uncertainty_preference_summary(TaskData, factor, short)
    for group, color, offset in [
        (low_label, COLOR_GREEN, -0.06),
        (high_label, COLOR_BLUE, 0.06),
    ]:
        group_data = summary[summary["Group"] == group]
        ax.errorbar(
            group_data["Block"] + offset,
            group_data["mean"],
            yerr=group_data["se"],
            fmt="-o",
            linewidth=2,
            markersize=5,
            capsize=2,
            elinewidth=1.0,
            label=group,
            color=color,
        )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Block")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, pad=8, weight="bold")
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, panel_label)
    sns.despine(ax=ax)
axes[0].set_ylabel("P(Selects Optimal Deck)")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "figure5_uncertainty_preference.svg", bbox_inches="tight")
fig.savefig(OUTPUT_FOLDER / "figure5_uncertainty_preference.png", bbox_inches="tight")
plt.close(fig)


factor_labels = {
    "EFA_Factor_1": "Moral",
    "EFA_Factor_2": "Stigma",
    "EFA_Factor_3": "Xeno",
    "EFA_Factor_4": "Disgust",
    "EFA_Factor_5": "Econ",
}
factor_tables = [
    (EFA_RESULTS / "baseline_policy_avoidance_regression.csv", "Baseline Policy Avoidance by Factor"),
    (EFA_RESULTS / "social_policy_avoidance_regression.csv", "Social Policy Avoidance by Factor"),
    (EFA_RESULTS / "cost_sensitivity_regression.csv", "Cost Sensitivity by Factor"),
]


# Figure 6: standardized factor-regression coefficients and 95% confidence intervals.
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (input_file, title), panel_label in zip(axes, factor_tables, ["a)", "b)", "c)"]):
    coefficients = pd.read_csv(input_file)
    coefficients["label"] = coefficients["factor"].map(factor_labels)
    positions = np.arange(len(coefficients))
    ax.errorbar(
        coefficients["estimate"],
        positions,
        xerr=np.vstack([
            coefficients["estimate"] - coefficients["l95"],
            coefficients["u95"] - coefficients["estimate"],
        ]),
        fmt="o",
        color=COLOR_BLUE,
        capsize=0,
    )
    ax.axvline(0, linestyle="--", color=COLOR_GREEN)
    ax.set_yticks(positions)
    ax.set_yticklabels(coefficients["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Regression Coefficient (β)")
    ax.set_title(title)
    add_panel_label(ax, panel_label)
    sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "figure6_psychological_factors.svg", bbox_inches="tight")
fig.savefig(OUTPUT_FOLDER / "figure6_psychological_factors.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved Figures 3, 5, and 6 to {OUTPUT_FOLDER}")
