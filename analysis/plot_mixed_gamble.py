from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "behavioral_task_trials.csv"
OUTPUT_FOLDER = ROOT / "results" / "figures" / "figure4_panels"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TaskData = pd.read_csv(INPUT_FILE)
TaskData["MGBinChoice"] = (
    TaskData["MGBinChoice"].replace({"a": 1, "b": 0}).apply(pd.to_numeric)
)


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


def plot_binary_by_trial_predictor(
    df,
    dv="Choice",
    x="TrialPredictor",
    idvar="IndDiff",
    idvarname=None,
    idvarshort=None,
    x_var_name=None,
    n_bins=20,
    low_color="#1f77b4",
    high_color="#ff7f0e",
    legend_loc="upper left",
    legend_anchor=(0.02, 0.98),
    figsize=(3.25, 2.8),
    ax=None,
):
    if idvarname is None:
        idvarname = idvar
    if x_var_name is None:
        x_var_name = x

    d = df[[dv, x, idvar]].copy()
    d[dv] = pd.to_numeric(d[dv], errors="coerce")
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[idvar] = pd.to_numeric(d[idvar], errors="coerce")
    d = d.dropna(subset=[dv, x, idvar])

    mean = d[idvar].mean()
    sd = d[idvar].std(ddof=1)
    low_cut = mean - sd
    high_cut = mean + sd
    high_mask = d[idvar] >= high_cut
    low_mask = d[idvar] <= low_cut

    if low_mask.sum() == 0 and high_mask.sum() > 0:
        high_proportion = float(high_mask.mean())
        low_quantile = float(d[idvar].quantile(high_proportion))
        low_mask = d[idvar] <= low_quantile

        d = d[low_mask | high_mask].copy()
        d["Group"] = np.where(
            d.index.isin(d.index[high_mask.loc[d.index]]),
            f"High {idvarshort} (+1 SD)",
            f"Low {idvarshort} (matched tail)",
        )
        low_label = f"Low {idvarshort} (matched tail)"
        high_label = f"High {idvarshort} (+1 SD)"
    else:
        d = d[low_mask | high_mask].copy()
        d["Group"] = np.where(
            d[idvar] <= low_cut,
            f"Low {idvarshort} (-1 SD)",
            f"High {idvarshort} (+1 SD)",
        )
        low_label = f"Low {idvarshort} (-1 SD)"
        high_label = f"High {idvarshort} (+1 SD)"

    d["x_bin"] = pd.cut(d[x], bins=n_bins)
    bin_mids = d.groupby("x_bin", observed=False)[x].mean()
    d["x_mid"] = d["x_bin"].map(bin_mids)

    summary = (
        d.groupby(["Group", "x_mid"], as_index=False, observed=False)
        .agg(p=(dv, "mean"), n=(dv, "size"), sd=(dv, "std"))
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary = summary.dropna(subset=["x_mid"]).sort_values(["Group", "x_mid"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = {low_label: low_color, high_label: high_color}
    for group, group_data in summary.groupby("Group", observed=False):
        color = colors.get(group, "#000000")
        ax.plot(group_data["x_mid"], group_data["p"], linewidth=2.0, color=color, label=group)
        ax.errorbar(
            group_data["x_mid"],
            group_data["p"],
            yerr=group_data["se"],
            fmt="none",
            capsize=2,
            elinewidth=1.0,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel(x_var_name, labelpad=6)
    ax.set_ylabel("P(Accepts Gamble)", labelpad=6)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc=legend_loc, bbox_to_anchor=legend_anchor, borderaxespad=0)
    sns.despine(ax=ax)
    return ax


individual_differences = [
    ("k_no_social", "Baseline Policy Avoidance (BPA)", "BPA"),
    ("k_social", "Social Policy Avoidance (SPA)", "SPA"),
    ("s", "Cost Sensitivity (CS)", "CS"),
]
trial_predictors = [
    ("MGGain", "Gain Amount", "gain"),
    ("MGLoss", "Loss Amount", "loss"),
    ("MGValue", "Net Value", "net_value"),
]

for idvar, idvarname, idvarshort in individual_differences:
    for x_var, x_var_name, output_name in trial_predictors:
        fig, ax = plt.subplots(figsize=(3.25, 2.8))

        if x_var == "MGLoss":
            legend_loc = "upper right"
            legend_anchor = (0.98, 0.98)
        else:
            legend_loc = "upper left"
            legend_anchor = (0.02, 0.98)

        plot_binary_by_trial_predictor(
            TaskData,
            dv="MGBinChoice",
            x=x_var,
            idvar=idvar,
            idvarname=idvarname,
            idvarshort=idvarshort,
            x_var_name=x_var_name,
            n_bins=25,
            low_color="#3b528b",
            high_color="#bddf26",
            legend_loc=legend_loc,
            legend_anchor=legend_anchor,
            ax=ax,
        )

        ax.set_title(idvarname, pad=8, weight="bold")
        plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)
        figure_file = OUTPUT_FOLDER / f"figure4_{idvarshort.lower()}_{output_name}.svg"
        fig.savefig(figure_file, dpi=300)
        plt.close(fig)
        print(f"Saved {figure_file}")
