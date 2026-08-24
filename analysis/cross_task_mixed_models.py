from pathlib import Path
import subprocess
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "behavioral_task_trials.csv"
R_HELPER = Path(__file__).with_name("glmer_binomial.R")
MIXED_GAMBLE_OUTPUT = ROOT / "results" / "mixed_gamble"
UP_OUTPUT = ROOT / "results" / "uncertainty_preference"
MIXED_GAMBLE_OUTPUT.mkdir(parents=True, exist_ok=True)
UP_OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)
mapping = {"a": 1, "b": 0}
for dependent_variable in ["MGBinChoice", "UniBinChoice"]:
    df[dependent_variable] = df[dependent_variable].map(mapping)


def zscore(array):
    mean = array.mean()
    sd = array.std(ddof=1)
    if sd == 0:
        raise ValueError("Cannot z-score array: standard deviation is zero.")
    return (array - mean) / sd


for variable in ["k_social", "k_no_social", "s"]:
    df[variable] = zscore(df[variable])

# Scaling the gamble amounts by 10 preserves the original reported coefficient units.
for variable in ["MGLoss", "MGGain", "MGValue"]:
    df[variable] = df[variable] / 10


models = [
    (MIXED_GAMBLE_OUTPUT / "gain_by_policy_avoidance.csv",
     "MGBinChoice ~ k_social * k_no_social * MGGain + (1 | ID)"),
    (MIXED_GAMBLE_OUTPUT / "loss_by_policy_avoidance.csv",
     "MGBinChoice ~ k_social * k_no_social * MGLoss + (1 | ID)"),
    (MIXED_GAMBLE_OUTPUT / "gain_by_cost_sensitivity.csv",
     "MGBinChoice ~ s * MGGain + (1 | ID)"),
    (MIXED_GAMBLE_OUTPUT / "loss_by_cost_sensitivity.csv",
     "MGBinChoice ~ s * MGLoss + (1 | ID)"),
    (UP_OUTPUT / "block_by_policy_avoidance.csv",
     "UniBinChoice ~ k_social * k_no_social * Block + (1 | ID)"),
    (UP_OUTPUT / "block_by_cost_sensitivity.csv",
     "UniBinChoice ~ s * Block + (1 | ID)"),
    (MIXED_GAMBLE_OUTPUT / "net_value_by_policy_avoidance.csv",
     "MGBinChoice ~ k_social * k_no_social * MGValue + (1 | ID)"),
    (MIXED_GAMBLE_OUTPUT / "net_value_by_cost_sensitivity.csv",
     "MGBinChoice ~ s * MGValue + (1 | ID)"),
]


with tempfile.TemporaryDirectory(prefix="policy_avoidance_models_") as temp_folder:
    model_input = Path(temp_folder) / "model_input.csv"
    df.to_csv(model_input, index=False)

    for output_file, formula in models:
        subprocess.run(
            ["Rscript", str(R_HELPER), str(model_input), str(output_file), formula],
            check=True,
        )
        print(f"Saved {output_file}")
