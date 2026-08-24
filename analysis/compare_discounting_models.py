from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import dirichlet


ROOT = Path(__file__).resolve().parents[1]
INPUT_FOLDER = ROOT / "data" / "model_comparison_bic"
OUTPUT_FILE = ROOT / "results" / "cmdt" / "table_s4_model_comparison.csv"


MODEL_FILES = {
    "Mazur": "mazur.csv",
    "Rachlin": "rachlin.csv",
    "Meyerson-Green": "meyerson_green.csv",
    "Mazur CW-k": "mazur_conditionwise_k.csv",
    "Meyerson-Green CW-k": "meyerson_green_conditionwise_k.csv",
    "Rachlin CW-k": "rachlin_conditionwise_k.csv",
    "Meyerson-Green CW-k+s": "meyerson_green_conditionwise_k_and_s.csv",
    "Rachlin CW-k+s": "rachlin_conditionwise_k_and_s.csv",
}


def vb_model_selection(log_evidences, alpha0=None, tol=1e-6, max_iter=1000):
    """
    Variational Bayesian Model Selection for multiple models and participants.
    This is the same iterative procedure used in the original analysis script.
    """
    n_participants, n_models = log_evidences.shape
    if alpha0 is None:
        alpha0 = np.ones(n_models)

    alpha = alpha0.copy()
    for iteration in range(max_iter):
        alpha_sum = np.sum(alpha)
        u = np.exp(log_evidences + psi(alpha) - psi(alpha_sum))
        g = u / np.sum(u, axis=1, keepdims=True)
        beta = np.sum(g, axis=0)
        alpha_new = alpha0 + beta

        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha, g, iteration + 1


def compute_exceedance_probability(alpha, n_samples=100000, random_state=123):
    samples = dirichlet.rvs(alpha, size=n_samples, random_state=random_state)
    winners = np.argmax(samples, axis=1)
    return np.bincount(winners, minlength=len(alpha)) / n_samples


model_tables = {
    model_name: pd.read_csv(INPUT_FOLDER / filename)
    for model_name, filename in MODEL_FILES.items()
}

participant_ids = model_tables["Mazur"]["Participant"].to_numpy()
for model_name, table in model_tables.items():
    if not np.array_equal(table["Participant"].to_numpy(), participant_ids):
        raise ValueError(f"Participant ordering differs for {model_name}.")

bic_matrix = np.column_stack([
    model_tables[model_name]["BIC"].to_numpy()
    for model_name in MODEL_FILES
])
log_evidences = -0.5 * bic_matrix

alpha, participant_model_probabilities, iterations = vb_model_selection(
    log_evidences,
    alpha0=np.ones(len(MODEL_FILES)),
    tol=1e-12,
    max_iter=50000,
)
expected_frequency = alpha / alpha.sum()
exceedance_probability = compute_exceedance_probability(alpha)

results = pd.DataFrame({
    "Model": list(MODEL_FILES),
    "Average BIC": bic_matrix.mean(axis=0),
    "alpha": alpha,
    "Expected Frequency": expected_frequency,
    "Exceedance Probability": exceedance_probability,
})

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)

print(results.round(3).to_string(index=False))
print(f"\nVBMS converged in {iterations} iterations.")
print(f"Saved {OUTPUT_FILE}")
