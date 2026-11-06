"""Check public inputs and regenerated survey results without private data."""

import csv
from collections import Counter
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class PublicReproductionTests(unittest.TestCase):
    def test_behavioral_project_columns(self):
        behavioral = rows("data/behavioral_task_trials.csv")
        columns = set(behavioral[0])
        self.assertEqual(len(behavioral), 24300)
        self.assertEqual(len(columns), 23)
        self.assertFalse(any(column.upper().startswith("SGT") for column in columns))
        self.assertTrue({
            "ID", "Block", "UniBinChoice", "UniReward",
            "MGBinChoice", "MGGain", "MGLoss", "MGValue",
            "k_social", "k_no_social", "s",
        } <= columns)
        counts = Counter(row["ID"] for row in behavioral)
        self.assertEqual(len(counts), 243)
        self.assertEqual(set(counts.values()), {100})

    def test_documentation_links(self):
        for relative in ["README.md", "data/README.md", "VALIDATION.md"]:
            document = ROOT / relative
            targets = re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8"))
            for target in targets:
                if not target.startswith(("http:", "https:", "#")):
                    with self.subTest(document=relative, link=target):
                        self.assertTrue((document.parent / target.split("#")[0]).exists())

    def test_survey_ids_and_public_links(self):
        survey = rows("data/survey_responses_deidentified.csv")
        behavioral = rows("data/behavioral_task_trials.csv")
        ids = {row["ID"] for row in survey}
        self.assertEqual(len(survey), 124)
        self.assertEqual(len(ids), 124)
        self.assertTrue(all(1 <= int(value) <= 243 for value in ids))
        self.assertTrue(ids <= {row["ID"] for row in behavioral})

    def test_survey_eligibility_and_orientation(self):
        survey = rows("data/survey_responses_deidentified.csv")
        flags = [int(row["analysis_included"]) for row in survey]
        self.assertTrue(set(flags) <= {0, 1})
        self.assertEqual(sum(flags), 119)
        self.assertTrue(all(int(row["political_orientation"]) in range(1, 6) for row in survey))
        direct_identifiers = {"mturkid", "workerid", "ipaddress", "email", "name"}
        self.assertTrue(direct_identifiers.isdisjoint(key.lower() for key in survey[0]))

    def test_efa_counts_and_fit(self):
        fit = rows("results/efa/fit_summary.csv")[0]
        expected = {
            "n_obs": 124, "n_items": 90, "n_regression_obs": 118,
            "parallel_analysis_n_factors": 7, "fitted_n_factors": 5,
            "parallel_analysis_seed": 42,
        }
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(float(fit[key]), value)
        self.assertAlmostEqual(float(fit["cumulative_var"]), 0.474862, places=5)
        self.assertAlmostEqual(float(fit["RMSEA"]), 0.0671, places=4)
        self.assertEqual(len(rows("results/efa/factor_loadings.csv")), 90)

    def test_factor_regressions(self):
        expected = {
            "social_policy_avoidance": [-0.307210, 0.414078, 0.192027, 0.012424, -0.006368],
            "baseline_policy_avoidance": [-0.375691, 0.296840, 0.174350, -0.048401, 0.003533],
            "cost_sensitivity": [-0.107625, -0.499407, -0.106016, -0.034253, 0.041722],
        }
        for name in ["social_policy_avoidance", "baseline_policy_avoidance", "cost_sensitivity"]:
            table = rows(f"results/efa/{name}_regression.csv")
            self.assertEqual([row["factor"] for row in table], [f"EFA_Factor_{i}" for i in range(1, 6)])
            if name in expected:
                for row, estimate in zip(table, expected[name]):
                    with self.subTest(model=name, factor=row["factor"]):
                        self.assertAlmostEqual(float(row["estimate"]), estimate, places=5)

    def test_all_condition_interactions(self):
        table = rows("results/efa/condition_interactions.csv")
        self.assertEqual(len(table), 5)
        for row in table:
            self.assertEqual(row["term"], row["factor"] + ":Conditionsocial")
        stigma = next(row for row in table if row["factor"] == "EFA_Factor_2")
        self.assertAlmostEqual(float(stigma["estimate"]), 0.118359, places=5)
        self.assertAlmostEqual(float(stigma["pval"]), 0.031239, places=5)
        significant = rows("results/efa/significant_condition_interactions.csv")
        self.assertEqual([row["factor"] for row in significant], ["EFA_Factor_2"])

    def test_political_orientation_results(self):
        table = rows("results/political_orientation/parameter_associations.csv")
        expected = {
            "baseline_policy_avoidance": (0.174543, 0.011861),
            "social_policy_avoidance": (0.260714, 0.000209),
            "social_minus_baseline": (0.086171, 0.040022),
            "cost_sensitivity": (0.046966, 0.512304),
        }
        self.assertEqual({row["outcome"] for row in table}, set(expected))
        for row in table:
            estimate, p_value = expected[row["outcome"]]
            with self.subTest(outcome=row["outcome"]):
                self.assertEqual(int(row["n_participants"]), 119)
                self.assertAlmostEqual(float(row["estimate"]), estimate, places=5)
                self.assertAlmostEqual(float(row["p_value"]), p_value, places=5)


if __name__ == "__main__":
    unittest.main()
