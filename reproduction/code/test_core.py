"""Portable unit checks requiring only the included anonymous scalar artifact."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import statistics_core as core
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class ScalarArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((ROOT / "data" / "cpu_inputs.json").read_text(encoding="utf-8"))
        cls.protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))

    def test_joined_count_grid_and_fixed_denominator(self):
        total = 0
        for name, expected in (("p5_main", 24), ("p5_corpus", 8), ("p5_model", 8)):
            setting = self.inputs["p5"][name]
            self.assertEqual(len(setting["blocks"]), expected)
            self.assertEqual(setting["denominator"], 255)
            self.assertEqual(setting["eligible_complete_k4"], 251)
            counts = np.asarray(setting["uncovered_count_k4"])
            self.assertEqual(counts.shape, (expected, 6, 8))
            self.assertTrue(np.all((counts >= 0) & (counts <= 251)))
            self.assertEqual(len({qid for block in setting["query_ids"] for qid in block}), expected*8)
            total += counts.size
        self.assertEqual(total, 1920)

    def test_same_background_reference(self):
        setting = self.inputs["p5"]["p5_main"]
        _, reference = core.p5_arrays(setting, "G20")
        oracle = np.asarray(setting["oracle"])
        np.testing.assert_array_equal(reference[:, 0], np.mean(oracle[:, [1, 2]], axis=1))
        self.assertFalse(np.array_equal(reference[:, 0], np.mean(oracle[:, [4, 5]], axis=1)))

    def test_ties_and_undefined_cells(self):
        self.assertEqual(core.ranks([1, 1, 4, 2]), [.5, .5, 3, 2])
        self.assertIsNone(core.rank_correlation([1]*8, list(range(8))))
        self.assertIsNone(core.rank_correlation([None]*8, list(range(8))))
        self.assertEqual(core.top_two_regret([1]*8, list(range(8))), 6)
        self.assertFalse(core.bootstrap([[1, np.nan], [2, 3]], replicates=20)["full_metric_defined"])

    def test_directions_and_median_not_mean(self):
        left = np.array([[0, 0, 100], [0, 0, 100]], dtype=float)
        right = np.array([[0, 100, 100], [0, 100, 100]], dtype=float)
        result = core.bootstrap(right, subtract=left, kind="median", replicates=20)
        self.assertEqual(result["estimate"], 100)
        self.assertNotEqual(result["estimate"], np.median(right-left))
        self.assertTrue(core.directional({"interval": [1, 2]}, "greater"))
        self.assertFalse(core.directional({"interval": [-1, 2]}, "greater"))

    def test_complete_primary_family_and_strata(self):
        self.assertEqual(len(self.protocol["bootstrap_check"]["comparisons"]), 11)
        self.assertEqual(self.protocol["bootstrap_check"]["sensitivity_replicates"], 100000)
        p4 = self.inputs["p4"]["p4_main"]
        self.assertEqual(len(p4["targets"]), 48)
        self.assertEqual(p4["strata"].count("higher_overlap"), 24)
        self.assertEqual(p4["strata"].count("lower_overlap"), 24)

    def test_exact_original_reproduction(self):
        for specification in self.protocol["bootstrap_check"]["comparisons"]:
            values, subtract, kind, strata = core.primary_cells(self.inputs, specification)
            result = core.bootstrap(values, subtract=subtract, kind=kind, strata=strata,
                                    confidence=1-.05/11, seed=specification["bootstrap_seed"], replicates=5000)
            self.assertEqual(result["estimate"], specification["original_estimate"])
            self.assertEqual(result["interval"], specification["original_interval"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScalarArtifactTests)
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
