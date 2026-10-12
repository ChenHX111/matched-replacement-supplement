"""Focused tests of the added saved-data diagnostic behavior."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import unittest

import numpy as np

import saved_diagnostics as diagnostics

HERE = Path(__file__).resolve().parent
DEFAULT_REPRODUCTION = Path(os.environ.get("REVIEW_REPRODUCTION", HERE.parent / "reproduction"))
CORE = diagnostics.load_core(DEFAULT_REPRODUCTION)


class DiagnosticTests(unittest.TestCase):
    def test_other_two_seed_excludes_held_out_and_other_background(self):
        theta = np.arange(48, dtype=float).reshape(1, 6, 8)
        reference = diagnostics.other_seed_target(theta)
        np.testing.assert_array_equal(reference[:, 0], np.mean(theta[:, [1, 2]], axis=1))
        modified = theta.copy()
        modified[:, 0] += 10000
        modified[:, 3:] -= 10000
        np.testing.assert_array_equal(reference[:, 0], diagnostics.other_seed_target(modified)[:, 0])
        with self.assertRaises(ValueError):
            diagnostics.other_seed_target(theta[:, :5])

    def test_component_pairs_exclude_same_realization(self):
        rng = np.random.default_rng(4)
        theta, nonfocal = rng.normal(size=(2, 6, 8)), rng.normal(size=(2, 6, 8))
        result = diagnostics.different_seed_components(nonfocal, theta, CORE)
        pairs = [(bg * 3 + a, bg * 3 + b) for bg in range(2)
                 for a, b in itertools.permutations(range(3), 2)]
        self.assertEqual(result.shape, (2, 12))
        for col, (a, b) in enumerate(pairs):
            for block in range(2):
                self.assertAlmostEqual(result[block, col],
                                       CORE.rank_correlation(nonfocal[block, a], theta[block, b]))

    def test_fixed_halves_retain_all_blocks_and_tie_order(self):
        blocks = ["B000", "B001", "B002", "B003"]
        halves = diagnostics.fixed_halves([.4, .1, .4, .2], blocks)
        self.assertEqual(halves, {"lower": [1, 3], "upper": [0, 2]})
        self.assertEqual(diagnostics.fixed_halves([.2] * 4, blocks),
                         {"lower": [0, 1], "upper": [2, 3]})
        with self.assertRaises(ValueError):
            diagnostics.fixed_halves([.1, np.nan, .3, .4], blocks)

    def test_control_coverage_counts_endpoints_not_unique_grams_or_eos(self):
        repeated = [7] * 254
        self.assertEqual(diagnostics.control_coverage(repeated, repeated), 251 / 255)
        self.assertEqual(diagnostics.control_coverage(repeated, [8] * 254), 0)
        self.assertEqual(diagnostics.control_coverage(repeated, [7] * 4 + [8] * 250), 251 / 255)
        with self.assertRaises(ValueError):
            diagnostics.control_coverage([7] * 256, repeated)

    def test_ties_regret_and_selection_stability_use_frozen_order(self):
        prediction = np.zeros((1, 6, 8))
        target = np.broadcast_to(np.arange(8), (1, 6, 8)).copy()
        metrics = diagnostics.metric_cells(prediction, target, CORE)
        self.assertTrue(np.isnan(metrics["rank"]).all())
        np.testing.assert_array_equal(metrics["regret"], np.full((1, 6), 6.0))
        np.testing.assert_array_equal(
            diagnostics.selection_stability(prediction)["identical_top_two_fraction"], np.ones((1, 6)))
        prediction[:, 0, 0] = np.nan
        self.assertIsNone(diagnostics.selected_two(prediction[0, 0]))

    def test_undefined_cells_do_not_disappear_in_bootstrap(self):
        cells = [[.5, .4], [.7, np.nan], [.8, .9]]
        result = CORE.bootstrap(cells, seed=5, replicates=30)
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["undefined_cells_count"], 1)

    def test_saved_grid_target_and_final_direction_are_preserved(self):
        data, extra = diagnostics.load_inputs(DEFAULT_REPRODUCTION, HERE / "review_inputs.json")
        for setting, count in zip(diagnostics.SETTINGS, (24, 8, 8)):
            current = data[setting]
            self.assertEqual(len(current["blocks"]), count)
            _, original_target = CORE.p5_arrays(current, "G20")
            np.testing.assert_array_equal(diagnostics.other_seed_target(current["oracle"]), original_target)
            self.assertEqual(np.shape(extra["p5"][setting]["control_coverage_k4"]), (count, 8))
            final = (np.asarray(current["scores"]["base_difficulty_nll"]) -
                     np.asarray(current["scores"]["audit_loss_drop"]))
            positive = diagnostics.rank_cells(final, original_target, CORE)
            negative = diagnostics.rank_cells(-final, original_target, CORE)
            np.testing.assert_allclose(positive, -negative, atol=1e-15)

    def test_projection_rejects_duplicate_or_missing_context(self):
        if not (HERE / "prepare_review_inputs.py").is_file():
            self.skipTest("Private projection helper is outside the portable artifact")
        from prepare_review_inputs import index_contexts
        rows = [{"setting": "test", "cluster": "B000", "background": bg, "seed": seed}
                for bg in range(2) for seed in range(3)]
        self.assertEqual(len(index_contexts(rows, "test", 1)[1]), 6)
        with self.assertRaises(ValueError):
            index_contexts(rows + [rows[0]], "test", 1)
        with self.assertRaises(ValueError):
            index_contexts(rows[:-1], "test", 1)


if __name__ == "__main__":
    unittest.main()
