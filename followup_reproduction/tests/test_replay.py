"""Focused tests of the portable scalar boundary; no neural or external work."""
import copy
from pathlib import Path
import statistics
import unittest

import replay as r

ROOT = Path(__file__).absolute().parent.parent


class PortableReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.data, cls.costs, cls.expected, cls.feasibility = r.load_inputs(ROOT)
        cls.index = r.validate_inputs(cls.protocol, cls.data, cls.costs)
        cls.result = r.replay(cls.protocol, cls.data, cls.costs, cls.feasibility)

    def test_complete_scalar_replay_matches_all_frozen_intervals_and_streams(self):
        checked = r.compare_expected(self.result, self.expected)
        self.assertEqual(checked["available_intervals"], 13)
        self.assertEqual(checked["maximum_absolute_numeric_difference"], 0.0)
        for setting, available, total in (("p5_main", 5, 11), ("p5_corpus", 5, 11), ("p5_model", 3, 7)):
            records = [x for x in self.result["endpoints"] if x["setting"] == setting]
            self.assertEqual(len(records), total)
            self.assertEqual(sum(x["status"] == "available" for x in records), available)

    def test_target_uses_other_two_individual_effects_not_prediction_seed(self):
        index = copy.deepcopy(self.index)
        cid = "p5_main-block000-b0-s0"
        expected = r.target_other_two(index, cid)
        index[cid]["original_individual_effect"][0] += 100
        self.assertEqual(r.target_other_two(index, cid), expected)
        index["p5_main-block000-b0-s1"]["original_individual_effect"][0] += 10
        self.assertAlmostEqual(r.target_other_two(index, cid)[0] - expected[0], 5)

    def test_missing_seed_or_changed_background_never_shrinks_population(self):
        data = copy.deepcopy(self.data)
        data["contexts"].pop()
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, data, self.costs)
        data = copy.deepcopy(self.data)
        data["contexts"][0]["background"] = 1
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, data, self.costs)

    def test_companion_endpoint_uses_mean_not_median_and_focal_order_breaks_ties(self):
        coherent, dispersed = [0.0] * 7 + [8.0], [0.0] * 8
        value = r.context_endpoint("E01_companion_error", {"original_individual_effect": [0.0] * 8},
                                   {"G5_coherent": coherent, "G5_dispersed": dispersed}, [0.0] * 8)
        self.assertEqual(value, 1.0)
        self.assertNotEqual(value, statistics.median(coherent))
        self.assertEqual(r.regret([1.0] * 8, list(range(8))), 6.0)

    def test_all_four_G5_fits_and_exact_two_focal_assignments_are_required(self):
        data = copy.deepcopy(self.data)
        row = next(x for x in data["contexts"] if x["setting"] == "p5_main")
        row["G5_partitions"]["G5_coherent"].pop()
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, data, self.costs)
        data = copy.deepcopy(self.data)
        row = next(x for x in data["contexts"] if x["setting"] == "p5_main")
        part = row["G5_partitions"]["G5_dispersed"][0]
        part["assigned_focal_indices"][1] = part["assigned_focal_indices"][0]
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, data, self.costs)

    def test_Q3_is_within_context_original_ALL_OUT_not_pooled_or_target_dependent(self):
        rows = [{"original_ALL_OUT_uncovered_count": list(range(8))},
                {"original_ALL_OUT_uncovered_count": list(range(244, 252))}]
        supports = [r.covered_fraction(row) for row in rows]
        self.assertAlmostEqual(supports[1][-1], 4 / 255)
        differences = [supports[0], [1000 + x for x in supports[1]]]
        values = [r.context_endpoint("E03_Q3", row,
                                    {"R_disj": diff, "G20": [0.0] * 8}, [999.0] * 8)
                  for row, diff in zip(rows, differences)]
        self.assertEqual(values, [1.0, 1.0])
        self.assertLess(r.correlation(differences[0] + differences[1], supports[0] + supports[1]), 0)

    def test_undefined_single_context_denies_whole_endpoint_without_complete_cases(self):
        predictions = {cid: r.scores(row, self.index) for cid, row in self.index.items()}
        targets = {cid: r.target_other_two(self.index, cid) for cid in self.index}
        predictions["p5_main-block000-b0-s0"]["R_disj"] = [0.0] * 8
        specification = next(x for x in self.protocol["endpoints"] if x["id"] == "E03_Q1")
        result = r.endpoint_record(specification, "p5_main", self.index, predictions, targets)
        self.assertEqual(result["status"], "blocked_undefined")
        self.assertEqual(result["population"]["blocks_expected"], 24)
        self.assertEqual(result["family_denominator"], 11)
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["interval"])
        self.assertIsNone(result["bootstrap"]["index_stream_sha256"])

    def test_reuse_is_next_original_block_including_development_block_donors(self):
        cid = "p5_main-block000-b0-s0"
        self.assertEqual(self.index[cid]["fixed_next_block_donor"], "p5_main-block001-b0-s0")
        self.assertEqual(self.index["p5_model-block007-b0-s2"]["fixed_next_block_donor"],
                         "p5_model-block000-b0-s2")
        data = copy.deepcopy(self.data)
        row = next(x for x in data["contexts"] if x["context_id"] == cid)
        row["fixed_next_block_donor"] = "p5_main-block002-b0-s0"
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, data, self.costs)

    def test_denied_selection_twenty_block_population_and_overlap_cannot_be_repaired(self):
        for record in self.result["endpoints"]:
            if record["id"].startswith("E02_"):
                self.assertEqual(record["population"]["blocks_expected"], 20 if record["setting"] == "p5_main" else 8)
                self.assertIsNone(record["estimate"])
        protocol = copy.deepcopy(self.protocol)
        protocol["E02_selection"].update(status="locked", selected_configuration="raw_0.001")
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(protocol, self.data, self.costs)
        costs = copy.deepcopy(self.costs)
        costs["accepted_scientific_job_hours"] = costs["reported_scientific_seconds"] / 3600
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(self.protocol, self.data, costs)
        protocol = copy.deepcopy(self.protocol)
        protocol["bootstrap_replicates"] = 5000
        with self.assertRaises(r.ReplayError):
            r.validate_inputs(protocol, self.data, self.costs)

    def test_contrary_selection_seed_exception_is_in_descriptive_outputs(self):
        record = next(x for x in self.result["endpoint_stability"]
                      if x["setting"] == "p5_main" and x["endpoint_id"] == "E01_selection")
        self.assertEqual(record["seed_means"]["0"], 0.0037440015281149652)
        self.assertLess(record["seed_means"]["1"], 0)
        self.assertLess(record["seed_means"]["2"], 0)
        self.assertEqual(record["block_sign_counts"], {"negative": 17, "positive": 6, "zero": 1})
        outputs = r.render_outputs(self.result, self.protocol)
        self.assertIn(b"0.0037440015281149652", outputs["followup_stability.csv"])
        for name, content in outputs.items():
            if name.endswith(".tex"):
                self.assertNotIn(b"\r", content)
                self.assertNotIn(b"E01_", content)
                self.assertNotIn(b"E03_", content)

    def test_original_two_background_feasibility_is_not_executed_background_zero(self):
        rows = self.result["input_feasibility"]["ranges"]
        original = next(x for x in rows if x["setting"] == "p5_corpus" and x["scope"].startswith("original"))
        executed = next(x for x in rows if x["setting"] == "p5_corpus" and x["scope"].startswith("executed"))
        self.assertNotEqual(original["minimum_net_support_fraction_contrast"],
                            executed["minimum_net_support_fraction_contrast"])
        self.assertNotEqual(original["maximum_net_support_fraction_contrast"],
                            executed["maximum_net_support_fraction_contrast"])
        self.assertFalse(original["original_gate_applies"])
        self.assertFalse(executed["original_gate_applies"])
        missing_background = copy.deepcopy(self.feasibility)
        missing_background["assignment_objectives"].pop()
        with self.assertRaises(r.ReplayError):
            r.feasibility_diagnostics(missing_background)


if __name__ == "__main__":
    unittest.main()
