"""Boundary, restart and checkpoint-provenance tests for the new stop rule."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.nmc_eden_stop_policy import POLICY, evaluate


def history(gains=(0.0, 0.0), reduction=950_000, end=1_000_000):
    rows = []
    for step in range(5_000, end + 1, 5_000):
        after = step > end - 50_000
        rows.append(dict(step=step, density_nmae=1 - gains[0] if after else 1.0,
                         occupation_relative_l1=1 - gains[1] if after else 1.0,
                         lr=0.005 if step <= reduction else 0.0025,
                         next_lr=0.005 if step < reduction else 0.0025))
    return rows


class StopPolicyTests(unittest.TestCase):
    def test_flat_both_stops_after_full_observation(self):
        result = evaluate(history())
        self.assertTrue(result["stop"])
        self.assertEqual(len(result["windows"]["baseline_steps"]), 10)
        self.assertEqual(len(result["windows"]["recent_steps"]), 10)
        self.assertFalse(set(result["windows"]["baseline_steps"]) & set(result["windows"]["recent_steps"]))

    def test_either_metric_improving_keeps_training(self):
        for gains in ((0.01, 0), (0, 0.01), (0.005, 0), (0, 0.005)):
            with self.subTest(gains=gains):
                self.assertFalse(evaluate(history(gains))["stop"])

    def test_both_below_half_percent_stop(self):
        self.assertTrue(evaluate(history((0.0049, 0.0049)))["stop"])

    def test_worsening_does_not_count_as_improvement(self):
        self.assertTrue(evaluate(history((-0.1, -0.2)))["stop"])

    def test_wait_fifty_thousand_steps_after_latest_reduction(self):
        self.assertFalse(evaluate(history(reduction=955_000))["eligible"])
        self.assertTrue(evaluate(history(reduction=950_000))["eligible"])

    def test_minimum_one_million_prevents_an_earlier_plateau_stop(self):
        result = evaluate(history(reduction=900_000, end=995_000))
        self.assertFalse(result["eligible"])
        self.assertFalse(result["stop"])
        self.assertEqual(result["reason"], "minimum_training_steps_not_reached")
        self.assertTrue(evaluate(history(reduction=900_000))["stop"])

    def test_new_reduction_resets_observation(self):
        rows = history()
        rows[-1]["next_lr"] = 0.00125
        self.assertFalse(evaluate(rows)["eligible"])

    def test_requires_a_learning_rate_reduction(self):
        self.assertEqual(evaluate(history(reduction=1_050_000))["reason"], "awaiting_first_lr_reduction")

    def test_incomplete_or_nonfinite_data_cannot_stop_training(self):
        self.assertFalse(evaluate(history(end=95_000))["stop"])
        for rows in (history()[1:], history() + [history()[-1]]):
            with self.assertRaises(ValueError):
                evaluate(rows)
        rows = history()
        rows[-1]["density_nmae"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_serialized_resume_produces_identical_decision(self):
        rows = history((0.004, 0.003))
        self.assertEqual(evaluate(rows), evaluate(json.loads(json.dumps(rows))))

    def test_uses_window_best_not_latest_outlier(self):
        rows = history()
        rows[-2]["density_nmae"] = 0.98
        self.assertFalse(evaluate(rows)["stop"])


class CheckpointTests(unittest.TestCase):
    def test_migration_preserves_training_states_and_is_idempotent(self):
        import torch
        from scripts import nmc_eden_early_stopping as trainer
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            old, new = {"recipe": "unchanged"}, {"recipe": "unchanged", "early_stopping": POLICY}
            before = dict(lock=old, step=1_000_000, model={"w": torch.tensor([1., 2.])},
                          optimizer={"momentum": torch.tensor([3.])}, scheduler={"best": 0.2},
                          cpu_rng=torch.get_rng_state(), cuda_rng=torch.tensor([4], dtype=torch.uint8),
                          history=history(), best_model={"w": torch.tensor([0.9, 1.8])})
            torch.save(before, run / "last.pt")
            for name, value in (("settings.json", old), ("launch.json", {}),
                                ("validation.json", history()), ("paired_validation.json", {})):
                (run / name).write_text(json.dumps(value))
            with patch.object(trainer, "RUN", run), patch.object(trainer.original, "settings", return_value=old):
                with self.assertRaises(ValueError):
                    trainer.load_or_adopt_checkpoint({}, new, False)
                adopted = trainer.load_or_adopt_checkpoint({}, new, True)
                self.assertTrue(torch.equal(adopted["model"]["w"], before["model"]["w"]))
                self.assertTrue(torch.equal(adopted["optimizer"]["momentum"], before["optimizer"]["momentum"]))
                self.assertEqual(adopted["scheduler"], before["scheduler"])
                self.assertTrue(torch.equal(adopted["cpu_rng"], before["cpu_rng"]))
                self.assertTrue(torch.equal(adopted["cuda_rng"], before["cuda_rng"]))
                resumed = trainer.load_or_adopt_checkpoint({}, new, False)
                self.assertEqual(resumed["step"], before["step"])
                self.assertEqual(resumed["history"], before["history"])
                self.assertEqual(resumed["lock"], new)
                with self.assertRaises(ValueError):
                    trainer.load_or_adopt_checkpoint({}, {"recipe": "changed"}, True)

    def test_freeze_uses_one_validation_selected_checkpoint_for_both_outputs(self):
        import torch
        from scripts.nmc_eden_early_stop_test import freeze_stopped
        rows = history()
        for row in rows:
            row["density_mae"] = 0.2 if row["step"] == 985_000 else 0.3
        lock = {"early_stopping": POLICY, "max_steps": 4_965_000}
        state = dict(step=1_000_000, history=rows, lock=lock, stop_reason="early_stopping",
                     early_stopping_decision=evaluate(rows), best_model={"w": torch.tensor([8.5])})
        with tempfile.TemporaryDirectory() as tmp:
            source, folder = Path(tmp) / "last.pt", Path(tmp) / "test"
            torch.save(state, source)
            selection = freeze_stopped(state, lock, source, folder)
            self.assertEqual(selection["step"], 1_000_000)
            self.assertEqual(selection["validation"]["step"], 985_000)
            self.assertEqual(freeze_stopped(state, lock, source, folder), selection)
            invalid = copy.deepcopy(state)
            invalid["early_stopping_decision"]["stop"] = False
            with self.assertRaises(ValueError):
                freeze_stopped(invalid, lock, source, folder)


if __name__ == "__main__":
    unittest.main()
