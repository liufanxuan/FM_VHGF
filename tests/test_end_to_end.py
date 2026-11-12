import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from examples.make_synthetic import create
from fm_vhgf.predict import main as predict
from fm_vhgf.train import main as train

torch.set_num_threads(1)


class EndToEndTests(unittest.TestCase):
    def test_both_tasks_train_five_folds_and_reload_best_loss_checkpoints(self):
        config = Path(__file__).resolve().parents[1] / "configs/smoke.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = create(root / "artificial")
            for task in ("isup", "epe"):
                output = root / task
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = train([
                        "--task", task, "--config", str(config),
                        "--data-root", str(data / "volumes"),
                        "--split-dir", str(data / "splits"), "--output", str(output),
                        "--device", "cpu", "--allow-random-init", "--epochs", "3", "--patience", "2",
                    ])
                self.assertEqual(len(summary["folds"]), 5)
                validation_cases = set()
                for fold in range(5):
                    fold_path = output / f"fold_{fold}"
                    checkpoint = torch.load(fold_path / "best.pt", weights_only=True)
                    with (fold_path / "history.csv").open() as stream:
                        history = list(csv.DictReader(stream))
                    best_epoch = min(history, key=lambda row: float(row["val_loss"]))
                    self.assertEqual(checkpoint["epoch"], int(best_epoch["epoch"]))
                    self.assertAlmostEqual(checkpoint["validation_loss"], float(best_epoch["val_loss"]))
                    self.assertEqual(checkpoint["task"], task)
                    with (fold_path / "predictions.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertNotIn(row["case_id"], validation_cases)
                        validation_cases.add(row["case_id"])
                        self.assertTrue(0 <= float(row["probability"]) <= 1)
                self.assertEqual(len(validation_cases), 10)
                predicted = predict([
                    "--checkpoint", str(output / "fold_0/best.pt"),
                    "--manifest", str(data / "splits/val_0.csv"),
                    "--data-root", str(data / "volumes"), "--output", str(output / "inference.csv"),
                    "--device", "cpu",
                ])
                with (output / "fold_0/predictions.csv").open() as stream:
                    reference = list(csv.DictReader(stream))
                for actual, expected in zip(predicted, reference):
                    self.assertEqual(actual["case_id"], expected["case_id"])
                    self.assertAlmostEqual(actual["probability"], float(expected["probability"]), places=6)
                run = json.loads((output / "run_config.json").read_text())
                self.assertEqual(run["cohort"]["patients"], 10)


if __name__ == "__main__":
    unittest.main()
