"""Create ten artificial volumes and predefined five-fold splits for a smoke test."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def create(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError("Choose a new directory for the synthetic example.")
    (root / "volumes").mkdir(parents=True)
    (root / "splits").mkdir()
    generator = np.random.default_rng(42)
    cases = []
    for index in range(10):
        case, label = f"synthetic_{index:02d}", index % 2
        with h5py.File(root / "volumes" / (case + ".h5"), "w") as h5:
            h5["pet"] = generator.normal(label * 0.2, 0.1, (1, 16, 16, 16)).astype("float32")
            h5["ct"] = generator.uniform(0, 1, (1, 16, 16, 16)).astype("float32")
            h5["label"] = np.int64(label)
        cases.append({"case_id": case, "path": case + ".h5", "label": label})
    for fold in range(5):
        for subset in ("train", "val"):
            rows = [row for index, row in enumerate(cases)
                    if (index // 2 == fold) == (subset == "val")]
            with (root / "splits" / f"{subset}_{fold}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["case_id", "path", "label"])
                writer.writeheader()
                writer.writerows(rows)
    (root / "SYNTHETIC_ONLY.txt").write_text(
        "Artificial data for software checks. Not prostate cancer patients or research results.\n"
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)
