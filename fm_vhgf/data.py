"""Patient-level HDF5 input and auditing of existing five-fold assignments."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    case_id: str
    path: Path
    label: int | None


def binary_label(value, encoding="binary", positive_index=None):
    if isinstance(value, str) and value.strip().startswith("["):
        value = json.loads(value)
    values = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Labels must be finite.")
    if encoding == "onehot":
        if positive_index not in (0, 1):
            raise ValueError("Legacy one-hot labels require an explicit positive_index of 0 or 1.")
        if values.size != 2 or not np.isin(values, [0, 1]).all() or values.sum() != 1:
            raise ValueError("Expected a two-element one-hot label.")
        return int(values[positive_index])
    if values.size != 1:
        raise ValueError("Vector labels require label_encoding='onehot' and positive_index.")
    value = float(values[0])
    if encoding == "grade_group":
        if value not in (1, 2, 3, 4, 5):
            raise ValueError("ISUP Grade Groups must be integers 1–5.")
        return int(value >= 3)
    if encoding != "binary" or value not in (0, 1):
        raise ValueError("Binary labels must be 0 or 1; positive class must be coded 1.")
    return int(value)


def read_records(csv_path, data_root, *, path_column="path", case_column="case_id",
                 label_column="label", label_encoding="binary", positive_index=None,
                 filename_suffix="", require_labels=True, pet_key="pet", ct_key="ct",
                 label_key="label"):
    root, csv_path = Path(data_root).resolve(), Path(csv_path)
    records = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or path_column not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing path column {path_column!r}")
        for row in reader:
            raw = Path(row[path_column])
            candidate = raw if raw.is_absolute() else root / raw
            if filename_suffix:
                candidate = root / (raw.stem + filename_suffix + ".h5")
            elif not candidate.is_file() and raw.is_absolute():
                candidate = root / raw.name
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"{csv_path}: missing HDF5 file {candidate}")
            case = row.get(case_column, "").strip() or raw.stem
            if not case:
                raise ValueError("Every row requires a patient identifier.")
            label = None
            csv_label = row.get(label_column, "").strip()
            with h5py.File(candidate, "r") as h5:
                if pet_key not in h5 or ct_key not in h5:
                    raise ValueError(f"{candidate}: missing paired PET/CT datasets.")
                if csv_label:
                    label = binary_label(csv_label, label_encoding, positive_index)
                elif require_labels:
                    if label_key not in h5:
                        raise ValueError(f"{candidate}: missing label.")
                    label = binary_label(h5[label_key][()], label_encoding, positive_index)
            if require_labels and label is None:
                raise ValueError(f"{case}: missing reference label.")
            records.append(Record(case, candidate, label))
    if not records:
        raise ValueError(f"{csv_path}: empty manifest.")
    if len({r.case_id for r in records}) != len(records):
        raise ValueError(f"{csv_path}: duplicate patient identifiers.")
    if len({r.path for r in records}) != len(records):
        raise ValueError(f"{csv_path}: duplicate HDF5 paths.")
    return records


def audit_five_folds(splits):
    if len(splits) != 5:
        raise ValueError("Exactly five predefined train/validation splits are required.")
    seen_validation, reference = set(), {}
    for index, (train, val) in enumerate(splits):
        train_ids, val_ids = {r.case_id for r in train}, {r.case_id for r in val}
        if train_ids & val_ids or {r.path for r in train} & {r.path for r in val}:
            raise ValueError(f"Fold {index}: patient leakage between training and validation.")
        if seen_validation & val_ids:
            raise ValueError("A patient appears in more than one validation fold.")
        seen_validation |= val_ids
        for records in (train, val):
            if {r.label for r in records} != {0, 1}:
                raise ValueError(f"Fold {index}: both binary classes are required in each subset.")
            for record in records:
                identity = (record.path, record.label)
                if record.case_id in reference and reference[record.case_id] != identity:
                    raise ValueError("Inconsistent patient paths or labels across folds.")
                reference[record.case_id] = identity
    cohort = set(reference)
    # Each patient's path must identify one and only one patient.
    if len({path for path, _ in reference.values()}) != len(reference):
        raise ValueError("The same HDF5 file is assigned to different patient identifiers.")
    if seen_validation != cohort:
        raise ValueError("Validation folds do not partition the entire cohort.")
    for index, (train, val) in enumerate(splits):
        if {r.case_id for r in train} != cohort - {r.case_id for r in val}:
            raise ValueError(f"Fold {index}: training set is not the other four folds.")
    return {"patients": len(cohort),
            "negative": sum(label == 0 for _, label in reference.values()),
            "positive": sum(label == 1 for _, label in reference.values())}


class H5VolumeDataset(Dataset):
    def __init__(self, records, input_size=(96, 224, 224), pet_key="pet", ct_key="ct"):
        self.records, self.input_size = records, tuple(input_size)
        self.pet_key, self.ct_key = pet_key, ct_key

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _volume(array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or array.shape[0] not in (1, 3):
            raise ValueError("Each HDF5 image must be (D,H,W), (1,D,H,W), or replicated RGB.")
        if array.shape[0] == 3:
            if not (np.array_equal(array[0], array[1]) and np.array_equal(array[0], array[2])):
                raise ValueError("Three-channel volumes must be copies of one modality.")
            array = array[:1]
        if not np.isfinite(array).all():
            raise ValueError("Image arrays contain NaN or infinity.")
        return torch.from_numpy(np.ascontiguousarray(array))

    def __getitem__(self, index):
        record = self.records[index]
        with h5py.File(record.path, "r") as h5:
            pet, ct = self._volume(h5[self.pet_key][()]), self._volume(h5[self.ct_key][()])
        if pet.shape != ct.shape:
            raise ValueError(f"{record.case_id}: paired image grids have different shapes.")
        if pet.shape[1] != self.input_size[0]:
            raise ValueError("Unexpected ROI depth; prepare crops at the configured physical spacing.")
        if tuple(pet.shape[1:]) != self.input_size:
            # Shared output geometry; ROI depth and physical crop are preserved.
            pet = F.interpolate(pet[None], size=self.input_size, mode="trilinear",
                                align_corners=False).squeeze(0)
            ct = F.interpolate(ct[None], size=self.input_size, mode="trilinear",
                               align_corners=False).squeeze(0)
        output = {"pet": pet, "ct": ct, "case_id": record.case_id}
        if record.label is not None:
            output["label"] = torch.tensor(record.label, dtype=torch.float32)
        return output
