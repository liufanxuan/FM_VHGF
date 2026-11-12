"""Image-only inference from an FM-VHGF checkpoint and a PET/CT HDF5 manifest."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import H5VolumeDataset, read_records
from .model import FMVHGF, ModelConfig
from .train import write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--filename-suffix", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pet-key", default="pet")
    parser.add_argument("--ct-key", default="ct")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = FMVHGF(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    records = read_records(
        args.manifest, args.data_root, path_column=args.path_column,
        case_column=args.case_column, filename_suffix=args.filename_suffix,
        label_column="__unused_for_inference__", require_labels=False,
        pet_key=args.pet_key, ct_key=args.ct_key,
    )
    loader = DataLoader(H5VolumeDataset(records, cfg.input_size, args.pet_key, args.ct_key),
                        batch_size=args.batch_size, shuffle=False)
    rows = []
    for batch in loader:
        probabilities = model.predict_proba(
            batch["pet"].to(args.device), batch["ct"].to(args.device)
        ).cpu().tolist()
        for case, probability in zip(batch["case_id"], probabilities):
            rows.append({"case_id": case, "probability": probability})
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    return rows


if __name__ == "__main__":
    main()
