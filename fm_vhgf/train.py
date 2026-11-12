"""Fixed-configuration, patient-level five-fold training for ISUP or EPE."""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .checkpoints import load_biomedclip_encoder
from .data import H5VolumeDataset, audit_five_folds, read_records
from .metrics import binary_metrics
from .model import FMVHGF, ModelConfig


@dataclass
class EarlyStopping:
    patience: int = 20
    best_loss: float = float("inf")
    bad_epochs: int = 0

    def update(self, loss):
        if not np.isfinite(loss):
            raise ValueError("Validation loss is not finite.")
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.bad_epochs = float(loss), 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_sampler(records, seed):
    labels = np.array([record.label for record in records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Training data must contain both classes.")
    weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(records), replacement=True, generator=generator)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss, rows = 0.0, []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = model(batch["pet"].to(device), batch["ct"].to(device))
            total_loss += nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            ).item()
            for case, label, probability in zip(
                batch["case_id"], labels.cpu().tolist(), logits.sigmoid().cpu().tolist()
            ):
                rows.append({"case_id": case, "label": int(label), "probability": probability})
    metrics = binary_metrics([row["label"] for row in rows],
                             [row["probability"] for row in rows], threshold)
    metrics["loss"] = total_loss / len(rows)
    return metrics, rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_fold(index, train_records, val_records, model_cfg, settings, args):
    seed = settings["seed"] + index
    seed_everything(seed)
    output = args.output / f"fold_{index}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"{output}: existing results; choose a new output directory.")
    model = FMVHGF(model_cfg)
    if args.pretrained:
        transfer = load_biomedclip_encoder(model, args.pretrained)
    else:
        transfer = {"source": "random initialization explicitly requested for a smoke test"}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select --device cpu for a CPU check.")
    model.to(device)
    train_set, val_set = (
        H5VolumeDataset(records, model_cfg.input_size, args.pet_key, args.ct_key)
        for records in (train_records, val_records)
    )
    sampler = balanced_sampler(train_records, seed) if settings["balanced_sampling"] else None
    train_loader = DataLoader(
        train_set, batch_size=settings["batch_size"], sampler=sampler,
        shuffle=sampler is None, num_workers=settings["num_workers"],
        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
    )
    # Validation is never sampled or rebalanced.
    val_loader = DataLoader(val_set, batch_size=settings["batch_size"], shuffle=False,
                            num_workers=settings["num_workers"], pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["lr"], weight_decay=settings["weight_decay"],
    )
    stop = EarlyStopping(settings["patience"])
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["pet"].to(device), batch["ct"].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            count += labels.numel()
        val_metrics, _ = evaluate(model, val_loader, device, settings["threshold"])
        improved, should_stop = stop.update(val_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        **{"val_" + key: value for key, value in val_metrics.items()}})
        write_csv(output / "history.csv", history)
        if improved:
            checkpoint = {
                "model": model.state_dict(), "model_config": model_cfg.to_dict(),
                "training": settings, "task": args.task, "fold": index, "epoch": epoch,
                "validation_loss": val_metrics["loss"], "encoder_transfer": transfer,
                "status": "manuscript-aligned implementation; historical results not reproduced",
            }
            temporary = output / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        print(json.dumps({"fold": index, **history[-1], "best_checkpoint": improved}), flush=True)
        if should_stop:
            break
    # Report only predictions from the checkpoint selected by validation BCE.
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    metrics, rows = evaluate(model, val_loader, device, settings["threshold"])
    write_csv(output / "predictions.csv", rows)
    result = {"fold": index, "best_epoch": best["epoch"], **metrics}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser(default_task=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["isup", "epe"], default=default_task,
                        required=default_task is None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Existing train_0.csv/val_0.csv through train_4.csv/val_4.csv.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, help="Local BiomedCLIP checkpoint.")
    parser.add_argument("--allow-random-init", action="store_true", help="Synthetic smoke tests only.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-encoding", choices=["binary", "grade_group", "onehot"], default="binary")
    parser.add_argument("--positive-index", type=int, choices=[0, 1])
    parser.add_argument("--filename-suffix", default="")
    parser.add_argument("--pet-key", default="pet")
    parser.add_argument("--ct-key", default="ct")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--expected-patients", type=int,
                        help="Optional cohort-size check against the manuscript.")
    for name, dtype in [("lr", float), ("batch-size", int), ("epochs", int),
                        ("patience", int), ("num-workers", int), ("seed", int)]:
        parser.add_argument("--" + name, type=dtype, default=None)
    return parser


def main(argv=None, default_task=None):
    args = build_parser(default_task).parse_args(argv)
    if not args.pretrained and not args.allow_random_init:
        raise ValueError("--pretrained is required; --allow-random-init is only for synthetic checks.")
    config = json.loads(args.config.read_text()) if args.config else {}
    if set(config) - {"task", "model", "training"}:
        raise ValueError("Unknown configuration sections.")
    if config.get("task", args.task) != args.task:
        raise ValueError("Task and configuration disagree.")
    if args.task == "epe" and args.label_encoding == "grade_group":
        raise ValueError("Grade Group conversion is only valid for the ISUP endpoint.")
    model_cfg = ModelConfig(**config.get("model", {}))
    settings = {"batch_size": 8, "lr": 1e-5, "weight_decay": 0.01, "epochs": 200,
                "patience": 20, "balanced_sampling": True, "threshold": 0.5,
                "num_workers": 4, "seed": 42}
    overrides = config.get("training", {})
    if set(overrides) - set(settings):
        raise ValueError("Unknown training settings: " + ", ".join(sorted(set(overrides) - set(settings))))
    settings.update(overrides)
    for key in ("lr", "batch_size", "epochs", "patience", "num_workers", "seed"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if settings["epochs"] < 1 or settings["epochs"] > 200 or settings["patience"] < 1:
        raise ValueError("Use 1–200 epochs and positive early-stopping patience.")
    if not 5e-6 <= settings["lr"] <= 2e-5:
        raise ValueError("Learning rate must be within the manuscript's 5e-6 to 2e-5 range.")
    if settings["batch_size"] < 1 or settings["num_workers"] < 0:
        raise ValueError("Invalid batch size or worker count.")
    if not 0 <= settings["threshold"] <= 1:
        raise ValueError("Classification threshold must be in [0,1].")
    record_options = {key: getattr(args, key) for key in (
        "path_column", "case_column", "label_column", "label_encoding", "positive_index",
        "filename_suffix", "pet_key", "ct_key", "label_key")}
    splits = [
        (read_records(args.split_dir / f"train_{index}.csv", args.data_root, **record_options),
         read_records(args.split_dir / f"val_{index}.csv", args.data_root, **record_options))
        for index in range(5)
    ]
    audit = audit_five_folds(splits)
    if args.expected_patients is not None and audit["patients"] != args.expected_patients:
        raise ValueError(f"Cohort size {audit['patients']} does not match {args.expected_patients}.")
    if any((args.output / f"fold_{index}" / "best.pt").exists() for index in range(5)):
        raise FileExistsError("This output already contains checkpoints; select a new output path.")
    args.output.mkdir(parents=True, exist_ok=True)
    run = {"task": args.task, "model": model_cfg.to_dict(), "training": settings,
           "cohort": audit, "label_encoding": args.label_encoding,
           "positive_index": args.positive_index,
           "positive_class": "GG3–5" if args.task == "isup" else "EPE-positive"}
    (args.output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    results = [fit_fold(index, train, val, model_cfg, settings, args)
               for index, (train, val) in enumerate(splits)]
    summary = {"folds": results, "sd_ddof": 1, "metrics": {
        key: {"mean": float(np.mean([result[key] for result in results])),
              "sd": float(np.std([result[key] for result in results], ddof=1))}
        for key in ("auc", "acc", "sen", "pre", "f1")
    }}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
