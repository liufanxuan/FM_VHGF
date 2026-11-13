# FM-VHGF

PyTorch code for PSMA PET/CT classification.

## Setup

Install [PyTorch 2.4.1](https://pytorch.org/get-started/previous-versions/), then run:

```bash
pip install -e .
```

Download `open_clip_pytorch_model.bin` from [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

## Data

Each patient needs an HDF5 file with aligned `pet` and `ct` volumes and a binary `label`. The label can instead be supplied in the CSV.

To prepare tumour-centred patches from aligned NIfTI images, provide PET in SUV units, CT in HU and a tumour mask. The mask is used only to locate the crop:

```bash
python -m fm_vhgf.preprocess --pet-suv /path/to/pet.nii.gz --ct-hu /path/to/ct.nii.gz --mask /path/to/mask.nii.gz --output /path/to/case.h5
```

Provide five predefined splits as `train_0.csv` and `val_0.csv` through `train_4.csv` and `val_4.csv`, with columns `case_id,path,label`. Label `1` denotes GG 3–5 for ISUP or EPE-positive for EPE.

## Training

```bash
python train_isup.py --config configs/isup.json --data-root /path/to/data --split-dir /path/to/isup_folds --pretrained /path/to/open_clip_pytorch_model.bin --output outputs/isup
```

For EPE, use `train_epe.py`, `configs/epe.json` and the EPE splits. Each fold's checkpoint, predictions and metrics are saved in the output directory.

## Inference

```bash
python -m fm_vhgf.predict --checkpoint outputs/isup/fold_0/best.pt --manifest /path/to/inference.csv --data-root /path/to/data --output outputs/predictions.csv
```

The inference CSV requires `case_id` and `path` columns. Add `--device cpu` to run on a CPU.

## Validation

```bash
python -m unittest discover -s tests -v
```

The BiomedCLIP checkpoint and reported clinical results have not yet been verified with this implementation. See [third-party notices](UPSTREAM_NOTICES.md).
