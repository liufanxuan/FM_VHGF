# FM-VHGF

PyTorch implementation for PET/CT classification.

## Installation

Create a Python environment and install the dependencies. For example, with CUDA 12.4:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

For a different CUDA version or CPU, install the appropriate [PyTorch build](https://pytorch.org/get-started/previous-versions/) first.

Download `open_clip_pytorch_model.bin` from [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) for encoder initialization.

## Data preparation

Patient data, fold assignments and trained checkpoints are not included.

Each patient requires an HDF5 file containing paired `pet` and `ct` arrays. Both arrays must be aligned and use the same grid. A binary `label` can be stored in the HDF5 file or provided in the CSV manifest.

To prepare a case from co-registered NIfTI images and a tumour mask:

```bash
python -m fm_vhgf.preprocess \
  --pet-suv /path/to/pet_suv.nii.gz \
  --ct-hu /path/to/ct_hu.nii.gz \
  --mask /path/to/tumour_mask.nii.gz \
  --label 1 \
  --output /path/to/data/patient_001.h5
```

PET must be in SUV units and CT in Hounsfield units. The script resamples the images to 1 mm spacing and extracts a tumour-centred 96 × 96 × 96 crop. The input mask is used to determine the crop centre.

Create `train_0.csv` and `val_0.csv` through `train_4.csv` and `val_4.csv`. Use one patient per row:

```csv
case_id,path,label
patient_001,patient_001.h5,1
patient_002,patient_002.h5,0
```

For ISUP, label `1` denotes GG 3–5 and `0` denotes GG 1–2. For EPE, `1` denotes EPE-positive and `0` EPE-negative. The training script checks for patient overlap across splits.

## Training

Run five-fold training for ISUP:

```bash
python train_isup.py \
  --config configs/isup.json \
  --data-root /path/to/data \
  --split-dir /path/to/isup_folds \
  --pretrained /path/to/open_clip_pytorch_model.bin \
  --output outputs/isup
```

For EPE:

```bash
python train_epe.py \
  --config configs/epe.json \
  --data-root /path/to/data \
  --split-dir /path/to/epe_folds \
  --pretrained /path/to/open_clip_pytorch_model.bin \
  --output outputs/epe
```

The scripts save the best checkpoint, validation predictions and metrics for each fold. Training settings are defined in `configs/isup.json` and `configs/epe.json`. Use a new output directory for each run.

## Inference

Prepare a CSV with `case_id` and `path` columns, then run:

```bash
python -m fm_vhgf.predict \
  --checkpoint outputs/isup/fold_0/best.pt \
  --manifest /path/to/inference.csv \
  --data-root /path/to/data \
  --output outputs/predictions.csv
```

Add `--device cpu` for CPU inference. For other options, run `python train_isup.py --help` or `python -m fm_vhgf.predict --help`.

## Tests

```bash
python -m unittest discover -s tests -v
```

**Validation:** The reported clinical results have not been reproduced with this implementation. Loading the full BiomedCLIP checkpoint has not yet been verified.

See [third-party notices](UPSTREAM_NOTICES.md) and [validation details](docs/VALIDATION.md).
