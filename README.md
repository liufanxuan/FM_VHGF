# FM-VHGF

Foundation-Model Transfer and Volumetric Hierarchical Hypergraph Fusion for PSMA PET/CT Prostate Cancer Risk Stratification.

This implementation was adapted from the recovered project code and aligned with the final manuscript's Methods and training protocol. Its software checks pass; the manuscript's clinical performance has **not** been reproduced with this modified implementation.

## Method

- Separate 3D PET/CT patch projections, class/register tokens and LayerNorms; shared positional embeddings and frozen pretrained attention/MLP weights.
- A descriptor from each Transformer block, obtained by normalizing and combining its class and register tokens.
- Separate adaptive PET/CT hypergraphs, within-modality aggregation, and two cross-modality updates using the other modality's depth-aligned participation structure.
- Concatenation of updated modality–depth nodes, depth-wise Conv1d pooling, and a scalar binary prediction head.

The model returns a logit during training. Binary cross-entropy is evaluated with the numerically stable logits formulation, and sigmoid converts the logit to a probability during evaluation and inference.

| Default model | Total parameters | Trainable parameters |
| --- | ---: | ---: |
| Volumetric encoder | 104,878,080 | 19,860,480 |
| Whole model, including fusion/prediction | 129,799,937 | 44,782,337 |

Encoder inputs have shape `(B,1,96,224,224)`; single-channel data are replicated internally across three channels. The 96³ physical ROI is resized in-plane to 224×224 before encoding, retaining its depth and common PET/CT geometry. This preserves the recovered encoder's token-grid convention. See [the implementation notes](docs/MANUSCRIPT_ALIGNMENT.md) for the distinction between specified manuscript settings and inherited defaults.

## Installation

Python 3.10 or later is required. The tests used Python 3.12 and PyTorch 2.4.1 on CPU.

```bash
python -m venv .venv
source .venv/bin/activate

# Choose the PyTorch build appropriate for your machine.
# Example matching the recovered CUDA 12.4 environment:
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

For a CPU environment, use PyTorch's CPU wheel index instead of the CUDA index.

## Pretrained visual weights

Obtain `open_clip_pytorch_model.bin` from the official [Microsoft BiomedCLIP model repository](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

The loader requires the complete pretrained visual attention/MLP parameters and transfers the pretrained normalization/class-token parameters into both branches. New volumetric projections, register tokens and volumetric positional embeddings remain trainable. A checkpoint missing required visual tensors raises an error; partial random initialization is not silently accepted.

Real BiomedCLIP weight transfer still requires validation with the supplied pretrained file. The automated transfer test uses a synthetic checkpoint with the same key layout.

## Prepare data

Clinical data, patient manifests, original fold assignments and model weights are not included.

Each patient has one HDF5 file with:

| Key | Shape / meaning |
| --- | --- |
| `pet` | `(D,H,W)` or `(1,D,H,W)`, preprocessed PET crop |
| `ct` | Same grid and shape as PET |
| `label` | Optional when provided in the CSV manifest; binary 0/1 |

Cached 2D features and clinical text are not required. Tumour annotations are used by the preprocessing step to select the crop center; the model itself consumes the cropped PET/CT pair.

An optional NIfTI preprocessing command accepts already co-registered PET in **SUV** units, CT in **HU** units, and a tumour mask:

```bash
python -m fm_vhgf.preprocess \
  --pet-suv /path/to/pet_suv.nii.gz \
  --ct-hu /path/to/ct_hu.nii.gz \
  --mask /path/to/tumour_mask.nii.gz \
  --label 1 \
  --output /path/to/prepared/example.h5
```

It places the images on a common 1 mm CT-derived grid, applies the CT window (width 350 HU, level 50 HU), and extracts a 96³ ROI around the bounding-box center of all annotated tumour regions. It does not calculate SUV from raw DICOM metadata or estimate registration.

## Existing five-fold assignments

Provide `train_0.csv`, `val_0.csv`, …, `train_4.csv`, `val_4.csv`. Keep the same patient assignments across all model comparisons; the training code does not generate new clinical folds.

The CSV format is:

```csv
case_id,path,label
example_patient_a,example_a.h5,0
example_patient_b,example_b.h5,1
```

`case_id` must identify a patient, with one paired HDF5 record per patient. A CSV label takes precedence over the HDF5 label. Set `1 = GG3–5` for ISUP and `1 = EPE-positive` for EPE.

- For raw Grade Group labels 1–5, use `--label-encoding grade_group` with the ISUP task.
- For legacy two-element one-hot labels, use `--label-encoding onehot --positive-index 0` or `1` according to the **actual original encoding**. The program does not infer the positive class from vector position.
- For the old `vol_names` CSV column, add `--path-column vol_names`.
- To map old `case.h5` entries to local `case_feature.h5`, add `--filename-suffix _feature`.

Before training, all five splits are checked for patient/path leakage, consistent labels, and a complete validation partition of the cohort.

## Train

```bash
python train_isup.py \
  --config configs/isup.json \
  --data-root /path/to/isup_h5 \
  --split-dir /path/to/isup_folds \
  --pretrained /path/to/open_clip_pytorch_model.bin \
  --expected-patients 476 \
  --output outputs/isup_run1

python train_epe.py \
  --config configs/epe.json \
  --data-root /path/to/epe_h5 \
  --split-dir /path/to/epe_folds \
  --pretrained /path/to/open_clip_pytorch_model.bin \
  --expected-patients 205 \
  --output outputs/epe_run1
```

Both commands execute all five folds with a fixed configuration. Defaults are AdamW, batch size 8, learning rate 1e-5, up to 200 epochs, and patience 20 based on validation BCE. Class balancing is applied to the training sampler only. Validation keeps its original distribution, and each fold's minimum-validation-loss checkpoint supplies the reported predictions. Reusing an output directory containing checkpoints is rejected.

The learning rate can be configured within the manuscript's 5e-6–2e-5 range. The supplied 1e-5 setting and other inherited implementation defaults are not asserted to be the exact historical configuration that produced the paper's tables.

Outputs include run/cohort configuration, per-epoch loss/metrics, each fold's best checkpoint, continuous positive-class validation probabilities, and five-fold mean ± sample SD (`ddof=1`). AUC uses the continuous probabilities; ACC/SEN/PRE/F1 use the fixed configured threshold (default 0.5). These are cross-validation results, without an additional independent test cohort.

## Image-only model inference

```bash
python -m fm_vhgf.predict \
  --checkpoint outputs/isup_run1/fold_0/best.pt \
  --manifest /path/to/inference_manifest.csv \
  --data-root /path/to/prepared_h5 \
  --output outputs/isup_run1/inference.csv
```

Inference manifests need patient IDs and paired image paths; labels are ignored. Preprocessing and crop-center requirements remain the same as for training.

## Software checks

```bash
python -m unittest discover -s tests -v

python examples/make_synthetic.py --output demo_data
python train_isup.py \
  --config configs/smoke.json \
  --data-root demo_data/volumes --split-dir demo_data/splits \
  --output smoke_run/isup --device cpu --allow-random-init
```

The smoke configuration uses small artificial volumes and a reduced model. Random initialization is permitted only when explicitly requested; this example does not assess clinical performance.

See [validation details](docs/VALIDATION.md), [manuscript alignment](docs/MANUSCRIPT_ALIGNMENT.md), and [source notices](UPSTREAM_NOTICES.md).
