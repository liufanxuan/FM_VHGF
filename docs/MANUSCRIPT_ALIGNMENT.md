# Implementation notes

- Input: co-registered PET and CT crops. The model resizes each crop to `(96, 224, 224)` before encoding.
- Configuration: the default encoder uses 12 layers; the fusion module uses 8 hyperedges. See `configs/isup.json` and `configs/epe.json` for all settings.
- Training: the default optimizer is AdamW with a learning rate of `1e-5`, batch size 8 and early stopping based on validation loss.
- Checkpoints: `best.pt` is selected using the lowest validation loss in each fold.
- Data: patient images, original fold assignments and trained checkpoints are not included.

