# Validation

Checked using Python 3.12 and PyTorch 2.4.1+cpu.

- Unit checks cover gradient flow through the frozen Transformer blocks, modality-specific adaptation parameters, normalized hyperedge participation, depth permutation behavior, each cross-modality direction, strict visual checkpoint loading, label coding, patient leakage, class-balanced sampling, continuous-score AUC, loss-based early stopping, paired HDF5 input and NIfTI crop/window preprocessing.
- The full default model passed an inference-mode forward call on paired tensors of shape `(1,1,96,224,224)`. Its scalar output and both `(1,12,8)` participation matrices were checked for shape, finite values and normalized participation.
- Default encoder parameter counts were checked exactly: 104,878,080 total and 19,860,480 trainable.
- End-to-end checks trained both ISUP and EPE task entry points across all five folds of an artificial ten-patient dataset using a reduced model. Saved checkpoints were checked against the minimum observed validation loss, and reloaded inference probabilities were compared with the saved validation probabilities.
- All automated examples and labels are synthetic. No patient images, reports, clinical tables or real patient split files were used.

Not validated: RTX 4090/GPU execution; loading an actual full BiomedCLIP checkpoint; full-size model training on the clinical cohort; reproduction of manuscript AUCs or other reported results.
