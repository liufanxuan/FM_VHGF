# Implementation changes and manuscript alignment

This is a modified implementation based on recovered project source, not a claim that the historical experiment-producing code has been recovered exactly. The specification was checked against `Manuscript_ESWA_final.docx`.

| Manuscript | Active implementation |
| --- | --- |
| 3.2: two volumetric branches | `VolumetricEncoder`: independent Conv3d patch projections and class/register tokens |
| 3.2: shared frozen pretrained weights | One set of attention/MLP blocks shared by both modalities, frozen by parameter flags; autograd remains enabled through them |
| 3.2: trainable adaptation components | Branch-specific LayerNorms, projections and tokens, plus shared volumetric positional embedding |
| 3.3: layer-wise descriptors | Final branch normalization of each block output, followed by class + register token readout |
| 3.4.1: adaptive hyperedges | Independent PET/CT generators, mean/max context, linear prototype offsets, multi-head similarities and softmax over depth nodes |
| 3.4.2: within-modality reasoning | Each modality uses its own participation matrix for node–hyperedge–node propagation |
| 3.4.3: bidirectional cross updating | CT participation reorganizes PET values; PET participation reorganizes CT values; separate directional projection parameters |
| 3.5: prediction | Updated node concatenation, depth-wise Conv1d over 2L nodes, MLP and scalar logit; sigmoid for probabilities |
| 3.5 / 4.3: BCE | Binary cross-entropy with logits, equivalent to sigmoid followed by BCE |
| 4.2: five folds | All five predefined folds are audited and trained; no replacement clinical splits are generated |
| 4.3: balancing | Class-balanced sampling for training only, no class weighting of validation loss |
| 4.3: model selection | Minimum validation BCE checkpoint, patience 20, at most 200 epochs |
| Evaluation | Explicit positive-label contract, probability-based AUC, fixed-threshold ACC/SEN/PRE/F1 |

## Details inherited from the recovered implementation

The supplied defaults retain 768-dimensional descriptors, 12 blocks, 12 Transformer attention heads, 16³ patches, 8 hyperedges, 4 hypergraph similarity heads, hypergraph dropout 0.1, and an MLP with hidden width 128 and dropout 0.5.

The 96³ crop is resized in-plane to 224×224, producing a 6×14×14 patch grid. This convention gives exactly 104,878,080 encoder parameters and 19,860,480 trainable encoder parameters, matching the rounded values in Section 3.2. The manuscript should state this resize explicitly; its preprocessing paragraph currently describes the 96³ crop and mentions resizing for 2D baselines but does not clearly specify the proposed model's in-plane resize.

The default learning rate is 1e-5 within the reported range. AdamW weight decay is 0.01. These settings, the MLP details, the 0.5 classification threshold, and sample SD convention are recorded configuration choices; the recovered material does not establish every exact historical experiment setting.

The final manuscript describes class-balanced training sampling without restricting it to one endpoint, so both supplied task configurations enable it. Changing the EPE policy would require an explicit configuration change and corresponding documentation.

## Pretrained transfer

The visual attention/MLP tensors retain the BiomedCLIP/timm parameter layout. Each original LayerNorm is copied into both modality branches, and the original class token initializes both class tokens. New 3D projections, register tokens and volumetric positional embedding are initialized for adaptation. The complete required visual tensor set is checked, rather than silently skipping missing frozen layers.

The original OpenCLIP text tower, final contrastive projection and experimental cached-feature inputs are not part of the active model. The Transformer descriptor dimension remains 768.

## Provenance and performance

Source reference: `open_clip_main_code_20260728_051401.tar.gz`, including the recovered `vision_transformer.py`, `HGYolo_my.py` and `model.py`.

The active model/trainer were organized into a small native PyTorch package to remove dependence on a broad experimental OpenCLIP/timm fork and hard-coded machine paths. The mathematical cross-modality update follows the recovered `HGFusion_CrossFusion` implementation and the final manuscript.

The historical clinical metrics, prediction files and trained checkpoints are not distributed or assigned to this modified code. They require rerunning the actual cohort with the correct pretrained weights and original patient folds.
