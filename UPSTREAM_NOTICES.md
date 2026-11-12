# Source provenance

This implementation was adapted from the user's recovered FM-VHGF project source. That project included modifications of OpenCLIP and timm, and an adaptive hyperedge implementation based on HyperACE/YOLOv13 as acknowledged in the manuscript.

Existing upstream license texts are retained under `LICENSES/`. They apply to their respective upstream material. A separate license for the project's original contributions has not been assigned by this code preparation.

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [BiomedCLIP weights](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

The active package is a compact native PyTorch implementation; it does not bundle the entire recovered experimental fork, patient data or pretrained weights.
