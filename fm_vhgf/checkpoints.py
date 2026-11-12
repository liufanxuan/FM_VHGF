"""Strict visual-weight transfer from a local BiomedCLIP checkpoint."""

import re
from pathlib import Path

import torch


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_biomedclip_encoder(model, checkpoint):
    """Require every frozen attention/MLP tensor, and both copies of pretrained LN.

    The new 3D projections, register tokens and volumetric positional embedding
    are intentionally initialized for this model, not silently shape-filtered.
    """
    source = read_checkpoint(checkpoint)
    if isinstance(source, dict):
        source = source.get("state_dict", source.get("model", source))
    if not isinstance(source, dict):
        raise ValueError("Expected a tensor state dictionary.")
    source = {k.removeprefix("module."): v for k, v in source.items()}
    encoder = model.encoder
    required = {}
    for key in encoder.state_dict():
        if key.startswith("blocks."):
            original = re.sub(r"\.(norm[12])\.(pet|ct)\.", r".\1.", key)
            required[key] = original
        elif key.startswith("norm."):
            required[key] = key.replace(".pet.", ".").replace(".ct.", ".")
        elif key.startswith("cls_token_"):
            required[key] = "cls_token"
    mapped, missing = {}, []
    target = encoder.state_dict()
    for destination, original in required.items():
        candidates = ("visual.trunk." + original, "trunk." + original, original)
        found = next((key for key in candidates if key in source), None)
        if found is None:
            missing.append(destination)
        elif source[found].shape != target[destination].shape:
            raise ValueError(
                f"Pretrained shape mismatch for {found}: "
                f"{tuple(source[found].shape)} versus {tuple(target[destination].shape)}"
            )
        else:
            mapped[destination] = source[found]
    if missing:
        raise ValueError("Incomplete BiomedCLIP visual encoder; missing: " + ", ".join(missing[:12]))
    encoder.load_state_dict(mapped, strict=False)
    return {"source": str(checkpoint), "loaded_tensors": len(mapped),
            "new_components": ["patch_embed_pet", "patch_embed_ct",
                               "reg_token_pet", "reg_token_ct", "pos_embed"]}
