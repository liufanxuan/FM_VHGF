import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from torch import nn

from fm_vhgf.checkpoints import load_biomedclip_encoder
from fm_vhgf.data import (
    H5VolumeDataset, Record, audit_five_folds, binary_label, read_records,
)
from fm_vhgf.hypergraph import HierarchicalHypergraphFusion
from fm_vhgf.metrics import binary_metrics
from fm_vhgf.model import FMVHGF, ModelConfig, VolumetricEncoder
from fm_vhgf.preprocess import prepare_case
from fm_vhgf.train import EarlyStopping, balanced_sampler

torch.set_num_threads(1)


def tiny_config():
    return ModelConfig(
        input_size=(16, 16, 16), patch_size=(8, 8, 8), dim=16,
        depth=2, num_heads=4, num_edges=3, hypergraph_heads=4,
        hypergraph_dropout=0, classifier_hidden=8, classifier_dropout=0,
    )


def fake_2d_state(model):
    """Export only the pretrained 2D-compatible weights, with one LN per block."""
    state = {}
    for key, value in model.encoder.state_dict().items():
        if key.startswith("blocks."):
            if ".ct." in key:
                continue
            key = key.replace(".pet.", ".")
        elif key.startswith("norm."):
            if ".ct." in key:
                continue
            key = key.replace(".pet.", ".")
        elif key == "cls_token_pet":
            key = "cls_token"
        else:
            continue
        state["visual.trunk." + key] = value.clone()
    return state


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_gradient_reaches_adapters_through_frozen_blocks(self):
        model = FMVHGF(tiny_config())
        x, y = torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 16, 16, 16)
        logits = model(x, y)
        self.assertEqual(tuple(logits.shape), (2,))
        nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
        for block in model.encoder.blocks:
            for module in (block.attn, block.mlp):
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        for parameter in (
            model.encoder.patch_embed_pet.weight, model.encoder.patch_embed_ct.weight,
            model.encoder.blocks[0].norm1.pet.weight, model.encoder.blocks[0].norm1.ct.weight,
            model.encoder.reg_token_pet, model.encoder.reg_token_ct, model.encoder.pos_embed,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        model.eval()
        self.assertTrue(torch.allclose(model.predict_proba(x, y), model(x, y).sigmoid()))

    def test_participation_and_depth_permutation(self):
        fusion = HierarchicalHypergraphFusion(16, 3, 4, 0).eval()
        pet, ct = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        p, c, weights = fusion(pet, ct, return_participation=True)
        for matrix in weights.values():
            self.assertTrue(torch.allclose(matrix.sum(1), torch.ones(2, 3), atol=1e-6))
        order = torch.tensor([2, 0, 3, 1])
        p2, c2 = fusion(pet[:, order], ct[:, order])
        self.assertTrue(torch.allclose(p2, p[:, order], atol=1e-6))
        self.assertTrue(torch.allclose(c2, c[:, order], atol=1e-6))

    def test_other_modality_structure_controls_each_direction(self):
        fusion = HierarchicalHypergraphFusion(16, 3, 4, 0)
        pet, ct = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        pet_out, _ = fusion(pet, ct)
        pet_out.square().mean().backward()
        # PET-only loss reaches the CT structure generator through CT-guided PET updating.
        self.assertGreater(fusion.ct_generator.context_net.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(fusion.ct_edge[0].weight.grad)
        fusion.zero_grad(set_to_none=True)
        _, ct_out = fusion(pet, ct)
        ct_out.square().mean().backward()
        self.assertGreater(fusion.pet_generator.context_net.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(fusion.pet_edge[0].weight.grad)

    def test_cross_update_uses_target_values_with_source_structure(self):
        fusion = HierarchicalHypergraphFusion(16, 3, 4, 0).eval()
        for module in fusion.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        pet, ct = torch.zeros(2, 4, 16), torch.randn(2, 4, 16)
        pet_out, _ = fusion(pet, ct)
        # CT-derived structure cannot inject CT values into zero-valued PET nodes.
        self.assertTrue(torch.equal(pet_out, torch.zeros_like(pet_out)))

    def test_strict_pretrained_transfer_and_branch_norm_copy(self):
        model = FMVHGF(tiny_config())
        state = fake_2d_state(model)
        preserved = model.encoder.pos_embed.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "encoder.pt"
            torch.save(state, checkpoint)
            load_biomedclip_encoder(model, checkpoint)
            self.assertTrue(torch.equal(model.encoder.cls_token_pet, model.encoder.cls_token_ct))
            self.assertTrue(torch.equal(
                model.encoder.blocks[0].norm1.pet.weight,
                model.encoder.blocks[0].norm1.ct.weight,
            ))
            self.assertTrue(torch.equal(model.encoder.pos_embed, preserved))
            state.pop("visual.trunk.blocks.0.attn.qkv.weight")
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_biomedclip_encoder(model, checkpoint)

    def test_default_encoder_parameter_counts_match_manuscript(self):
        encoder = VolumetricEncoder(ModelConfig())
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 104_878_080)
        self.assertEqual(sum(p.numel() for p in encoder.parameters() if p.requires_grad), 19_860_480)


class ProtocolTests(unittest.TestCase):
    def test_auc_uses_scores_when_every_prediction_is_negative(self):
        metrics = binary_metrics([0, 0, 1, 1], [.1, .2, .3, .4])
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["acc"], .5)

    def test_stopping_counts_non_improving_validation_losses(self):
        stop = EarlyStopping(patience=2)
        self.assertEqual(stop.update(.6), (True, False))
        self.assertEqual(stop.update(.5), (True, False))
        self.assertEqual(stop.update(.51), (False, False))
        self.assertEqual(stop.update(.52), (False, True))
        self.assertEqual(stop.best_loss, .5)

    def test_binary_and_legacy_label_contracts(self):
        self.assertEqual(binary_label(2, "grade_group"), 0)
        self.assertEqual(binary_label(3, "grade_group"), 1)
        self.assertEqual(binary_label([1, 0], "onehot", positive_index=0), 1)
        with self.assertRaises(ValueError):
            binary_label([1, 0], "onehot")
        with self.assertRaises(ValueError):
            binary_label([1, 1], "onehot", 1)

    @staticmethod
    def splits():
        records = [Record(f"case_{i}", Path(f"/synthetic/{i}.h5"), i % 2) for i in range(10)]
        return [(records[:2*i] + records[2*i+2:], records[2*i:2*i+2]) for i in range(5)]

    def test_five_fold_partition_and_balancing(self):
        splits = self.splits()
        self.assertEqual(audit_five_folds(splits), {"patients": 10, "negative": 5, "positive": 5})
        records = splits[0][0][:3]
        sampler = balanced_sampler(records, 42)
        totals = [sum(sampler.weights[i].item() for i, r in enumerate(records) if r.label == k)
                  for k in (0, 1)]
        self.assertAlmostEqual(totals[0], totals[1])
        splits[0] = (splits[0][0] + [splits[0][1][0]], splits[0][1])
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_five_folds(splits)

    def test_hdf5_input_does_not_require_cached_2d_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with h5py.File(root / "case.h5", "w") as h5:
                h5["pet"] = np.ones((1, 16, 12, 12), dtype="float32")
                h5["ct"] = np.zeros((1, 16, 12, 12), dtype="float32")
                h5["label"] = 1
            with (root / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["case_id", "path"])
                writer.writeheader()
                writer.writerow({"case_id": "case", "path": "case.h5"})
            records = read_records(root / "manifest.csv", root)
            sample = H5VolumeDataset(records, (16, 16, 16))[0]
            self.assertEqual(tuple(sample["pet"].shape), (1, 16, 16, 16))
            self.assertEqual(sample["label"].item(), 1)
            self.assertTrue(torch.all(sample["pet"] == 1))
            self.assertTrue(torch.all(sample["ct"] == 0))

    def test_preprocessing_window_and_annotation_bounding_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (12, 12, 12)
            pet = np.full(shape, 5, dtype="float32")
            ct = np.full(shape, 50, dtype="float32")
            mask = np.zeros(shape, dtype="float32")
            # Separate regions: their joint bounding box determines the crop.
            mask[3:5, 4:6, 5:7] = 1
            mask[7, 8, 9] = 1
            for name, array in [("pet", pet), ("ct", ct), ("mask", mask)]:
                nib.save(nib.Nifti1Image(array, np.eye(4)), root / (name + ".nii.gz"))
            prepare_case(root / "pet.nii.gz", root / "ct.nii.gz",
                         root / "mask.nii.gz", root / "prepared.h5", label=1, crop_size=4)
            with h5py.File(root / "prepared.h5") as h5:
                self.assertTrue(np.allclose(h5.attrs["crop_center_xyz"], [5, 6, 7]))
                self.assertEqual(h5["pet"].shape, (1, 4, 4, 4))
                self.assertTrue(np.allclose(h5["pet"][()], 5))
                self.assertTrue(np.allclose(h5["ct"][()], .5))


if __name__ == "__main__":
    unittest.main()
