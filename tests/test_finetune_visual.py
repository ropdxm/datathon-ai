"""Supervised visual training mechanics on synthetic pixels, without downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from datathon_ml.finetune_visual import (FinetuneConfig, VisualMil, candidate_grid,
    configure_trainable_visual, evaluate_pack, fit_visual_selection, load_finetuned,
    point_bag_loss, read_frame_pack, score_visual_development, training_indices,
    uniform_indices)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(3, 8)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.randn(8, 512) * .02)

    def forward(self, pixels):
        x = self.conv1(pixels.mean(dim=(2, 3)))
        for block in self.transformer.resblocks:
            x = x + block(x).tanh()
        return self.ln_post(x) @ self.proj


def tiny_visual(_checkpoint):
    with torch.random.fork_rng():
        torch.manual_seed(137)
        return TinyVisual()


def pack_fixture(root, key, positive=False):
    directory = Path(root) / key
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for index in range(3):
        name = f"{index:06d}_full.jpg"
        Image.new("RGB", (224, 224), (200 if positive else 30, 80, 80 if positive else 180)).save(directory / name)
        images.append([name])
    metadata = {"source_sha256": key, "timestamps": [0., 1., 2.], "images": images,
                "views": ["full"], "preprocessing": "raw_rgb_uint8_224_letterbox"}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    members = [directory / "metadata.json", *[directory / names[0] for names in images]]
    receipt = {"completed": True, "members_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in members}}
    (directory / "_integrity.json").write_text(json.dumps(receipt))
    return metadata


class FinetuneVisualTests(unittest.TestCase):
    def test_frame_packs_require_source_integrity_increasing_times_and_contained_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            key = "a" * 64
            metadata = pack_fixture(directory, key)
            pack = read_frame_pack(directory, key)
            self.assertEqual(pack["views"], ["full"])
            Image.new("RGB", (224, 224), (1, 2, 3)).save(Path(directory) / key / metadata["images"][0][0])
            with self.assertRaisesRegex(ValueError, "integrity receipt"):
                read_frame_pack(directory, key)
            metadata = pack_fixture(directory, key)
            metadata["images"][0][0] = "../outside.jpg"
            (Path(directory) / key / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_frame_pack(directory, key)
            metadata = pack_fixture(directory, key)
            metadata["timestamps"] = [0., 1., 1.]
            (Path(directory) / key / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "increasing"):
                read_frame_pack(directory, key)

    def test_only_selected_final_visual_blocks_receive_gradients(self):
        visual = tiny_visual(None)
        trainable = configure_trainable_visual(visual, 1)
        model = VisualMil(visual)
        model(torch.randn(2, 3, 16, 16)).sum().backward()
        self.assertTrue(any(name.startswith("transformer.resblocks.2") for name in trainable))
        self.assertFalse(any(name.startswith("transformer.resblocks.0") for name in trainable))
        self.assertIsNone(visual.conv1.weight.grad)
        self.assertIsNone(visual.transformer.resblocks[0].weight.grad)
        self.assertIsNotNone(visual.transformer.resblocks[2].weight.grad)
        self.assertIsNotNone(visual.proj.grad)
        self.assertIsNotNone(model.head[0].weight.grad)
        with self.assertRaisesRegex(ValueError, "completely frozen"):
            configure_trainable_visual(visual, 3)

    def test_training_anchor_sampling_and_loss_never_make_unmarked_positives_negative(self):
        config = FinetuneConfig(train_frames_per_bag=4)
        times = np.arange(20, dtype=float)
        indices, mask = training_indices(times, {"label": 1, "event_times": [7.]}, config, np.random.default_rng(3))
        self.assertIn(7, indices)
        self.assertTrue(mask[list(indices).index(7)])
        self.assertEqual(len(indices), 4)
        positive = torch.tensor([3., 2., 1., 0.], requires_grad=True)
        losses = point_bag_loss([positive], [1], [torch.tensor([False, False, True, False])], top_k=1)
        losses["total"].backward()
        self.assertEqual(float(losses["normal"].detach()), 0.)
        self.assertEqual(float(positive.grad[1]), 0.)
        self.assertEqual(float(positive.grad[3]), 0.)
        negative = torch.tensor([3., 2., 1., -1.], requires_grad=True)
        point_bag_loss([negative], [0], [torch.zeros(4, dtype=torch.bool)], top_k=1)["total"].backward()
        self.assertTrue(torch.all(negative.grad > 0))

    def test_uniform_evaluation_does_not_consult_labels_or_point_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            key = "b" * 64
            pack_fixture(directory, key)
            pack = read_frame_pack(directory, key)
            model = VisualMil(tiny_visual(None)).eval()
            config = FinetuneConfig(device="cpu", evaluation_frames_per_bag=2, image_batch_size=2)
            before = evaluate_pack(model, pack, config)
            pack.update(label=1, event_times=[10000.], video_id="never_model_input")
            after = evaluate_pack(model, pack, config)
            self.assertEqual(before, after)
            self.assertEqual(before["window_timestamps"], [0., 2.])
            dense = evaluate_pack(model, pack, FinetuneConfig(device="cpu", image_batch_size=2))
            self.assertEqual(dense["window_timestamps"], [0., 1., 2.])
            np.testing.assert_array_equal(uniform_indices(100, 3), [0, 49, 99])

    def test_two_fixed_visual_fits_export_delta_and_leave_development_packs_unopened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(16):
                key, label = f"{index + 1001:064x}", index % 2
                records.append({"sha256": key, "group_id": key, "group_verified": True, "label": label,
                    "event_times": [1.] if label else [], "label_status": "human_verified",
                    "event_times_verified": True, "evaluation_ground_truth_eligible": True})
                if index < 12:
                    pack_fixture(root / "frames", key, bool(label))
            split = {"train": [row["sha256"] for row in records[:8]],
                     "validation": [row["sha256"] for row in records[8:12]],
                     "test": [row["sha256"] for row in records[12:]], "protocol": {"type": "synthetic"}}
            config = FinetuneConfig(device="cpu", threads=1, train_frames_per_bag=2,
                evaluation_frames_per_bag=2, image_batch_size=4, bags_per_batch=4)
            with patch("datathon_ml.finetune_visual.load_visual", side_effect=tiny_visual):
                path = fit_visual_selection(records, root / "frames", split, root / "not_downloaded.pt", root / "run", config)
                selection = json.loads((root / "run" / "visual_selection.json").read_text())
                self.assertEqual(selection["candidates"], candidate_grid())
                self.assertFalse(selection["development_partition_opened_for_selection"])
                self.assertEqual(len(selection["leaderboard"]), 2)
                model, artifact = load_finetuned(path, root / "not_downloaded.pt", torch.device("cpu"))
                self.assertFalse(any(name.startswith("conv1") for name in artifact["visual_delta_state"]))
                self.assertEqual(artifact["training_sha256"], split["train"])
                self.assertFalse(set(artifact["frame_pack_metadata_sha256"]) & set(split["test"]))
                for row in records[12:]:
                    pack_fixture(root / "frames", row["sha256"], bool(row["label"]))
                report = score_visual_development(records, root / "frames", split, root / "not_downloaded.pt", path, root / "run")
                self.assertFalse(report["independent_test"])
                self.assertEqual(report["metrics"]["n"], 4)
                self.assertIn("previously_exposed", report["evaluation"])


if __name__ == "__main__":
    unittest.main()
