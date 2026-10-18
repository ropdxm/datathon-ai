"""Temporal order, source ownership and real-gradient protocol on tiny video modules."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from datathon_ml.temporal_video_finetune import (TemporalVideoConfig, VideoOnly,
    TemporalVideoMil, candidate_grid, clip_tensor, configure_trainable_video,
    evaluation_centers, evaluate_video_pack, fit_temporal_video_selection,
    load_finetuned_temporal, score_temporal_video_development,
    temporal_indices, training_centers)
from datathon_ml.finetune_visual import read_frame_pack
from tests.test_finetune_visual import pack_fixture


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(3, 8)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.post_layernorm = nn.LayerNorm(8)

    def forward(self, pixel_values, return_dict=True):
        x = self.patch_embedding(pixel_values.mean(dim=(2, 3)))
        for layer in self.encoder.layers:
            x = x + layer(x).tanh()
        return SimpleNamespace(pooler_output=self.post_layernorm(x))


class TinyMit(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.linspace(.1, 1.6, 16)[None, :, None])

    def forward(self, values, return_dict=True):
        return SimpleNamespace(pooler_output=(values * self.position_embedding).mean(dim=1))


def tiny_video(_base_dir):
    with torch.random.fork_rng():
        torch.manual_seed(481)
        return VideoOnly(TinyVision(), nn.Linear(8, 512, bias=False), TinyMit())


class TemporalVideoTests(unittest.TestCase):
    def test_clip_indices_keep_temporal_order_and_repeat_only_observed_edge_frames(self):
        config = TemporalVideoConfig()
        times = np.arange(24, dtype=float)
        indices = temporal_indices(times, 10.5, config)
        np.testing.assert_array_equal(indices, np.arange(3, 19))
        edge = temporal_indices(times, 0., config)
        self.assertTrue(np.all(np.diff(edge) >= 0))
        self.assertEqual(int(edge[0]), 0)
        self.assertEqual(len(edge), 16)
        np.testing.assert_array_equal(indices, temporal_indices(times + 100., 110.5, config))
        np.testing.assert_array_equal(evaluation_centers(np.array([0., 10.]), config), [0., 4., 8., 10.])

    def test_anchor_rotation_is_training_only_and_never_becomes_negative_supervision(self):
        config = TemporalVideoConfig(clips_per_bag=3)
        times = np.arange(30, dtype=float)
        seen = set()
        rng = np.random.default_rng(42)
        for _ in range(12):
            centers, masks = training_centers(times, {"label": 1, "event_times": [5., 20.]}, config, rng)
            self.assertEqual(len(centers), 3)
            self.assertEqual(int(masks.sum()), 1)
            seen.update(centers[masks])
        self.assertEqual(seen, {5., 20.})
        _, masks = training_centers(times, {"label": 0, "event_times": []}, config, rng)
        self.assertFalse(masks.any())
        self.assertEqual(len(candidate_grid()), 2)

    def test_temporal_backbone_changes_with_frame_order_and_only_final_blocks_train(self):
        backbone = tiny_video(None)
        trainable = configure_trainable_video(backbone, 1)
        pixels = torch.arange(16, dtype=torch.float32)[None, :, None, None, None].expand(1, 16, 3, 4, 4) / 16
        before = backbone(pixels)
        reverse = backbone(pixels.flip(1))
        self.assertFalse(torch.allclose(before, reverse))
        model = TemporalVideoMil(backbone)
        model(pixels).sum().backward()
        self.assertIsNone(backbone.vision_model.patch_embedding.weight.grad)
        self.assertIsNone(backbone.vision_model.encoder.layers[0].weight.grad)
        self.assertIsNotNone(backbone.vision_model.encoder.layers[-1].weight.grad)
        self.assertIsNotNone(backbone.mit.position_embedding.grad)
        self.assertTrue(any(name.startswith('mit.') for name in trainable))
        self.assertFalse(any('text' in name for name, _ in model.named_parameters()))

    def test_clip_augmentation_is_consistent_across_repeated_frames_and_eval_has_no_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            key = 'c' * 64
            pack_fixture(directory, key)
            pack = read_frame_pack(directory, key)
            config = TemporalVideoConfig(device='cpu', clip_batch_size=1)
            values = clip_tensor(pack, 1., 0, config, training=True, rng=np.random.default_rng(4))
            # All fixture frames have the same pixels. A clip-wide augmentation
            # preserves that identity instead of injecting synthetic motion.
            torch.testing.assert_close(values[0], values[-1], rtol=0, atol=0)
            model = TemporalVideoMil(tiny_video(None)).eval()
            before = evaluate_video_pack(model, pack, config)
            pack.update(label=1, event_times=[12345.], video_id='ignored')
            self.assertEqual(before, evaluate_video_pack(model, pack, config))
            self.assertEqual(before['window_timestamps'], [0., 2.])

    def test_two_temporal_fits_export_restore_and_never_read_exposed_partition_for_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(8):
                key, label = f'{index + 2001:064x}', index % 2
                rows.append({'sha256':key,'group_id':key,'group_verified':True,'label':label,
                    'event_times':[1.] if label else [],'event_times_verified':True,
                    'label_status':'human_verified','evaluation_ground_truth_eligible':True})
                if index < 6: pack_fixture(root/'frames',key,bool(label))
            split = {'train':[row['sha256'] for row in rows[:4]],'validation':[row['sha256'] for row in rows[4:6]],
                'test':[row['sha256'] for row in rows[6:]],'protocol':{'type':'synthetic'}}
            config = TemporalVideoConfig(device='cpu',threads=1,clips_per_bag=1,clip_batch_size=1)
            with patch('datathon_ml.temporal_video_finetune.load_video_only',side_effect=tiny_video):
                path = fit_temporal_video_selection(rows,root/'frames',split,root/'not_downloaded',root/'run',config)
                selection = json.loads((root/'run/temporal_video_selection.json').read_text())
                self.assertEqual(selection['candidates'], candidate_grid())
                self.assertFalse(selection['development_partition_opened_for_selection'])
                model, artifact = load_finetuned_temporal(path,root/'not_downloaded',torch.device('cpu'))
                self.assertEqual(set(artifact['training_sha256']),set(split['train']))
                self.assertFalse(set(artifact['frame_pack_metadata_sha256'])&set(split['test']))
                self.assertTrue(any(name.startswith('mit.') for name in artifact['backbone_delta_state']))
                self.assertFalse(any('patch_embedding' in name for name in artifact['backbone_delta_state']))
                for row in rows[6:]: pack_fixture(root/'frames',row['sha256'],bool(row['label']))
                report = score_temporal_video_development(rows,root/'frames',split,root/'not_downloaded',path,root/'run')
                self.assertFalse(report['independent_test'])
                self.assertEqual(report['metrics']['n'],2)


if __name__=='__main__': unittest.main()
