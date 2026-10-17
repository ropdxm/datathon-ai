"""Fixed-prompt projection, view alignment and derived-cache integrity."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datathon_ml.clip_features import BACKBONE as IMAGE_BACKBONE, EXPECTED_WEIGHT_SHA256, HUB_REPO
from datathon_ml.features import _canonical_json, _configuration_hash, load_feature_cache
from datathon_ml.semantic_clip import (
    CONCEPT_NAMES, TEMPERATURE, convert_clip_cache, prompt_protocol, semantic_scores,
)


def prototypes():
    positive = np.eye(512, dtype=np.float32)[:18]
    return positive, -positive


class SemanticClipTests(unittest.TestCase):
    def test_schema_and_fixed_prompt_protocol(self):
        self.assertEqual(len(CONCEPT_NAMES), 18)
        self.assertEqual(CONCEPT_NAMES[0], "cashier_present")
        self.assertEqual(CONCEPT_NAMES[-1], "bag_or_seal_handling")
        protocol = prompt_protocol()
        self.assertEqual(tuple(protocol["prompts"]), CONCEPT_NAMES)
        self.assertEqual(protocol["temperature"], .07)
        for positive, negative in protocol["prompts"].values():
            self.assertEqual(len(positive), 2)
            self.assertEqual(len(negative), 2)

    def test_cosine_contrast_and_view_names_determine_projection(self):
        positive, negative = prototypes()
        features = np.zeros((2, 4, 512), dtype=np.float32)
        features[:, :, 19] = np.sqrt(1. - .05**2)
        features[:, 0, 0] = .05  # full
        features[:, 2, 0] = -.05  # middle
        expected_margin = .7 * .1 + .3 * -.1
        expected = 1. / (1. + np.exp(-expected_margin / TEMPERATURE))
        scores = semantic_scores(features, ("full", "top", "middle", "bottom"), positive, negative)
        self.assertEqual(scores.shape, (2, 1, 18))
        np.testing.assert_allclose(scores[:, 0, 0], expected, atol=1e-6)
        np.testing.assert_allclose(scores[:, 0, 1:], .5, atol=1e-6)
        permutation = [3, 2, 1, 0]
        reordered = semantic_scores(features[:, permutation], ("bottom", "middle", "top", "full"), positive, negative)
        np.testing.assert_allclose(scores, reordered, atol=1e-7)

    def test_invalid_embeddings_are_rejected(self):
        positive, negative = prototypes()
        with self.assertRaisesRegex(ValueError, "Zero"):
            semantic_scores(np.zeros((1, 1, 512)), ("full",), positive, negative)
        with self.assertRaisesRegex(ValueError, "512"):
            semantic_scores(np.ones((1, 1, 768)), ("full",), positive, negative)

    def test_conversion_keeps_times_and_rebuilds_if_source_embedding_bytes_change(self):
        positive, negative = prototypes()
        key = "a" * 64
        protocol_hash = _configuration_hash(_canonical_json(prompt_protocol()))
        text_metadata = {"protocol_sha256": protocol_hash, "prototypes_sha256": "b" * 64,
                         "weights_sha256": EXPECTED_WEIGHT_SHA256, "text_cache_reused": True}
        metadata = {"schema_version": 1, "backbone": IMAGE_BACKBONE, "sample_fps": 2.,
            "image_size": 224, "views": ["full", "middle"], "preprocessing": "test-original",
            "decoder": "test-decoder", "weights_sha256": EXPECTED_WEIGHT_SHA256, "hub_repo": HUB_REPO}
        config_json = _canonical_json(metadata)
        features = np.zeros((3, 2, 512), dtype=np.float32)
        features[:, :, 20] = 1.
        with tempfile.TemporaryDirectory() as directory:
            source_dir, destination = Path(directory) / "original", Path(directory) / "semantic"
            source_dir.mkdir()
            source = source_dir / f"{key}.npz"
            def save():
                np.savez_compressed(source, features=features, timestamps=np.array([0., .5, 1.], np.float32),
                    sha256=np.array(key), config_json=np.array(config_json),
                    config_hash=np.array(_configuration_hash(config_json)), provenance_json=np.array("{}"))
            save()
            with patch("datathon_ml.semantic_clip._get_text_prototypes", return_value=(positive, negative, text_metadata)):
                paths = convert_clip_cache(source_dir, destination, device="cpu", offline=True)
                cached = load_feature_cache(paths[key])
                self.assertEqual(cached["features"].shape, (3, 1, 18))
                self.assertEqual(cached["config"]["concept_names"], list(CONCEPT_NAMES))
                np.testing.assert_array_equal(cached["timestamps"], [0., .5, 1.])
                with patch("datathon_ml.semantic_clip.semantic_scores", side_effect=AssertionError("cache should be reused")):
                    self.assertEqual(convert_clip_cache(source_dir, destination, offline=True), paths)
                features[:, :, 0] = 1.
                save()
                convert_clip_cache(source_dir, destination, offline=True)
                updated = load_feature_cache(paths[key])
                self.assertFalse(np.array_equal(updated["features"], cached["features"]))
                self.assertFalse(updated["provenance"]["external_inference_api_used"])


if __name__ == "__main__":
    unittest.main()
