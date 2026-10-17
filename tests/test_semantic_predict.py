"""Fresh-video dispatch/configuration without labels, weights or real footage."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib

import semantic_predict_experiment as entry
from datathon_ml import semantic_heads, temporal
from datathon_ml.clip_features import BACKBONE as IMAGE_BACKBONE, EXPECTED_WEIGHT_SHA256
from datathon_ml.features import _canonical_json, _configuration_hash
from datathon_ml.model import semantic_feature_config
from datathon_ml.semantic_clip import BACKBONE, TEMPERATURE, prompt_protocol


def artifact(kind="semantic_temporal"):
    source = {"schema_version": 1, "backbone": IMAGE_BACKBONE, "sample_fps": 1.25,
        "image_size": 224, "views": ["full", "middle"], "preprocessing": "fixed-test-source",
        "decoder": "fixed-test-decoder", "weights_sha256": EXPECTED_WEIGHT_SHA256, "hub_repo": "pinned-clip"}
    protocol = _configuration_hash(_canonical_json(prompt_protocol()))
    metadata = {"backbone": BACKBONE, "source_feature_config": source, "sample_fps": 1.25,
        "image_size": 224, "views": ["semantic"], "temperature": TEMPERATURE,
        "prompts_sha256": protocol, "weights_sha256": EXPECTED_WEIGHT_SHA256,
        "preprocessing": f"semantic-clip18-fixed-v1;protocol={protocol};source={_configuration_hash(_canonical_json(source))}",
        "continuous_window_coverage": False}
    temporal_sha = hashlib.sha256(Path(temporal.__file__).read_bytes()).hexdigest()
    code_sha = temporal_sha if kind == "semantic_temporal" else hashlib.sha256(Path(semantic_heads.__file__).read_bytes()).hexdigest()
    return {"kind": kind, "feature_config": metadata, "feature_semantic_config": semantic_feature_config(metadata),
        "continuous_window_coverage": False, "coverage_certification": None,
        "provenance": {"code_sha256": code_sha, "temporal_code_sha256": temporal_sha}}


class SemanticPredictTests(unittest.TestCase):
    def test_fresh_video_pipeline_restores_source_and_is_offline_without_annotations(self):
        for kind in ("semantic_temporal", "semantic_head"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video, model = root / "unlabeled.mov", root / "model.joblib"
                video.write_bytes(b"synthetic-unlabeled-video")
                joblib.dump(artifact(kind), model)
                digest = hashlib.sha256(video.read_bytes()).hexdigest()
                expected = {"verdict": "no_violation", "score": .2, "prediction": 0, "events": []}
                predictor = "datathon_ml.temporal.predict_temporal_features" if kind == "semantic_temporal" else "datathon_ml.semantic_heads.predict_semantic_head_features"
                stdout = io.StringIO()
                with patch("semantic_predict_experiment.extract_clip_features") as extract, \
                     patch("semantic_predict_experiment.convert_clip_cache", return_value={digest: root / "states.npz"}) as convert, \
                     patch(predictor, return_value=expected.copy()) as predict, redirect_stdout(stdout):
                    result = entry.main(["--video", str(video), "--model", str(model), "--output-dir", str(root / "out"), "--batch-size", "7"])
                records, _, config = extract.call_args.args
                self.assertEqual(records, [{"path": str(video.resolve()), "sha256": digest}])
                self.assertEqual((config.sample_fps, config.views, config.batch_size), (1.25, ("full", "middle"), 7))
                self.assertFalse(config.allow_download)
                self.assertTrue(convert.call_args.kwargs["offline"])
                predict.assert_called_once_with(root / "states.npz", model.resolve())
                self.assertTrue(result["offline_requested"])
                self.assertTrue((root / "out" / "prediction.json").is_file())
                self.assertEqual(set(json.loads(stdout.getvalue())), {"verdict", "score", "event_count", "elapsed_seconds"})

    def test_bad_backbone_protocol_and_code_reject_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, model = root / "unlabeled.mov", root / "model.joblib"
            video.write_bytes(b"synthetic")
            for change in ("backbone", "prompt", "code"):
                saved = artifact()
                if change == "backbone":
                    saved["feature_config"]["backbone"] = "unsupported_qwen"
                elif change == "prompt":
                    saved["feature_config"]["prompts_sha256"] = "wrong-protocol"
                else:
                    saved["provenance"]["code_sha256"] = "0" * 64
                joblib.dump(saved, model)
                with patch("semantic_predict_experiment.extract_clip_features") as extract:
                    with self.assertRaises(ValueError):
                        entry.main(["--video", str(video), "--model", str(model), "--output-dir", str(root / "out")])
                    extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
