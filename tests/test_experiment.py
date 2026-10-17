"""CLI-to-extractor protocol integration without weights, footage or downloads."""
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import experiment


class ExperimentProtocolTests(unittest.TestCase):
    def test_qwen_predict_restores_protocol_then_applies_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, artifact = root / "synthetic.mov", root / "model.joblib"
            video.write_bytes(b"synthetic-not-footage")
            artifact.write_bytes(b"mocked-model")
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            trained = {"feature_config": {"backbone": "qwen3_vl_embedding_2b", "sample_fps": .25,
                "image_size": 448, "views": ["full"], "context_config": {
                    "clip_seconds": 6., "frames_per_window": 5, "instruction": "Fixed embedding instruction.",
                    "quantize_4bit": False}}}
            argv = ["predict", "--video", str(video), "--model", str(artifact), "--output-dir", str(root / "output")]
            numeric_prediction = {"score": .4, "prediction": 0, "verdict": "no_violation", "events": []}
            with patch("joblib.load", return_value=trained), \
                 patch("datathon_ml.local_vlm.extract_features", return_value={digest: root / "cache.npz"}) as extract, \
                 patch("datathon_ml.model.predict_features", return_value=numeric_prediction), redirect_stdout(io.StringIO()):
                experiment.main(argv)
                restored = extract.call_args.args[2]
                self.assertEqual((restored.clip_seconds, restored.frames_per_window), (6., 5))
                self.assertEqual(restored.instruction, "Fixed embedding instruction.")
                experiment.main(argv + ["--clip-seconds", "4", "--frames-per-window", "8", "--sample-fps", ".5"])
                overridden = extract.call_args.args[2]
                self.assertEqual((overridden.clip_seconds, overridden.frames_per_window, overridden.sample_fps), (4., 8, .5))
                self.assertEqual(overridden.instruction, restored.instruction)

    def test_qwen_context_flags_reject_other_encoders_and_invalid_values(self):
        for flags in (["--backbone", "dinov2_vits14", "--clip-seconds", "4"],
                      ["--backbone", "qwen3_vl_embedding_2b", "--clip-seconds", "nan"],
                      ["--backbone", "qwen3_vl_embedding_2b", "--clip-seconds", "0"],
                      ["--backbone", "qwen3_vl_embedding_2b", "--frames-per-window", "1"]):
            with self.assertRaises(ValueError):
                experiment.main(["extract", *flags])


if __name__ == "__main__":
    unittest.main()
