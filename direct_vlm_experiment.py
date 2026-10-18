"""Native direct Qwen video binary scoring; public setup is separate/offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from datathon_ml.direct_vlm import DirectVLMConfig, LocalDirectVLM, fit_direct, score_direct


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "fit", "score"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--originals-from", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/direct_vlm_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    config = DirectVLMConfig(device=args.device, threads=args.threads)
    if args.command == "probe":
        started = time.perf_counter()
        engine = LocalDirectVLM(args.checkpoint, config)
        bundle = {"source_sha256": "0" * 64, "frames": np.zeros((16, 224, 224, 3), dtype=np.uint8),
                  "timestamps": list(map(float, range(16)))}
        probabilities = {"zero_shot": engine.score(bundle, [])}
        demo_bundle = {"source_sha256": "1" * 64, "frames": np.zeros((4, 224, 224, 3), dtype=np.uint8),
                       "timestamps": list(map(float, range(4)))}
        probabilities["two_shot_train_owned"] = engine.score(bundle, [
            {"sha256": "1" * 64, "label": 0, "bundle": demo_bundle},
            {"sha256": "2" * 64, "label": 1, "bundle": demo_bundle}])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"synthetic_probe": True, "completed": True, "conditional_A_probabilities": probabilities,
                  "elapsed_seconds": time.perf_counter() - started, "model_provenance": engine.provenance}
        (args.output_dir / "probe.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"synthetic_probe": True, "completed": True,
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        return report
    if args.manifest is None or args.split is None or args.frame_root is None:
        raise ValueError("fit/score need manifest, split and the private native frame root")
    records, split = json.loads(args.manifest.read_text()), json.loads(args.split.read_text())
    if args.command == "fit":
        return fit_direct(records, args.frame_root, split, args.checkpoint, args.output_dir,
                          config, originals_from=args.originals_from)
    return score_direct(records, args.frame_root, split, args.checkpoint, args.output_dir,
                        originals_from=args.originals_from)


if __name__ == "__main__":
    main()
