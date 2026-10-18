"""Native dense/context Qwen experiment; download public weights separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.direct_context_vlm import ContextVLMConfig, fit_context, probe_context, score_context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "fit", "score"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--detail-frame-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--originals-from", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/context_vlm_v1"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-resource-fallback", action="store_true",
                        help="probe only: try24 query frames if synthetic32-frame CUDA probe runs out of memory")
    args = parser.parse_args(argv)
    config = ContextVLMConfig(device=args.device, threads=args.threads)
    if args.command == "probe":
        return probe_context(args.checkpoint, args.output_dir, config,
                             allow_resource_fallback=args.allow_resource_fallback)
    if args.allow_resource_fallback:
        raise ValueError("Resource fallback is allowed only during the synthetic probe before private fitting")
    if args.manifest is None or args.split is None or args.detail_frame_root is None:
        raise ValueError("fit/score need manifest, split and the private native detail frame root")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if args.command == "fit":
        return fit_context(records, args.detail_frame_root, split, args.checkpoint, args.output_dir,
                           config, originals_from=args.originals_from)
    return score_context(records, args.detail_frame_root, split, args.checkpoint, args.output_dir,
                         originals_from=args.originals_from)


if __name__ == "__main__":
    main()
