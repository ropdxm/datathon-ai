"""Prepare silent, smaller video derivatives without changing source recordings.

Only FFmpeg processes pixels. No audio is decoded, transcribed, or retained in
derivatives. Inventories contain media metadata and checksums, never frames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".m4v"}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Settings:
    long_edge: int = 1280
    max_fps: int = 15
    crf: int = 23
    preset: str = "fast"
    threads: int = 4

    def validate(self):
        if self.long_edge < 2 or self.max_fps < 1 or not 0 <= self.crf <= 51 or self.threads < 1:
            raise ValueError("Invalid size, frame rate, CRF, or thread limit")
        if self.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
            raise ValueError("Unsupported encoder preset")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path, ffprobe="ffprobe"):
    result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def video_info(metadata):
    videos = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if len(videos) != 1:
        raise ValueError("Exactly one real video stream is required")
    video = videos[0]
    fps = Fraction(video.get("avg_frame_rate", "0/1"))
    if fps <= 0:
        raise ValueError("Missing or invalid average frame rate")
    duration = float(video.get("duration", metadata.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Missing or invalid video duration")
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-5:
        raise ValueError("Non-right-angle rotation needs explicit review")
    width, height = int(video["width"]), int(video["height"])
    if round(rotation / 90) % 2:
        width, height = height, width
    sar = video.get("sample_aspect_ratio", "1:1")
    if sar not in {"1:1", "N/A", "0:1"}:
        raise ValueError("Non-square source pixels need explicit aspect-ratio handling")
    return {"stream_index": video["index"], "width": width, "height": height,
            "coded_width": int(video["width"]), "coded_height": int(video["height"]),
            "rotation": rotation, "fps": str(fps), "duration_seconds": duration,
            "start_time_seconds": float(video.get("start_time", 0)), "time_base": video.get("time_base"),
            "codec": video.get("codec_name"), "pixel_format": video.get("pix_fmt"),
            "color_transfer": video.get("color_transfer"), "color_primaries": video.get("color_primaries"),
            "audio_streams": sum(s.get("codec_type") == "audio" for s in metadata.get("streams", [])),
            "stream_types": [s.get("codec_type") for s in metadata.get("streams", [])]}


def output_geometry(info, settings):
    scale = min(1.0, settings.long_edge / max(info["width"], info["height"]))
    result = tuple(2 * math.floor(info[key] * scale / 2) for key in ("width", "height"))
    if min(result) < 2:
        raise ValueError("Source dimensions are too small")
    return result


def encode_command(source, output, info, settings, ffmpeg="ffmpeg"):
    settings.validate()
    if info.get("color_transfer") in {"smpte2084", "arib-std-b67"} or info.get("color_primaries") == "bt2020":
        raise ValueError("HDR/wide-gamut video requires an explicit tested tone-map policy")
    width, height = output_geometry(info, settings)
    filters = ["setpts=PTS-STARTPTS"]
    # Preserve a low-rate source's original frames. Never raise its nominal FPS.
    if Fraction(info["fps"]) > settings.max_fps:
        filters.append(f"fps={settings.max_fps}:round=near")
    filters += [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source),
            "-map", f"0:{info['stream_index']}", "-an", "-sn", "-dn", "-map_metadata", "-1",
            # -vsync also supports FFmpeg 4.x shipped by some Colab runtimes.
            "-map_chapters", "-1", "-vf", ",".join(filters), "-vsync", "vfr",
            "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
            "-threads", str(settings.threads), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-metadata:s:v:0", "rotate=0", "-f", "mp4", str(output)]


def verify_output(source_info, output_metadata, settings):
    result = video_info(output_metadata)
    if result["stream_types"] != ["video"] or result["audio_streams"]:
        raise ValueError("Prepared file must contain exactly one video stream and no audio/data/subtitles")
    if result["codec"] != "h264" or result["pixel_format"] != "yuv420p":
        raise ValueError("Unexpected output codec/pixel format")
    if (result["width"], result["height"]) != output_geometry(source_info, settings) or result["rotation"]:
        raise ValueError("Output orientation or geometry differs from plan")
    fps = float(Fraction(result["fps"]))
    source_fps = float(Fraction(source_info["fps"]))
    if fps > min(settings.max_fps, source_fps) + .01:
        raise ValueError("Output FPS exceeds source or requested cap")
    tolerance = max(.15, 2 / fps, 2 / source_fps)
    duration_error = result["duration_seconds"] - source_info["duration_seconds"]
    if abs(duration_error) > tolerance or abs(result["start_time_seconds"]) > 1 / fps + .001:
        raise ValueError("Transcoding changed the video timeline beyond tolerance")
    if not result["time_base"] or Fraction(result["time_base"]) <= 0:
        raise ValueError("Invalid output time base")
    private_keys = {"creation_time", "location", "location-eng", "com.apple.quicktime.location.ISO6709",
                    "comment", "title", "description", "artist", "author"}
    tags = [output_metadata.get("format", {}).get("tags", {})]
    tags += [s.get("tags", {}) for s in output_metadata.get("streams", [])]
    if any(private_keys.intersection(t) for t in tags):
        raise ValueError("Source metadata was retained unexpectedly")
    return {"duration_error_seconds": duration_error, "duration_tolerance_seconds": tolerance,
            "annotation_time_mapping": "output_elapsed_seconds = source_video_elapsed_seconds",
            "source_video_start_time_seconds": source_info["start_time_seconds"], "verified": True}


def resolve_directories(source_dir, output_dir):
    source, output = Path(source_dir).resolve(strict=True), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate, with neither inside the other")
    return source, output


def inventory(source_dir, ffprobe="ffprobe"):
    source_dir = Path(source_dir).resolve(strict=True)
    records = []
    for path in sorted(source_dir.rglob("*"), key=lambda p: p.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.is_symlink() or source_dir not in path.resolve().parents:
            raise ValueError("Linked video paths are not accepted")
        metadata = probe(path, ffprobe)
        records.append({"source_relative_path": path.relative_to(source_dir).as_posix(),
                        "source_sha256": sha256(path), "source_bytes": path.stat().st_size,
                        "source": video_info(metadata), "source_ffprobe": metadata})
    if not records:
        raise ValueError("No supported source videos found")
    return records


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate(records):
    source_bytes = sum(r["source_bytes"] for r in records)
    prepared = [r for r in records if r.get("output_sha256")]
    output_bytes = sum(r["output_bytes"] for r in prepared)
    return {"videos": len(records), "unique_source_sha256": len({r["source_sha256"] for r in records}),
            "source_bytes": source_bytes, "source_seconds": sum(r["source"]["duration_seconds"] for r in records),
            "source_with_audio": sum(bool(r["source"]["audio_streams"]) for r in records),
            "source_resolutions": dict(Counter(f"{r['source']['width']}x{r['source']['height']}" for r in records)),
            "prepared_videos": len(prepared), "output_bytes": output_bytes,
            "output_with_audio": sum(bool(r["output"]["audio_streams"]) for r in prepared),
            "output_seconds": sum(r["output"]["duration_seconds"] for r in prepared),
            "output_resolutions": dict(Counter(f"{r['output']['width']}x{r['output']['height']}" for r in prepared)),
            "storage_reduction_percent": 100 * (1 - output_bytes / source_bytes) if len(prepared) == len(records) else None,
            "max_duration_error_seconds": max((abs(r["verification"]["duration_error_seconds"]) for r in prepared), default=None)}


def write_reports(output_dir, report):
    atomic_json(output_dir / "preparation_manifest.json", report)
    # This CSV is lineage, never an annotation or a training feature table.
    fields = ["source_relative_path", "source_sha256", "source_bytes", "output_relative_path", "output_sha256", "output_bytes"]
    with (output_dir / "preparation_lineage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["videos"])


def prepare(source_dir, output_dir, settings=Settings(), ffmpeg="ffmpeg", ffprobe="ffprobe", inventory_only=False):
    settings.validate()
    source_dir, output_dir = resolve_directories(source_dir, output_dir)
    records = inventory(source_dir, ffprobe)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preparation_manifest.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    # Keep authentication records across interrupted resume runs. A partially
    # refreshed inventory must not erase checksums for not-yet-visited outputs.
    prior = previous.get("resume_outputs") or {r["source_relative_path"]: r for r in previous.get("videos", [])}
    known_outputs = dict(prior) if previous.get("settings") == asdict(settings) else {}
    report = {"schema_version": SCHEMA_VERSION, "source_dir": str(source_dir), "output_dir": str(output_dir),
              "settings": asdict(settings), "audio_policy": "video stream only; no audio decoding or inference",
              "originals_preserved": True, "labels_modified": False,
              "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
              "videos": records, "resume_outputs": known_outputs, "complete": False}
    completed_by_sha = {}
    start = time.perf_counter()
    for record in records:
        if inventory_only:
            continue
        source = source_dir / record["source_relative_path"]
        name = f"{source.stem}__{source.suffix[1:].lower()}__{record['source_sha256'][:12]}.mp4"
        output = output_dir / name
        old = prior.get(record["source_relative_path"], {})
        if output.exists():
            if (previous.get("settings") != asdict(settings) or old.get("source_sha256") != record["source_sha256"]
                    or old.get("output_relative_path") != name or old.get("output_sha256") != sha256(output)):
                raise FileExistsError(f"Refusing to overwrite unverified existing derivative: {output}")
            status = "reused_verified_existing"
        else:
            temporary = output.with_name(output.stem + ".partial-" + uuid.uuid4().hex + ".mp4")
            command = encode_command(source, temporary, record["source"], settings, ffmpeg)
            try:
                if record["source_sha256"] in completed_by_sha:
                    shutil.copyfile(completed_by_sha[record["source_sha256"]], temporary)
                    status = "reused_exact_source_duplicate"
                else:
                    subprocess.run(command, capture_output=True, text=True, check=True)
                    status = "encoded"
                verify_output(record["source"], probe(temporary, ffprobe), settings)
                if sha256(source) != record["source_sha256"]:
                    raise RuntimeError("Source changed during preparation")
                # Atomic no-clobber publication: even a concurrent run cannot replace an existing file.
                os.link(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
        metadata = probe(output, ffprobe)
        record.update({"output_relative_path": name, "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
                       "output": video_info(metadata), "output_ffprobe": metadata,
                       "verification": verify_output(record["source"], metadata, settings), "status": status})
        completed_by_sha[record["source_sha256"]] = output
        known_outputs[record["source_relative_path"]] = dict(record)
        report["summary"] = aggregate(records)
        write_reports(output_dir, report)
        print(json.dumps({"prepared": len(completed_by_sha), "file": source.name, "status": status}), flush=True)
    report["summary"] = aggregate(records)
    report["elapsed_seconds"] = time.perf_counter() - start
    report["complete"] = not inventory_only and all(r.get("output_sha256") for r in records)
    # Inventory never replaces a completed manifest used to authenticate resumable derivatives.
    if inventory_only:
        atomic_json(output_dir / "source_inventory.json", report)
    else:
        write_reports(output_dir, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "prepare"))
    parser.add_argument("--source-dir", type=Path, default=Path("open/customVideos"))
    parser.add_argument("--output-dir", type=Path, default=Path("open/customVideos_prepared"))
    parser.add_argument("--long-edge", type=int, default=1280)
    parser.add_argument("--max-fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    report = prepare(args.source_dir, args.output_dir, Settings(args.long_edge, args.max_fps, args.crf, args.preset, args.threads),
                     args.ffmpeg, args.ffprobe, args.command == "inventory")
    print(json.dumps({**report["summary"], "complete": report["complete"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
