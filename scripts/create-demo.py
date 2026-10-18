"""Prepare a private browser demo from a supplied video and its source annotation.

This command never invokes a model. Importing it never generates media.
Example: python scripts/create-demo.py --video /private/source/Х-26.mov --timecode 9
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]


def _default_output() -> Path:
    return Path(os.environ.get("FORTE_DATA_DIR", ".forte-runtime")) / "demo"


def _destination(output: Path, video: Path) -> Path:
    output = output.expanduser().resolve()
    public = ROOT / "webapp/static"
    if output.is_relative_to(public) or public.is_relative_to(output):
        raise ValueError("Демо должно храниться в приватном каталоге, вне публичного webapp/static.")
    if video.is_relative_to(output):
        raise ValueError("Каталог вывода не должен содержать исходное видео.")
    if output.exists() and (not output.is_dir() or any(path.name not in {"video.mp4", "result.json"} for path in output.iterdir())):
        raise ValueError("Каталог вывода должен быть пустым либо содержать только video.mp4 и result.json предыдущего демо.")
    return output


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    homebrew = Path("/opt/homebrew/bin/ffmpeg")
    if homebrew.is_file():
        return str(homebrew)
    raise RuntimeError("Для подготовки браузерного MP4 нужен ffmpeg с кодеком libx264. Установите ffmpeg и повторите команду.")


def _publish(stage: Path, output: Path) -> None:
    """Swap complete directories; never expose a video paired with stale JSON."""
    previous = output.with_name(f".{output.name}-previous-{uuid.uuid4().hex}")
    existed = output.exists()
    if existed:
        output.replace(previous)
    try:
        stage.replace(output)
    except BaseException:
        if existed:
            previous.replace(output)
        raise
    if existed:
        shutil.rmtree(previous)


def _timestamp(value, field: str) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{field}: требуется конечное число секунд.")
    return float(value)


def _optional_text(value: str | None, field: str, limit: int) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > limit):
        raise ValueError(f"{field}: требуется непустая строка до {limit} символов.")
    return value.strip() if value is not None else None


def prepare_demo(video: Path, *, timecode: float = 9.0, label: str = "violation",
                 output_dir: Path | None = None, title: str | None = None,
                 description: str | None = None, evidence_times: list[float] | None = None) -> Path:
    """Transcode the entire source and attach source-PTS JPEGs to its given label.

    The only accepted label values are the API verdicts. No specific violation
    type or normative source is inferred. Evidence comes from the original,
    before the display copy is normalized to 30 fps for browser playback.
    Optional text and context-frame times are supplied annotation, not inference.
    """
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Исходное видео не найдено: {video}")
    if label not in {"violation", "no_violation"}:
        raise ValueError("Разметка должна быть violation или no_violation.")
    timecode = _timestamp(timecode, "Время исходной разметки")
    title = _optional_text(title, "Заголовок", 200)
    description = _optional_text(description, "Описание", 4000)
    if evidence_times is not None:
        if not isinstance(evidence_times, (list, tuple)) or not 1 <= len(evidence_times) <= 3:
            raise ValueError("Укажите от 1 до 3 временных отметок кадров.")
        evidence_times = [_timestamp(value, "Время кадра") for value in evidence_times]
    output = _destination(Path(output_dir) if output_dir is not None else _default_output(), video)
    # Reuse the service's bounded, original-PTS decoder; no inference is called.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from webapp.analysis import _attach_evidence, _video_duration

    source_duration = _video_duration(video)
    if not 0 <= timecode <= source_duration:
        raise ValueError(f"Время разметки {timecode:g} с выходит за длительность исходного видео ({source_duration:g} с).")
    if evidence_times is not None and any(not 0 <= value <= source_duration for value in evidence_times):
        raise ValueError("Время кадра выходит за длительность исходного видео.")
    executable = _ffmpeg()
    result = {
        "job_id": "dataset-demo", "status": "done", "demo": True,
        "demo_source": "provided_video", "filename": video.name,
        "video_url": "/api/ui/demo/video", "verdict": label,
        "duration_sec": source_duration, "elapsed_sec": 0,
        "model_name": "Исходная разметка",
        "limitations": [
            "Вердикт перенесён из предоставленной исходной разметки; модель его не определяла.",
            "Это демонстрация предоставленного видео и разметки, а не результат ML-анализа.",
            "Конкретный тип нарушения и нормативное основание в исходной разметке не указаны.",
        ],
        "violations": [],
    }
    if label == "violation":
        result["limitations"].append(f"Время фрагмента взято из исходной разметки: {timecode:g} с.")
        supplied_description = description or (f"Фрагмент около {timecode:g} с отмечен в исходной разметке как возможное нарушение. "
                                               "Конкретный тип события не указан.")
        result["violations"] = [{
            "type": title or "Фрагмент, отмеченный в исходной разметке",
            "timecode_sec": float(timecode),
            "description": supplied_description + " Нормативное основание не определялось. "
                "Это показ предоставленной разметки, а не результат ML-анализа.",
            "rule": None,
        }]
        if evidence_times is None:
            _attach_evidence(video, result)
        else:
            context = {"duration_sec": source_duration,
                       "violations": [{"timecode_sec": value} for value in evidence_times]}
            _attach_evidence(video, context)
            selected = [min(event["evidence_frames"], key=lambda frame: abs(frame["timestamp_sec"] - requested))
                        for event, requested in zip(context["violations"], evidence_times)]
            unique = {frame["timestamp_sec"]: frame for frame in selected}
            result["violations"][0]["evidence_frames"] = [unique[value] for value in sorted(unique)]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary)
        process = subprocess.run([
            executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-map", "0:v:0", "-an", "-map_metadata", "-1",
            "-vf", "setpts=PTS-STARTPTS,fps=30", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(stage / "video.mp4"),
        ], capture_output=True, text=True, check=False)
        if process.returncode:
            raise RuntimeError("ffmpeg не смог создать браузерный MP4: " + process.stderr.strip()[-2000:])
        result["duration_sec"] = _video_duration(stage / "video.mp4")
        if timecode > result["duration_sec"]:
            raise ValueError("После преобразования время разметки выходит за длительность браузерного видео.")
        (stage / "video.mp4").chmod(0o600)
        manifest = stage / "result.json"
        manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        manifest.chmod(0o600)
        _publish(stage, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Путь к предоставленному исходному видео")
    parser.add_argument("--timecode", type=float, default=9.0, help="Время из исходной разметки, секунды (по умолчанию 9)")
    parser.add_argument("--label", choices=("violation", "no_violation"), default="violation")
    parser.add_argument("--output", type=Path, default=None, help="Приватный каталог результата (FORTE_DATA_DIR/demo или .forte-runtime/demo)")
    parser.add_argument("--title", help="Заголовок из предоставленной разметки, до 200 символов")
    parser.add_argument("--description", help="Фактическое описание из разметки, до 4000 символов")
    parser.add_argument("--evidence-times", type=float, nargs="+", help="От 1 до 3 отметок исходного видео для контекстных кадров, секунды")
    arguments = parser.parse_args(argv)
    try:
        output = prepare_demo(arguments.video, timecode=arguments.timecode, label=arguments.label, output_dir=arguments.output,
                              title=arguments.title, description=arguments.description, evidence_times=arguments.evidence_times)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Демо по исходной разметке подготовлено: {output}. ML-анализ не выполнялся.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
