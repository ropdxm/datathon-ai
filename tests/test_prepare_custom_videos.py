"""Video-only preparation safety checks, including synthetic FFmpeg integration."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_custom_videos as prep


def metadata(width=1920, height=1080, fps="60/1", duration="2.0", rotation=None, audio=True):
    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
             "width": width, "height": height, "avg_frame_rate": fps, "duration": duration,
             "start_time": "0", "time_base": "1/15360", "sample_aspect_ratio": "1:1"}
    if rotation is not None:
        video["side_data_list"] = [{"rotation": rotation}]
    return {"streams": [video] + ([{"index": 1, "codec_type": "audio"}] if audio else []), "format": {}}


class PreparationTests(unittest.TestCase):
    def test_stream_mapping_removes_audio_metadata_and_caps_geometry_fps(self):
        info = prep.video_info(metadata())
        args = prep.encode_command("source.mov", "new.mp4", info, prep.Settings())
        self.assertEqual(args[args.index("-map") + 1], "0:0")
        for flag in ("-an", "-sn", "-dn", "-map_metadata", "-map_chapters"):
            self.assertIn(flag, args)
        self.assertIn("fps=15:round=near", args[args.index("-vf") + 1])
        self.assertIn("scale=1280:720", args[args.index("-vf") + 1])
        low = prep.video_info(metadata(width=320, height=240, fps="10/1"))
        low_command = prep.encode_command("source", "out", low, prep.Settings())
        self.assertNotIn("fps=", low_command[low_command.index("-vf") + 1])
        self.assertEqual(prep.output_geometry(low, prep.Settings()), (320, 240))

    def test_rotation_and_unsafe_formats(self):
        rotated = prep.video_info(metadata(rotation=90))
        self.assertEqual(prep.output_geometry(rotated, prep.Settings()), (720, 1280))
        with self.assertRaises(ValueError):
            prep.video_info(metadata(rotation=27))
        hdr = dict(rotated, color_transfer="smpte2084")
        with self.assertRaisesRegex(ValueError, "HDR"):
            prep.encode_command("source", "out", hdr, prep.Settings())

    def test_verification_rejects_audio_changed_timing_and_wrong_geometry(self):
        source = prep.video_info(metadata())
        valid = metadata(1280, 720, "15/1", audio=False)
        self.assertTrue(prep.verify_output(source, valid, prep.Settings())["verified"])
        for bad in (metadata(1280, 720, "15/1", audio=True), metadata(1280, 720, "15/1", "3", audio=False),
                    metadata(1920, 1080, "15/1", audio=False), metadata(1280, 720, "30/1", audio=False)):
            with self.assertRaises(ValueError):
                prep.verify_output(source, bad, prep.Settings())

    def test_overlap_source_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for output in (root, root / "child", root.parent):
                with self.assertRaises(ValueError):
                    prep.resolve_directories(root, output)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_synthetic_rotation_and_low_frame_rate_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            unrotated = root / "unrotated.mp4"
            rotated = source / "rotated.mov"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=320x240:rate=10:duration=1.2", "-c:v", "libx264", str(unrotated)],
                           check=True, capture_output=True)
            help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True, check=True).stdout
            if "-display_rotation" in help_text:
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-display_rotation:v:0", "90",
                           "-i", str(unrotated), "-c", "copy", str(rotated)]
            else:
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(unrotated), "-c", "copy",
                           "-metadata:s:v:0", "rotate=90", str(rotated)]
            subprocess.run(command, check=True, capture_output=True)
            report = prep.prepare(source, root / "out")
            result = report["videos"][0]
            self.assertEqual((result["output"]["width"], result["output"]["height"]), (240, 320))
            self.assertEqual(Fraction(result["output"]["fps"]), Fraction(10))
            self.assertEqual(result["output"]["rotation"], 0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_synthetic_audio_stripping_duplicate_reuse_resume_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            recording = source / "sample.mov"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=320x240:rate=30:duration=1.2", "-f", "lavfi", "-i",
                            "sine=frequency=1000:duration=1.2", "-c:v", "libx264", "-c:a", "aac",
                            "-metadata", "comment=private synthetic comment", "-shortest", str(recording)],
                           check=True, capture_output=True)
            shutil.copyfile(recording, source / "duplicate.mov")
            original_sha = prep.sha256(recording)
            report = prep.prepare(source, root / "out")
            self.assertEqual(report["summary"]["source_with_audio"], 2)
            self.assertEqual(report["summary"]["output_with_audio"], 0)
            self.assertEqual(report["summary"]["unique_source_sha256"], 1)
            self.assertEqual({r["status"] for r in report["videos"]}, {"encoded", "reused_exact_source_duplicate"})
            self.assertEqual(len({r["output_sha256"] for r in report["videos"]}), 1)
            self.assertTrue(report["complete"])
            self.assertEqual(prep.sha256(recording), original_sha)
            # Read-only inventory does not destroy the resume manifest.
            prep.prepare(source, root / "out", inventory_only=True)
            resumed = prep.prepare(source, root / "out")
            self.assertEqual({r["status"] for r in resumed["videos"]}, {"reused_verified_existing"})
            original_write = prep.write_reports
            def interrupted_write(directory, contents):
                original_write(directory, contents)
                raise InterruptedError("synthetic interruption after durable checkpoint")
            with patch.object(prep, "write_reports", side_effect=interrupted_write):
                with self.assertRaises(InterruptedError):
                    prep.prepare(source, root / "out")
            # Checksums for later unvisited existing outputs survive interruption.
            resumed = prep.prepare(source, root / "out")
            self.assertEqual({r["status"] for r in resumed["videos"]}, {"reused_verified_existing"})
            output = root / "out" / resumed["videos"][0]["output_relative_path"]
            output.write_bytes(b"untrusted replacement")
            with self.assertRaises(FileExistsError):
                prep.prepare(source, root / "out")
            self.assertEqual(output.read_bytes(), b"untrusted replacement")
            self.assertEqual(prep.sha256(recording), original_sha)


if __name__ == "__main__":
    unittest.main()
