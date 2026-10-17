"""Numeric-only sanitizer diagnostics on two first-inner-training note owners.

Never writes or prints raw/sanitized notes, model answers, words or media.
Existing translation/vector caches are neither read nor modified.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment


def answer_statistics(answer, parser, vocabulary):
    """Return ONLY counts/booleans; no dictionary keys derived from private text."""
    result = {"answer_characters": len(answer), "fenced_json": answer.strip().startswith("```"),
              "json_valid": False, "json_is_list": False, "list_length": None,
              "strict_parser_accepted_phrases": len(parser(answer)), "phrases": []}
    try:
        payload = json.loads(answer.strip())
        result["json_valid"] = True
    except (ValueError, TypeError):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE).strip()
        try:
            payload = json.loads(stripped)
            result["json_valid_after_removing_fences"] = True
        except (ValueError, TypeError):
            result["json_valid_after_removing_fences"] = False
            return result
    result["json_is_list"] = isinstance(payload, list)
    if not isinstance(payload, list):
        return result
    result["list_length"] = len(payload)
    for value in payload:
        if not isinstance(value, str):
            result["phrases"].append({"is_string": False})
            continue
        phrase = " ".join(value.strip().lower().split())
        words = phrase.split()
        stripped_terminal = phrase.rstrip(".,;:!?")
        result["phrases"].append({"is_string": True, "characters": len(phrase), "word_count": len(words),
            "word_count_valid": 2 <= len(words) <= 12,
            "punctuation_or_non_ascii_characters": len(re.findall(r"[^a-z ]", phrase)),
            "terminal_punctuation_only": phrase != stripped_terminal and bool(re.fullmatch(r"[a-z ]+", stripped_terminal)),
            "out_of_vocabulary_word_count": sum(word not in vocabulary for word in words),
            "out_of_vocabulary_after_terminal_strip": sum(word not in vocabulary for word in stripped_terminal.split())})
    return result


def self_test():
    from tools.prepare_privileged_notes import WORDS, parse_phrases
    result = answer_statistics('```json\n["a worker counts cash."]\n```', parse_phrases, WORDS)
    assert result["fenced_json"] and not result["json_valid"]
    assert result["json_valid_after_removing_fences"]
    assert result["phrases"][0]["terminal_punctuation_only"]
    assert result["phrases"][0]["out_of_vocabulary_after_terminal_strip"] == 0
    assert "cash" not in json.dumps(result)
    print("PRIVILEGED_DIAGNOSTIC_SELF_TEST", json.dumps({"passed": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--notes-dir", type=Path)
    parser.add_argument("--folds-from", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not all((args.original_manifest, args.notes_dir, args.output)):
        parser.error("--original-manifest, --notes-dir and --output are required")
    configure_offline_environment()
    report = {"schema_version": 1, "purpose": "sanitizer_format_diagnosis_not_accuracy",
              "raw_text_emitted": False, "existing_caches_modified": False, "owners": []}
    started, counts, failed = time.perf_counter(), None, False
    try:
        with blocked_network() as counts, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report["phase"] = "load_manifest_and_folds"
            from datathon_ml import model as base, temporal
            from tools import prepare_privileged_notes as prep
            records = json.loads(args.original_manifest.read_text(encoding="utf-8-sig"))
            records = records["records"] if isinstance(records, dict) else records
            records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
            splits, fold_source = temporal.frozen_outer_splits(records, args.folds_from, temporal.TemporalConfig())
            outer_train_idx, outer_test_idx = splits[0]
            outer_train = [records[i] for i in outer_train_idx]
            inner_train_idx, inner_val_idx = base.grouped_splits(outer_train, 3, 43)[0]
            inner_train = [outer_train[i] for i in inner_train_idx]
            allowed = frozenset(r["sha256"] for r in inner_train)
            chosen = [next(r for r in inner_train if r["label"] == label) for label in (0, 1)]
            report["selection"] = {"outer_fold": 0, "inner_fold": 0,
                "allowed_training_sha256": sorted(allowed),
                "outer_test_sha256": [records[i]["sha256"] for i in outer_test_idx],
                "inner_validation_sha256": [outer_train[i]["sha256"] for i in inner_val_idx],
                "selected_owner_sha256": [r["sha256"] for r in chosen], "fold_source": fold_source}
            assert not allowed.intersection(report["selection"]["outer_test_sha256"])
            assert not allowed.intersection(report["selection"]["inner_validation_sha256"])
            checkpoint = args.checkpoint
            report["phase"] = "resolve_cached_checkpoint"
            if checkpoint is None:
                from huggingface_hub import snapshot_download
                checkpoint = Path(snapshot_download(prep.MODEL_ID, revision=prep.MODEL_REVISION, local_files_only=True))
            report["phase"] = "load_native_sanitizer"
            sanitizer = prep.NativeSanitizer(checkpoint, args.device)
            original_parser = prep.parse_phrases
            original_generate = sanitizer.model.generate
            active_stats = {}
            def inspected_generate(**kwargs):
                output = original_generate(**kwargs)
                active_stats["generated_tokens"] = int(output.shape[1] - kwargs["input_ids"].shape[1])
                active_stats["hit_max_new_tokens"] = active_stats["generated_tokens"] >= kwargs.get("max_new_tokens", prep.MAX_NEW_TOKENS)
                return output
            def inspected_parser(answer):
                active_stats.update(answer_statistics(answer, original_parser, prep.WORDS))
                return original_parser(answer)
            sanitizer.model.generate = inspected_generate
            prep.parse_phrases = inspected_parser
            try:
                for record in chosen:
                    report["phase"] = "read_allowed_owner"
                    owner = record["sha256"]
                    texts = prep.read_owner_notes(args.notes_dir, owner, allowed_owners=allowed)
                    cleaned = [prep.remove_nonvisual_context(text) for text in texts]
                    bounded, remaining = [], 6000
                    for text in cleaned:
                        if text and remaining > 0:
                            bounded.append(text[:remaining])
                            remaining -= len(bounded[-1])
                    active_stats = {"owner_sha256": owner, "training_label": record["label"],
                        "raw_fields": len(texts), "raw_characters": sum(map(len, texts)),
                        "raw_clauses": sum(len(re.split(r"[\r\n;]+|(?<=[.!?])\s+", text)) for text in texts),
                        "cleaned_nonempty_fields": sum(bool(text) for text in cleaned),
                        "cleaned_characters": sum(map(len, cleaned)),
                        "cleaned_clauses": sum(len(re.split(r"[\r\n;]+|(?<=[.!?])\s+", text)) for text in cleaned if text),
                        "bounded_fields": len(bounded), "bounded_characters": sum(map(len, bounded)),
                        "bounded_clauses": sum(len(re.split(r"[\r\n;]+|(?<=[.!?])\s+", text)) for text in bounded)}
                    if bounded:
                        report["phase"] = "native_generation_and_numeric_parse"
                        sanitizer(bounded)
                    else:
                        active_stats["sanitizer_skipped_empty_input"] = True
                    report["owners"].append(dict(active_stats))
            finally:
                prep.parse_phrases = original_parser
                sanitizer.model.generate = original_generate
            report["completed"] = True
            report["phase"] = "complete"
    except Exception as error:
        failed = True
        report["completed"] = False
        report["error_type"] = type(error).__name__  # Exception messages can contain private inputs.
    report["network"] = counts
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PRIVILEGED_NUMERIC_DIAGNOSIS", json.dumps(report, allow_nan=False))
    if failed:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
