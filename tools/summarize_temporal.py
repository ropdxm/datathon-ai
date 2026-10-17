"""Summarize numeric temporal experiments on the original frozen 40-video folds.

Only numeric prediction/metric fields are used. State vectors, rule descriptions,
event explanations and pixels are never inspected or copied into the summary.
Model files are hashed as bytes, never unpickled. All comparisons are development
results; binomial intervals and paired p-values are descriptive diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TARGET = .87
EXPECTED_N = 40
SCALAR_METRICS = ("n", "accuracy", "balanced_accuracy", "recall", "specificity", "precision",
                  "f1", "f2", "average_precision", "roc_auc", "brier_score")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson95(correct, total):
    if total < 1 or correct < 0 or correct > total:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {"low": max(0., center-radius), "high": min(1., center+radius),
            "method": "Wilson 95%; illustrative independent-binomial approximation"}


def exact_mcnemar(baseline, current):
    baseline_only = sum(baseline[key]["prediction"] == baseline[key]["label"]
                        and current[key]["prediction"] != current[key]["label"] for key in baseline)
    current_only = sum(baseline[key]["prediction"] != baseline[key]["label"]
                       and current[key]["prediction"] == current[key]["label"] for key in baseline)
    discordant = baseline_only + current_only
    p = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(baseline_only, current_only)+1)) / 2**discordant)
         if discordant else 1.)
    return {"baseline_correct_model_wrong": baseline_only, "model_correct_baseline_wrong": current_only,
            "discordant_pairs": discordant, "exact_two_sided_p": p,
            "multiple_comparison_adjusted": False,
            "interpretation": "Descriptive paired development diagnostic; not independent confirmation or causal evidence"}


def _numeric_rows(rows, *, require_fold=True):
    result = {}
    for source in rows:
        key = str(source["sha256"])
        label, prediction = source["label"], source["prediction"]
        if key in result or label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Duplicate SHA or invalid binary outcome in report")
        score = float(source["score"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite report score")
        row = {"label": int(label), "prediction": int(prediction), "score": score}
        if require_fold:
            fold = source["fold"]
            if not isinstance(fold, int):
                raise ValueError("Report fold IDs must be integers")
            row["fold"] = fold
        result[key] = row
    return result


def _verify(rows, baseline, identifier, *, require_fold=True):
    if len(rows) != EXPECTED_N or set(rows) != set(baseline):
        raise ValueError(f"{identifier}: expected exact original 40-SHA coverage")
    if any(rows[key]["label"] != baseline[key]["label"] for key in baseline):
        raise ValueError(f"{identifier}: labels differ from original baseline")
    if require_fold and any(rows[key]["fold"] != baseline[key]["fold"] for key in baseline):
        raise ValueError(f"{identifier}: outer fold assignments differ from original baseline")


def _metrics(rows, reported):
    confusion = [[0, 0], [0, 0]]
    for row in rows.values():
        confusion[row["label"]][row["prediction"]] += 1
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    n = len(rows)
    recall, specificity = tp/(tp+fn), tn/(tn+fp)
    precision = tp/(tp+fp) if tp+fp else 0.
    measured = {"n": n, "accuracy": (tn+tp)/n, "balanced_accuracy": (recall+specificity)/2,
        "recall": recall, "specificity": specificity, "precision": precision,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "confusion_matrix": confusion}
    for field in ("n", "accuracy", "balanced_accuracy", "recall", "specificity", "precision", "f1"):
        if field in reported and reported[field] is not None and not math.isclose(float(reported[field]), measured[field], abs_tol=1e-10):
            raise ValueError(f"Reported {field} disagrees with numeric predictions")
    if "confusion_matrix" in reported and reported["confusion_matrix"] != confusion:
        raise ValueError("Reported confusion matrix disagrees with numeric predictions")
    for field in SCALAR_METRICS:
        if field not in measured and isinstance(reported.get(field), (int, float)):
            value = float(reported[field])
            if not math.isfinite(value):
                raise ValueError("Nonfinite scalar report metric")
            measured[field] = value
    localization = reported.get("localization", {})
    point_recall = localization.get("point_recall_at_tolerance", {})
    measured["localization"] = {"annotated_points": localization.get("annotated_points"),
        "point_recall_at_tolerance": {key: point_recall.get(key) for key in ("1s", "2s", "5s")},
        "median_error_sec_conditional": localization.get("median_error_sec_conditional"),
        "p90_error_sec_conditional": localization.get("p90_error_sec_conditional")}
    return measured


def _artifact(path):
    path = Path(path).resolve()
    return {"artifact_path": str(path) if path.is_file() else None,
            "artifact_sha256": _sha(path) if path.is_file() else None,
            "artifact_available": path.is_file(), "expected_artifact_path": str(path)}


def _entry(identifier, label, role, perception, report_path, rows, reported, baseline, artifact_path=None):
    _verify(rows, baseline, identifier, require_fold=role != "frozen_rules")
    metrics = _metrics(rows, reported)
    correct = sum(row["label"] == row["prediction"] for row in rows.values())
    entry = {"model": identifier, "label": label, "role": role, "perception_family": perception,
        "metrics": metrics, "correct": correct, "accuracy_wilson95": wilson95(correct, len(rows)),
        "paired_vs_original_clip": exact_mcnemar(baseline, rows),
        "source_report_path": str(Path(report_path).resolve()), "source_report_sha256": _sha(report_path),
        "original_SHA_and_labels_verified": True,
        "original_outer_folds_verified": True if role != "frozen_rules" else None,
        "evaluation": "repeated_development_comparison_not_independent_test",
        "selection_is_independent_test": False, "target_accuracy": TARGET,
        "target_achieved_on_development": metrics["accuracy"] >= TARGET, "case_ready": False}
    entry.update(_artifact(artifact_path) if artifact_path else
                 {"artifact_path": None, "artifact_sha256": None, "artifact_available": False,
                  "artifact_kind": "frozen_observable_rules_no_fitted_head"})
    return entry


def summarize(results_root, baseline_path):
    results_root, baseline_path = Path(results_root).resolve(), Path(baseline_path).resolve()
    reference = _read(baseline_path)
    baseline = _numeric_rows(reference["predictions"])
    if len(baseline) != EXPECTED_N or len({row["fold"] for row in baseline.values()}) != 5:
        raise ValueError("Original baseline must contain all 40 unique sources on five outer folds")
    models = [_entry("original_clip_b32", "Original CLIP baseline", "baseline", "original_clip",
        baseline_path, baseline, reference["metrics"], baseline, baseline_path.parent.parent / "model" / "model.joblib")]
    frozen_rules, rule_sources = {}, {}
    temporal_paths = sorted(results_root.rglob("temporal_cv_report.json"))
    head_paths = sorted(results_root.rglob("semantic_head_cv_report.json"))
    if not temporal_paths and not head_paths:
        raise ValueError("No completed temporal or semantic-head reports found")
    for path in temporal_paths:
        report = _read(path)
        run = path.parent.parent
        perception = run.name
        short = "CLIP" if "clip" in perception.lower() else "Qwen4B" if "qwen" in perception.lower() else perception
        rows = _numeric_rows(report["predictions"])
        models.append(_entry(perception + "/linear_temporal", short + " · linear temporal",
            "linear_temporal", perception, path, rows, report["metrics"], baseline, run / "model" / "model_temporal.joblib"))
        rule = report.get("frozen_rules", {})
        if rule.get("predictions") is not None:
            rule_rows = _numeric_rows(rule["predictions"], require_fold=False)
            frozen_rules[perception] = _entry(perception + "/frozen_rules", short + " · frozen observable rules",
                "frozen_rules", perception, path, rule_rows, rule["metrics"], baseline)
            rule_sources[perception] = rule["metrics"]
    for path in head_paths:
        report = _read(path)
        run = path.parent.parent
        perception = run.parent.name if run.name == "nonlinear_heads" else run.name
        short = "CLIP" if "clip" in perception.lower() else "Qwen4B" if "qwen" in perception.lower() else perception
        rows = _numeric_rows(report["predictions"])
        models.append(_entry(perception + "/inner_selected_heads", short + " · inner-selected family policy",
            "inner_selected_family_policy", perception, path, rows, report["metrics"], baseline,
            run / "model" / "model_semantic_head.joblib"))
        for family, comparison in sorted(report.get("family_comparisons", {}).items()):
            family_rows = _numeric_rows(comparison["predictions"])
            models.append(_entry(perception + "/family_" + family, short + " · family " + family,
                "family_specific_nested_oof", perception, path, family_rows, comparison["metrics"], baseline,
                run / "model" / ("model_semantic_" + family + ".joblib")))
        if perception in rule_sources:
            for field in ("accuracy", "balanced_accuracy", "confusion_matrix"):
                if report.get("frozen_rules", {}).get("metrics", {}).get(field) != rule_sources[perception].get(field):
                    raise ValueError("Frozen-rule outcomes differ between linear and family-comparison reports")
    if len({row["model"] for row in models}) != len(models):
        raise ValueError("Duplicate model identifiers; use one extracted run root")
    best = max(models, key=lambda row: (row["metrics"]["balanced_accuracy"], row["metrics"]["accuracy"]))
    digest_rows = sorted((key, row["label"], row["fold"]) for key, row in baseline.items())
    return {"created_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": _sha(__file__), "results_root": str(results_root),
        "baseline_report_path": str(baseline_path), "baseline_report_sha256": _sha(baseline_path),
        "n": EXPECTED_N, "positive_videos": sum(row["label"] for row in baseline.values()),
        "normal_videos": sum(1-row["label"] for row in baseline.values()),
        "all_learned_models_original_SHA_labels_and_folds_verified": True,
        "original_fold_assignment_sha256": hashlib.sha256(json.dumps(digest_rows, separators=(",", ":")).encode()).hexdigest(),
        "model_comparisons": models, "frozen_rule_comparisons": list(frozen_rules.values()),
        "posthoc_best_development_model": {**best,
            "selection_basis": "Posthoc maximum balanced accuracy, then accuracy; frozen rules excluded",
            "posthoc_selection": True, "is_primary_inner_selected_policy": best["role"] == "inner_selected_family_policy"},
        "target_accuracy": TARGET, "target_scope": "learned models; frozen-rule diagnostics reported separately",
        "target_achieved_on_development": any(row["metrics"]["accuracy"] >= TARGET for row in models),
        "case_ready": False, "selection_is_independent_test": False,
        "limitations": ["All models reuse the same 40 staged development recordings; no independent or new-camera test is established.",
            "The posthoc best model is distinct from the primary policy that selects its family inside training folds.",
            "Wilson intervals assume independent binomial trials and are illustrative; related recordings and model selection add uncertainty.",
            "Exact McNemar p-values are unadjusted descriptive paired diagnostics across repeated comparisons, not independent confirmation or causal evidence.",
            "Capped state dwell/run counts and max/top-k statistics remain sensitive to recording exposure length.",
            "Observable-rule scores and machine annotation proposals are not verified semantic or legal ground truth.",
            "Achieving measured development accuracy of 87% does not make the case ready."]}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_rows(summary):
    for row in [*summary["model_comparisons"], *summary["frozen_rule_comparisons"]]:
        metrics, interval, paired = row["metrics"], row["accuracy_wilson95"], row["paired_vs_original_clip"]
        yield {"model": row["model"], "role": row["role"], "n": metrics["n"], "correct": row["correct"],
            **{key: metrics.get(key) for key in ("accuracy", "balanced_accuracy", "recall", "specificity", "precision", "f1")},
            "accuracy_ci95_low": interval["low"], "accuracy_ci95_high": interval["high"],
            "point_recall_5s": metrics["localization"]["point_recall_at_tolerance"]["5s"],
            "baseline_correct_model_wrong": paired["baseline_correct_model_wrong"],
            "model_correct_baseline_wrong": paired["model_correct_baseline_wrong"],
            "mcnemar_exact_two_sided_p": paired["exact_two_sided_p"],
            "target_achieved_on_development": row["target_achieved_on_development"], "case_ready": False,
            "artifact_path": row["artifact_path"], "artifact_sha256": row["artifact_sha256"]}


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    learned, rules = summary["model_comparisons"], summary["frozen_rule_comparisons"]
    panels = [learned, rules] if rules else [learned]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, max(5., .43*(len(learned)+len(rules))+2.5)),
        gridspec_kw={"height_ratios": [max(len(rows), 2) for rows in panels]}, squeeze=False)
    colors = {"baseline": "#7b8490", "linear_temporal": "#208b84", "inner_selected_family_policy": "#2f62ad",
              "family_specific_nested_oof": "#91afcf", "frozen_rules": "#c58a34"}
    for ax, rows in zip(axes[:, 0], panels):
        for index, row in enumerate(rows):
            accuracy = row["metrics"]["accuracy"]*100
            interval = row["accuracy_wilson95"]
            ax.barh(index, accuracy, color=colors[row["role"]], height=.64)
            ax.errorbar(accuracy, index, xerr=[[accuracy-100*interval["low"]], [100*interval["high"]-accuracy]],
                        fmt="none", color="#253242", capsize=3, linewidth=1.1)
            ax.text(min(accuracy+1.2, 94), index-.18, f"{accuracy:.1f}% ({row['correct']}/40)", fontsize=8.5)
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.axvline(87, color="#b33737", linestyle="--", linewidth=1.3, label="87% target")
        ax.grid(axis="x", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_xlabel("Development OOF accuracy (%)" if rows is learned else "Frozen-rule accuracy on the same development videos (%)")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Temporal model comparison · same original 40-video folds", fontsize=13)
    fig.text(.015, .012, "Bars span the full 0–100% scale. Error bars: illustrative Wilson 95% intervals.\nRepeated development comparisons; no independent test or case-readiness claim.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .07, 1, .965))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    return fig


def write_outputs(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "temporal_experiment_summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    rows = list(_csv_rows(summary))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(output_dir / "temporal_model_comparison.csv", buffer.getvalue())
    fig = plot_summary(summary, output_dir / "temporal_model_comparison.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def self_test():
    class SummaryTests(unittest.TestCase):
        def test_exact_pair_and_interval(self):
            base = {str(i): {"label": 1, "prediction": 1} for i in range(6)}
            current = {key: {"label": 1, "prediction": 0} for key in base}
            paired = exact_mcnemar(base, current)
            self.assertEqual(paired["baseline_correct_model_wrong"], 6)
            self.assertEqual(paired["model_correct_baseline_wrong"], 0)
            self.assertEqual(paired["exact_two_sided_p"], .03125)
            interval = wilson95(35, 40)
            self.assertAlmostEqual(interval["low"], .738879, places=5)
            self.assertAlmostEqual(interval["high"], .945405, places=5)
        def test_fixture_flatten_guards_artifact_hashes_and_chart_scale(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def predictions(errors, folds=True):
                    return [{"sha256": f"{i:064x}", "label": int(i<21),
                        "prediction": int(i<21) ^ int(i in errors), "score": .8 if (int(i<21)^int(i in errors)) else .2,
                        **({"fold": i%5} if folds else {}), "explanation": "FORBIDDEN_OBSERVATION"} for i in range(40)]
                def metrics(rows):
                    return _metrics(_numeric_rows(rows, require_fold="fold" in rows[0]), {})
                def save(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding="utf-8")
                baseline_rows = predictions(set(range(17)))
                baseline = root / "baseline/evaluation/cv_report.json"
                save(baseline, {"predictions": baseline_rows, "metrics": metrics(baseline_rows)})
                run = root / "runs/semantic_clip"
                linear = predictions(set(range(10)))
                rules = predictions(set(range(17)), False)
                save(run / "evaluation/temporal_cv_report.json", {"predictions": linear, "metrics": metrics(linear),
                    "frozen_rules": {"predictions": rules, "metrics": metrics(rules)}})
                families = {"linear": linear, "rbf": predictions(set(range(8))), "extra_trees": predictions(set(range(5)))}
                head = run / "nonlinear_heads/evaluation/semantic_head_cv_report.json"
                save(head, {"predictions": families["rbf"], "metrics": metrics(families["rbf"]),
                    "family_comparisons": {name: {"predictions": rows, "metrics": metrics(rows)} for name, rows in families.items()},
                    "frozen_rules": {"metrics": metrics(rules)}})
                paths = [baseline.parent.parent / "model/model.joblib", run / "model/model_temporal.joblib",
                    head.parent.parent / "model/model_semantic_head.joblib",
                    *[head.parent.parent / ("model/model_semantic_"+name+".joblib") for name in families]]
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(("fixture-model-do-not-unpickle:" + path.name).encode())
                summary = summarize(root / "runs", baseline)
                self.assertEqual(len(summary["model_comparisons"]), 6)
                self.assertEqual(len(summary["frozen_rule_comparisons"]), 1)
                best = summary["posthoc_best_development_model"]
                self.assertTrue(best["target_achieved_on_development"])
                self.assertFalse(best["case_ready"])
                self.assertEqual(best["artifact_path"], str(paths[-1].resolve()))
                self.assertEqual(best["artifact_sha256"], _sha(paths[-1]))
                self.assertNotIn("FORBIDDEN_OBSERVATION", json.dumps(summary))
                output = root / "output"
                write_outputs(summary, output)
                self.assertTrue((output / "temporal_model_comparison.png").is_file())
                self.assertTrue((output / "temporal_model_comparison.csv").is_file())
                self.assertFalse((output / "model_comparison.png").exists())
                self.assertFalse((output / "model_comparison.csv").exists())
                fig = plot_summary(summary, root / "chart.png")
                self.assertTrue(all(tuple(ax.get_xlim()) == (0., 100.) for ax in fig.axes))
                import matplotlib.pyplot as plt
                plt.close(fig)
                changed = _read(head)
                changed["predictions"][0]["fold"] = 4
                save(head, changed)
                with self.assertRaisesRegex(ValueError, "fold assignments"):
                    summarize(root / "runs", baseline)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("dtFinal/runs/clip_b32/evaluation/cv_report.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.results_root is None:
        parser.error("--results-root is required")
    summary = summarize(args.results_root, args.baseline)
    write_outputs(summary, args.output_dir)
    best = summary["posthoc_best_development_model"]
    print(json.dumps({"original_40_SHA_labels_and_folds_verified": True,
        "learned_comparisons": len(summary["model_comparisons"]), "frozen_rule_comparisons": len(summary["frozen_rule_comparisons"]),
        "posthoc_best_development_model": best["model"],
        "metrics": {key: best["metrics"].get(key) for key in ("accuracy", "balanced_accuracy", "recall", "specificity")},
        "accuracy_wilson95": best["accuracy_wilson95"],
        "target_achieved_on_development": summary["target_achieved_on_development"], "case_ready": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
