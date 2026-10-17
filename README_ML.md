# Datathon model experiments

This repository contains model training and offline inference code, without a web interface or any hosted inference API. Experiments compare frozen image, motion and multimodal encoders with small temporal classifiers. All model forward passes run in the selected Python runtime.

## Temporal follow-up: 18 October 2026

**Best development accuracy is now 62.5% (25/40), below the 87% target.** The CLIP semantic-state model with temporal features improves violation recall from 33.3% to 52.4%, with specificity decreasing from 84.2% to 73.7%. It still misses 10 of 21 positive videos. All original 40 video hashes, labels and outer test-fold assignments were verified against the report supplied in `dtFinal`; no labels were changed. The illustrative Wilson 95% accuracy interval is 47.0–75.8%. These repeated development comparisons are not an independent test.

| New representation / selection policy | Accuracy | Violation recall | Specificity |
|---|---:|---:|---:|
| Previous CLIP embedding baseline | 57.5% | 33.3% | 84.2% |
| CLIP observable states + linear temporal head | **62.5%** | **52.4%** | **73.7%** |
| CLIP states + inner-selected linear/SVM/tree head | 60.0% | 42.9% | 78.9% |
| Local Qwen4B states + linear temporal head | 40.0% | 57.1% | 21.1% |
| Local Qwen4B states + inner-selected linear/SVM/tree head | 50.0% | 71.4% | 26.3% |

The conservative frozen rules scored 47.5% with CLIP states and 42.5% with Qwen states. CLIP rules emitted no events; Qwen rules identified only 1/22 supplied points within five seconds. **The best binary classifier also localizes none of the supplied points. This is not a ready event detector.** No event-type accuracy, interval mAP or legal correctness can be established from the available annotations. State-detection accuracy itself remains unmeasured because the dataset supplies no verified state labels.

The implemented pipeline replaces generic embedding-only reasoning with 18 fixed observable signals, including cashier/customer presence, exposed cash, counting, handover, storage, departure and visibility. CLIP uses fixed positive/negative text prototypes; the separate Qwen experiment executes the original [Qwen3-VL-4B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) directly with PyTorch/Transformers. It requests only 18 numeric observation codes, never a legal verdict, identity or caption. No hosted LLM API or model server is used. Videos, frames and generated observations are never sent to an external AI service.

Temporal rules require confident observations, two-second dwell, hysteresis and continuous sampling support. They propose unattended cash, departure during visible service and leaving exposed cash with open secure access. A missing-count rule is implemented but disabled for these sampled-frame caches: it needs independently certified continuous coverage of a complete visible transaction. Unknown observations remain unknown; missing frames do not establish an omitted action. Phone interaction is a measured candidate feature, not an automatic legal violation. Rule component intervals/confidence remain separate when events merge.

The small heads use 162 fixed state/order/rule descriptors. Scaling, regularization, optional family selection, sigmoid calibration and thresholds use training folds only. The nonlinear comparison fixes seven candidates in advance: three regularized linear heads, three RBF SVM heads and one shallow ExtraTrees head. Both the inner-selected policy and each separately selected family are reported. Recording names, source IDs, annotation points and duration columns are not predictors. Capped dwell, run counts and max/top-mean summaries can still correlate with exposure length; this limitation remains explicit. An original SHA-to-fold map is loaded with `--folds-from`, because the same seed produced different assignments across scikit-learn versions.

Annotation review found 42 workbook rows representing 40 unique recordings and 22 valid positive points, with no supplied event types or boundaries. Four-second context is clipped near the end for X-18 and X-1. Exact duplicate X-4/X-47 retains both 57s and 68s; a person must verify whether these are separate events or alternate anchors. `artifacts/annotation_review_v2.csv` prioritizes all 40 recordings without relabeling them. Each semantic run exports `annotation_machine_proposals.csv`, containing local numeric observations around the points and any overlapping rule candidates. Reviewed types/bounds stay blank, and every proposal requires human verification.

All **82 regression tests passed locally and in the Colab T4 runtime**. Fresh-process inference with the best CLIP temporal head on the longest 142.5-second recording took **31.3 seconds**, with fresh image/state caches, no label/annotation input and zero Python network attempts under a blocking guard. This verifies execution and latency on a supplied recording; it is not an accuracy test. The final fitted heads enforce source/protocol fingerprints; use their matching source archive or refit after changing temporal/head code. Sequential Qwen extraction took 337.9 seconds on that recording, exceeding the case's five-minute budget. The optional batch-2 fresh-process offline benchmark also exceeded its enforced 300-second deadline. Qwen therefore has neither acceptable detection quality nor demonstrated latency compliance here. No completed batch-2 accuracy or equivalence claim is made.

Reproduce with the self-contained `datathon_temporal.ipynb`, or run:

```powershell
python tools/run_temporal_colab.py --workspace C:/projects/datathon --data-root C:/projects/datathon/open
```

That driver runs extraction, frozen-fold validation, final fitting, machine annotation proposals and export. Use `--skip-qwen` for the faster CLIP experiment or `--skip-heads` to omit the nonlinear comparison. For inference on an already extracted unlabeled semantic cache:

```powershell
python temporal_experiment.py predict --feature path/to/SHA.npz --model path/to/model_temporal.joblib --output-dir artifacts/prediction
```

For a fresh unlabeled video with the exported CLIP temporal/head model, use the offline-by-default entry tested in Colab:

```powershell
python semantic_predict_experiment.py --video path/to/unseen.mov --model artifacts/temporal_results/runs/semantic_temporal_v1/semantic_clip/model/model_temporal.joblib --output-dir artifacts/new_prediction
```

No workbook or timestamp is required. Download the original CLIP checkpoint once beforehand, or explicitly add `--allow-download` for that public-weight setup step. Detailed predictions stay in `prediction.json`. The clean notebook defaults to CLIP linear; the slower comparison runs are optional.

Measured reports, artifact hashes, paired comparisons and the full-scale chart are in `artifacts/temporal_experiment_summary.json`, `artifacts/temporal_model_comparison.csv` and `artifacts/temporal_model_comparison.png`. `artifacts/datathon_temporal_results.zip` contains the fitted heads, matching source and numeric caches; encoder weights are a separate one-time download. The best head is `artifacts/temporal_results/runs/semantic_temporal_v1/semantic_clip/model/model_temporal.joblib`.

The next justified step is human verification of the 22 event anchors and matched normal actions, including event types, full intervals, object/role labels and related filming groups. Then train and validate the task-specific perception layer before using procedural rules to aggregate events. Reserve new takes and a new camera angle for acceptance. More generic backbone swaps and threshold searches have not established an 87% model.

## Previous embedding baselines: 17 October 2026

**The 87% accuracy target was not achieved.** The best primary selection policy was CLIP at **57.5% (23/40)**, balanced accuracy 58.8%, violation recall 33.3% and specificity 84.2%. It missed 14 of 21 violation recordings. The majority baseline is 52.5%. These are development comparisons, not an independent test.

| Encoder / objective | Accuracy | Violation recall | Specificity | Point recall within 5s |
|---|---:|---:|---:|---:|
| DINOv2-S, PCA only | 45.0% | 28.6% | 63.2% | 0.0% |
| DINOv2-S, PCA/raw | 50.0% | 47.6% | 52.6% | 0.0% |
| DINOv2-B-reg, PCA/raw | 50.0% | 52.4% | 47.4% | 0.0% |
| CLIP ViT-B/32 | 57.5% | 33.3% | 84.2% | 0.0% |
| X3D-S | 42.5% | 52.4% | 31.6% | 36.4% |
| Qwen embedding, 4 frames | 55.0% | 61.9% | 47.4% | 40.9% |
| Qwen embedding, 8 frames | 45.0% | 47.6% | 42.1% | 27.3% |
| CLIP + MIL | 50.0% | 71.4% | 26.3% | 54.5% |
| X3D-S + MIL | 40.0% | 66.7% | 10.5% | 54.5% |
| Qwen 4 frames + MIL | 47.5% | 85.7% | 5.3% | 45.5% |
| Qwen 8 frames + MIL | 35.0% | 52.4% | 15.8% | 31.8% |

Point recall is an unofficial diagnostic over 22 supplied anchors; it is not event precision or interval mAP. The CLIP primary policy chooses whole-video pooling and cannot localize. Its separately exported point head also scored 57.5% video accuracy, with 57.1% violation recall and 6/22 points within five seconds.

All 53 regression tests passed in the Colab T4 runtime. Fresh offline inference on the longest 142.5-second recording took **27.8 seconds on T4** and **142.7 seconds on CPU**, using the CLIP primary model. The CPU test blocked network connections and observed zero connection attempts. These are execution checks on supplied recordings, not accuracy tests or a guarantee for every future recording.

The error audit found matching hashes, labels and fold assignments across the early CLIP/X3D comparisons. CLIP missed all three positive recordings longer than 30 seconds. This is consistent with rare actions being diluted by global pooling, but it is a modeling hypothesis rather than an observation about the footage. MIL often collapsed to nearly constant decisions; its weak within-fold ranking means threshold calibration alone is insufficient. Fold composition ranged from 2 positives/6 normals to 6 positives/2 normals; a 75% fold accuracy could result from predicting every example normal.

Review priorities are X-44 and X-23, which all four early CLIP/X3D policies got wrong, plus the long positives X-4, X-13 and X-38. These are review candidates; the supplied labels have not been changed. Recordings over 30 seconds contain 3 positives/9 normals, versus 18 positives/10 normals below that boundary, so duration must not become a shortcut feature.

The practical next step is reviewed event types and intervals, matched normal actions, workstation/object predicates and explicit procedural state rules, followed by a locked new-take/new-angle test. The current binary baselines are not a reliable compliance detector.

![Development model comparison](artifacts/model_comparison.png)

Full machine-readable metrics: `artifacts/experiment_summary.json`. Trained baselines and all eight post-reset reports: `artifacts/datathon_results.zip`. The selected head is `artifacts/colab_results/runs/best_development/model.joblib`; its point head is adjacent. The exported heads exclude encoder weights; retain/download the original checkpoint once for offline operation. The executed cloud notebook is backed up at `artifacts/colab_executed.ipynb`. Early DINO head files were lost when the first temporary Colab runtime reset; their metrics were recovered from saved outputs, and the source supports retraining them.



## Modeling and measurement decisions

This is weakly supervised temporal event detection with a binary recording verdict. It also requires semantic event classification and regulatory grounding for the eventual case submission. The supplied workbook teaches only the verdict and a few event points. Those labels cannot establish event types, full event boundaries, required actions omitted from an entire service sequence, or the legal reason for a flag.

With only 40 unique recordings, train a small head over frozen pretrained representations before attempting any backbone fine-tuning. DINOv2 and CLIP test appearance; X3D tests short motion; Qwen embeddings test a longer ordered visual context. A point-guided multiple-instance objective combines recording labels with event anchors and trains against normal activity throughout each normal recording. Once sufficient reviewed event labels exist, a small temporal convolution head is a reasonable next step; training a large video transformer from scratch is not justified by this corpus.

Normal-only anomaly detection can rank unusual clips for review, but unusual appearance is not equivalent to a procedural violation. A different camera angle can be statistically unusual while an often-repeated prohibited action can look ordinary. An autoencoder or one-class detector should therefore be an auxiliary signal rather than the source of a regulatory verdict.

Procedural omissions need a separate temporal state representation: whether service has begun, which observable actions occurred, and whether it ended before a required action. Reviewed person/object tracks, workstation zones and explicit state transitions can support those rules. A short-window classifier cannot establish that an action never happened outside its window. This hybrid perception-and-rules layer is a stronger architectural fit for the full case than a single opaque binary score; its event vocabulary and visual predicates still require validation.

All model choices and thresholds must use grouped inner validation. The outer folds hold out complete source recordings; duplicate files and related filming takes belong to one group. Independent new takes and new camera angles are still needed. Because several representations and objectives were compared after viewing earlier results, the best score from this corpus is development evidence, not an independent final test. Predicting the majority class already gives 52.5% accuracy (21/40); exceeding 87% on 40 recordings means at least 35 correct, and each mistake changes accuracy by 2.5 percentage points.

Report accuracy together with balanced accuracy, violation recall, precision, specificity, F1/F2 and the raw confusion matrix. The case asks for both few false alarms and very few misses, so a high accuracy with poor violation recall is unsuitable. Choose the operating threshold inside training validation against a stated recall/false-alarm tradeoff. AP and ROC AUC are secondary ranking diagnostics; scores from different fitted folds are uncalibrated, so pooled rankings have limits. Do not report window accuracy as recording accuracy.

For localization, the available labels support point recall at explicitly stated tolerances and timing error conditional on a detection. Always count missed points against recall. They do not support trustworthy event precision, interval IoU or temporal mAP because the annotation is not exhaustive and has no event intervals. After reviewed intervals and types are available, add event F1/mAP, type macro-F1, per-class recall and evidence-frame relevance. The case provides no official timestamp tolerance; the +/-1/2/5-second diagnostics here are internal choices.

Before deployment, measure complete uncached video latency, peak memory, failure rate and disconnected inference on the intended hardware; the case's deadline is 300 seconds per recording. A longer-context encoder can propose a candidate region, followed by denser local sampling to select the evidence frame. Sampling less often improves speed but directly limits timestamp precision. Any object/occupancy tracking or procedural rule must use reviewed scene geometry and validated objects; a generic person detector does not automatically recognize cash handling.

The eventual rule layer should map a reviewed visible event type to a real Kazakhstan source and exact clause, including exceptions. A binary anomaly score is insufficient for that mapping. Keep visible observations distinct from missing context and legal conclusions. The practical path is frozen local visual/action representations plus a small point-guided temporal head, reviewed event annotations and explicit procedural rules, then a locked new-angle evaluation. The experiments below determine which representation supports that path; none can guarantee the requested accuracy in advance.

`artifacts/annotation_review.csv` is a review scaffold containing only supplied verdicts and event points. Its type, interval, visible-action, normative-source and filming-group columns are deliberately empty. A human reviewer can complete those fields; they are not inferred annotations or model inputs.

## Files

- `datathon_ml/data.py`: label parsing, source hashes, explicit duplicate aliases and validated manifests.
- `datathon_ml/features.py`: sequential video decoding, timestamp-preserving sampling, full-frame letterboxing, three overlapping crops and frozen DINOv2 embeddings.
- `datathon_ml/model.py`: point-supervised and pooled models, grouped nested cross-validation, training-only preprocessing/selection, model export and inference.
- `datathon_ml/motion.py`: optional frozen X3D action features from short video clips.
- `datathon_ml/clip_features.py`: optional original CLIP image-feature comparison.
- `datathon_ml/local_vlm.py`: optional directly executed Qwen3-VL-Embedding-2B window encoder; numeric embeddings only.
- `datathon_ml/mil.py`, `mil_experiment.py`: optional point-guided multiple-instance objective on existing frozen caches.
- `experiment.py`: command-line entry point.
- `datathon.ipynb`: self-contained Colab/Kaggle notebook generated by `tools/build_notebook.py`.

## Install and audit

Use the GPU-enabled PyTorch already installed in Colab/Kaggle. For another machine, install a matching PyTorch/torchvision build before the remaining requirements.

```bash
python -m pip install -r requirements-ml.txt
python -m unittest discover -s tests -v
python experiment.py audit --data-root . --output-dir artifacts/run --verified-aliases duplicate_aliases.json
```

The local corpus currently has 40 video files and 42 annotation rows. The earlier SHA-256 audit verified X-47 as an exact duplicate of X-4 and X-48 as an exact duplicate of X-5. Those two files have since been removed locally. `duplicate_aliases.json` declares those aliases explicitly; it does not infer them from similar names. X-4 retains both supplied anchors, 57 and 68 seconds. An original archive containing all 42 files is deduplicated by actual SHA-256 instead. Conflicting binary labels fail validation.

## Extract, validate and fit

```bash
python experiment.py extract --data-root . --output-dir artifacts/run --verified-aliases duplicate_aliases.json --backbone dinov2_vits14 --sample-fps 2 --batch-size 32
python experiment.py cv --data-root . --output-dir artifacts/run --verified-aliases duplicate_aliases.json
python experiment.py fit --data-root . --output-dir artifacts/run --verified-aliases duplicate_aliases.json
```

`pipeline` runs those stages together. To compare ViT-B, use `--backbone dinov2_vitb14` and a **different output directory**. Reuse the same predefined folds and report every attempted configuration; repeated tuning to outer-fold results turns them into development data.

Each cache is `<source_sha256>.npz` with `features[T,V,D]`, `timestamps[T]`, encoder configuration and weight/source provenance. Input labels and event anchors never determine feature sampling. Cache identity includes extraction settings, encoder weights and numerical provenance; incompatible caches are rebuilt. Configuration and source provenance are recorded in the run directory.

The temporal classifier sees multiscale local appearance, variability, feature change and signed before/after context. Positive training windows come from supplied event points; unmarked spans of positive videos remain unlabeled. Negative windows come from compliant videos. Sampling weights limit long-video/class dominance. The baseline pools a complete recording and cannot localize events.

Outer grouped folds estimate the fixed selection policy's development performance. Inner grouped folds choose PCA versus raw scaled embeddings, classifier regularization, temporal aggregation and decision threshold. Raw features use dual logistic regression; PCA features compare logistic regression and RBF SVM. Scaling and PCA are fitted only on the relevant training split. `--pca-components 0` forces raw features. The default primary selection metric is balanced accuracy. Reports include accuracy, precision, recall, specificity, F1/F2, raw confusion counts, AP/AUC and point-localization diagnostics. Different fitted models have uncalibrated score scales, limiting pooled AP/AUC interpretation.

If filming-session relationships are known, pass `--groups-csv path.csv` with columns `video_id,group_id`. All related takes must share a group. Exact duplicate grouping alone does not establish independence.

## Optional representation comparisons

```bash
python -m pip install -r requirements-experiments.txt
python experiment.py pipeline --data-root . --output-dir artifacts/clip --verified-aliases duplicate_aliases.json --backbone clip_vit_b32_openai
python experiment.py pipeline --data-root . --output-dir artifacts/motion --verified-aliases duplicate_aliases.json --backbone x3d_s
```

CLIP supplies frozen 512-dimensional image features with an enforced official checkpoint hash. X3D-S supplies frozen 2048-dimensional motion features from 13 frames over 2.6 seconds, at one-second window centers, using full and middle views. Its classifier head is removed. Each representation uses its own cache/run directory and the same grouped evaluation policy.

Original CLIP and its loader have MIT licenses; its model card treats deployment as outside its intended scope, so this branch is an evaluated research comparison. PyTorchVideo code is Apache 2.0 and links the X3D weights officially; a separate checkpoint-specific license was not found. The motion cache records that provenance rather than inventing a weight-specific license. DINOv2 remains the explicitly commercially licensed default backbone. None of these branches uses face recognition or an LLM API.

An optional multimodal representation comparison uses original [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), Apache 2.0, pinned to revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`:

```bash
python -m pip install -r requirements-vlm.txt
python experiment.py pipeline --data-root . --output-dir artifacts/qwen_embedding --verified-aliases duplicate_aliases.json --backbone qwen3_vl_embedding_2b
```

It downloads public weights, then calls the native PyTorch model directly with four ordered frames per eight-second context, centered every four seconds. Last-token numeric embeddings are extracted without generating descriptions, calling a hosted model or running remote helper code. Frames remain in the selected runtime. This ordered-image adaptation and its classifier require task-specific validation; the model card's general benchmarks do not establish cashier-video accuracy. Retain the complete downloaded model directory and use `--offline --local-checkpoint /path/to/model` for disconnected inference. The optional NF4 code path is not enabled by the CLI and has not been tested.

The temporal-coverage comparison uses `--sample-fps .5 --clip-seconds 4 --frames-per-window 8` in a separate run directory. Prediction restores the trained context settings automatically. These flags are Qwen-specific; using an incompatible override with an existing trained head is rejected by the semantic feature check.

The optional MIL objective trains a small linear/MLP temporal scoring head on complete recordings. It combines top-three-window bag classification, positive-point loss, dense normal-window loss and light temporal smoothness. This addresses the mismatch between limited negative-window sampling during training and searching the entire video at inference:

```bash
python mil_experiment.py pipeline --manifest artifacts/clip/manifest.json --feature-dir artifacts/clip/features --output-dir artifacts/mil_clip
python experiment.py predict --video /path/to/new.mov --model artifacts/mil_clip/model/model_mil.joblib --output-dir artifacts/inference
```

Scaling and PCA use only the relevant training folds. Two fixed head/epoch candidates and a decision threshold are selected by inner grouped validation; no outer-test early stopping is used. Unmarked positive windows receive no negative instance target. The ordinary inference CLI recognizes exported MIL artifacts. This objective was introduced after observing earlier encoder results, so all comparisons on this corpus remain development results.

## Colab / Kaggle

Select a free GPU runtime. Open `datathon.ipynb`, run its embedded source bootstrap, and upload `artifacts/datathon_data.zip` using Colab's Files pane. Alternatively attach a **private** Kaggle input containing the label workbook and videos. Do not publish the dataset or notebook outputs containing recordings.

The notebook discovers the input directory, validates the manifest, extracts the selected encoder, performs grouped nested evaluation, fits the development model and exports reports plus the trained head. It defaults to the best measured CLIP baseline; the unsuccessful MIL comparison is optional. Each encoder/context uses its own run directory. The exported results archive excludes videos and feature caches. Download the result before the temporary runtime ends. Keep the original encoder source and checkpoint separately for offline video inference.

The live Colab run can download the organizer's already-shared archive directly from Drive instead of waiting for a browser upload. This is a dataset download into your runtime; no frames are sent to an inference API.

## Unlabeled video inference

```bash
python experiment.py predict --video /path/to/new.mov --model artifacts/run/model/model.joblib --output-dir artifacts/inference --prediction-output artifacts/inference/prediction.json
```

The CLI derives backbone, image size, views and sampling defaults from the trained artifact. Models return a binary verdict, uncalibrated score, selected threshold and, for a point model, candidate event times and a score trace. If the best binary model is pooled, `fit` also preserves the separately selected point model in `model_point.joblib`; see `final_selection.json`.

For fully offline inference, retain the pinned original DINOv2 repository and its matching checkpoint, then pass:

```bash
python experiment.py predict --video /path/to/new.mov --model artifacts/run/model/model.joblib --offline --local-repo /path/to/dinov2 --local-checkpoint /path/to/dinov2_vits14_pretrain.pth
```

GPU/CPU precision and library provenance differences are reported without rejecting semantically compatible features. A different backbone, weight hash, crop geometry or sampling configuration is rejected. Validate exported models on representative videos after moving environments. Load only trusted joblib artifacts.

## What these models do not establish yet

- Real-data development evaluation is complete: the best primary policy scored 57.5%; the requested 87% target was not achieved.
- Synthetic regression-test accuracy verifies implementation behavior, not performance on cashier videos.
- The supplied labels cannot teach semantic violation types or legal reasoning; those require reviewed annotations and a separate rule registry.
- Point timestamps do not define exhaustive event intervals. Localization reports use unofficial +/-1/2/5-second diagnostics; event precision and temporal IoU are not identifiable from these labels.
- New filming takes, real second-angle footage and unseen event families need separate locked tests.
- No face recognition, audio models or external inference API is used. The optional Qwen branch uses frozen local multimodal hidden states without text generation.

Original DINOv2 code and model weights are Apache 2.0: https://github.com/facebookresearch/dinov2 . The extractor pins the source revision and records the actual checkpoint SHA-256.
