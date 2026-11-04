# FieldBCQ

FieldBCQ is a YOLO11x-seg model for agricultural parcel instance segmentation in imagery from multiple sources. Shared-prototype boundary supervision combines boundary Dice supervision (BDS, A) and boundary-aware pixel supervision (BAPS, B). Region–boundary quality scoring (RBQS, C) estimates mask coverage and contour agreement, producing the joint confidence `class_score * sqrt(region_quality * boundary_quality)` before box-based NMS. The auxiliary boundary predictor is used only during training.

**Manuscript:** FieldBCQ for agricultural parcel delineation with boundary supervision and region boundary quality scoring.

The eight configurations are Baseline, A, B, C, A+B, A+C, B+C, and FieldBCQ. The current manuscript contains five tables, seven figures, and supplementary Tables S1–S3. The [reported result snapshot](reported_results/manuscript_20261104/README.md) preserves the displayed segmentation results and their source hashes; it is a transcription of existing results, not a new evaluation. Each configuration uses one checkpoint with recorded training seed 0.

See [segmentation source notes](src/README.md) for package compatibility and licensing.

## Manuscript results

AgriParcel-MS contains 15,000 images and 403,199 patch-level instance annotations selected from FBIS, FTW, and FHAPD. The fixed train / validation / test splits contain 12,000 / 1,000 / 2,000 images and 353,352 / 28,657 / 21,190 annotations, respectively. All sources are harmonized to one field class. The test set includes 83 empty-label images and seven nominal-resolution groups from 0.3 to 10 m; the collection's 0.25 m category has no test images.

The six-model comparison in manuscript Table 2 reports mask AP as percentages:

| Model | AP50 | AP75 | AP50–95 |
|---|---:|---:|---:|
| Mask R-CNN R50-FPN | 47.57 | 29.83 | 29.17 |
| Mask2Former R50 | 52.51 | 31.68 | 32.51 |
| YOLO8x-seg | 60.40 | 38.25 | 37.78 |
| YOLO11x-seg (Baseline) | 60.71 | 39.66 | 38.65 |
| YOLO26x-seg | 61.69 | 40.14 | 39.30 |
| FieldBCQ | 64.01 | 43.50 | 41.99 |

FieldBCQ improves AP50–95 by 3.34 percentage points over the baseline. At confidence 0.25, F1 changes from 62.09% to 64.58% and Boundary-F1 from 40.29% to 43.22%; small-parcel recall decreases from 30.65% to 28.77%, and foreground-union area WAPE increases from 6.56% to 7.20%. With identical full-model weights, joint scoring improves AP50–95 by 0.492 percentage points over class-only scoring, calculated before rounding.

Section 3.3 reports the following computational measurements under the protocol in Section 2.6.2:

| Model | Parameters (M) | GFLOPs | Raw forward (ms) | Prediction pipeline (ms) |
|---|---:|---:|---:|---:|
| YOLO11x-seg (Baseline) | 62.051 | 297.03 | 9.953 | 14.275 |
| FieldBCQ | 63.970 | 305.34 | 10.311 | 14.354 |

Parameters increase by 3.09% and computation by 2.80%. Parameter counts include auxiliary training branches, while evaluation-mode GFLOPs exclude the training-only boundary predictor. Timing uses RTX 4090, FP32, 640×640 inputs and batch size 1. These are the manuscript's recorded measurements; the snapshot is not a new benchmark. Cross-model complexity and latency values for the other four architectures are not reported in the current manuscript.

See the [complete tables, captions, and computational measurements](reported_results/manuscript_20261104/README.md). The comparison-model AP values are transcribed results; the bundled training and evaluation entry points implement the eight YOLO11x-seg ablation configurations, not training recipes for the other architectures.

## Installation

Install the PyTorch and torchvision builds appropriate for your CPU or CUDA environment first, then install this repository from its root:

```bash
python -m pip install -e ./src
```

Python 3.10 or later is required. Training, evaluation, and verification entry points are under `src/scripts/`. They prioritize this repository's `src/ultralytics`, and relative command-line paths resolve against the project root. The retained Ultralytics framework supplies the model classes and runtime needed by the bundled checkpoints.

The repository contains dataset configuration in `data/`, eight checkpoints in `model/`, segmentation code in `src/`, reported CSV summaries in `reported_results/`, and a reserved figure directory `fig/`. Images, labels, and generated experiment outputs must be supplied or generated separately.

## Segmentation evaluation and manuscript mapping

The evaluation CLI retains legacy task IDs 4–10 and filenames `table4.csv` through `table10.csv` for compatibility. These IDs are not the current manuscript table numbers. All tasks use the same segmentation test split.

| Legacy task ID (`--tables`) | Contents | Current manuscript location |
|---|---|---|
| 4 | Native mask P, R, F1 and AP for eight configurations | Table 4 and Table S1 |
| 5 | Boundary F1, merge rate, oversegmentation and score–IoU ECE | Table 4 |
| 6 | Identical-checkpoint class / joint scoring | Table 5 |
| 7 | Fixed-threshold detection, count and area metrics | Table 3 |
| 8 | Source-group results | Table S2 |
| 9 | Nominal-resolution groups | Table S3 |
| 10 | Parameters, GFLOPs and latency | Section 3.3; measurement protocol in Section 2.6.2 |

Current Table 2 contains the separate cross-model comparison; it is not generated by this eight-configuration evaluation entry point. Table 1 describes the dataset.

### Checkpoints

The eight **best.pt** checkpoints were copied from the round-2 experiments in `F:\project\二轮`. The original files are retained. Evaluation uses neither `best_fp32.pt` nor `last.pt`. Training seeds are read from checkpoint contents rather than inferred from folder names.

| Configuration | Round-2 directory | Default project path |
|---|---|---|
| Baseline | exp | model/baseline/best.pt |
| A | a_seed01 | model/a/best.pt |
| B | b_seed01 | model/b/best.pt |
| C | c_seed01 | model/c/best.pt |
| A+B | ab_seed01 | model/ab/best.pt |
| A+C | ac_seed01 | model/ac/best.pt |
| B+C | bc_seed01 | model/bc/best.pt |
| FieldBCQ | abc_seed01 | model/abc/best.pt |

Checkpoint origins and SHA256 hashes are recorded in `model/checkpoints.json`. Evaluation verifies each configuration against the actual model head and training arguments and rejects mismatches. Configurations containing C use joint scoring. Legacy task 6 (current Table 5) reuses the complete FieldBCQ model's joint results and performs one additional class-scoring evaluation with the same checkpoint, verifying its SHA256 hash to separate scoring effects from weight changes.

These tasks define the current segmentation reproduction protocol. Segmentation results should be obtained with this protocol on the target dataset and hardware.

### Server Dataset Paths

Add `path` to `data/datasets.yaml` to specify the server's dataset root. If `path` is omitted, the directory containing the YAML file is used as the root. Set the training, validation, and test splits through `train`, `val`, and `test`, respectively. The evaluation data should contain `images/test` and corresponding `labels/test` directories. Labels use YOLO instance segmentation polygon format with a single `field` class; negative images require explicit empty label files. For example:

```yaml
path: /data/AgriParcel
train: images/train
val: images/val
test: images/test
nc: 1
names: [field]
```

The repository's `data/` directory contains only `datasets.yaml`; images and labels must be supplied separately. Legacy tasks 8 and 9 (Tables S2 and S3) require an external manifest specified through `--metadata-csv`, with the required columns `image`, `source`, and `resolution_m`. The `image` column accepts paths relative to the dataset root, absolute paths, or unambiguous filenames. Supported sources are `FBIS`, `FHAPD` (also accepting `DDFHAPD`), and `FTW`. Resolution values must be positive and expressed in meters.

The manifest must match the test images one-to-one. Optional columns are `instances`, `image_sha256`, and `label_sha256`. The program verifies complete coverage and any supplied instance counts and file hashes. Groups are determined from the manifest; evaluation areas and distances remain in pixel units.

### Execution

After configuring the data on the server, check the files before running evaluation. Legacy tasks 4, 5, 6, 7, and 10 can run without the source and resolution manifest:

```bash
python src/scripts/val_paper.py --tables 4 5 6 7 10 --check-only
python src/scripts/val_paper.py --tables 4 5 6 7 10 --device 0 --project results/paper_round2
```

`--check-only` rejects empty checkpoints and Git LFS pointer files whose weights have not been downloaded, without deserializing models or running inference. If evaluation reports `Git LFS pointer` or PyTorch's `invalid load key, 'v'`, download the weights into the working tree first:

```bash
git lfs install --local
git lfs fetch origin main --include="model/**/*.pt" --exclude=""
git lfs checkout
python src/scripts/val_paper.py --tables 4 5 6 7 10 --check-only
```

Alternatively, copy the original round-2 `best.pt` files to the server using the paths above. Each complete checkpoint is approximately 125–129 MB. Files beginning with `version https://git-lfs.github.com/spec/v1` are pointer text. Failed evaluation directories retain their provenance records; use a new `--project` path when rerunning.

Legacy tasks 4–10 are selected by default, so the complete protocol requires an external manifest:

```bash
python src/scripts/val_paper.py --metadata-csv /path/to/metadata.csv --check-only
python src/scripts/val_paper.py --metadata-csv /path/to/metadata.csv --device 0 --project results/paper_round2_tables4_10
```

Use a new output directory for each evaluation run. To rerun individual legacy tasks:

```bash
python src/scripts/val_paper.py --tables 6 --device 0 --project results/paper_round2_table6
python src/scripts/val_paper.py --tables 10 --device 0 --project results/paper_round2_table10
```

Use `--data`, `--metadata-csv`, and `--weights-root` to specify alternative locations. Override an individual configuration with, for example, `--weights 'A+B=/path/to/best.pt'`. Native and fixed-threshold metrics share one inference pass per checkpoint. Grouped results aggregate the same per-image counts without rerunning inference for each group.

### Evaluation Protocol

- Native mask metrics: FP32, imgsz=640, batch=8, rect=True, and augmentation disabled; minimum confidence 0.001, NMS IoU=0.70, and max_det=710. P/R use the framework's native best-F1 operating point; F1 is their harmonic mean.
- Fixed-threshold metrics: confidence 0.25, with predictions matched one-to-one to same-class GT masks in descending score order at mask IoU≥0.5. Full-resolution binary prediction masks are mapped to a 640×640 square evaluation grid using the loader's recorded resize and padding. GT normalized polygons are rasterized with integer coordinates at the same size and position. Rectangular validation may add padding up to 672×672; conversion translates masks and crops padding outside the evaluation grid without resampling prediction masks.
- Boundary F1: two-sided boundaries from 3×3 morphological operations, with a square tolerance of 2 px. Unmatched instance boundaries contribute to the denominators. Merge rate uses the number of predicted instances as its denominator; oversegmentation rate uses the number of GT instances.
- Merging: one prediction covers at least two same-class GT instances, covering at least 50% of each GT's area. Oversegmentation: one GT contains at least two same-class predictions, each with at least 50% of its own area inside that GT and covering at least 5% of the GT's area.
- Small parcels: GT area≤1024 px on the evaluation grid. ECE uses 10 equal-width bins and retains all predictions, including false positives, comparing each score with its best same-class GT mask IoU. A score of 1 belongs to the final bin.
- Count MAE covers all images, including negatives. Area MAPE includes only successfully matched instances. WAPE is the sum of per-image absolute errors in foreground union area divided by total GT foreground union area; net bias retains the error sign. Grouped F1 and Boundary F1 are computed from aggregated group counts.
- Legacy task 10 (Section 3.3; measurement protocol in Section 2.6.2): FP32, 640×640, and batch=1, with 50 warm-up iterations and 200 timed iterations for each measurement, using CUDA synchronization. Parameter counts include auxiliary training branches. GFLOPs are twice the THOP MAC count of the unfused evaluation graph and exclude the training-only boundary predictor. Raw forward timing measures only the model forward pass on resident tensors. Prediction pipeline timing includes preprocessing, inference, NMS, and mask generation, excluding disk reads. The first real test image is used, and the input and hardware are recorded in provenance.

Undefined values caused by zero denominators are left empty and displayed as `—` in Markdown. P/R/F1, AP, recall, merge and oversegmentation rates, and relative area errors are reported as percentages in CSV and Markdown. Differences in legacy tasks 6 and 9 (Table 5 and Table S3) are calculated before rounding and expressed in percentage points; ECE differences retain the 0–1 scale.

### Outputs

```text
results/paper_round2_tables4_10/
  table4.csv ... table10.csv
  tables4_10.md
  provenance.json
```

`provenance.json` records the actual arguments, seeds, model/data/code hashes, environment, and timing scope. Failed runs are marked `failed` with the reason retained. The example above corresponds to all legacy tasks 4–10. When a subset is selected, only the corresponding `tableN.csv` files are generated; the summary Markdown remains named `tables4_10.md`. This entry point produces quantitative segmentation results and reproduction records.

## Training

The training entry point is `src/scripts/train_paper.py`. The complete model uses boundary Dice+BCE supervision and region/boundary quality supervision. The following command uses 80 epochs, batch=8, imgsz=640, and seed=0, writing training results to `results/train/`:

```bash
python src/scripts/train_paper.py --stage fieldbcq --data data/datasets.yaml --weights model/pretrained/yolo11x-seg.pt --epochs 80 --batch 8 --imgsz 640 --seed 0
```

Supply the pretrained initialization checkpoint `model/pretrained/yolo11x-seg.pt` separately, or specify its location through `--weights`. The eight bundled `best.pt` files are provided for evaluating existing models. Training configurations correspond to the manuscript ablations as follows:

| Manuscript configuration | `--stage` | Active modules |
|---|---|---|
| Baseline | `baseline` | Original segmentation model |
| A | `pcsc` (alias `bds`) | BDS |
| B | `baps` | BAPS |
| C | `rbqs` | RBQS |
| A+B | `baps_pcsc` | BDS+BAPS |
| A+C | `bds_rbqs` | BDS+RBQS |
| B+C | `baps_rbqs` | BAPS+RBQS |
| FieldBCQ | `fieldbcq` | BDS+BAPS+RBQS |

The current training entry point sets the boundary supervision weight to 0.20 when enabled and the quality supervision weight to 0.25 when enabled. The complete model uses `boundary_mode=dice_bce`, with the quality branch's boundary target width set to 2. Actual training arguments and the environment are saved with the training records. Historical directory names do not determine the actual seed used by the command above. Select a device with `--device`, resume from the relevant experiment's `last.pt` with `--resume`, or switch to full-precision training with `--no-amp`.

### Boundary and quality targets

BDS (Dice) and BAPS (pixel BCE) supervise the same auxiliary boundary prediction from the shared 32×160×160 prototype tensor. Their target is the 3×3 morphological gradient of the GT foreground union at prototype resolution. The union target preserves external contours and inter-parcel gaps, but does not encode every interface between directly touching instances; instance identities remain supervised by the original mask loss.

RBQS predicts region and boundary quality from multiscale neck features. Matched positive candidates receive detached soft mask-IoU and tolerant Boundary-F1 targets within their assigned GT box crops. Both outputs use BCE supervision. The quality target's tolerance is two prototype pixels, whereas the reported evaluation Boundary-F1 uses two pixels on the 640×640 grid. These are different spatial scales. At inference, `class_score * sqrt(region_quality * boundary_quality)` is applied before box-based NMS; mask reconstruction remains unchanged for fixed weights.

The manuscript uses automatic optimizer selection and AMP for all eight configurations, with mosaic disabled during the final 10 epochs. For the full model, automatic selection resolved to SGD with initial learning rate 0.01, momentum 0.9 and weight decay 0.0005. A new run must record its own resolved settings; the current command does not establish the exact historical configuration of every comparison model.

## Verification

Run the configuration, positive/empty training-loss, gradient, and checkpoint inference checks on CPU:

```bash
python src/scripts/check_paper_models.py --checkpoints
```

This verifies that all eight configurations and bundled checkpoints execute. It does not reproduce the manuscript's accuracy or GPU latency measurements.

## Evaluation scope

The fixed evaluation split contains 2,000 images and 21,190 annotations. These images have also been used in prior exploratory work; the evaluation is not an untouched blind test. All three data sources occur in both training and testing, so source-group gains describe the represented collection rather than transfer to an unseen source. Source-pool separation alone does not establish geographic independence. Repeated training and previously unused source-image or spatial-block evaluation with duplicate checks remain necessary to assess stability and transfer.

At confidence 0.25, improved F1 and boundary agreement coexist with reduced small-parcel recall and increased foreground-area WAPE. Threshold choice should reflect the intended application. The repository's current commit must not be assigned retroactively to historical training or evaluation records.

## License

Use original images and annotations under their respective source licenses. The model framework retains the AGPL-3.0 license; see [src/LICENSE](src/LICENSE).
