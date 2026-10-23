# FieldBCQ

**FieldBCQ for agricultural parcel instance segmentation through shared prototype boundary supervision and region boundary quality scoring**

FieldBCQ builds on **YOLO11x-seg** to improve agricultural parcel masks and candidate selection. A training-only boundary predictor supervises the shared mask prototypes with **Boundary Dice Supervision (BDS)** and **Boundary-Aware Pixel Supervision (BAPS)**. **Region–Boundary Quality Scoring (RBQS)** predicts region overlap and boundary agreement from multiscale neck features and adjusts candidate scores before non-maximum suppression (NMS).

On the fixed **AgriParcel-MS** test set of **2,000 images and 21,190 patch-level instance annotations**, the manuscript reports **mask mAP50–95 of 0.420 ± 0.004** over three experimental rounds. The paired improvement over the baseline is **2.97 ± 0.95 AP points**. An identical-weight intervention isolates **0.51 ± 0.04 AP points** from inference score fusion. Fixed-threshold evaluation also reveals lower recall and degradation on submeter imagery.

This repository provides the implementation, training and native evaluation entry points, four archived checkpoints, and dataset/result summaries. The underlying imagery and annotations must be obtained from the original sources.

The five-way ablation includes an independently trainable **BAPS-only** configuration (`--stage baps`). BAPS-only requires a new training run; no trained BAPS-only checkpoint or result is bundled.

## Method

### Shared Prototype Boundary Supervision and Candidate Quality Scoring

The base model reconstructs each candidate mask from shared prototypes and candidate-specific coefficients. FieldBCQ retains this reconstruction rule and adds the following supervision and scoring paths:

| Objective or head | Input | Function |
|---|---|---|
| **BDS** | Shared mask prototypes | Soft Dice supervision of an auxiliary boundary prediction. |
| **BAPS** | The same boundary prediction as BDS | Pixelwise binary cross-entropy (BCE) against the same target. |
| **RBQS** | Multiscale neck features | Predicts region and boundary quality; fuses both with classification confidence before box-based NMS. |

**BDS and BAPS are two supervision objectives on one boundary branch.** They are enabled together in the combined configuration; their ablation order does not describe sequential training within one model. The boundary predictor is skipped at inference. The quality head runs in parallel with the original prediction heads rather than consuming their outputs in sequence.

The boundary target is the two-sided morphological boundary of the union of ground-truth masks at prototype resolution:

```text
U = union of ground-truth instance masks, resized to prototype resolution
T = Dilate_3x3(U) - Erode_3x3(U)
L_boundary = L_BDS + L_BAPS
L_total = L_base + 0.20 * L_boundary + 0.25 * L_quality
```

This target preserves foreground outlines and visible gaps. Interfaces between directly touching parcels disappear from the union, so the target supplies contour information without encoding complete instance adjacency. The original instance-mask objective continues to supervise individual parcels.

RBQS learns two sigmoid quality estimates from matched positive candidates: a **smoothed soft mask-IoU target** and a **tolerant Boundary-F1 target**. Both targets are detached before quality supervision. Boundary matching uses a tolerance of two prototype pixels. At inference:

```text
fused_score = class_score * sqrt(region_quality * boundary_quality)
```

Fusion can change ranking, threshold survival, and NMS outcomes. For fixed weights and a fixed candidate, it leaves the prototypes, coefficients, and reconstructed mask pixels unchanged. Quality-head training can still affect the shared feature extractor.

The contribution is the placement and integration of these objectives within a shared-prototype parcel segmentation pipeline. Boundary learning, Dice/BCE objectives, and learned mask scoring have established foundations: [Boundary-preserving Mask R-CNN](https://doi.org/10.1007/978-3-030-58568-6_39), [V-Net](https://arxiv.org/abs/1606.04797), and [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241). BAPS-only is available for new experiments. The archived results do not include that experiment or separately trained region-only and boundary-only quality heads.

### Configurations and Checkpoints

| Manuscript configuration | Enabled objectives | Training option | Model definition | Archived checkpoint |
|---|---|---|---|---|
| **Baseline** | Base detection and mask losses | `--stage baseline` | [YOLO11x-seg](./src/ultralytics/cfg/models/11/yolo11x-seg.yaml) | [model/baseline/best.pt](./model/baseline/best.pt) |
| **A: BDS** | Base + BDS | `--stage pcsc` | [Boundary Dice model](./src/ultralytics/cfg/models/11/yolo11x-seg-boundary-dice.yaml) | [model/a/best.pt](./model/a/best.pt) |
| **B: BAPS-only** | Base + boundary pixel BCE, without boundary Dice | `--stage baps` | Same joint boundary model, `boundary_mode=bce` | Requires training |
| **A+B: BDS + BAPS** | Base + BDS + BAPS | `--stage baps_pcsc` | [Joint boundary model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas.yaml) | [model/ab/best.pt](./model/ab/best.pt) |
| **FieldBCQ** | Base + BDS + BAPS + RBQS | `--stage fieldbcq` | [Complete model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml) | [model/abc/best.pt](./model/abc/best.pt) |

Legacy CLI/output identifiers `pcsc` and `baps_pcsc` correspond to BDS and the joint boundary configuration. Historical names PCSC and RBQC correspond to BDS and RBQS. The summary CSV calls A+B `BAPS + BDS`; both names denote the same two losses. Existing model definitions, checkpoint folders, and Python class paths are retained for compatibility.

One checkpoint is included for each of the four archived configurations. Their identities have not been matched to all three historical rounds, so evaluating these files does not recreate the manuscript's repeated-run summary. The repository does not include all 12 historical main-experiment checkpoints, a BAPS-only checkpoint, or reference-model checkpoints.

## AgriParcel-MS Dataset

**AgriParcel-MS (Multi-Source Agricultural Parcel Collection)** combines selected subsets of FBIS, Fields of The World (FTW), and FHAPD. It contains **15,000 images**, **15,000 corresponding annotation files**, and **403,199 patch-level instance records** across **26 countries** and **eight nominal resolution categories (0.25–10 m)**. All parcels use one `field` class and normalized polygon annotations. Instance counts refer to records in image patches rather than geographically deduplicated physical fields.

### Sources and Fixed Split

| Source | Images | Instance records | Training | Validation | Test |
|---|---:|---:|---:|---:|---:|
| FBIS | 8,250 | 275,357 | 6,600 | 550 | 1,100 |
| FTW | 1,500 | 42,182 | 1,200 | 100 | 200 |
| FHAPD | 5,250 | 85,660 | 4,200 | 350 | 700 |
| **Total** | **15,000** | **403,199** | **12,000** | **1,000** | **2,000** |

The source proportions are **55% FBIS, 10% FTW, and 35% FHAPD in each split**. Training and validation samples come from the original source training/validation pools. Selected samples from an original source test pool remain exclusively in the fixed test set. Validation supports model development and checkpoint selection; the test list is reserved for final comparisons.

This protocol preserves original source test membership. It evaluates the specified test pools from the same three sources used for training; it does not constitute a country-held-out or source-held-out experiment. Independent spatial overlap auditing across sources additionally requires footprints or parcel identifiers and split manifests.

Source references: [FBIS](https://arxiv.org/abs/2504.02534), [FTW](https://doi.org/10.1609/aaai.v39i27.35034), and [FHAPD](https://doi.org/10.1016/j.isprsjprs.2025.01.034). FHAPD is available through the authors' [HBGNet repository](https://github.com/NanNanmei/HBGNet); archived filenames may retain `DDFHAPD`.

| Split | Images | Instance records | Mean instances/image |
|---|---:|---:|---:|
| Training | 12,000 | 353,352 | 29.45 |
| Validation | 1,000 | 28,657 | 28.66 |
| Test | 2,000 | 21,190 | 10.60 |
| **Total** | **15,000** | **403,199** | **26.88** |

The test set retains **83 empty-label images from the Xinjiang subset of FHAPD** as negative samples. Its lower instance density follows the retained source test membership; the split was not redrawn to match training density. Aggregate counts are also available in [dataset_split.csv](./results/dataset_split.csv).

### Nominal Spatial Resolutions

Resolution values describe the source imagery in metres, independently of the network input size.

| Nominal resolution | Images | Instance records | Training | Validation | Test |
|---:|---:|---:|---:|---:|---:|
| 0.25 m | 60 | 118 | 55 | 5 | 0 |
| 0.30 m | 769 | 1,740 | 248 | 21 | 500 |
| 0.50 m | 255 | 561 | 65 | 5 | 185 |
| 1.00 m | 3,983 | 69,594 | 2,917 | 243 | 823 |
| 1.20 m | 287 | 2,367 | 231 | 19 | 37 |
| 2.00 m | 1,632 | 20,876 | 1,448 | 121 | 63 |
| 3.00 m | 1,778 | 41,977 | 1,519 | 126 | 133 |
| 10.00 m | 6,236 | 265,966 | 5,517 | 460 | 259 |
| **Total** | **15,000** | **403,199** | **12,000** | **1,000** | **2,000** |

FBIS retains nominal resolution metadata from filenames. FTW belongs to the nominal 10 m Sentinel-2 category. The local FHAPD subsets use 1 m for Hebei, Henan, Jiangsu, Sichuan, Hunan, and Xinjiang, and 2 m for Heilongjiang (local code `DB`), Chongqing, and Gansu.

The test set covers seven resolution categories: **0.25 m has no test samples**, and **10 m accounts for 46.0% of training images versus 13.0% of test images**. Source, country, density, and resolution are coupled, so grouped results do not isolate a causal effect of spatial resolution.

### Country Distribution

China, France, Slovakia, and Ukraine contribute 5,250, 3,389, 2,885, and 1,869 images, respectively. The remaining 22 countries contribute 1,607 images. Cross-source country records are combined, with Corsica counted under France.

**Twenty-one countries contribute test images.** India, Kenya, Portugal, Rwanda, and South Africa occur only in training or validation. Uneven sample sizes limit country-level conclusions.

<details>
<summary>Full country distribution (manuscript Appendix A)</summary>

| Country | Images | Instance records | Training | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Austria | 244 | 11,515 | 154 | 13 | 77 |
| Belgium | 49 | 1,582 | 39 | 3 | 7 |
| Brazil | 42 | 171 | 35 | 2 | 5 |
| Cambodia | 9 | 4,337 | 7 | 1 | 1 |
| China | 5,250 | 85,660 | 4,200 | 350 | 700 |
| Croatia | 78 | 4,718 | 63 | 5 | 10 |
| Denmark | 93 | 1,287 | 77 | 6 | 10 |
| Estonia | 118 | 959 | 94 | 8 | 16 |
| Finland | 113 | 2,204 | 93 | 7 | 13 |
| France | 3,389 | 212,940 | 3,066 | 254 | 69 |
| Germany | 4 | 72 | 0 | 1 | 3 |
| India | 44 | 192 | 38 | 6 | 0 |
| Kenya | 4 | 5 | 4 | 0 | 0 |
| Latvia | 141 | 1,307 | 113 | 9 | 19 |
| Lithuania | 117 | 2,166 | 92 | 8 | 17 |
| Luxembourg | 33 | 1,588 | 21 | 2 | 10 |
| Netherlands | 145 | 4,528 | 100 | 9 | 36 |
| Portugal | 1 | 53 | 1 | 0 | 0 |
| Rwanda | 2 | 46 | 2 | 0 | 0 |
| Slovakia | 2,885 | 38,112 | 2,553 | 212 | 120 |
| Slovenia | 75 | 3,561 | 46 | 4 | 25 |
| South Africa | 15 | 186 | 15 | 0 | 0 |
| Spain | 121 | 7,929 | 82 | 6 | 33 |
| Sweden | 150 | 2,812 | 112 | 10 | 28 |
| Ukraine | 1,869 | 12,725 | 986 | 83 | 800 |
| Vietnam | 9 | 2,544 | 7 | 1 | 1 |
| **Total** | **15,000** | **403,199** | **12,000** | **1,000** | **2,000** |

</details>

### Annotation Format and Preparation

Obtain the selected imagery and labels subject to the original datasets' access conditions. The full collection and exact per-image split manifests are not bundled. Aggregate counts alone do not reconstruct the evaluated subset; retain verified source membership when preparing the data.

```text
data/
|-- datasets.yaml
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

Each image has a corresponding label file with the same stem. YOLO segmentation labels contain one normalized polygon per line; class `0` is `field`:

```text
0 x1 y1 x2 y2 ... xn yn
```

The [dataset configuration](./data/datasets.yaml) uses:

```yaml
train: images/train
val: images/val
test: images/test
nc: 1
names: ['field']
```

With `path` omitted, the YAML file's directory is the dataset root. Update the split entries for another storage location. Local images and labels are excluded from Git.

## Reported Results

### Main Experiment: Three Rounds

All results below use the fixed test set. Values are **mean ± sample standard deviation** over three experimental rounds. Mask AP is reported on a 0–1 scale; **0.01 equals one AP point**. Precision, recall, and F1 use the native evaluator's selected best-F1 operating point, independently of the confidence-0.25 analysis below.

| Configuration | Precision | Recall | F1 | mAP50 | mAP75 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.663 ± 0.012 | 0.600 ± 0.001 | 0.630 ± 0.005 | 0.611 ± 0.007 | 0.400 ± 0.008 | 0.390 ± 0.006 |
| A: BDS | 0.672 ± 0.004 | 0.606 ± 0.004 | 0.637 ± 0.004 | 0.622 ± 0.004 | 0.411 ± 0.002 | 0.400 ± 0.002 |
| A+B: BDS + BAPS | 0.673 ± 0.001 | 0.614 ± 0.003 | 0.642 ± 0.001 | 0.628 ± 0.004 | 0.415 ± 0.002 | 0.405 ± 0.003 |
| **FieldBCQ** | **0.694 ± 0.007** | **0.615 ± 0.002** | **0.652 ± 0.003** | **0.640 ± 0.004** | **0.434 ± 0.003** | **0.420 ± 0.004** |

Within-round mAP50–95 gains average **1.03 AP points** for BDS over the baseline, another **0.47 AP points** when adding BAPS, and **1.47 ± 0.15 AP points** for FieldBCQ over A+B. The complete-model gain over baseline is **2.97 ± 0.95 AP points**. Paired gains are computed before rounding the displayed means.

The final ablation step includes both quality-head training and inference fusion. Seed identifiers and resume histories are incompletely aligned across rounds, so the observed variation is descriptive and does not establish statistical significance. The main summary is provided in [metrics_summary.csv](./results/metrics_summary.csv).

### Identical-Weight Scoring Intervention

C0 disables quality fusion (`quality_blend = 0`); C2 enables it (`quality_blend = 1`). Each pair uses identical trained full-model weights and the same evaluation settings.

| Setting | mAP50 | mAP75 | mAP50–95 |
|---|---:|---:|---:|
| C0: fusion disabled | 0.6389 ± 0.0030 | 0.4273 ± 0.0030 | 0.4145 ± 0.0031 |
| C2: full RBQS | 0.6404 ± 0.0036 | 0.4339 ± 0.0031 | 0.4196 ± 0.0035 |
| **Paired gain (AP points)** | **0.14 ± 0.07** | **0.66 ± 0.05** | **0.51 ± 0.04** |

This isolates the contribution of inference scoring. C0 retains the representation learned with quality supervision, so the gain does not measure every effect of the quality branch.

### Fixed-Threshold Behavior

The supplementary comparison uses archived `baseline_seed0` and `abc_seed02` checkpoints from the third supplementary group at **confidence 0.25**, with same-class, one-to-one mask matching at **IoU ≥ 0.50**. Their complete training pairing has not been reconstructed; this is a descriptive comparison, separate from the identical-weight C0/C2 intervention.

| Metric | Baseline | FieldBCQ |
|---|---:|---:|
| Precision (%) ↑ | 62.67 | 67.01 |
| Recall (%) ↑ | 60.75 | 59.61 |
| F1 (%) ↑ | 61.69 | 63.09 |
| Boundary-F1 (%) ↑ | 63.90 | 64.87 |
| Boundary-band IoU (%) ↑ | 40.80 | 41.45 |
| Merge rate (%) ↓ | 8.98 | 8.31 |
| Oversegmentation rate (%) ↓ | 9.16 | 7.75 |
| Small-parcel recall (%) ↑ | 44.49 | 43.63 |
| Score–IoU ECE ↓ | 0.0729 | 0.0522 |

Boundary metrics use a two-pixel evaluation tolerance. Small parcels have mask area ≤ 1,024 pixels in the supplementary evaluation grid (10,373 test annotations); this is an image-space definition. Score–IoU ECE measures alignment with mask IoU over retained predictions, rather than binary correctness calibration.

Precision and structural errors improve alongside reduced overall and small-parcel recall. Since the quality factor lies in [0, 1], fusion can remove candidates at an unchanged numerical threshold. Deployment thresholds should be selected using validation data.

### Source and Resolution Dependence

For the same supplementary checkpoint pair at confidence 0.25:

| Source | Test images | Test instances | Baseline F1 (%) | FieldBCQ F1 (%) |
|---|---:|---:|---:|---:|
| FHAPD | 700 | 5,256 | 77.48 | 79.43 |
| FBIS | 1,100 | 9,608 | 59.21 | 59.95 |
| FTW | 200 | 6,326 | 52.04 | 53.67 |

F1 improves for all three sources, but other metrics vary: FTW Boundary-F1 falls from 74.23% to 73.98%, and FHAPD score–IoU ECE rises from 0.0560 to 0.0962.

| Nominal resolution | Test images | Baseline F1 (%) | FieldBCQ F1 (%) |
|---|---:|---:|---:|
| 0.25 m | 0 | N/A | N/A |
| 0.30 m | 500 | 68.08 | 64.02 |
| 0.50 m | 185 | 73.43 | 63.15 |
| 1.00 m | 823 | 69.20 | 71.26 |
| 1.20 m | 37 | 60.98 | 61.89 |
| 2.00 m | 63 | 77.12 | 79.26 |
| 3.00 m | 133 | 58.79 | 61.05 |
| 10.00 m | 259 | 52.85 | 54.60 |

**Submeter imagery is a failure condition at this operating point:** pooled F1 falls from **69.38% to 63.79%** over 685 images and 1,592 instances. Group AP is unavailable, so these F1 changes should not be interpreted as measured group AP changes. The results support improvement under the evaluated protocol rather than uniform gains across resolutions or unseen countries.

### Reference Models and Computational Cost

The manuscript also reports the following single-checkpoint native mask results on the same test set:

| Reference model | AP50 | AP75 | AP50–95 |
|---|---:|---:|---:|
| Mask R-CNN R50-FPN | 0.4757 | 0.2983 | 0.2917 |
| Mask2Former R50 | 0.5251 | 0.3168 | 0.3251 |
| YOLOv8x-seg | 0.6140 | 0.3925 | 0.3878 |
| YOLO26x-seg | 0.6369 | 0.4114 | 0.4030 |

These provide context rather than a fully aligned repeated benchmark. Mask2Former uses a 100-instance output cap versus 710 for the other references, and training histories are not fully aligned. At confidence 0.25, the YOLO26x-seg reference has F1 64.62%, above the selected FieldBCQ checkpoint's 63.09%.

Archived complexity records report **62.00 M parameters / 295.9 GFLOPs** for the baseline and **63.92 M / 304.2 GFLOPs** for FieldBCQ. Recorded pure-model latency changes from 8.7 to 9.0 ms/image, but incomplete timing conditions preclude end-to-end throughput or common-device deployment claims.

### Evidence and Reproducibility Scope

The figures above summarize manuscript-reported experiments. Historical boundary/structural measurements and supplementary measurements use incompletely aligned protocols; their absolute values must be compared within each series. The two public CSV files contain aggregate summaries rather than original test logs or per-image predictions.

Exact reproduction of the historical rounds additionally requires per-image split manifests, complete training/resume configurations, evaluator versions, and identified checkpoint hashes. The commands below support new experiments with the current implementation. Geographic/source holdouts, independent overlap auditing, and a fully controlled repeated benchmark remain further evaluation needs.

## Setup and Experiments

### Environment

From the directory containing the project:

```powershell
cd ./FieldBCQ
```

Use Python 3.10 with PyTorch and torchvision versions suitable for your hardware. Install the local Ultralytics source:

```powershell
python -m pip install -e ./src
```

The public configurations use PyTorch layers only; optional experimental backbones and native extensions have been removed. The distribution and import name remain `ultralytics` for checkpoint compatibility. Use a dedicated environment to avoid replacing another Ultralytics installation. Package dependencies are listed in [pyproject.toml](./src/pyproject.toml). A complete pinned training environment is not included. Check that Python imports the project's custom implementation:

```powershell
python -c "import ultralytics; print(ultralytics.__file__)"
```

The path should point to this project's `src/ultralytics` directory. Training and evaluation arguments resolve relative to the project root, including when the scripts are launched from another directory.

### Training

Use [train_paper.py](./src/scripts/train_paper.py) with the training option in the [configuration table](#configurations-and-checkpoints). For pretrained initialization, prepare the original YOLO11x-seg pretrained weights at `./model/pretrained/yolo11x-seg.pt`. This pretrained file is separate from the four included trained checkpoints and is not bundled.

To train the complete FieldBCQ configuration:

```powershell
python src/scripts/train_paper.py --stage fieldbcq --data data/datasets.yaml --weights model/pretrained/yolo11x-seg.pt --epochs 80 --batch 8 --imgsz 640 --seed 0 --project results/train --name fieldbcq_seed0
```

Select `--stage baseline`, `--stage pcsc`, or `--stage baps_pcsc` for the corresponding comparison, and use a matching run name. `--no-pretrained` trains from random initialization, which is a different experimental setting.

Current training defaults are:

| Setting | Value |
|---|---|
| Input size | 640 × 640 |
| Epochs / batch size | 80 / 8 |
| Optimizer / precision | `auto` / AMP enabled |
| Mosaic closing interval | Final 10 epochs |
| Boundary / quality loss weights | 0.20 / 0.25 |
| Quality-target boundary tolerance | 2 prototype pixels |

These defaults do not establish the complete settings of the historical rounds. The command launches a new training run; it is not a verified reproduction of the historical paper runs. For repeated experiments, retain the seed, full arguments, code revision, and checkpoint hash. Select checkpoints on the validation set and evaluate the fixed test set afterward.

### Test Evaluation

Use [val_paper.py](./src/scripts/val_paper.py) for routine evaluation. It makes one native validation pass per checkpoint and writes a single `paper_metrics.csv` containing mask P, R, F1, AP50, AP75 and AP50–95 (all on a 0–1 scale). Plots, secondary prediction passes, CPU contour/area analysis and efficiency benchmarks are disabled by default. Validation without plots does not check or download plotting fonts.

```powershell
python src/scripts/val_paper.py --weights baseline_seed0=results/train/baseline_seed0/weights/best.pt --data data/datasets.yaml --split test --device 0 --imgsz 640 --batch 8
```

Repeat `--weights NAME=CHECKPOINT` to evaluate multiple models into the same CSV. Results go into a new timestamped directory under `results/validation/`; `--name` chooses a new directory explicitly. A small `validation_config.json` retains seeds read from checkpoints, checkpoint hashes, evaluator version, runtime versions and evaluation settings. Existing result directories are never overwritten. Use `--plots` only when plots are needed. The default is FP32, square letterbox input (`rect=False`), confidence floor 0.001, NMS IoU 0.70 and at most 710 instances per image, matching the native-AP settings of the full collector.

Already have a full report? Extract its main-table values without loading a model or rerunning validation:

```powershell
python src/scripts/val_paper.py --report results/paper_round2/round2_exp/paper_report.json
```

Repeat `--report` to collect multiple reports. F1 is the harmonic mean of native mask P/R at the framework operating point; it is not the full collector's fixed-confidence F1. Do not mix the two definitions. New evaluations of individual checkpoints do not recreate the paper's three-run summaries. The older `val_paper_all_models.py` remains available for the four archived checkpoints.

For country- or nominal-resolution evaluation, prepare a dataset YAML referencing the verified test image list for each group and pass it through `--data`. Use the same group membership for every configuration.

For supplementary boundary, merge, oversegmentation, small-parcel, calibration and agricultural area/count metrics, run `collect_paper_metrics.py` when those measurements are needed; it performs additional inference and CPU analysis. Use `benchmark_paper.py` separately for efficiency measurements. These are supplementary workflows rather than the routine main-table validation command.

Scripts store writable Ultralytics settings under `./results/runtime_config` unless `YOLO_CONFIG_DIR` is already defined.

### Verification and Outputs

Check path portability, configuration construction, inference, and synthetic training losses:

```powershell
python -m unittest discover -s tests -v
```

Training writes checkpoints, metrics, and plots into each run directory. The `results` directory also contains the paper's [performance summary](./results/metrics_summary.csv) and [dataset split counts](./results/dataset_split.csv). Archive selected outputs together with their run metadata after each experiment.

## Repository Structure

```text
FieldBCQ/
|-- README.md
|-- data/
|   `-- datasets.yaml                  # AgriParcel-MS configuration
|-- model/                            # four archived checkpoints (Git LFS)
|-- src/
|   |-- scripts/                      # training, evaluation, and analysis
|   |-- ultralytics/                  # base framework and public FieldBCQ implementation
|   |   |-- nn/modules/head.py        # RBQS head, BASMQCSegment
|   |   |-- nn/extra_modules/head.py  # historical boundary-head import paths
|   |   `-- utils/loss.py             # BDS, BAPS + BDS, and FieldBCQ losses
|   |-- README.md                     # Python package description
|   |-- pyproject.toml
|   `-- LICENSE
|-- tests/                            # portability and model regression tests
`-- results/
    |-- metrics_summary.csv
    `-- dataset_split.csv
```

Only `baseline`, `pcsc`, `baps_pcsc`, and `fieldbcq` are accepted by the training script. Archived checkpoint folders `a`, `ab`, and `abc`, and the original Python class paths, are retained so existing weights continue to load. Upstream framework utilities remain available to support training, validation, prediction, and checkpoint loading; unrelated experimental backbones and native extensions are excluded.

Experiment output directories such as `./results/train` and `./results/runs` are created when training or evaluation is run. Generated outputs and temporary runtime settings are ignored by Git.

To also verify inference from all four archived checkpoints:

```powershell
python src/scripts/check_paper_models.py --checkpoints
```

## Data Access and License

FBIS, FTW, and FHAPD imagery and annotations remain subject to their original access and usage conditions. AgriParcel-MS describes the selected, harmonized collection used in the manuscript.

The local Ultralytics distribution declares **AGPL-3.0-or-later**; see [src/LICENSE](./src/LICENSE) and [src/pyproject.toml](./src/pyproject.toml).
