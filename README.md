# FieldBCQ: Boundary Supervision, Contour Consistency, and Quality Calibration for Multi-Region Farmland Parcel Instance Segmentation

**A boundary-aware framework for farmland parcel instance segmentation in heterogeneous remote-sensing imagery.**

FieldBCQ extends YOLO11x-seg through three progressive improvement mechanisms: Boundary-Aware Pixel Supervision (BAPS), Parcel Contour Structural Consistency Constraint (PCSC), and Region-Boundary Quality Calibration (RBQC). BAPS improves boundary-pixel discrimination, PCSC strengthens overall contour consistency, and RBQC estimates regional and boundary quality to recalibrate instance confidence scores. Together, these improvements enhance boundary-aware segmentation and prediction ranking across heterogeneous imaging conditions.

The study uses **FieldSeg-15K: Global Multi-Region Farmland Parcel Instance Segmentation Dataset**, comprising **15,000 remote-sensing images**, **15,000 matching annotation files**, and **403,199 field instances** across **26 countries** and multiple sensor spatial resolutions. On the fixed test set, the reported three-run mean Mask mAP50-95 for FieldBCQ is **0.420 +/- 0.004**.

This repository provides the paper-aligned Baseline, PCSC ablation, BAPS + PCSC, and FieldBCQ configurations, together with their archived checkpoints, implementation code, architecture figures, and paper-reported evaluation summaries. The CLI and checkpoint aliases `baseline`, `a`, `ab`, and `abc` are retained for reproducibility and are mapped to the paper configurations under [Model Preparation](#model-preparation).

## Prerequisites

### Project Preparation

From the directory containing the project, enter `./FieldBCQ` before following the setup instructions:

```powershell
cd ./FieldBCQ
```

File and output-directory arguments in the training and test evaluation entry points are relative to the project root. These entry points locate that root from their own files, so moving the project or launching them from another working directory does not change these references. Dataset split entries are relative to the directory containing `./data/datasets.yaml`.

### Environment Setup

Use Python 3.10 with a PyTorch and torchvision installation suitable for your hardware. Install the local Ultralytics source and the additional Python packages imported by its custom layers:

```powershell
python -m pip install -e ./src
python -m pip install timm einops
```

Dependencies declared by the local package are listed in [pyproject.toml](./src/pyproject.toml). Custom modules can require additional packages; the archive does not contain a complete pinned training environment. For historical reproduction, reuse the environment associated with the original checkpoint.

The custom heads and losses are part of the local source. Check that Python imports this copy:

```powershell
python -c "import ultralytics; print(ultralytics.__file__)"
```

The printed path should point to this project's `src/ultralytics` directory.

Check path handling without loading model weights or running inference:

```powershell
python -m unittest discover -s tests -v
```

## Model

### Paper Configurations

The paper evaluates four progressive configurations:

| Paper configuration | Components | Purpose |
|---|---|---|
| Baseline | YOLO11x-seg | Reference instance segmentation model |
| PCSC ablation | Parcel Contour Structural Consistency Constraint (PCSC) | Preliminary contour supervision ablation |
| BAPS + PCSC | Boundary-Aware Pixel Supervision (BAPS) + Parcel Contour Structural Consistency Constraint (PCSC) | Joint boundary representation |
| FieldBCQ | BAPS + PCSC + Region-Boundary Quality Calibration (RBQC) | Boundary representation and quality-aware confidence calibration |

### Model Preparation

The following trained checkpoints are included under `./model`. Model YAML files are stored under `./src/ultralytics/cfg/models/11`.

| Paper Configuration | CLI Stage | Model Configuration | Trained Checkpoint |
|---|---|---|---|
| Baseline | `baseline` | [yolo11x-seg.yaml](./src/ultralytics/cfg/models/11/yolo11x-seg.yaml) | [best.pt](./model/baseline/best.pt) |
| PCSC ablation | `a` | [yolo11x-seg-boundary-dice.yaml](./src/ultralytics/cfg/models/11/yolo11x-seg-boundary-dice.yaml) | [best.pt](./model/a/best.pt) |
| BAPS + PCSC | `ab` | [yolo11x-seg-bas.yaml](./src/ultralytics/cfg/models/11/yolo11x-seg-bas.yaml) | [best.pt](./model/ab/best.pt) |
| FieldBCQ | `abc` | [yolo11x-seg-bas-mqc.yaml](./src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml) | [best.pt](./model/abc/best.pt) |

The paper evaluates three progressive improvement mechanisms: BAPS, PCSC, and RBQC. The `abc` checkpoint combines all three mechanisms and represents the complete FieldBCQ configuration. The archived implementation realizes BAPS with BCE and PCSC with Boundary Dice and uses internal names such as `BAS` and `MQC`; these names are retained in the code and file paths for reproducibility. A checkpoint-to-paper-run mapping has not been verified, so the paper's three-run test values are reported separately from these archived checkpoints.

The archive contains one `best.pt` per stage. It does not contain all 12 checkpoints used for the paper's three-run summaries or checkpoints for comparison models.

## Dataset Preparation

### FieldSeg-15K Dataset

FieldSeg-15K contains **15,000 remote-sensing images**, **15,000 matching annotation files**, and **403,199 field instances**. Each field parcel is represented as an independent instance polygon, and each annotation line corresponds to one field instance. The dataset covers multiple countries, agricultural landscapes, and sensor spatial resolutions, providing a unified benchmark for multi-region field instance segmentation.

The study uses the following split:

| Split | Images | Field Instances | Share | Use |
|---|---:|---:|---:|---|
| Train | 12,000 | 353,352 | 80.0% | Parameter learning |
| Val | 1,000 | 28,657 | 6.7% | Hyperparameter and checkpoint selection |
| Test | 2,000 | 21,190 | 13.3% | Final evaluation |
| **Total** | **15,000** | **403,199** | **100%** | |

The fixed test set includes **83 empty-label images**. These negative samples remain in the evaluation set, and the paper uses the `test.cache` counting convention. Retain the same image list when comparing results.

The split counts are also available in [dataset_split.csv](./results/dataset_split.csv).

The sensor spatial-resolution distribution is:

| Sensor spatial resolution | Images | Field instances |
|---:|---:|---:|
| 0.25 m | 60 | 118 |
| 0.30 m | 769 | 1,740 |
| 0.50 m | 255 | 561 |
| 1.00 m | 285 | 4,012 |
| 1.20 m | 287 | 2,367 |
| 2.00 m | 80 | 798 |
| 3.00 m | 1,778 | 41,977 |
| 10.00 m | 11,486 | 351,626 |
| **Total** | **15,000** | **403,199** |

The 1.00 m category combines 1 m and 100 cm products; the 3.00 m category combines 3 m and 300 cm products. The country distribution is:

| Country | Images | Field instances | Country | Images | Field instances |
|---|---:|---:|---|---:|---:|
| China | 5,250 | 85,660 | France | 3,389 | 212,940 |
| Slovakia | 2,885 | 38,112 | Ukraine | 1,869 | 12,725 |
| Austria | 244 | 11,515 | Netherlands | 145 | 4,528 |
| Sweden | 150 | 2,812 | Latvia | 141 | 1,307 |
| Spain | 121 | 7,929 | Estonia | 118 | 959 |
| Lithuania | 117 | 2,166 | Finland | 113 | 2,204 |
| Denmark | 93 | 1,287 | Croatia | 78 | 4,718 |
| Slovenia | 75 | 3,561 | Belgium | 49 | 1,582 |
| India | 44 | 192 | Brazil | 42 | 171 |
| Luxembourg | 33 | 1,588 | South Africa | 15 | 186 |
| Cambodia | 9 | 4,337 | Vietnam | 9 | 2,544 |
| Germany | 4 | 72 | Kenya | 4 | 5 |
| Rwanda | 2 | 46 | Portugal | 1 | 53 |

The dataset covers **26 countries** after consolidating regional identifiers; the Corsica region is counted under France. This geographic and sensor diversity supports evaluation of model generalization across countries, agricultural landscapes, and image resolutions.

### Dataset Configuration

Prepare the dataset with matching image and label filenames:

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

Each label file uses YOLO segmentation format, with one field instance per line:

```text
0 x1 y1 x2 y2 ... xn yn
```

Coordinates are normalized to the image dimensions. Class `0` represents a field parcel.

Place the images and polygon labels under `./data/images` and `./data/labels`. The [dataset configuration](./data/datasets.yaml) uses the following relative entries and field class name:

```yaml
train: images/train
val: images/val
test: images/test
nc: 1
names: ['field']
```

The `path` key is intentionally omitted so Ultralytics uses the YAML file's directory as the dataset root. For another dataset location, edit the three split entries relative to that directory. The full dataset is not bundled and local images and labels are excluded from Git.

## Training

### Main Configurations

The training entry point is [train_paper.py](./src/scripts/train_paper.py). The archive includes model YAML files for `baseline`, `a`, `ab`, and `abc`; use these four stage names.

For pretrained initialization, prepare an original YOLO11x-seg pretrained checkpoint at `./model/pretrained/yolo11x-seg.pt`. This pretrained file and directory are not included in the archive. The stage-specific `best.pt` files listed above are trained checkpoints.

Run the following from the project directory:

```powershell
python src/scripts/train_paper.py --stage baseline --data data/datasets.yaml --weights model/pretrained/yolo11x-seg.pt --epochs 80 --batch 8 --imgsz 640 --seed 0 --project results/train --name baseline_seed0
```

Replace `baseline` with `a`, `ab`, or `abc` and assign a corresponding run name. The command shows the current training entry point; it is not a verified reproduction of the historical paper runs. Training from random initialization is available through `--no-pretrained` and constitutes a different experimental setting.

### Training Outputs

Ultralytics writes checkpoints, metrics, and plots together inside the generated training run directory. After a run is complete, archive the selected checkpoint under `./model/<stage>`, numerical outputs under `./results`, and figures under `./fig/<stage>`.

For repeated experiments, retain the seed, full run arguments, code revision, and checkpoint hash for each run. Select checkpoints using the validation split and evaluate the fixed test set afterward.

## Results

### Test-Set Performance

The following Mask metrics are reported in the research material for the fixed test set of **2,000 images** and **21,190 instances**. Values are **mean +/- sample standard deviation** over three independent runs; they are not 95% confidence intervals.

| Configuration | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.663 +/- 0.012 | 0.600 +/- 0.001 | 0.630 +/- 0.005 | 0.611 +/- 0.007 | 0.400 +/- 0.008 | 0.390 +/- 0.006 |
| Contour-overlap ablation | 0.672 +/- 0.004 | 0.606 +/- 0.004 | 0.637 +/- 0.004 | 0.622 +/- 0.004 | 0.411 +/- 0.002 | 0.400 +/- 0.002 |
| Combined boundary supervision | 0.673 +/- 0.001 | 0.614 +/- 0.003 | 0.642 +/- 0.001 | 0.628 +/- 0.004 | 0.415 +/- 0.002 | 0.405 +/- 0.003 |
| FieldBCQ: three-mechanism configuration | **0.694 +/- 0.007** | **0.615 +/- 0.002** | **0.652 +/- 0.003** | **0.640 +/- 0.004** | **0.434 +/- 0.003** | **0.420 +/- 0.004** |

The reported paired Mask mAP50-95 gain is **2.97 +/- 0.95 AP** over Baseline and **1.47 +/- 0.15 AP** over the combined boundary supervision configuration. Here, an increase of 1 AP means an increase of 0.01 in the normalized mAP value.

The machine-readable table is available in [metrics_summary.csv](./results/metrics_summary.csv). These are paper-reported summaries; original test logs and per-image predictions are not included.

### Test Evaluation

To evaluate an included checkpoint, run the following from the project root after completing dataset and environment preparation:

```powershell
python src/scripts/val_paper.py --weights model/abc/best.pt --data data/datasets.yaml --split test --imgsz 640 --batch 8 --max-det 710 --project results/runs --name abc
```

Replace `abc` with the stage to evaluate. The script generates `paper_metrics.json`, evaluation curves, confusion matrices, and batch visualizations in the run directory. This is a new evaluation of one archived checkpoint; it does not regenerate the three-run paper table.

For sequential evaluation, [val_paper_all_models.py](./src/scripts/val_paper_all_models.py) defaults to the four checkpoints under `./model`. Override them with repeated `--weights NAME=PATH` arguments when needed:

```powershell
python src/scripts/val_paper_all_models.py --data data/datasets.yaml --project results/runs
```

Scripts default to `./results/runtime_config` for writable Ultralytics settings. An existing `YOLO_CONFIG_DIR` environment variable takes precedence. The generated multi-model summary stores checkpoint and data references relative to the project root.

### Source-Specific Evaluation

For source, domestic, and foreign subsets, run [val_by_source.py](./src/scripts/val_by_source.py) from the project root:

```powershell
python src/scripts/val_by_source.py --weights model/abc/best.pt --data data/datasets.yaml --split test --sources DDFHAPD,FBIS,FTW --max-det 710 --project results/source_eval --name abc
```

The script groups images using source tokens in their paths or filenames. Preserve these tokens or configure `--source-token` for your dataset. Generated group YAML files and image lists also contain relative paths. Source-based evaluation alone does not recreate the sensor and resolution groups reported in the paper.

### Results Archive

The current `./results` directory contains:

| File | Contents |
|---|---|
| [metrics_summary.csv](./results/metrics_summary.csv) | Paper-reported mean and standard deviation for four configurations |
| [dataset_split.csv](./results/dataset_split.csv) | Train, validation, and test image and instance counts |

Paper result source: the research document titled `FieldSeg-15K_全球多区域田块实例分割研究内容`. Its test summaries are archived in the CSV files above; the implementation and figures are included under `./src` and `./fig`.

## Figures

### FieldBCQ Architecture (Archived `abc`)

The following diagram follows the archived `yolo11x-seg-bas-mqc.yaml` graph and the `BASMQCSegment` head. The boundary supervision branch implements BAPS and PCSC during training; the archived implementation uses BCE and Boundary Dice for these objectives. RBQC predicts Mask-IoU and Boundary-F1 quality and uses their geometric mean to calibrate class confidence before NMS.

![YOLO11x-seg FieldBCQ architecture](./fig/fieldbcq_architecture_en.png)

Vector and print-ready versions are available as [SVG](./fig/fieldbcq_architecture_en.svg) and [PDF](./fig/fieldbcq_architecture_en.pdf). A Chinese-labelled version is available as [PNG](./fig/fieldbcq_architecture_zh.png), [SVG](./fig/fieldbcq_architecture_zh.svg), and [PDF](./fig/fieldbcq_architecture_zh.pdf).

The [figure notes](./fig/fieldbcq_architecture_notes.md) include bilingual captions, source-code mappings, and the simplifications used in this module-level diagram. PNG exports use 600 dpi; Chinese SVG text requires a compatible CJK font when edited.

Regenerate the figures with [draw_fieldbcq_architecture.py](./fig/draw_fieldbcq_architecture.py):

```powershell
python fig/draw_fieldbcq_architecture.py --language both --dpi 600
```

### Figure Generation

New evaluations with `val_paper.py` enable plots automatically. Preserve the numerical outputs under `./results` and move selected test figures to a separate `./fig/test` directory when they are available. No test images are included in the current figure archive.

The legacy [plot_result.py](./src/scripts/plot_result.py) uses a manually configured run list and historical paths. Update those inputs before using it to plot new experiments.

## Project Structure

```text
FieldBCQ/
|-- README.md
|-- tests/
|   `-- test_relative_paths.py    # path portability checks
|-- src/
|   |-- scripts/                  # training, evaluation, analysis, and figures
|   |-- ultralytics/              # local source, custom heads, and model YAMLs
|   |-- pyproject.toml
|   `-- LICENSE
|-- model/
|   |-- baseline/best.pt
|   |-- a/best.pt
|   |-- ab/best.pt
|   `-- abc/best.pt
|-- data/
|   `-- datasets.yaml             # dataset configuration template
|-- results/
|   |-- metrics_summary.csv
|   `-- dataset_split.csv
`-- fig/
    |-- draw_fieldbcq_architecture.py
    |-- fieldbcq_architecture_notes.md
    |-- fieldbcq_architecture_en.png
    |-- fieldbcq_architecture_zh.png
    |-- fieldbcq_architecture_*.svg
    `-- fieldbcq_architecture_*.pdf
```

The structure above describes the current archive. Directories such as `./results/train`, `./results/runs`, and `./fig/test` are created only when new experiments are run and archived.
