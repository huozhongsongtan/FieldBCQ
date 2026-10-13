# FieldBCQ

**Boundary Supervision, Contour Consistency, and Quality Calibration for Multi-Region Farmland Parcel Instance Segmentation**

FieldBCQ is a farmland parcel instance segmentation framework built on YOLO11x-seg. It combines **Boundary-Aware Pixel Supervision (BAPS)**, **Parcel Contour Structural Consistency Constraint (PCSC)**, and **Region-Boundary Quality Calibration (RBQC)** to supervise parcel boundaries and calibrate instance confidence.

The study uses **AgriParcel-MR26: A Multi-Country, Multi-Resolution Farmland Parcel Instance Segmentation Dataset**, containing **15,000 remote-sensing images**, **15,000 matching annotation files**, and **403,199 parcel instances** across **26 countries**. It spans **8 sensor spatial resolution categories**, ranging from **0.25 m to 10 m**.

On the fixed AgriParcel-MR26 test split, the paper reports a three-run mean **Mask mAP50-95 of 0.420 +/- 0.004** for FieldBCQ. This repository includes the implementation, four trained checkpoints, dataset statistics, paper-reported performance summaries, and bilingual architecture figures.

## Method

### Three Improvement Mechanisms

| Mechanism | Full Name | Role in FieldBCQ |
|---|---|---|
| **BAPS** | Boundary-Aware Pixel Supervision | Uses pixel-wise BCE supervision to distinguish boundary pixels in an auxiliary boundary prediction map. |
| **PCSC** | Parcel Contour Structural Consistency Constraint | Uses Boundary Dice to encourage overlap consistency between the predicted boundary map and its target. |
| **RBQC** | Region-Boundary Quality Calibration | Estimates region and boundary quality and uses their geometric mean to recalibrate class confidence before NMS. |

BAPS and PCSC supervise the same auxiliary boundary branch, which takes mask prototypes as input and operates during training. The boundary target is constructed from the union of ground-truth instance masks using dilation and erosion. PCSC denotes the Boundary Dice objective in this implementation; it does not introduce a separate contour encoder or topology loss.

RBQC operates on the three feature scales used by the segmentation head. Its region and boundary quality targets are detached Soft IoU and tolerant Boundary-F1 values computed from matched predicted and ground-truth instance masks. At inference, the predicted qualities calibrate confidence before box NMS, and the retained mask coefficients and prototypes decode parcel masks.

### Configurations and Checkpoints

The four configurations use the same YOLO11x-seg backbone. Their names are used consistently in the method description, results, and experiment commands.

| Configuration | Added Mechanisms | Training Option | Model Definition | Trained Checkpoint |
|---|---|---|---|---|
| **Baseline** | None | `--stage baseline` | [YOLO11x-seg](./src/ultralytics/cfg/models/11/yolo11x-seg.yaml) | [Baseline checkpoint](./model/baseline/best.pt) |
| **PCSC** | PCSC | `--stage pcsc` | [Contour supervision model](./src/ultralytics/cfg/models/11/yolo11x-seg-boundary-dice.yaml) | [PCSC checkpoint](./model/a/best.pt) |
| **BAPS + PCSC** | BAPS and PCSC | `--stage baps_pcsc` | [Joint boundary supervision model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas.yaml) | [BAPS + PCSC checkpoint](./model/ab/best.pt) |
| **FieldBCQ** | BAPS, PCSC, and RBQC | `--stage fieldbcq` | [Complete FieldBCQ model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml) | [FieldBCQ checkpoint](./model/abc/best.pt) |

Each configuration includes one trained checkpoint. The individual checkpoints have not been matched to the paper's three repeated runs, so the reported three-run results are presented separately from new evaluations of these files. The repository does not contain all 12 checkpoints used for the repeated-run summaries or checkpoints for comparison methods.

## AgriParcel-MR26 Dataset

**AgriParcel-MR26: A Multi-Country, Multi-Resolution Farmland Parcel Instance Segmentation Dataset** is the dataset used throughout this study. It contains **15,000 images**, **15,000 corresponding annotation files**, and **403,199 farmland parcel instances**. Each parcel is represented by an instance polygon, with one instance per annotation line.

The name reflects the dataset's defining properties: **AgriParcel** identifies agricultural parcel instances, **MR** denotes multiple sensor spatial resolutions, and **26** denotes the number of countries. Its eight sensor-resolution categories span 0.25-10 m across varied agricultural landscapes.

### Dataset Split

| Split | Images | Parcel Instances | Image Share | Purpose |
|---|---:|---:|---:|---|
| Train | 12,000 | 353,352 | 80.0% | Model training |
| Validation | 1,000 | 28,657 | 6.7% | Hyperparameter and checkpoint selection |
| Test | 2,000 | 21,190 | 13.3% | Final evaluation |
| **Total** | **15,000** | **403,199** | **100%** | |

The test split includes **83 empty-label images**, which remain in the evaluation set as negative samples. The paper uses the `test.cache` counting convention. Keep the same image membership when comparing configurations. Split counts are also provided in [dataset_split.csv](./results/dataset_split.csv).

### Sensor Spatial Resolutions

Resolution values refer to **sensor spatial resolution in metres**.

| Sensor Spatial Resolution | Images | Parcel Instances |
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

Equivalent units are consolidated: 1 m and 100 cm are counted together, as are 3 m and 300 cm.

### Country Distribution

AgriParcel-MR26 covers **26 countries**. Regional identifiers are consolidated at the country level; Corsica is counted under France.

| Country | Images | Parcel Instances | Country | Images | Parcel Instances |
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

The geographic coverage and sensor-resolution range provide varied conditions for evaluating parcel segmentation across agricultural landscapes.

### Annotation Format and Preparation

The full AgriParcel-MR26 image and annotation files are not bundled in this repository. Prepare matching image and label filenames under the following structure:

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

Labels use YOLO segmentation format, with one parcel polygon per line:

```text
0 x1 y1 x2 y2 ... xn yn
```

Coordinates are normalized to image dimensions. Class `0` represents a farmland parcel. The [AgriParcel-MR26 configuration](./data/datasets.yaml) contains:

```yaml
train: images/train
val: images/val
test: images/test
nc: 1
names: ['field']
```

The split paths are relative to the YAML file's directory. The `path` key is omitted so Ultralytics uses that directory as the dataset root. For a different storage location, update the split entries accordingly. Local images and labels are excluded from Git.

## Reported Results

The following **Mask metrics** are reported for the fixed AgriParcel-MR26 test split of **2,000 images** and **21,190 parcel instances**. Values are **mean +/- sample standard deviation** over three independent runs.

| Configuration | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.663 +/- 0.012 | 0.600 +/- 0.001 | 0.630 +/- 0.005 | 0.611 +/- 0.007 | 0.400 +/- 0.008 | 0.390 +/- 0.006 |
| PCSC | 0.672 +/- 0.004 | 0.606 +/- 0.004 | 0.637 +/- 0.004 | 0.622 +/- 0.004 | 0.411 +/- 0.002 | 0.400 +/- 0.002 |
| BAPS + PCSC | 0.673 +/- 0.001 | 0.614 +/- 0.003 | 0.642 +/- 0.001 | 0.628 +/- 0.004 | 0.415 +/- 0.002 | 0.405 +/- 0.003 |
| **FieldBCQ** | **0.694 +/- 0.007** | **0.615 +/- 0.002** | **0.652 +/- 0.003** | **0.640 +/- 0.004** | **0.434 +/- 0.003** | **0.420 +/- 0.004** |

The reported paired Mask mAP50-95 improvement is **2.97 +/- 0.95 AP** over Baseline and **1.47 +/- 0.15 AP** over BAPS + PCSC. One AP corresponds to 0.01 in normalized mAP. Paired improvements are computed from individual runs before rounding the summary values.

The machine-readable results are provided in [metrics_summary.csv](./results/metrics_summary.csv). These are paper-reported summaries; original test logs and per-image predictions are not included.

## Setup and Experiments

### Environment

From the directory containing the project:

```powershell
cd ./FieldBCQ
```

Use Python 3.10 with PyTorch and torchvision versions suitable for your hardware. Install the local Ultralytics source and the additional packages used by its custom layers:

```powershell
python -m pip install -e ./src
python -m pip install timm einops
```

Package dependencies are listed in [pyproject.toml](./src/pyproject.toml). A complete pinned training environment is not included. Check that Python imports the project's custom implementation:

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

The command launches a new training run; it is not a verified reproduction of the historical paper runs. For repeated experiments, retain the seed, full arguments, code revision, and checkpoint hash. Select checkpoints on the validation split and evaluate the fixed test split afterward.

### Test Evaluation

After preparing AgriParcel-MR26 and the environment, evaluate all four included checkpoints with [val_paper_all_models.py](./src/scripts/val_paper_all_models.py):

```powershell
python src/scripts/val_paper_all_models.py --data data/datasets.yaml --split test --imgsz 640 --batch 8 --project results/runs
```

The evaluation names are `baseline`, `pcsc`, `baps_pcsc`, and `fieldbcq`. The script writes per-model `paper_metrics.json` files, a combined `paper_metrics_all.json`, and evaluation plots. Checkpoint and dataset references in the combined summary are relative to the project root.

For a single checkpoint, use [val_paper.py](./src/scripts/val_paper.py) with `--weights` pointing to a checkpoint from the configuration table and `--name` set to the configuration name. New evaluations of individual checkpoints do not recreate the paper's three-run summaries.

For country- or sensor-resolution evaluation, prepare a dataset YAML referencing the verified test image list for each group and pass it through `--data`. Use the same group membership for every configuration.

Scripts store writable Ultralytics settings under `./results/runtime_config` unless `YOLO_CONFIG_DIR` is already defined.

### Verification and Outputs

Check path portability and the documented configuration names without loading weights or running inference:

```powershell
python -m unittest discover -s tests -v
```

Training writes checkpoints, metrics, and plots into each run directory. The `results` directory also contains the paper's [performance summary](./results/metrics_summary.csv) and [dataset split counts](./results/dataset_split.csv). Archive selected outputs together with their run metadata after each experiment.

## Architecture Figures

The diagram follows the complete FieldBCQ model definition and its segmentation head. It shows the shared boundary prediction branch, the BAPS and PCSC training objectives, and the RBQC quality prediction and confidence calibration path.

![FieldBCQ architecture with BAPS, PCSC, and RBQC](./fig/fieldbcq_architecture_en.png)

| Version | High-Resolution Image | Editable Vector | Print-Ready PDF |
|---|---|---|---|
| English | [PNG](./fig/fieldbcq_architecture_en.png) | [SVG](./fig/fieldbcq_architecture_en.svg) | [PDF](./fig/fieldbcq_architecture_en.pdf) |
| Chinese | [PNG](./fig/fieldbcq_architecture_zh.png) | [SVG](./fig/fieldbcq_architecture_zh.svg) | [PDF](./fig/fieldbcq_architecture_zh.pdf) |

The [figure notes](./fig/fieldbcq_architecture_notes.md) provide bilingual captions, source-code mappings, and diagram simplifications. PNG files are exported at 600 dpi. Chinese SVG text requires a compatible CJK font when edited.

Regenerate both versions with [draw_fieldbcq_architecture.py](./fig/draw_fieldbcq_architecture.py):

```powershell
python fig/draw_fieldbcq_architecture.py --language both --dpi 600
```

## Repository Structure

```text
FieldBCQ/
|-- README.md
|-- data/
|   `-- datasets.yaml                  # AgriParcel-MR26 configuration
|-- model/                            # four trained configuration checkpoints
|-- src/
|   |-- scripts/                      # training, evaluation, and analysis
|   |-- ultralytics/                  # custom model heads, losses, and definitions
|   |-- pyproject.toml
|   `-- LICENSE
|-- tests/
|   `-- test_relative_paths.py
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

Experiment output directories such as `./results/train` and `./results/runs` are created when training or evaluation is run.
