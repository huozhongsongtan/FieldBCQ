# FieldBCQ

**Boundary Supervision, Contour Consistency, and Quality Calibration for Multi-Region Farmland Parcel Instance Segmentation**

FieldBCQ is a farmland parcel instance segmentation framework built on YOLO11x-seg. Its three improvements target the segmentation head, the training loss, and inference-time post-processing: **Boundary-Aware Pixel Supervision (BAPS)**, **Parcel Contour Structural Consistency Constraint (PCSC)**, and **Region-Boundary Quality Calibration (RBQC)**.

The study uses **AgriParcel-MR26: A Multi-Country, Multi-Resolution Farmland Parcel Instance Segmentation Dataset**, containing **15,000 remote-sensing images**, **15,000 matching annotation files**, and **403,199 parcel instances** across **26 countries**. It spans **8 sensor spatial resolution categories**, ranging from **0.25 m to 10 m**.

On the fixed AgriParcel-MR26 test split, the paper reports a three-run mean **Mask mAP50-95 of 0.420 +/- 0.004** for FieldBCQ. This repository includes the implementation, four trained checkpoints, dataset statistics, and paper-reported performance summaries.

## Method

### Three Improvement Mechanisms

The three contributions are classified by their primary effect in the implemented method:

| Mechanism | Primary Category | Implementation and Effect |
|---|---|---|
| **BAPS** | **Segmentation-head enhancement** | Adds an auxiliary boundary prediction branch to the segmentation head. The branch predicts a boundary map from mask prototypes and is trained with pixel-wise BCE, improving boundary-pixel discrimination. |
| **PCSC** | **Contour-consistency loss** | Adds a Boundary Dice loss between the predicted boundary map and the ground-truth boundary target, encouraging boundary overlap and contour consistency. |
| **RBQC** | **Quality-aware inference post-processing** | Uses region and boundary quality estimates to calibrate instance confidence before NMS, so suppression and ranking use quality-aware scores. |

**BAPS: segmentation-head enhancement.** The boundary branch takes mask prototypes as input and operates during training. Its auxiliary prediction is supervised with BCE on a boundary target constructed by applying dilation and erosion to the union of ground-truth instance masks. This adds a boundary prediction task alongside parcel detection and mask prediction.

**PCSC: contour-consistency loss.** PCSC is implemented as a Boundary Dice term on the same boundary prediction and target used by BAPS. It improves overlap consistency at the mask-boundary level. It does not add a separate network module or explicitly construct pairwise boundaries between individual parcels.

**RBQC: quality-aware inference post-processing.** A multi-scale quality prediction head is trained with detached Soft IoU and tolerant Boundary-F1 targets computed from matched predicted and ground-truth masks. At inference, the predicted region and boundary qualities are combined by their geometric mean to calibrate class confidence before box NMS. Thus, its primary contribution is confidence calibration at the inference decision stage, supported by a learned quality head and its training loss.

Together, the primary categories are segmentation-head enhancement (BAPS), contour-consistency loss (PCSC), and quality-aware post-processing (RBQC). The method does not modify the dataset or annotations, backbone, neck, or label-assignment strategy.

### Configurations and Checkpoints

The four configurations use the same YOLO11x-seg backbone. Their names are used consistently in the method description, results, and experiment commands.

| Configuration | Primary Improvement Category | Mechanisms | Training Option | Model Definition | Trained Checkpoint |
|---|---|---|---|---|---|
| **Baseline** | Reference | None | `--stage baseline` | [YOLO11x-seg](./src/ultralytics/cfg/models/11/yolo11x-seg.yaml) | [Baseline checkpoint](./model/baseline/best.pt) |
| **PCSC** | Loss function | PCSC | `--stage pcsc` | [Contour supervision model](./src/ultralytics/cfg/models/11/yolo11x-seg-boundary-dice.yaml) | [PCSC checkpoint](./model/a/best.pt) |
| **BAPS + PCSC** | Segmentation head + loss function | BAPS, PCSC | `--stage baps_pcsc` | [Joint boundary supervision model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas.yaml) | [BAPS + PCSC checkpoint](./model/ab/best.pt) |
| **FieldBCQ** | Segmentation head + loss function + inference post-processing | BAPS, PCSC, RBQC | `--stage fieldbcq` | [Complete FieldBCQ model](./src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml) | [FieldBCQ checkpoint](./model/abc/best.pt) |

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

AgriParcel-MR26 covers **26 countries**. To keep the summary consistent with the distribution chart and avoid overemphasizing very small groups, the main table reports the 10 countries with the most images individually and aggregates the remaining 16 countries. Regional identifiers are consolidated at the country level; Corsica is counted under France.

| Country / group | Images | Parcel Instances |
|---|---:|---:|
| China | 5,250 | 85,660 |
| France | 3,389 | 212,940 |
| Slovakia | 2,885 | 38,112 |
| Ukraine | 1,869 | 12,725 |
| Austria | 244 | 11,515 |
| Sweden | 150 | 2,812 |
| Netherlands | 145 | 4,528 |
| Latvia | 141 | 1,307 |
| Spain | 121 | 7,929 |
| Estonia | 118 | 959 |
| Other 16 countries | 688 | 24,712 |
| **Total** | **15,000** | **403,199** |

The **Other 16 countries** group contains Lithuania, Finland, Denmark, Croatia, Slovenia, Belgium, India, Brazil, Luxembourg, South Africa, Cambodia, Vietnam, Germany, Kenya, Rwanda, and Portugal.

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
|   `-- datasets.yaml                  # AgriParcel-MR26 configuration
|-- model/                            # four archived checkpoints (Git LFS)
|-- src/
|   |-- scripts/                      # training, evaluation, and analysis
|   |-- ultralytics/                  # base framework and public FieldBCQ implementation
|   |   |-- nn/modules/head.py        # RBQC head, BASMQCSegment
|   |   |-- nn/extra_modules/head.py  # historical boundary-head import paths
|   |   `-- utils/loss.py             # PCSC, BAPS + PCSC, and FieldBCQ losses
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
