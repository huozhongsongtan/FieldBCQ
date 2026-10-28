# FieldBCQ

FieldBCQ addresses agricultural parcel instance segmentation and vector boundary representation in remote sensing imagery from multiple sources. The study has two stages: first, a YOLO11x-seg model combines boundary supervision and quality scoring to predict parcel masks; second, the masks are vectorized and processed through geometric simplification, visual tool selection, and local editing to reduce redundant boundary vertices.

The segmentation model includes three modules evaluated through ablation: A, boundary Dice supervision (BDS), learns boundary shape; B, boundary pixel BCE supervision (BAPS), supervises boundary pixel classification; and C, region and boundary quality scoring (RBQS), predicts region and boundary quality and produces the joint confidence `class_score * sqrt(region_quality * boundary_quality)`. The eight configurations are Baseline, A, B, C, A+B, A+C, B+C, and the complete FieldBCQ model.

Manuscript Tables 4–10 evaluate instance segmentation accuracy, boundary quality, the effect of scoring, and inference cost. The postprocessing experiments separately assess the tradeoff between vector complexity and boundary quality. Completed experiments show that visual tool selection and local editing can further reduce vertex counts, with a cost in average boundary quality.

Implementation notes: [Segmentation source](src/README.md) · [Postprocessing module](src/postprocessing/INTEGRATION.md)

## Installation

From the project root, use a PyTorch environment compatible with the server's CUDA installation:

```bash
python -m pip install -e ./src
```

For postprocessing, install the additional dependencies for geometry processing, raster vectorization, and runtime monitoring:

```bash
python -m pip install -e "./src[postprocess]"
```

Python 3.10 or later is required. Install compatible PyTorch and CUDA versions for the target server first. Training, evaluation, and the unified postprocessing entry point are under `src/scripts/`. These scripts prioritize this repository's `src/ultralytics`; relative command-line paths are resolved against the project root. File paths inside an input JSON manifest are resolved relative to that manifest.

The repository contains `data/`, `model/`, `src/`, and the reserved figure directory `fig/`. Only dataset configuration is distributed under `data/`; experiment outputs are created when the corresponding commands run.

## Evaluation for Manuscript Tables 4–10

Tables 4–10 use the same test split to evaluate segmentation performance, module contributions, performance across data sources, and computational cost:

| Table | Results | Models |
|---|---|---|
| 4 | Native mask P, R, F1, AP50, AP75, and AP50–95 | All eight configurations |
| 5 | Boundary F1, merge rate, oversegmentation rate, and score–IoU ECE | All eight configurations |
| 6 | AP75, AP50–95, fixed-threshold metrics, and differences for class / joint scoring with identical weights | FieldBCQ |
| 7 | Fixed-threshold P, R, F1, Boundary F1, small-parcel recall, count MAE, and area MAPE/WAPE/net bias | Baseline, FieldBCQ |
| 8 | Image counts, instance counts, F1, and Boundary F1 by data source | Baseline, FieldBCQ |
| 9 | Image counts, instance counts, F1, and F1 differences by resolution | Baseline, FieldBCQ |
| 10 | Parameter counts, GFLOPs, mean raw forward latency, and mean prediction pipeline latency | Baseline, FieldBCQ |

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

Checkpoint origins and SHA256 hashes are recorded in `model/checkpoints.json`. Evaluation verifies each configuration against the actual model head and training arguments and rejects mismatches. Configurations containing C use joint scoring. Table 6 reuses the complete FieldBCQ model's joint results and performs one additional class-scoring evaluation with the same checkpoint, verifying its SHA256 hash to separate scoring effects from weight changes.

These tables define the current reproduction protocol. Segmentation results should be obtained with this protocol on the target dataset and hardware; historical postprocessing results do not replace segmentation evaluation.

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

The repository's `data/` directory contains only `datasets.yaml`; images and labels must be supplied separately. Tables 8 and 9 require an external manifest specified through `--metadata-csv`, with the required columns `image`, `source`, and `resolution_m`. The `image` column accepts paths relative to the dataset root, absolute paths, or unambiguous filenames. Supported sources are `FBIS`, `FHAPD` (also accepting `DDFHAPD`), and `FTW`. Resolution values must be positive and expressed in meters.

The manifest must match the test images one-to-one. Optional columns are `instances`, `image_sha256`, and `label_sha256`. The program verifies complete coverage and any supplied instance counts and file hashes. Groups are determined from the manifest; evaluation areas and distances remain in pixel units.

### Execution

After configuring the data on the server, check the files before running evaluation. Tables 4, 5, 6, 7, and 10 can run without the source and resolution manifest:

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

Tables 4–10 are selected by default, so the complete protocol requires an external manifest:

```bash
python src/scripts/val_paper.py --metadata-csv /path/to/metadata.csv --check-only
python src/scripts/val_paper.py --metadata-csv /path/to/metadata.csv --device 0 --project results/paper_round2_tables4_10
```

Use a new output directory for each evaluation run. To rerun individual tables:

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
- Table 10: FP32, 640×640, and batch=1, with 50 warm-up iterations and 200 timed iterations for each measurement, using CUDA synchronization. Parameter counts include auxiliary training branches. GFLOPs are twice the THOP MAC count of the unfused evaluation graph. Raw forward timing measures only the model forward pass on resident tensors. Prediction pipeline timing includes preprocessing, inference, NMS, and mask generation, excluding disk reads. The first real test image is used, and the input and hardware are recorded in provenance.

Undefined values caused by zero denominators are left empty and displayed as `—` in Markdown. P/R/F1, AP, recall, merge and oversegmentation rates, and relative area errors are reported as percentages in CSV and Markdown. Differences in Tables 6 and 9 are calculated before rounding and expressed in percentage points; ECE differences retain the 0–1 scale.

### Outputs

```text
results/paper_round2_tables4_10/
  table4.csv ... table10.csv
  tables4_10.md
  provenance.json
```

`provenance.json` records the actual arguments, seeds, model/data/code hashes, environment, and timing scope. Failed runs are marked `failed` with the reason retained. The example above corresponds to all Tables 4–10. When a subset is selected, only the corresponding `tableN.csv` files are generated; the summary Markdown remains named `tables4_10.md`. This entry point produces quantitative results and reproduction records; postprocessing vectors and model-call records are generated separately by the postprocessing entry point.

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

## Postprocessing Experiments After Visual Segmentation

Postprocessing addresses redundant vertices, preservation of local turns, and boundary displacement in segmentation predictions. It follows the final FieldAgent protocol: **segmentation masks → vectorization without simplification → independent generation of three tool candidates → visual tool selection per parcel → local proposals, execution, and review → offline evaluation**.

The five primary methods and the stage ablation are:

| Method | Identifier | Settings |
|---|---|---|
| Raw predictions | `raw` | Vectorize each instance independently, preserving components, holes, and overlap between instances |
| Fixed-stride sampling | `nth_point` | Stride 12 |
| Radial-distance thinning | `radial_distance` | Distance threshold 8 px |
| DP | `dp` | Topology-preserving Douglas–Peucker simplification with a 1 px tolerance |
| Visual tool selection + editing | `tool_routed_edit` | Select a candidate per parcel, then propose and review local edits for selected boundary segments |
| Tool selection only (stage ablation) | `tool_routed` | Save the mixed draft after tool selection and report it separately |

All three candidates independently process the same raw predictions. Stride 12 and the radial threshold of 8 px were selected on 50 development validation images to approximately match the total predicted vertex count of DP at 1 px, without using GT or F1 for parameter selection. The 2,000-image experiment retains these parameters. The DP tolerance applies only to that candidate relative to its input geometry; it is not an error bound for the final edited result.

Visual tool selection processes up to 24 parcels per batch. Subsequent editing examines at most 12 segments per image and 3 per instance. The model can select an existing candidate, propose a local path, or invoke edge search. Local execution produces an actual preview, after which the model decides whether to commit or discard the edit. Checks cover geometric validity, component/hole identities, displacement, area differences, and newly introduced overlap. Neither GT nor offline accuracy metrics enter selection or editing.

The unified entry point is `src/scripts/run_postprocess.py`, supporting `predict / local / gpt / report`. After installing the dependencies, export vectors in original-image pixel coordinates and the postprocessing input manifest from segmentation predictions:

```bash
python src/scripts/run_postprocess.py predict --weights model/abc/best.pt --data data/datasets.yaml --split test --device 0 --output /path/to/predictions --include-labels
```

By default, `predict` uses the complete FieldBCQ checkpoint with joint scoring: FP32, imgsz=640, conf=0.25, NMS IoU=0.70, and max_det=710, with instance masks restored at the original image dimensions. Each instance is vectorized without simplification. A mask–vector rasterization round-trip check verifies consistency, and the original masks, GeoJSON, inference configuration, and file hashes are saved.

`--include-labels` links labels for offline evaluation. Omit it for unlabeled data, for which GT-dependent quality metrics cannot be computed. The exported `inputs.json` records image, prediction vector, and optional label locations. Existing vectors in pixel coordinates can also be supplied through a manifest in the same format; remove `label` for unlabeled data:

```json
{
  "coordinate_space": "image_pixels",
  "labels_offline_only": true,
  "inputs": [{
    "scene": "example",
    "domain": "external",
    "image": "/path/to/image.png",
    "prediction_vector": "/path/to/prediction.pixel.geojson",
    "label": "/path/to/label.txt"
  }]
}
```

Specify a service configuration file through `--planner`, for example:

```json
{
  "api": "responses",
  "base_url": "https://your-service.example/v1",
  "model": "your-vision-model",
  "api_key_env": "OPENAI_API_KEY",
  "vision": true,
  "timeout_seconds": 90,
  "max_output_tokens": 8192,
  "max_instances": 1024
}
```

Replace the endpoint and model name with an actual service supporting image input and structured responses, and set the key in the corresponding environment variable. The configuration stores only the environment variable name. The historical study used a third-party compatible interface with the model alias `gpt-6.1-sol`. This alias records the service configuration at that time; reproduction should also retain the actual requests and responses.

Run the local candidate stage before the visual agent stage. Use the same experiment directory for `--output`. Scene concurrency is controlled by `--workers`, accepting 1–4 with a default of 4:

```bash
python src/scripts/run_postprocess.py local --inputs /path/to/predictions/inputs.json --planner /path/to/planner.json --output /path/to/postprocess_run
python src/scripts/run_postprocess.py gpt --output /path/to/postprocess_run
python src/scripts/run_postprocess.py report --output /path/to/postprocess_run
```

The local stage freezes inputs, service configuration, parameters, and source hashes, and generates the four local controls. The `gpt` stage reads the frozen configuration, makes actual visual service requests, and saves selection and review records. The `report` stage summarizes existing outputs. Reusing an experiment requires its frozen contents to remain unchanged; use a new directory after changing inputs or the protocol.

Outputs include `MAIN_METRICS.csv`, `ABLATION_METRICS.csv`, `summary.json`, and a Markdown report. Per-image, per-method `final.pixel.geojson` files, completion hashes, and edit traces are stored under `scenes/`. GT is used only for offline evaluation after vectors have been saved. Postprocessing BF1@2px is computed per image in original pixel coordinates and then macro-averaged. Vertex counts include all exterior and interior rings without counting the closing vertex twice. This protocol is reported separately from the 640×640 evaluation grid used in Tables 4–10.

**Existing experimental results.** The source run `final-test2000-20261025-v1` completed the five-method comparison on 2,000 images and 18,561 predicted parcels: 700 images from DDFHAPD, 1,100 from FBIS, and 200 from FTW. The table below summarizes the original FieldAgent experiment's `summary.json`; the 50-image development validation results are reported separately:

| Method | Total vertices | Mean vertices per instance | BF1@2px↑ |
|---|---:|---:|---:|
| Raw predictions | 3,099,830 | 167.01 | 0.570760 |
| Fixed-stride sampling | 738,245 | 39.77 | 0.512513 |
| Radial-distance thinning | 593,606 | 31.98 | 0.553995 |
| DP 1px | 552,418 | 29.76 | 0.568944 |
| Visual tool selection + editing | 499,728 | 26.92 | 0.567379 |

The complete method removes 83.88% of vertices relative to raw predictions and an additional 52,690 vertices (9.54%) relative to DP. Mean vertices per instance decrease from 29.76 to 26.92, while BF1 decreases from 0.568944 to 0.567379, a drop of 0.1565 percentage points. Tool selection alone produces 528,297 vertices with BF1=0.567249. Subsequent visual editing removes another 28,569 vertices, accounting for 54.22% of the additional reduction relative to DP. Mean BF1 slightly recovers, but the exploratory paired interval crosses zero.

These results support further compression of vector complexity with a cost in average boundary quality. All three data sources show additional vertex reduction and lower mean BF1. A sufficiently tuned DP comparison at matched vertex counts, visual-input ablations, and independent human verification remain to be completed. The results therefore do not establish lossless simplification, an overall accuracy improvement, or improved manual editing efficiency. The 2,000-image split had been used in earlier exploration and is not presented as a fully blind test first accessed after protocol selection.

Historical requests, per-image vectors, and original research records remain in the original FieldAgent project. This repository retains the integrated implementation and reproduction entry points, without bundling those results or example data. The values above summarize existing experiments and do not represent a new full evaluation after integration.

Use original images and annotations under their respective source licenses. The model framework retains the repository's AGPL-3.0 license.
