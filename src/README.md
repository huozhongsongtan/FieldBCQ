# FieldBCQ Python Package

FieldBCQ is a focused Ultralytics fork for farmland parcel instance segmentation
with YOLO11x-seg. It provides all eight combinations of boundary Dice (BDS),
boundary pixel BCE (BAPS), and region-boundary quality scoring (RBQS): Baseline,
BDS, BAPS, BDS+BAPS, RBQS, BDS+RBQS, BAPS+RBQS, and FieldBCQ.
Eight round-2 best.pt checkpoints are bundled under ../model/.
See [the project README](../README.md) for checkpoint provenance and current evaluation commands.

Install the source in a dedicated Python 3.10+ environment:

```powershell
python -m pip install -e .
```

Install the PyTorch and torchvision builds appropriate for your CPU or GPU before
installing this package. The package uses the distribution and import name
`ultralytics` to preserve compatibility with the archived checkpoints.
No timm, einops, or custom CUDA extensions are needed for the public models.

See the [FieldBCQ repository](https://github.com/huozhongsongtan/FieldBCQ)
for dataset preparation, model definitions and archived checkpoints, training
and evaluation commands, reported results, and verification instructions.

The base framework is derived from [Ultralytics](https://github.com/ultralytics/ultralytics).
The source is distributed under AGPL-3.0; see LICENSE.
