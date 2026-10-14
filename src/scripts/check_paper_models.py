"""Check the four public configurations and optional archived checkpoints on CPU."""

import argparse
import gc

from project_paths import project_path
from train_paper import STAGES

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel


EXPECTED = {
    "baseline": ("Segment", "v8SegmentationLoss", "baseline"),
    "pcsc": ("BoundaryDiceSegment", "v8BoundaryDiceSegmentationLoss", "a"),
    "baps_pcsc": ("Segment_BAS", "v8BASSegmentationLoss", "ab"),
    "fieldbcq": ("BASMQCSegment", "v8BASMQCSegmentationLoss", "abc"),
}


def synthetic_batch(empty=False):
    """Include one parcel per image, or two empty images, at 64x64 resolution."""
    masks = torch.zeros(2, 16, 16)
    masks[:, 4:12, 4:12] = 1
    return {
        "img": torch.linspace(0, 1, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64),
        "batch_idx": torch.empty(0) if empty else torch.arange(2),
        "cls": torch.empty(0, 1) if empty else torch.zeros(2, 1),
        "bboxes": torch.empty(0, 4) if empty else torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(2, 1),
        "masks": torch.zeros_like(masks) if empty else masks,
    }


def check_configuration(name):
    stage = STAGES[name]
    model = SegmentationModel(str(project_path("src") / stage["model"]), nc=1, verbose=False)
    model.args = get_cfg(overrides=stage["overrides"])
    head = model.model[-1]
    expected_head, expected_loss, _ = EXPECTED[name]
    assert type(head).__name__ == expected_head
    assert type(model.init_criterion()).__name__ == expected_loss
    assert model.stride.tolist() == [8, 16, 32]
    model.eval()
    with torch.no_grad():
        prediction, auxiliary = model(torch.zeros(1, 3, 64, 64))
    assert prediction.shape == (1, 37, 84)
    assert auxiliary[-1].shape == (1, 32, 16, 16)
    assert torch.isfinite(prediction).all()

    model.train()
    for empty in (False, True):
        model.zero_grad(set_to_none=True)
        loss, items = model(synthetic_batch(empty))
        assert items.shape == (4,)
        assert torch.isfinite(loss) and torch.isfinite(items).all()
        loss.backward()
        if hasattr(head, "boundary_head"):
            gradient = head.boundary_head[-1].weight.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert empty or gradient.abs().sum() > 0
        if hasattr(head, "cv5"):
            gradient = head.cv5[0][-1].weight.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert empty or gradient.abs().sum() > 0
    return sum(parameter.numel() for parameter in model.parameters())


def check_checkpoint(name):
    expected_head, _, directory = EXPECTED[name]
    model = YOLO(str(project_path("model") / directory / "best.pt")).model.eval()
    assert type(model.model[-1]).__name__ == expected_head
    with torch.no_grad():
        prediction, _ = model(torch.zeros(1, 3, 64, 64))
    assert prediction.shape == (1, 37, 84)
    assert torch.isfinite(prediction).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", action="store_true", help="Also load all four archived checkpoints.")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for name in STAGES:
        parameters = check_configuration(name)
        gc.collect()
        if args.checkpoints:
            check_checkpoint(name)
            gc.collect()
        print(f"{name}: configuration, inference, positive/empty loss and gradients passed; parameters={parameters:,}")
    print("All four public configurations passed.")


if __name__ == "__main__":
    main()
