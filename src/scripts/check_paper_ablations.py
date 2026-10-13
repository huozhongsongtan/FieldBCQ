"""Build and verify the four newly added paper ablation variants."""

import argparse

from project_paths import project_path

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from train_paper import STAGES


EXPECTED = {
    "b": ("BASSegment", "v8BASOnlySegmentationLoss"),
    "c": ("MQCSegment", "v8MQCSegmentationLoss"),
    "bc": ("BASOnlyMQCSegment", "v8BASOnlyMQCSegmentationLoss"),
    "ac": ("BoundaryDiceMQCSegment", "v8BoundaryDiceMQCSegmentationLoss"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Also run one 64x64 inference forward pass.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    for stage_name, (expected_head, expected_loss) in EXPECTED.items():
        model = YOLO(str(project_path("src") / STAGES[stage_name]["model"]))
        core = model.model
        core.args = get_cfg(overrides={**core.args, **STAGES[stage_name]["overrides"]})
        head_name = type(core.model[-1]).__name__
        loss_name = type(core.init_criterion()).__name__
        if head_name != expected_head or loss_name != expected_loss:
            raise AssertionError(
                f"{stage_name}: expected {expected_head}/{expected_loss}, got {head_name}/{loss_name}"
            )
        if args.forward:
            core.to(args.device).eval()
            with torch.no_grad():
                core(torch.zeros(1, 3, 64, 64, device=args.device))
        parameters = sum(parameter.numel() for parameter in core.parameters())
        print(f"{stage_name}: head={head_name}, loss={loss_name}, parameters={parameters:,}")
        del model, core
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("All four ablation variants passed.")


if __name__ == "__main__":
    main()
