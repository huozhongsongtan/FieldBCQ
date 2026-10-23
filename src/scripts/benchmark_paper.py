"""Collect parameters, FLOPs, CUDA peak memory and scoped latency/throughput."""

import argparse
from project_paths import project_path
from quality_scoring import QUALITY_MODES
from paper_runtime import benchmark, code_provenance, environment, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--images", nargs="+", help="Representative image paths, decoded before timing")
    scoring = parser.add_mutually_exclusive_group()
    scoring.add_argument("--quality-blend", type=float, choices=(0.0, 1.0))
    scoring.add_argument("--quality-mode", choices=QUALITY_MODES)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32 or min(args.batch, args.iterations) < 1 or args.warmup < 0:
        parser.error("imgsz must be a positive multiple of 32; batch/iterations positive; warmup nonnegative")
    images = None
    if args.images:
        import cv2
        images = [cv2.imread(str(project_path(path))) for path in args.images]
        if any(image is None for image in images):
            parser.error("A representative image could not be read")
    weights = project_path(args.weights)
    report = {"checkpoint": str(weights), "checkpoint_sha256": sha256(weights),
              "environment": environment(), "code": code_provenance(),
              "benchmark": benchmark(weights, args.imgsz, args.batch, args.device, args.precision == "fp16",
                                     args.warmup, args.iterations, images=images, quality_blend=args.quality_blend, quality_mode=args.quality_mode)}
    write_json(project_path(args.output), report)
    print(f"Saved {project_path(args.output)}")


if __name__ == "__main__":
    main()
