"""FieldBCQ segmentation -> FieldAgent five-arm post-processing experiment."""
import argparse
import sys

from project_paths import PROJECT_ROOT, project_path

POSTPROCESS_ROOT = PROJECT_ROOT / "src/postprocessing"
sys.path.insert(0, str(POSTPROCESS_ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict", "local", "gpt", "report"])
    parser.add_argument("--inputs", help="Input manifest; required for local")
    parser.add_argument("--output", required=True, help="Experiment output directory")
    parser.add_argument("--planner", help="External service configuration; required for local")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args, extra = parser.parse_known_args(argv)
    if args.stage == "predict":
        from export_segmentation_vectors import main as predict
        return predict(["--output", args.output, *extra])
    if extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if args.stage == "local" and (not args.inputs or not args.planner):
        parser.error("local requires --inputs and --planner")
    # Use the actual five-arm protocol, including visual routing and editing.
    from final_protocol import main as protocol
    command = [args.stage, "--output", str(project_path(args.output)),
               "--workers", str(args.workers)]
    for flag, value in [("--inputs", args.inputs), ("--planner", args.planner)]:
        if value is not None:
            command.extend([flag, str(project_path(value))])
    return protocol(command)


if __name__ == "__main__":
    raise SystemExit(main())
