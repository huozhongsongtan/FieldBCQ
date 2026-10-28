"""Public command line; importing this module never calls an external API."""
import argparse
import json
from pathlib import Path

from . import activate_engine, PROJECT_ROOT
from .io import import_cache, load, planner_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="FieldAgent: parcel boundary vertex reduction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the local numerical runtime, without network requests")
    prepare = commands.add_parser("prepare", help="Create an input manifest from the existing prediction cache")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=PROJECT_ROOT/"data/inputs.json")
    prepare.add_argument("--limit", type=int, default=0, help="0 imports all images; a positive limit is a smoke subset")
    run = commands.add_parser("run", help="Run local/GPT stages or regenerate the paired report")
    run.add_argument("--inputs", type=Path, default=PROJECT_ROOT/"data/inputs.json")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, default=PROJECT_ROOT/"data/experiment.json")
    run.add_argument("--planner", type=Path, default=PROJECT_ROOT/"data/planner.example.json")
    run.add_argument("--stage", choices=("local", "gpt", "all", "report"), default="local")
    run.add_argument("--methods", nargs="+", choices=("public_edge", "local_adaptive_strict", "local_adaptive_balanced", "gpt_autonomous"))
    run.add_argument("--resume-after-error", action="store_true", help="One explicit GPT continuation, preserving original attempt caps")
    figure = commands.add_parser("figure", help="Plot actual saved outlines and vertex counts")
    figure.add_argument("--output", type=Path, required=True, help="An existing FieldAgent run")
    figure.add_argument("--scene", required=True)
    figure.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        count = import_cache(args.cache, args.output, limit=args.limit)
        print(json.dumps({"images": count, "manifest": str(args.output.resolve())}, ensure_ascii=False))
        return
    activate_engine()
    if args.command == "doctor":
        from .native_runtime import preflight
        import cv2, rasterio, scipy, shapely
        report = preflight()
        report["dependencies"] = {"opencv": cv2.__version__, "rasterio": rasterio.__version__,
                                  "scipy": scipy.__version__, "shapely": shapely.__version__}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "figure":
        from .visualize import comparison_figure
        comparison_figure(args.output, args.scene, args.save)
        return
    import cv2
    cv2.setNumThreads(1)
    from .experiment import freeze, run_local, run_gpt, build_report
    if args.stage == "report":
        manifest = load(args.output / "manifest.json")
        result = build_report(manifest, args.output)
    else:
        config = load(args.config)
        methods = args.methods or config["methods"]
        planner = planner_config(args.planner) if "gpt_autonomous" in methods else None
        manifest = freeze(args.inputs, args.output, methods, config["calls_per_target"], planner)
        try:
            if args.stage in ("local", "all"):
                run_local(manifest, args.output)
            if args.stage in ("gpt", "all"):
                run_gpt(manifest, args.output, resume_after_error=args.resume_after_error)
        finally:
            result = build_report(manifest, args.output)
    print(json.dumps({"complete": result["complete"], "counts": result["complete_by_method"],
                      "results": str((args.output/"RESULTS.md").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

