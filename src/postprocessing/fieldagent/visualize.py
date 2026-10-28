"""Actual saved vector overlays, including all remaining vertices."""
from pathlib import Path
from .io import load


def comparison_figure(out, scene, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvr_agent import read_items, vertices
    from fieldbqc_geometry import polygons
    from run_boundary_tools import load_image
    out = Path(out)
    manifest = load(out / "manifest.json")
    entry = next((e for e in manifest["entries"] if e["scene"] == scene), None)
    if entry is None:
        raise ValueError("Unknown scene")
    image = load_image(entry["image"])[..., ::-1]
    methods = manifest["methods"]
    figure, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5), squeeze=False)
    for axis, method in zip(axes[0], methods):
        axis.imshow(image)
        folder = out / "scenes" / scene / method
        if not (folder / "completed.json").exists():
            axis.set_title(method + "\nNOT COMPLETED")
            axis.axis("off")
            continue
        items = read_items(folder / "final.pixel.geojson")
        for i, item in enumerate(items):
            color = plt.get_cmap("tab20")(i % 20)
            for polygon in polygons(item.geometry):
                for ring in [polygon.exterior, *polygon.interiors]:
                    x, y = ring.xy
                    axis.plot(x, y, color=color, linewidth=1.2)
                    axis.scatter(x[:-1], y[:-1], s=5, color=[color])
            center = item.geometry.representative_point()
            axis.text(center.x, center.y, f"{item.identifier}:{vertices(item.geometry)}", fontsize=7,
                      color="white", bbox={"facecolor": "black", "alpha": .6, "pad": 1})
        axis.set_title(f"{method}\n{sum(vertices(p.geometry) for p in items)} vertices")
        axis.axis("off")
    figure.suptitle(scene)
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(destination.resolve())

