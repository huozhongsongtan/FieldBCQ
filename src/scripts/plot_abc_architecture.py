"""Export publication figures for the archived FieldBCQ architecture.

All CLI paths are relative to the project root. No model weights or dataset
files are loaded. Requires matplotlib; the Chinese version needs a CJK font.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#202B33",
    "muted": "#5D6A73",
    "line": "#495A66",
    "border": "#B9C5CC",
    "blue": "#4B6E8A",
    "blue_fill": "#EFF4F8",
    "teal": "#207F78",
    "teal_fill": "#EAF5F2",
    "coral": "#B45D47",
    "coral_fill": "#FCF0EB",
    "gray_fill": "#F6F8FA",
}

TEXT = {
    "en": {
        "overall": "(a) FieldBCQ: overall architecture",
        "shared": "Training and inference",
        "training": "Training only",
        "input": "RGB image\n" + r"$H\times W\times 3$",
        "backbone": "YOLO11x backbone",
        "stem": "Conv + C3k2",
        "c3": "C3k2 (C3, stride 8)",
        "c4": "C3k2 (C4, stride 16)",
        "c5": "C3k2 / SPPF\nC2PSA (C5, stride 32)",
        "neck": "FPN + PAN neck",
        "neck_ops": "Upsample / Concat\nC3k2 / stride-2 Conv",
        "quality": "Quality head\n" + r"$q_r,\ q_b=\sigma(z)$",
        "detect": "Detection head\n" + r"$b_i,\ s_i$",
        "coefficient": "Mask coefficients\n" + r"$c_i\in\mathbb{R}^{32}$",
        "proto": "Prototype branch\n" + r"$P\in\mathbb{R}^{32\times H/4\times W/4}$",
        "calibration": "Score calibration\n" + r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$",
        "nms": "Box NMS",
        "decode": "Mask decoding\n" + r"$M=\sigma(cP)$",
        "instances": "Field instances\n" + r"$\{M_i,\ s'_i\}$",
        "mini_pbas": "BAPS + PCSC\nTraining only",
        "pbas_title": "(b) BAPS + PCSC: boundary supervision",
        "pbas_input": "Prototypes\n" + r"$P$ (32 channels)",
        "boundary_head": "3x3 Conv (32 to 32)\n1x1 Conv (32 to 1)",
        "boundary_logits": "Boundary logits\n" + r"$\hat B$",
        "gt": "GT instance\nmasks",
        "boundary_target": "Mask union\n3x3 dilation - erosion",
        "target": "Boundary target\n" + r"$B$",
        "rbqc_title": "(c) RBQC: region-boundary quality calibration",
        "features": "Feature maps\n" + r"$P_3,\ P_4,\ P_5$",
        "quality_head": "Per scale: 2 x 3x3 Conv\n1x1 Conv (2 logits)",
        "qualities": "Quality estimates\n" + r"$(q_r,q_b)=\sigma(z)$",
        "matched": "Matched masks\n" + r"$M_i,\ Y_i$",
        "quality_targets": "Soft IoU / Boundary-F1\nDetached targets " + r"$(t_r,t_b)$",
        "quality_loss": "Quality supervision\n" + r"$\mathcal{L}_{\rm RBQC}=\operatorname{BCE}(z,t)$",
        "inference": "Inference: score calibration before NMS",
        "index": "kept indices",
    },
    "zh": {
        "overall": "(a) FieldBCQ 总体架构",
        "shared": "训练与推理",
        "training": "仅训练",
        "input": "输入影像\n" + r"$H\times W\times 3$",
        "backbone": "YOLO11x 主干网络",
        "stem": "Conv + C3k2",
        "c3": "C3k2（C3，步长 8）",
        "c4": "C3k2（C4，步长 16）",
        "c5": "C3k2 / SPPF\nC2PSA (C5, /32)",
        "neck": "FPN + PAN 特征融合",
        "neck_ops": "上采样 / 拼接\nC3k2 / 步长 2 卷积",
        "quality": "质量预测头\n" + r"$q_r,\ q_b=\sigma(z)$",
        "detect": "检测头\n" + r"$b_i,\ s_i$",
        "coefficient": "掩膜系数预测头\n" + r"$c_i\in\mathbb{R}^{32}$",
        "proto": "原型掩膜分支\n" + r"$P\in\mathbb{R}^{32\times H/4\times W/4}$",
        "calibration": "置信度重标定\n" + r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$",
        "nms": "边界框 NMS",
        "decode": "掩膜解码\n" + r"$M=\sigma(cP)$",
        "instances": "田块实例输出\n" + r"$\{M_i,\ s'_i\}$",
        "mini_pbas": "BAPS + PCSC\n仅训练",
        "pbas_title": "(b) BAPS + PCSC：边界监督",
        "pbas_input": "原型掩膜\n" + r"$P$（32 通道）",
        "boundary_head": "3x3 卷积（32 到 32）\n1x1 卷积（32 到 1）",
        "boundary_logits": "边界预测 logits\n" + r"$\hat B$",
        "gt": "GT 实例\n掩膜",
        "boundary_target": "掩膜并集\n3x3 膨胀减腐蚀",
        "target": "边界监督目标\n" + r"$B$",
        "rbqc_title": "(c) RBQC：区域-边界质量校准",
        "features": "多尺度特征\n" + r"$P_3,\ P_4,\ P_5$",
        "quality_head": "每尺度：两层 3x3 卷积\n1x1 卷积（2 个 logits）",
        "qualities": "区域 / 边界质量\n" + r"$(q_r,q_b)=\sigma(z)$",
        "matched": "匹配实例掩膜\n" + r"$M_i,\ Y_i$",
        "quality_targets": "Soft IoU / Boundary-F1\n停止梯度的目标 " + r"$(t_r,t_b)$",
        "quality_loss": "质量监督\n" + r"$\mathcal{L}_{\rm RBQC}=\operatorname{BCE}(z,t)$",
        "inference": "推理阶段：在 NMS 前重标定置信度",
        "index": "保留索引",
    },
}


class Diagram:
    def __init__(self, ax):
        self.ax = ax
        self.nodes = []
        self.routes = []
        self.texts = []

    def text(self, x, y, value, size=16, color="ink", bold=False, ha="center"):
        item = self.ax.text(
            x, y, value, fontsize=size, color=COLORS.get(color, color),
            weight="bold" if bold else "normal", ha=ha, va="center",
            linespacing=1.35, zorder=5,
        )
        self.texts.append(item)
        return item

    def box(self, x, y, w, h, value, kind="blue", size=16):
        edge = COLORS[kind] if kind in ("blue", "teal", "coral") else COLORS["border"]
        fill = COLORS.get(kind + "_fill", COLORS["gray_fill"])
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
            linewidth=0.95, facecolor=fill, edgecolor=edge, zorder=2,
        ))
        item = self.text(x + w / 2, y + h / 2, value, size=size)
        self.nodes.append(((x, y, w, h), item))

    def group(self, x, y, w, h, title, kind="blue"):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0.7, facecolor="white", edgecolor=COLORS["border"], zorder=0,
        ))
        self.text(x + w / 2, y + h - 0.55, title, size=16, color=kind, bold=True)

    def route(self, points, kind="line", dashed=False, arrow=True):
        color = COLORS.get(kind, kind)
        self.routes.append(points)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            last = index == len(points) - 2
            if arrow and last:
                self.ax.add_patch(FancyArrowPatch(
                    start, end, arrowstyle="-|>", mutation_scale=10,
                    linewidth=1.05, color=color, shrinkA=0, shrinkB=2,
                    linestyle=(0, (4, 3)) if dashed else "solid", zorder=3,
                ))
            else:
                self.ax.plot(
                    [start[0], end[0]], [start[1], end[1]], color=color,
                    lw=1.05, linestyle=(0, (4, 3)) if dashed else "solid", zorder=3,
                )

    def validate(self):
        # Check rendered labels, then reject connectors passing through nodes.
        self.ax.figure.canvas.draw()
        renderer = self.ax.figure.canvas.get_renderer()
        for (x, y, w, h), item in self.nodes:
            bottom = self.ax.transData.transform((x + 0.12, y + 0.07))
            top = self.ax.transData.transform((x + w - 0.12, y + h - 0.07))
            while True:
                extent = item.get_window_extent(renderer)
                ratio = min((top[0] - bottom[0]) / extent.width, (top[1] - bottom[1]) / extent.height)
                if ratio >= 0.99:
                    break
                new_size = item.get_fontsize() * min(ratio * 0.98, 0.96)
                if new_size < 12.0:
                    raise ValueError("Label does not fit: " + item.get_text())
                item.set_fontsize(new_size)
        self.ax.figure.canvas.draw()
        renderer = self.ax.figure.canvas.get_renderer()
        for index, item in enumerate(self.texts):
            extent = item.get_window_extent(renderer)
            for other in self.texts[index + 1:]:
                overlap = extent.intersection(extent, other.get_window_extent(renderer))
                if overlap is not None and overlap.width > 1 and overlap.height > 1:
                    raise ValueError("Overlapping labels: " + item.get_text() + " / " + other.get_text())
        for points in self.routes:
            for start, end in zip(points, points[1:]):
                if start[0] != end[0] and start[1] != end[1]:
                    raise ValueError("Expected an orthogonal connector")
                for (x, y, w, h), _ in self.nodes:
                    if start[0] == end[0]:
                        hits = x + 0.04 < start[0] < x + w - 0.04 and max(min(start[1], end[1]), y + 0.04) < min(max(start[1], end[1]), y + h - 0.04)
                    else:
                        hits = y + 0.04 < start[1] < y + h - 0.04 and max(min(start[0], end[0]), x + 0.04) < min(max(start[0], end[0]), x + w - 0.04)
                    if hits:
                        raise ValueError("Connector crosses a node: " + str((start, end)))


def draw(output_dir: Path, language: str, dpi: int):
    labels = TEXT[language]
    available = {font.name for font in font_manager.fontManager.ttflist}
    if language == "zh":
        family = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans SC") if name in available), None)
        if family is None:
            raise RuntimeError("Install Noto Sans SC or Microsoft YaHei for Chinese labels.")
    else:
        family = "Arial" if "Arial" in available else "DejaVu Sans"
    with plt.rc_context({
        "font.family": family,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }):
        fig = plt.figure(figsize=(14, 9.6), facecolor="white")
        ax = fig.add_axes((0.015, 0.015, 0.97, 0.97))
        ax.set(xlim=(0, 34), ylim=(0, 23.3))
        ax.axis("off")
        d = Diagram(ax)

        d.text(0.6, 22.65, labels["overall"], size=20, bold=True, ha="left")
        d.route([(22.0, 22.65), (23.0, 22.65)], arrow=False)
        d.text(23.3, 22.65, labels["shared"], size=12, ha="left")
        d.route([(28.4, 22.65), (29.4, 22.65)], dashed=True, arrow=False)
        d.text(29.7, 22.65, labels["training"], size=12, ha="left")

        d.box(0.55, 18.6, 2.7, 1.5, labels["input"], kind="gray", size=16)
        d.group(3.85, 13.1, 5.5, 7.9, labels["backbone"])
        d.box(4.3, 18.85, 4.6, 1.0, labels["stem"])
        d.box(4.3, 17.1, 4.6, 1.25, labels["c3"], size=15)
        d.box(4.3, 15.45, 4.6, 1.25, labels["c4"], size=15)
        d.box(4.3, 13.8, 4.6, 1.25, labels["c5"], size=14)
        d.route([(3.25, 19.35), (4.3, 19.35)])
        for upper, lower in ((18.85, 18.35), (17.1, 16.7), (15.45, 15.05)):
            d.route([(6.6, upper), (6.6, lower)], kind="blue")

        d.group(10.0, 13.1, 4.85, 7.9, labels["neck"])
        d.text(12.425, 19.25, labels["neck_ops"], size=12)
        centers = (17.725, 16.075, 14.425)
        for index, (y, divisor) in enumerate(zip((17.1, 15.45, 13.8), (8, 16, 32))):
            value = r"$P_" + str(index + 3) + "$\n" + r"$H/" + str(divisor) + r"\times W/" + str(divisor) + "$"
            d.box(10.7, y, 3.45, 1.25, value, size=14.5)
            d.route([(8.9, centers[index]), (10.7, centers[index])], kind="blue")
            d.route([(14.15, centers[index]), (15.5, centers[index])], arrow=False)
        for upper, lower in ((17.1, 16.7), (15.45, 15.05)):
            d.route([(11.9, lower), (11.9, upper)], kind="blue")
            d.route([(13.0, upper), (13.0, lower)], kind="blue")

        d.route([(15.5, 14.425), (15.5, 20.3)], arrow=False)
        for y in (20.3, 18.3, 16.3):
            d.route([(15.5, y), (16.7, y)])
        d.text(19.075, 21.65, "BASMQCSegment", size=15, bold=True)
        d.box(16.7, 19.55, 4.75, 1.5, labels["quality"], kind="coral")
        d.box(16.7, 17.55, 4.75, 1.5, labels["detect"])
        d.box(16.7, 15.55, 4.75, 1.5, labels["coefficient"])
        d.box(16.7, 13.55, 4.75, 1.5, labels["proto"], size=15)
        d.route([(12.425, 13.1), (12.425, 12.6), (16.0, 12.6), (16.0, 14.3), (16.7, 14.3)])
        d.text(14.1, 12.95, r"$P_3$", size=13)

        d.box(22.7, 17.55, 4.5, 1.5, labels["calibration"], kind="coral", size=15)
        d.route([(21.45, 20.3), (24.95, 20.3), (24.95, 19.05)], kind="coral")
        d.route([(21.45, 18.3), (22.7, 18.3)])
        d.box(28.7, 17.55, 4.6, 1.5, labels["nms"])
        d.route([(27.2, 18.3), (28.7, 18.3)], kind="coral")
        d.box(28.7, 15.55, 4.6, 1.5, labels["decode"], size=15)
        d.route([(31.0, 17.55), (31.0, 17.05)])
        d.text(31.25, 17.3, labels["index"], size=10.5, ha="left")
        d.route([(21.45, 16.3), (28.7, 16.3)])
        d.text(23.3, 16.65, r"$c_i$", size=13)
        d.route([(21.45, 14.3), (27.6, 14.3), (27.6, 15.95), (28.7, 15.95)])
        d.text(23.3, 14.65, r"$P$", size=13)
        d.box(28.7, 13.55, 4.6, 1.5, labels["instances"], kind="teal")
        d.route([(31.0, 15.55), (31.0, 15.05)])

        d.box(22.7, 11.65, 4.5, 1.2, labels["mini_pbas"], kind="teal", size=15)
        d.route([(19.075, 13.55), (19.075, 12.25), (22.7, 12.25)], kind="teal", dashed=True)
        ax.plot([0.55, 33.45], [10.85, 10.85], color=COLORS["border"], lw=0.75)

        d.text(0.65, 10.0, labels["pbas_title"], size=17, bold=True, color="teal", ha="left")
        d.box(0.9, 7.325, 3.5, 1.4, labels["pbas_input"], kind="teal")
        d.box(5.15, 7.325, 5.1, 1.4, labels["boundary_head"], kind="teal", size=15)
        d.box(11.0, 7.325, 4.3, 1.4, labels["boundary_logits"], kind="teal")
        d.route([(4.4, 8.025), (5.15, 8.025)], kind="teal", dashed=True)
        d.route([(10.25, 8.025), (11.0, 8.025)], kind="teal", dashed=True)
        d.box(0.9, 4.725, 3.5, 1.4, labels["gt"], kind="gray")
        d.box(5.15, 4.725, 5.1, 1.4, labels["boundary_target"], kind="gray", size=15)
        d.box(11.0, 4.725, 4.3, 1.4, labels["target"], kind="gray")
        d.route([(4.4, 5.425), (5.15, 5.425)], dashed=True)
        d.route([(10.25, 5.425), (11.0, 5.425)], dashed=True)
        boundary_loss = r"$\mathcal{L}_{\rm boundary}=\operatorname{BCE}(\hat B,B)+\operatorname{Dice}(\sigma(\hat B),B)$"
        d.box(5.15, 2.1, 10.15, 1.45, boundary_loss, kind="teal", size=17)
        d.route([(15.3, 8.025), (15.95, 8.025), (15.95, 2.825), (15.3, 2.825)], kind="teal", dashed=True)
        d.route([(13.15, 4.725), (13.15, 3.55)], dashed=True)

        ax.plot([16.65, 16.65], [1.7, 10.15], color=COLORS["border"], lw=0.75)
        d.text(17.35, 10.0, labels["rbqc_title"], size=17, bold=True, color="coral", ha="left")
        d.box(17.6, 7.325, 3.5, 1.4, labels["features"], kind="coral")
        d.box(21.85, 7.325, 5.4, 1.4, labels["quality_head"], kind="coral", size=14.5)
        d.box(28.1, 7.325, 4.9, 1.4, labels["qualities"], kind="coral", size=15)
        d.route([(21.1, 8.025), (21.85, 8.025)], kind="coral")
        d.route([(27.25, 8.025), (28.1, 8.025)], kind="coral")
        d.box(17.6, 4.725, 3.5, 1.4, labels["matched"], kind="gray")
        d.box(21.85, 4.725, 5.4, 1.4, labels["quality_targets"], kind="gray", size=14.2)
        d.box(28.1, 4.725, 4.9, 1.4, labels["quality_loss"], kind="coral", size=15)
        d.route([(21.1, 5.425), (21.85, 5.425)], dashed=True)
        d.route([(27.25, 5.425), (28.1, 5.425)], dashed=True)
        d.route([(24.55, 7.325), (24.55, 6.7), (30.55, 6.7), (30.55, 6.125)], kind="coral", dashed=True)
        d.text(26.8, 6.95, r"$z$", size=13, color="coral")
        d.box(17.6, 2.1, 15.4, 1.45, labels["inference"] + "\n" + r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$", kind="coral", size=17)
        d.route([(33.0, 8.025), (33.55, 8.025), (33.55, 2.825), (33.0, 2.825)], kind="coral")

        d.text(17.0, 0.65, r"$\mathcal{L}=\mathcal{L}_{\rm YOLO\!\! -\! seg}+\lambda_b\mathcal{L}_{\rm boundary}+\lambda_q\mathcal{L}_{\rm RBQC}$", size=19)
        d.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "abc_architecture" if language == "en" else "abc_architecture_zh"
        for extension in ("png", "svg", "pdf"):
            path = output_dir / (stem + "." + extension)
            fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Title": "FieldBCQ architecture"})
        plt.close(fig)
        print("Exported " + stem + " (PNG, SVG, PDF); layout checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("fig/abc"))
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    for language in (("en", "zh") if args.language == "both" else (args.language,)):
        draw(output_dir.resolve(), language, args.dpi)


if __name__ == "__main__":
    main()
