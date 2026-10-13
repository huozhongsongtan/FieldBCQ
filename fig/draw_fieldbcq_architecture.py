"""Draw FieldBCQ architecture and three mechanisms without loading model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#26343D",
    "muted": "#65727A",
    "line": "#576872",
    "border": "#BAC8CF",
    "blue": "#527E99",
    "green": "#218577",
    "orange": "#B77B27",
    "red": "#C55748",
}
STYLES = {
    "conv": ("#E5F1EC", "#719489"),
    "c3": ("#DFECF6", "#527E99"),
    "concat": ("#F9E9D4", "#B1936A"),
    "up": ("#F0F2F8", "#8D99AC"),
    "neutral": ("#F4F6F8", "#9AAAB4"),
    "baps": ("#E5F5EF", COLORS["green"]),
    "pcsc": ("#FFF1D8", COLORS["orange"]),
    "rbqc": ("#FCE8E3", COLORS["red"]),
}

LABELS = {
    "zh": {
        "subtitle": "田块实例分割网络结构及三项改进机制",
        "overall": "(a) 总体网络结构",
        "shared": "训练与推理",
        "training": "仅训练",
        "backbone": "Backbone · 主干网络",
        "neck": "Neck · FPN / PAN 特征融合",
        "head": "分割与质量预测头",
        "output": "校准与实例输出",
        "input": "输入影像\n" + r"$H\times W\times 3$",
        "quality": "RBQC 质量预测\n每尺度预测 " + r"$(q_r,q_b)$",
        "detect": "检测分支\n边界框 " + r"$b_i$" + " / 类别分数 " + r"$s_i$",
        "coefficient": "掩膜系数分支\n" + r"$c_i\in\mathbb{R}^{32}$",
        "proto": "原型掩膜分支\n" + r"$P\in\mathbb{R}^{32\times H/4\times W/4}$",
        "calibration": "置信度校准\n" + r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$",
        "nms": "边界框 NMS",
        "index": "保留索引",
        "decode": "掩膜解码\n" + r"$M_i=\sigma(c_iP)$",
        "instances": "田块实例输出\n" + r"$\{M_i,s'_i\}$",
        "boundary": "共享边界预测\n3×3 Conv → 1×1 Conv",
        "mini_baps": "BAPS · BCE 像素监督",
        "mini_pcsc": "PCSC · Boundary Dice 约束",
        "baps": "边界感知像素监督",
        "pcsc": "田块轮廓结构一致性约束",
        "rbqc": "区域-边界质量校准",
        "prototypes": "原型掩膜\n" + r"$P$ (32 通道)",
        "boundary_conv": "3×3 Conv (32→32)\n1×1 Conv (32→1)",
        "boundary_logits": "边界 logits\n" + r"$z_b$",
        "gt": "GT 实例\n掩膜",
        "target_ops": "掩膜并集\n3×3 膨胀−腐蚀",
        "target": "边界目标\n" + r"$B$",
        "shared_pred": "共享边界预测\n" + r"$p_b=\sigma(z_b)$",
        "shared_gt": "共享边界目标\n" + r"$B$",
        "pcsc_note": "与 BAPS 共用同一预测分支及监督目标",
        "q_features": "多尺度特征\n" + r"$P_3,P_4,P_5$",
        "q_conv": "每尺度：2×3×3 Conv\n1×1 Conv (2 logits)",
        "q_scores": "Sigmoid\n" + r"$q_r,\ q_b$",
        "matched": "匹配实例掩膜\n" + r"$M_i,Y_i$",
        "q_target": "Soft IoU / Boundary-F1\nStop-gradient",
        "q_loss": "BCE 监督\n" + r"$\mathcal{L}_{\rm RBQC}$",
    },
    "en": {
        "subtitle": "Field instance segmentation with three improvement mechanisms",
        "overall": "(a) Overall network architecture",
        "shared": "Training & inference",
        "training": "Training only",
        "backbone": "YOLO11x backbone",
        "neck": "Neck · FPN / PAN feature fusion",
        "head": "Segmentation & quality heads",
        "output": "Calibration & instance output",
        "input": "Input image\n" + r"$H\times W\times 3$",
        "quality": "RBQC quality heads\nPer-scale " + r"$(q_r,q_b)$" + " predictions",
        "detect": "Detection heads\nBoxes " + r"$b_i$" + " / class scores " + r"$s_i$",
        "coefficient": "Mask coefficient heads\n" + r"$c_i\in\mathbb{R}^{32}$",
        "proto": "Mask prototype branch\n" + r"$P\in\mathbb{R}^{32\times H/4\times W/4}$",
        "calibration": "Confidence calibration\n" + r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$",
        "nms": "Box NMS",
        "index": "Kept indices",
        "decode": "Mask decoding\n" + r"$M_i=\sigma(c_iP)$",
        "instances": "Field instances\n" + r"$\{M_i,s'_i\}$",
        "boundary": "Shared boundary prediction\n3×3 Conv → 1×1 Conv",
        "mini_baps": "BAPS · Pixel-wise BCE supervision",
        "mini_pcsc": "PCSC · Boundary Dice constraint",
        "baps": "Boundary-aware pixel supervision",
        "pcsc": "Parcel contour structural consistency",
        "rbqc": "Region-boundary quality calibration",
        "prototypes": "Prototypes\n" + r"$P$ (32 channels)",
        "boundary_conv": "3×3 Conv (32→32)\n1×1 Conv (32→1)",
        "boundary_logits": "Boundary logits\n" + r"$z_b$",
        "gt": "GT instance\nmasks",
        "target_ops": "Mask union\n3×3 dilation − erosion",
        "target": "Boundary target\n" + r"$B$",
        "shared_pred": "Shared boundary prediction\n" + r"$p_b=\sigma(z_b)$",
        "shared_gt": "Shared boundary target\n" + r"$B$",
        "pcsc_note": "Same prediction branch and boundary target as BAPS",
        "q_features": "Feature maps\n" + r"$P_3,P_4,P_5$",
        "q_conv": "Per scale: 2×3×3 Conv\n1×1 Conv (2 logits)",
        "q_scores": "Sigmoid\n" + r"$q_r,\ q_b$",
        "matched": "Matched masks\n" + r"$M_i,Y_i$",
        "q_target": "Soft IoU / Boundary-F1\nStop-gradient",
        "q_loss": "BCE loss\n" + r"$\mathcal{L}_{\rm RBQC}$",
    },
}


class Diagram:
    def __init__(self, ax):
        self.ax = ax
        self.nodes = []
        self.texts = []
        self.routes = []

    def text(self, x, y, value, size=10.5, color="ink", bold=False, ha="center"):
        item = self.ax.text(
            x, y, value, ha=ha, va="center", fontsize=size,
            weight="bold" if bold else "normal", color=COLORS.get(color, color),
            linespacing=1.25, zorder=8,
        )
        self.texts.append(item)
        return item

    def box(self, x, y, w, h, value, style="c3", size=10.5):
        fill, edge = STYLES[style]
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.55",
            facecolor=fill, edgecolor=edge, linewidth=0.85, zorder=4,
        ))
        item = self.text(x + w / 2, y + h / 2, value, size=size)
        self.nodes.append(((x, y, w, h), item))

    def group(self, x, y, w, h, title, fill="white"):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
            facecolor=fill, edgecolor=COLORS["border"], linewidth=0.8,
            linestyle=(0, (4, 3)), zorder=0,
        ))
        self.text(x + w / 2, y + 1.9, title, size=11.5, bold=True)

    def route(self, points, color="line", dashed=False, arrow=True, bridge=False):
        stroke = COLORS.get(color, color)
        effects = [patheffects.Stroke(linewidth=3.2, foreground="white"), patheffects.Normal()] if bridge else []
        self.routes.append(points)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            is_last = index == len(points) - 2
            if arrow and is_last:
                item = FancyArrowPatch(
                    start, end, arrowstyle="-|>", mutation_scale=8.5,
                    linewidth=0.95, color=stroke,
                    linestyle=(0, (4, 2.5)) if dashed else "solid",
                    shrinkA=0, shrinkB=1.5, zorder=6 if bridge else 2,
                )
                item.set_path_effects(effects)
                self.ax.add_patch(item)
            else:
                self.ax.plot(
                    [start[0], end[0]], [start[1], end[1]], color=stroke,
                    lw=0.95, linestyle=(0, (4, 2.5)) if dashed else "solid",
                    zorder=6 if bridge else 2, path_effects=effects,
                )

    def dot(self, x, y, color="line"):
        self.ax.plot(x, y, "o", ms=2.5, color=COLORS.get(color, color), zorder=7)

    def divider(self, x1, y1, x2, y2):
        self.ax.plot([x1, x2], [y1, y2], lw=0.65, color=COLORS["border"], zorder=0)

    def validate(self):
        # Validate rendered text bounds and orthogonal connectors before export.
        self.ax.figure.canvas.draw()
        renderer = self.ax.figure.canvas.get_renderer()
        for (x, y, w, h), item in self.nodes:
            p1 = self.ax.transData.transform((x + 0.45, y + 0.22))
            p2 = self.ax.transData.transform((x + w - 0.45, y + h - 0.22))
            allowed_w, allowed_h = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            for _ in range(4):
                extent = item.get_window_extent(renderer)
                ratio = min(allowed_w / extent.width, allowed_h / extent.height)
                if ratio >= 1:
                    break
                item.set_fontsize(item.get_fontsize() * ratio * 0.97)
            if item.get_fontsize() < 8.3:
                raise ValueError(f"Label too small: {item.get_text()!r}: {item.get_fontsize():.2f}")
        self.ax.figure.canvas.draw()
        renderer = self.ax.figure.canvas.get_renderer()
        for i, first in enumerate(self.texts):
            a = first.get_window_extent(renderer)
            for second in self.texts[i + 1:]:
                b = second.get_window_extent(renderer)
                overlap = a.intersection(a, b)
                if overlap is not None and overlap.width > 1 and overlap.height > 1:
                    raise ValueError(f"Overlapping labels: {first.get_text()!r} / {second.get_text()!r}")
        for points in self.routes:
            for start, end in zip(points, points[1:]):
                if start[0] != end[0] and start[1] != end[1]:
                    raise ValueError(f"Non-orthogonal connector: {start}, {end}")
                for (x, y, w, h), label in self.nodes:
                    if start[0] == end[0]:
                        hits = x + .06 < start[0] < x + w - .06 and max(min(start[1], end[1]), y + .06) < min(max(start[1], end[1]), y + h - .06)
                    else:
                        hits = y + .06 < start[1] < y + h - .06 and max(min(start[0], end[0]), x + .06) < min(max(start[0], end[0]), x + w - .06)
                    if hits:
                        raise ValueError(f"Connector {start}, {end} crosses node {label.get_text()!r}")


def draw(output_dir, language, dpi):
    labels = LABELS[language]
    available = {font.name for font in font_manager.fontManager.ttflist}
    if language == "zh":
        family = next((name for name in ("Microsoft YaHei", "Noto Sans SC", "SimHei") if name in available), None)
        if family is None:
            raise RuntimeError("A Chinese font (Microsoft YaHei, Noto Sans SC, SimHei) is required.")
    else:
        family = "Arial" if "Arial" in available else "DejaVu Sans"

    with plt.rc_context({
        "font.family": family, "mathtext.fontset": "stix",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }):
        fig = plt.figure(figsize=(18, 14.8), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 180), ylim=(148, 0), aspect="equal")
        ax.axis("off")
        d = Diagram(ax)

        d.text(3, 4.9, "FieldBCQ", size=25, bold=True, ha="left")
        d.text(3, 9.1, labels["subtitle"], size=11, color="muted", ha="left")
        d.route([(125, 4.8), (131, 4.8)], arrow=False)
        d.text(133, 4.8, labels["shared"], size=9.5, ha="left")
        d.route([(125, 9.1), (131, 9.1)], dashed=True, arrow=False)
        d.text(133, 9.1, labels["training"], size=9.5, ha="left")
        d.text(3, 13.6, labels["overall"], size=13.5, bold=True, ha="left")

        d.group(3, 17, 25, 64, labels["backbone"], "#F8FAFA")
        d.group(31, 17, 67, 66, labels["neck"], "#F5FAFD")
        d.group(104, 17, 29, 64, labels["head"])
        d.group(136, 17, 41, 66, labels["output"])
        d.text(50, 24.5, "Top-down FPN", size=9.5, color="muted")
        d.text(73, 24.5, "Bottom-up PAN", size=9.5, color="muted")

        backbone = [
            (22, labels["input"], "neutral", 4.0),
            (27, "Conv /2", "conv", 3.2),
            (31.7, "Conv /2", "conv", 3.2),
            (36.4, "C3k2 ×2", "c3", 3.2),
            (41.1, "Conv /2", "conv", 3.2),
            (45.8, "C3k2 ×2", "c3", 3.2),
            (50.5, "Conv /2", "conv", 3.2),
            (55.2, "C3k2 ×2", "c3", 3.2),
            (59.9, "Conv /2", "conv", 3.2),
            (64.6, "C3k2 ×2", "c3", 3.2),
            (69.3, "SPPF", "conv", 3.2),
            (74, "C2PSA ×2", "c3", 3.4),
        ]
        for center, value, style, height in backbone:
            d.box(7, center - height / 2, 17, height, value, style=style, size=10.3)
        for upper, lower in zip(backbone, backbone[1:]):
            d.route([(15.5, upper[0] + upper[3] / 2), (15.5, lower[0] - lower[3] / 2)])

        fpn = [
            (64.5, "Upsample", "up"),
            (57, "Concat", "concat"),
            (51.5, "C3k2 ×2", "c3"),
            (45.2, "Upsample", "up"),
            (38.8, "Concat", "concat"),
            (32.5, "C3k2 ×2", "c3"),
        ]
        for y, value, style in fpn:
            d.box(43, y, 14, 3.5, value, style=style, size=10.1)
        for lower, upper in zip(fpn, fpn[1:]):
            d.route([(50, lower[0]), (50, upper[0] + 3.5)], color="blue")
        d.route([(24, 45.8), (30, 45.8), (30, 40.55), (43, 40.55)], color="blue")
        d.route([(24, 55.2), (35, 55.2), (35, 58.75), (43, 58.75)], color="blue")
        d.route([(24, 74), (50, 74), (50, 68)], color="blue")
        d.route([(38, 74), (38, 81), (61, 81), (61, 71.25), (65, 71.25)], color="blue")
        d.dot(38, 74, "blue")

        pan = [
            (43.8, "Conv /2", "conv"),
            (51.5, "Concat", "concat"),
            (57, "C3k2 ×2", "c3"),
            (63, "Conv /2", "conv"),
            (69.5, "Concat", "concat"),
            (75, "C3k2 ×2", "c3"),
        ]
        for y, value, style in pan:
            d.box(65, y, 14, 3.5, value, style=style, size=10.1)
        for upper, lower in zip(pan, pan[1:]):
            d.route([(72, upper[0] + 3.5), (72, lower[0])], color="blue")
        d.route([(57, 34.25), (85, 34.25)], color="blue")
        d.route([(72, 34.25), (72, 43.8)], color="blue")
        d.dot(72, 34.25, "blue")
        d.route([(57, 53.25), (65, 53.25)], color="blue")
        for y in (58.75, 76.75):
            d.route([(79, y), (85, y)], color="blue")
        for y, scale, stride in ((32.5, 3, 8), (57, 4, 16), (75, 5, 32)):
            d.box(85, y, 10, 3.5, f"$P_{scale}$\n$1/{stride}$", style="c3", size=9.2)

        # This bus denotes separate predictions at each scale, not a Concat operation.
        d.route([(100, 31), (100, 76.75)], color="blue", arrow=False)
        for y in (34.25, 58.75, 76.75):
            d.route([(95, y), (100, y)], color="blue", arrow=False)
            d.dot(100, y, "blue")
        for y in (31, 45, 58):
            d.route([(100, y), (106, y)], color="blue")
        d.box(106, 28, 25, 6, labels["quality"], style="rbqc", size=10.8)
        d.box(106, 42, 25, 6, labels["detect"], size=10.5)
        d.box(106, 55, 25, 6, labels["coefficient"], size=10.8)
        d.box(106, 72.5, 25, 6, labels["proto"], size=10.6)
        d.route([(95, 34.25), (97, 34.25), (97, 84), (103, 84), (103, 75.5), (106, 75.5)], color="blue", bridge=True)
        d.dot(97, 34.25, "blue")
        d.text(99.8, 86.3, r"$P_3$", size=10, color="blue")

        d.box(140, 40, 34, 8, labels["calibration"], style="rbqc", size=12)
        d.route([(131, 31), (157, 31), (157, 40)], color="red")
        d.route([(131, 45), (140, 45)])
        d.box(148, 54, 26, 4.5, labels["nms"], style="neutral", size=11)
        d.route([(157, 48), (157, 54)], color="red")
        d.box(148, 65, 26, 6, labels["decode"], style="conv", size=11)
        d.route([(161, 58.5), (161, 65)])
        d.text(165.7, 61.7, labels["index"], size=8.8, color="muted")
        d.route([(131, 58), (142, 58), (142, 67), (148, 67)])
        d.text(136.5, 56.1, r"$c_i$", size=11)
        d.route([(131, 75.5), (144, 75.5), (144, 69), (148, 69)])
        d.text(137.2, 73.5, r"$P$", size=11)
        d.box(148, 77, 26, 5, labels["instances"], style="baps", size=11)
        d.route([(161, 71), (161, 77)])

        d.box(106, 86, 26, 6, labels["boundary"], style="baps", size=10.2)
        d.route([(118.5, 78.5), (118.5, 86)], color="green", dashed=True)
        d.box(141, 84.5, 34, 4, labels["mini_baps"], style="baps", size=10.8)
        d.box(141, 90.5, 34, 4, labels["mini_pcsc"], style="pcsc", size=10.8)
        d.route([(132, 89), (137, 89)], dashed=True, arrow=False)
        d.route([(137, 89), (137, 86.5), (141, 86.5)], color="green", dashed=True)
        d.route([(137, 89), (137, 92.5), (141, 92.5)], color="orange", dashed=True)
        d.dot(137, 89)

        d.divider(3, 98, 177, 98)
        d.divider(59, 101, 59, 139)
        d.divider(114, 101, 114, 139)
        for x, title, subtitle, color in (
            (3, "(b) BAPS", labels["baps"], "green"),
            (62, "(c) PCSC", labels["pcsc"], "orange"),
            (117, "(d) RBQC", labels["rbqc"], "red"),
        ):
            d.text(x, 102.3, title, size=15.5, color=color, bold=True, ha="left")
            d.text(x, 106.3, subtitle, size=11.3, color="muted", ha="left")

        d.box(3, 111, 13, 5, labels["prototypes"], style="baps", size=9.7)
        d.box(20, 111, 16, 5, labels["boundary_conv"], style="baps", size=9.1)
        d.box(40, 111, 14, 5, labels["boundary_logits"], style="baps", size=10)
        d.route([(16, 113.5), (20, 113.5)], color="green", dashed=True)
        d.route([(36, 113.5), (40, 113.5)], color="green", dashed=True)
        d.box(3, 121, 13, 5, labels["gt"], style="neutral", size=10)
        d.box(20, 121, 16, 5, labels["target_ops"], style="neutral", size=9.5)
        d.box(40, 121, 14, 5, labels["target"], style="neutral", size=10)
        d.route([(16, 123.5), (20, 123.5)], dashed=True)
        d.route([(36, 123.5), (40, 123.5)], dashed=True)
        d.box(20, 133, 34, 5, r"$\mathcal{L}_{\rm BAPS}=\operatorname{BCE}(z_b,B)$", style="baps", size=13)
        d.route([(54, 113.5), (56.5, 113.5), (56.5, 135.5), (54, 135.5)], color="green", dashed=True)
        d.route([(47, 126), (47, 133)], dashed=True)

        d.box(62, 111, 22, 5, labels["shared_pred"], style="pcsc", size=10.1)
        d.box(89, 111, 22, 5, labels["shared_gt"], style="neutral", size=10.1)
        d.route([(73, 116), (73, 122)], color="orange", dashed=True)
        d.route([(100, 116), (100, 122)], dashed=True)
        dice = "Boundary Dice\n" + r"$\mathcal{L}_{\rm PCSC}=1-\frac{2\sum p_b B+\epsilon}{\sum p_b+\sum B+\epsilon}$"
        d.box(62, 122, 49, 11, dice, style="pcsc", size=13)
        d.text(86.5, 137.1, labels["pcsc_note"], size=9.5, color="muted")

        d.box(117, 111, 14, 5, labels["q_features"], style="rbqc", size=10)
        d.box(135, 111, 20, 5, labels["q_conv"], style="rbqc", size=9.7)
        d.box(160, 111, 17, 5, labels["q_scores"], style="rbqc", size=10.5)
        d.route([(131, 113.5), (135, 113.5)], color="red")
        d.route([(155, 113.5), (160, 113.5)], color="red")
        d.box(117, 122, 17, 6, labels["matched"], style="neutral", size=10)
        d.box(139, 122, 19, 6, labels["q_target"], style="neutral", size=9.3)
        d.box(163, 122, 14, 6, labels["q_loss"], style="rbqc", size=10.3)
        d.route([(134, 125), (139, 125)], dashed=True)
        d.route([(158, 125), (163, 125)], dashed=True)
        d.route([(157.5, 113.5), (157.5, 119), (170, 119), (170, 122)], color="red", dashed=True)
        d.dot(157.5, 113.5, "red")
        d.text(161.7, 118, r"$z_q$", size=9, color="red")
        d.box(117, 133, 60, 5, r"$s'_i=s_i\sqrt{q_{r,i}q_{b,i}}$" + "   →   NMS", style="rbqc", size=13)
        d.route([(177, 113.5), (179, 113.5), (179, 135.5), (177, 135.5)], color="red")

        d.divider(3, 141, 177, 141)
        d.text(90, 144.5,
               r"$\mathcal{L}=\mathcal{L}_{\rm YOLO-seg}+\lambda_b(\mathcal{L}_{\rm BAPS}+\mathcal{L}_{\rm PCSC})+\lambda_q\mathcal{L}_{\rm RBQC}$",
               size=17)

        d.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"fieldbcq_architecture_{language}"
        metadata = {"Title": "FieldBCQ architecture: BAPS, PCSC, RBQC"}
        for extension in ("svg", "pdf", "png"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=dpi,
                        facecolor="white", metadata=metadata)
        fig.savefig(output_dir / f"{stem}_preview.png", dpi=150, facecolor="white", metadata=metadata)
        plt.close(fig)
        print(f"Exported {stem}: PNG ({dpi} dpi), editable SVG, vector PDF. Layout checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    for language in (("zh", "en") if args.language == "both" else (args.language,)):
        draw(args.output_dir.resolve(), language, args.dpi)


if __name__ == "__main__":
    main()
