# FieldBCQ 网络结构及三项改进机制

## 成图文件

- `fieldbcq_architecture_zh.png`：中文高清图，600 dpi，10800 × 8880 像素。
- `fieldbcq_architecture_zh.svg`：中文可编辑矢量图，文字保留为文本。
- `fieldbcq_architecture_zh.pdf`：中文矢量 PDF，嵌入字体。
- `fieldbcq_architecture_en.*`：同一结构的英文版本。
- `*_preview.png`：150 dpi 预览图。
- `draw_fieldbcq_architecture.py`：可重新生成成图的绘图脚本。

## 图注

**中文：** 图 X. FieldBCQ 总体网络结构及三项改进机制。（a）YOLO11x 骨干网络与 FPN/PAN 生成 P3、P4、P5 三尺度特征，分别用于检测、掩膜系数预测和区域-边界质量估计；原型掩膜分支仅以 P3 为输入。共享边界预测分支在训练时从原型掩膜产生辅助边界 logits，并分别采用（b）BAPS 的像素级 BCE 和（c）PCSC 的 Boundary Dice 进行监督。（d）RBQC 利用匹配预测实例掩膜与 GT 的 Soft IoU 和容差 Boundary-F1 构建停止梯度的质量目标，以区域质量和边界质量的几何均值重标定类别置信度，再执行边界框 NMS，并利用保留的掩膜系数与原型解码田块实例掩膜。实线表示训练与推理使用的计算路径，虚线表示仅训练时使用的路径。

**English:** Figure X. Overall architecture of FieldBCQ and its three improvement mechanisms. (a) The YOLO11x backbone and FPN/PAN neck produce P3, P4, and P5 features for detection, mask coefficient prediction, and region-boundary quality estimation. The prototype branch uses P3 only. A shared training-only boundary prediction branch takes the prototypes as input and is supervised by (b) pixel-wise BCE in BAPS and (c) Boundary Dice in PCSC. (d) RBQC uses detached Soft-IoU and tolerant Boundary-F1 targets computed from matched predicted instance masks and ground truth. The geometric mean of the estimated region and boundary qualities calibrates class confidence before box NMS, after which the retained coefficients and prototypes decode field instance masks. Solid and dashed arrows indicate paths used in training and inference and paths used only in training, respectively.

## 实现对应与简化说明

1. 总体连接依据 `../src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml`；骨干网络与 FPN/PAN 保留模块级逐层连接，最终输出为第 16、19、22 层。
2. 预测头依据 `BASMQCSegment`（`../src/ultralytics/nn/modules/head.py`）。检测、系数和质量分支在三个尺度分别执行。图中三尺度输入总线表示这组特征的并列输入，不表示额外的拼接、求和或特征融合模块。重复次数 `×2` 对应 x 模型的深度配置；`Conv /2` 表示步长为 2 的卷积。
3. 原型掩膜具有 32 个通道，辅助边界分支由 `Conv(32,32,3)` 和 `Conv2d(32,1,1)` 构成，仅在训练时执行。
4. BAPS 与 PCSC 的命名与 README 一致，分别对应同一辅助边界图上的 BCE 和 Boundary Dice。当前代码对 GT 实例掩膜先求并集，再使用 3×3 膨胀减腐蚀生成边界目标；该过程不额外提取逐个实例之间的共享分界。PCSC 图表示 Boundary Dice 约束，不引入独立的轮廓编码器、图网络或拓扑损失。
5. 边界损失依据 `v8BASSegmentationLoss`、`v8BASMQCSegmentationLoss`（`../src/ultralytics/utils/loss.py`）。图中 BCE 接收 logits，等同于 `binary_cross_entropy_with_logits`；Dice 接收 Sigmoid 后的概率。Dice 先按图像计算再取均值，平滑项 epsilon 为 1。
6. RBQC 每尺度使用两层 3×3 Conv 和一层输出 2 通道的 1×1 Conv。区域质量目标为 Soft IoU，边界质量目标为容差 Boundary-F1；监督目标在匹配 GT 框内由预测实例掩膜和 GT 计算，并停止梯度。辅助边界 logits 不直接作为质量目标。
7. 推理校准公式对应默认 `quality_blend=1`：`s' = s * sqrt(q_r*q_b)`。边界框 NMS 在校准之后执行，保留索引用于筛选掩膜系数。掩膜解码公式省略框裁剪、上采样与二值化等标准步骤。
8. 总损失表达为 `L_YOLO-seg + lambda_b*(L_BAPS + L_PCSC) + lambda_q*L_RBQC`，符合实现中两个边界损失共用同一权重的方式；归档训练入口的默认权重为 `lambda_b=0.20`、`lambda_q=0.25`。
9. 本图描述 README 中的完整 FieldBCQ 配置，不包含 BPR、拓扑 Mask-NMS 或其他未启用分支，也不验证现存 checkpoint 与论文三次重复实验之间的对应关系。

## 编辑与重新生成

建议将 SVG 导入 Inkscape、Illustrator 或其他支持 SVG 的编辑器。中文 SVG 需要 Microsoft YaHei、Noto Sans SC 或 SimHei；PDF 已嵌入字体。

在当前目录执行：

```powershell
python draw_fieldbcq_architecture.py --language both --dpi 600
```

依赖：Python、Matplotlib，以及中文字体。脚本不加载模型、权重或数据集。若需较快生成预览，可指定 `--dpi 150`。
