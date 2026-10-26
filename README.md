# FieldBCQ

基于 YOLO11x-seg 的田块实例分割。A 为边界 Dice 监督（BDS），B 为边界像素 BCE 监督（BAPS），C 为区域与边界质量评分（RBQS）。联合评分为 `class_score * sqrt(region_quality * boundary_quality)`。

## 安装

在项目根目录使用与服务器 CUDA 匹配的 PyTorch 环境：

```bash
python -m pip install -e ./src
```

脚本优先导入本项目的 `src/ultralytics`，所有相对参数路径均相对于项目根目录解析。

## 论文表4—10测试

测试仅输出以下结果：

| 表 | 结果 | 模型 |
|---|---|---|
| 4 | 原生掩膜 P、R、F1、AP50、AP75、AP50–95 | 八组消融 |
| 5 | Boundary F1、合并率、过分割率、分数–IoU ECE | 八组消融 |
| 6 | 同权重 class / joint 的 AP75、AP50–95 及固定阈值指标和差值 | FieldBCQ |
| 7 | 固定阈值 P、R、F1、Boundary F1、小地块召回、数量MAE、面积MAPE/WAPE/净偏差 | Baseline、FieldBCQ |
| 8 | 分来源影像数、实例数、F1、Boundary F1 | Baseline、FieldBCQ |
| 9 | 分辨率影像数、实例数、F1及差值 | Baseline、FieldBCQ |
| 10 | 参数量、GFLOPs、原始forward和预测流程平均耗时 | Baseline、FieldBCQ |

### 权重

使用从 `F:\project\二轮` 复制的八组 **best.pt**。原文件保留，测试不读取 `best_fp32.pt` 或 `last.pt`。文件夹名称不作为训练种子的依据；种子从检查点内部读取。

| 配置 | 二轮目录 | 项目默认路径 |
|---|---|---|
| Baseline | exp | model/baseline/best.pt |
| A | a_seed01 | model/a/best.pt |
| B | b_seed01 | model/b/best.pt |
| C | c_seed01 | model/c/best.pt |
| A+B | ab_seed01 | model/ab/best.pt |
| A+C | ac_seed01 | model/ac/best.pt |
| B+C | bc_seed01 | model/bc/best.pt |
| FieldBCQ | abc_seed01 | model/abc/best.pt |

八组配置会根据实际模型头和训练参数核验；配置不匹配直接报错。表6复用 FieldBCQ 的 joint 结果，仅额外运行一次相同权重的 class 模式，并校验检查点 SHA256。

### 服务器数据路径

将 `data/paper_test.yaml` 中的 `path` 修改为服务器数据集根目录。该目录应包含 `images/test` 和对应的 `labels/test`；负样本必须有空的标签文件。例如：

```yaml
path: /data/AgriParcel
train: images/train
val: images/val
test: images/test
nc: 1
names: [field]
```

`data/paper_test_metadata.csv` 来自现有固定测试集清单，共2000张影像、21190个实例，保留来源、分辨率、实例数及影像/标签校验值。程序验证完整覆盖和文件哈希，防止漏图、错配或混入其他测试集。影像文件名应与清单一致；同名文件存在歧义时，`image` 列须改为相对于数据集根目录的路径。

### 执行

服务器先核对文件，随后正式评测：

```bash
python src/scripts/val_paper.py --check-only
python src/scripts/val_paper.py --device 0 --project results/paper_round2_tables4_10
```

`--check-only` 会拒绝空权重和尚未下载的 Git LFS 指针文件，不执行模型反序列化或推理。若遇到 `Git LFS pointer` 或 PyTorch 的 `invalid load key, 'v'`，先将权重下载并写入工作区：

```bash
git lfs install --local
git lfs fetch origin main --include="model/**/*.pt" --exclude=""
git lfs checkout
python src/scripts/val_paper.py --check-only
```

也可将二轮的原始 `best.pt` 按上表复制到服务器。真实权重每份约125—129 MB；以 `version https://git-lfs.github.com/spec/v1` 开头的文件是指针文本。失败的评测目录会保留provenance记录，重跑时使用新的 `--project` 路径。

默认计算表4—10。重复运行时指定新的输出目录。需要重跑其中一张表时：

```bash
python src/scripts/val_paper.py --tables 6 --device 0 --project results/paper_round2_table6
python src/scripts/val_paper.py --tables 10 --device 0 --project results/paper_round2_table10
```

可通过 `--data`、`--metadata-csv`、`--weights-root` 指定其他位置；`--weights 'A+B=/path/to/best.pt'` 可覆盖指定配置。原生指标及固定阈值指标共享每个检查点的一次推理，分组结果从同一批逐图计数汇总，不重新按组推理。

### 评价口径

- 原生掩膜指标：FP32，imgsz=640，batch=8，rect=True，增强关闭；分数下限0.001，NMS IoU=0.70，max_det=710。P/R来自框架原生最优F1工作点，F1为两者调和均值。
- 固定阈值指标：置信度0.25；按分数降序，在同类别中以 mask IoU≥0.5一对一匹配。预测全分辨率二值掩膜恢复到640×640方形补边网格，GT由归一化多边形整数栅格化。
- Boundary F1：3×3双侧形态学边界，方形容差2 px；未匹配实例边界计入分母。合并率以预测实例数为分母，过分割率以GT实例数为分母。
- 合并：一个预测覆盖至少两个同类GT，各覆盖至少50%的GT面积。过分割：一个GT中至少两个同类预测，各至少50%的自身面积位于该GT内，且各覆盖至少5%的GT面积。
- 小地块：评价网格上GT面积≤1024 px。ECE：10个等宽分箱，保留所有预测（含误检），比较分数与最佳同类GT的mask IoU；分数1进入最后一箱。
- 数量MAE覆盖全部影像（含负样本）。面积MAPE只统计成功匹配实例；WAPE为逐图前景并集面积绝对误差之和除以GT前景并集总面积；净偏差使用有符号误差。分组F1及Boundary F1由组内计数汇总。
- 表10：FP32、640×640、batch=1；各预热50次、计时200次，CUDA同步。参数量包含训练辅助分支，GFLOPs为未融合eval图的2×THOP MACs。forward仅计驻留张量上的原始模型前向；预测流程包括预处理、推理、NMS、掩膜生成，不计磁盘读取。使用首张真实测试影像，输入与硬件记录在provenance中。

分母为零时输出空值，Markdown中显示 `—`。CSV和Markdown中的P/R/F1、AP、召回率、合并率、过分割率和面积相对误差均使用百分数；表6/表9差值使用未舍入值计算，ECE差值保持0—1尺度。

### 输出

```text
results/paper_round2_tables4_10/
  table4.csv ... table10.csv
  tables4_10.md
  provenance.json
```

`provenance.json` 保存实际参数、种子、模型/数据/代码哈希、环境及耗时范围；失败时标记failed并保留原因。无预测图片、箱线图、国家分组、距离、物理面积或重复轮次汇总输出。

## 训练

训练入口保留在 `src/scripts/train_paper.py`。需要训练新权重时：

```bash
python src/scripts/train_paper.py --stage fieldbcq --data data/datasets.yaml --weights model/pretrained/yolo11x-seg.pt --epochs 80 --batch 8 --imgsz 640 --seed 0
```

八组训练选项依次为 `baseline`、`pcsc`、`baps`、`rbqs`、`baps_pcsc`、`bds_rbqs`、`baps_rbqs`、`fieldbcq`。模型结构、损失和历史检查点类路径保留兼容。

## 开发验证

数值、表格、路径及小模型回归测试位于 `tests/`，可在服务器运行：

```bash
python -m unittest discover -s tests
```

本次按要求未在本地执行评测或推理回归。原始影像与标注按各来源数据许可使用；模型框架沿用仓库的AGPL-3.0许可。
