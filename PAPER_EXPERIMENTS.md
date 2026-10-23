# BAPS-only 与论文实验采集

所有命令在项目根目录执行，使用本项目修改过的 Ultralytics，不能用官方安装包替代。实际影像、标注和初始预训练权重需要自行准备。保留原来的训练/验证/测试划分，并使用已有三轮实验的真实 seed 集合；下文 seed=0 只是命令示例。

## 1. 五组设置

| 论文名称 | `--stage` | 额外目标 | 边界权重 |
|---|---|---|---:|
| Baseline | `baseline` | 无 | — |
| BDS-only | `bds` 或 `pcsc` | 边界 Dice | 0.20 |
| BAPS-only | `baps` 或 `baps_only` | 边界像素 BCE | 0.20 |
| BDS+BAPS | `bds_baps` 或 `baps_pcsc` | 同一边界分支的 Dice+BCE | 0.20 |
| FieldBCQ | `fieldbcq` | Dice+BCE 与 RBQS 质量监督 | 0.20；质量项 0.25 |

BAPS-only 与 BDS+BAPS 复用 `Segment_BAS` 网络结构，区别由 `boundary_mode=bce` / `dice_bce` 决定，并保存在检查点中。新增 `v8BAPSSegmentationLoss` 只计算边界 BCE，不计算边界 Dice，也没有质量头。原有类名中的 BAS 是历史代码标识，不能据此判断某个旧检查点是 BAPS-only。新增设置需要独立训练，不能将旧的联合边界结果改名为 BAPS-only。

## 2. 训练与基本验证

正式训练前可以先对五组做一次 smoke（默认使用论文 x 规模、随机初始化、FP32）：

```bash
python src/scripts/smoke_paper.py --data data/datasets.yaml --device 0 --epochs 1 --batch 2 --imgsz 320 --train-images 16 --val-images 8
```

脚本按 seed 固定抽取少量 train/val 图像，为各组依次运行训练、原生验证、检查点保存、配置重载检查及预测，最后汇总 `SMOKE RESULT: 5/5 passed`。选取样本中需至少包含一个非空标注；否则增加样本数或更换 seed。输入数据、标签和原始 YAML 不会复制或改写，结果及临时子集配置保存到带时间戳的 `results/smoke/` 子目录。每组检查通过不要求 AP 达到某个数值；这是流程检查，不能填入论文结果。

默认关闭 AMP 和绘图，并用 Matplotlib 自带字体避免字体下载阻塞。增加 `--amp` 可进一步检查混合精度，`--weights PATH` 可检查指定本地预训练权重的加载。`--scale n` 只检查较小网络的流程，不能替代默认 x 模型检查。只跑部分组可使用 `--stages baps fieldbcq`。独立的模型图、正/空标注梯度检查仍可运行 `python src/scripts/check_paper_models.py`。

```powershell
# 先激活原有实验使用的 Python / PyTorch 环境
python -m pip install -e ./src
python src/scripts/check_paper_models.py
python src/scripts/train_paper.py --stage baps --seed 0 --device 0 --epochs 80 --batch 8 --imgsz 640
```

默认初始权重是 `model/pretrained/yolo11x-seg.pt`。如原实验使用了其他初始权重，显式传入 `--weights` 并保持各组一致。`--no-pretrained` 表示从头训练，只有原实验也如此时才使用。`--no-amp` 关闭训练混合精度，默认开启 AMP。训练输出为 `results/train/baps_seed0/`；对原实验每个 seed 分别训练，并保持轮数、增强、优化器及其他设置一致。

```powershell
python src/scripts/val_paper.py --weights results/train/baps_seed0/weights/best.pt --device 0
```

上述脚本保留现有原生验证入口。完整论文指标建议用下一节的采集脚本；所有组都用同一版本重新评估。`val_paper_all_models.py` 默认仍只验证四个已有归档权重，BAPS-only 需显式传入 `--weights NAME=PATH`，或直接使用新的采集脚本。

每次训练自动保存 `args.yaml`（框架原有）和 `run_metadata.json`（新增）：包括实际 trainer 参数、seed、是否恢复训练、实际 AMP 状态、优化器和参数组、软硬件环境、源代码 SHA256、best/last 检查点 SHA256，以及 CUDA 训练峰值显存。恢复训练时回调会检查实际设置与 `--stage` 是否一致。

## 3. 一次获取论文需要的值

```powershell
python src/scripts/collect_paper_metrics.py --weights baps_seed0=results/train/baps_seed0/weights/best.pt --data data/datasets.yaml --split test --device 0 --precision fp32 --batch 8 --imgsz 640 --conf 0.25 --warmup 20 --iterations 100 --plots
```

可重复传入 `--weights NAME=PATH` 顺序评估五组模型。原来四个归档权重分别为 `model/baseline/best.pt`、`model/a/best.pt`、`model/ab/best.pt`、`model/abc/best.pt`；它们不能替代全部随机种子的检查点。输出目录是 `results/paper/NAME/`，已存在时会报错，避免混合结果；重新评估请更换 `--project` 或 NAME。仅验证精度可加 `--skip-benchmark`。CPU 使用 `--device cpu --precision fp32`；FP16 需要 CUDA。

| 输出值 | `paper_report.json` 中的位置 | 用途 |
|---|---|---|
| mask AP50、AP75、AP50–95 | `native.mask` | 主结果与消融表 |
| 原生 Precision、Recall、F1 | `native.mask` | Ultralytics 原生 P/R 统计对应的 F1 |
| 固定阈值 Precision、Recall、F1 | `fixed.precision/recall/f1` | 比较同一置信度阈值下的检出表现 |
| 边界 P/R/F1、边界带 IoU | `fixed.boundary_*` | 边界与轮廓质量 |
| 合并率、过分割率、小田块召回 | `fixed.merge_rate/oversegmentation_rate/small_parcel_recall` | 田块结构误差 |
| score–IoU ECE、10 个 bin 计数 | `fixed.score_iou_ece/calibration_bins` | 质量分数校准；不是分类 ECE |
| 匹配轮廓平均对称距离 | `fixed.matched_boundary_mean_distance_*` | 轮廓位置误差 |
| 匹配田块面积 MAE、MAPE | `fixed.matched_parcel_area_*` | 匹配实例的面积误差 |
| 每图田块数量 MAE、前景总面积 MAE/WAPE | `fixed.parcel_count_* / foreground_area_*` | 农业应用的数量与面积统计 |
| 来源/国家/分辨率分组指标 | `groups` | 需要真实逐图 metadata；这里是固定阈值指标 |
| 参数量、GFLOPs、延迟、吞吐量、显存 | `efficiency` | 模型复杂度与推理效率 |
| 环境、seed、训练配置、权重/数据/代码 hash、evaluator 版本 | 报告顶层及 `evaluation_protocol` | 可复现记录 |

同时保存 `per_image.csv` 与 `dataset_manifest.json`。图像和标注逐文件计算 SHA256；空白图像必须有显式空标签文件，缺标签会报错。`--plots` 保存原生 PR/F1 曲线等图。原生 F1 随框架的 P/R 工作点计算，与固定 `--conf` 的 F1 不应混写。旧稿数字若使用其他评价口径，需统一重算后再填表。

## 4. 指标口径与分组元数据

固定指标在 `imgsz × imgsz` 的保持长宽比 letterbox 网格上计算，预测为二值实例 mask，GT 为 YOLO 多边形栅格化。匹配按分数降序、同类别、最高可用 mask IoU 进行一对一分配，默认 `--match-iou 0.5`；这与原生 AP 评价器分别记录。固定 `--conf 0.25`，不能根据测试结果选择阈值。

边界采用 3×3 膨胀与腐蚀的双侧轮廓，默认方形容差 `--boundary-tolerance 2` 个评价网格像素。未匹配实例的轮廓计入分母。边界带 IoU 汇总匹配实例交集和所有实例的边界带面积。平均对称距离按匹配轮廓像素汇总，面积 MAE/MAPE 也仅针对匹配实例，必须结合召回与 FP/FN 阅读；无匹配时这些误差为 null，不能当作 0。这里的轮廓距离不等同于 HD95，边界带 IoU 也不应直接改称其他论文定义的 Boundary IoU。

合并率：一个预测覆盖至少两个同类 GT，各 GT 覆盖比例 ≥0.5，除以预测实例数。过分割率：一个 GT 至少含两个预测，每个预测至少 50% 面积位于该 GT 内且覆盖该 GT 至少 5%，除以 GT 数。小田块默认 GT 栅格面积 ≤1024 像素，可用 `--small-area` 调整；跨分辨率物理面积不同，论文需说明此限制。

可选 `--metadata-csv data/eval_metadata.csv`，CSV 格式如下，`image` 相对数据 YAML 的数据根目录：

```csv
image,source,country,resolution_m,pixel_size_x_m,pixel_size_y_m
images/test/example.png,FTW,France,10,10,10
```

提供 metadata 时，每张评价图像必须有一条唯一记录。来源、国家、名义分辨率可以只填写已确认的列；物理单位计算需要同时填写真实的 `pixel_size_x_m/pixel_size_y_m`（原图每像素多少米），脚本会进一步校正 letterbox 缩放。仅有名义分辨率时不会猜测平方米或米制距离。若影像曾重采样，必须使用重采样后的实际像元尺寸；经纬度像元不能直接当米。没有这些信息时仍输出像素面积和像素距离。分组原生 AP 可另外使用已有 `val_by_source.py`；新 `groups` 不包含分组 AP。

## 5. 效率测量与完整模型的同权重干预

```powershell
python src/scripts/benchmark_paper.py --weights results/train/baps_seed0/weights/best.pt --device 0 --precision fp32 --batch 1 --imgsz 640 --warmup 20 --iterations 100 --images data/images/test/example.png --output results/baps_efficiency.json
```

计时分别记录：① 已驻留设备张量的未融合模型前向；② 已解码 BGR 图像经预处理、预测（predictor 可融合层）、NMS、mask 后处理、Results 的流程。不包含磁盘读取、指标计算、模型加载。每个流程单独预热，CUDA 计时前后同步，输出 batch 平均/P50/P95 延迟、每图摊销延迟、images/s、峰值 allocated/reserved 显存。CPU 显存为 null。参数量统计模型总参数，包含训练期边界分支；GFLOPs 按实际推理前向统计，为单图、FP32、`2 × THOP MACs`，不能保证覆盖所有自定义算子，需写明统计方法；失败时为 null 并记录原因。

采集脚本从真实评价图像提取计时输入；单独 benchmark 不传 `--images` 时只用空白合成图，并明确标注其后处理耗时不代表真实输入。论文表使用相同硬件、软件、输入尺寸、batch、精度、阈值和 max-det；batch=1 延迟与 batch=8 吞吐量可分别报告。训练 AMP 与推理 FP16/FP32 分开记录。

FieldBCQ 同一检查点可分别采集 `--quality-blend 0`（C0）和 `--quality-blend 1`（C2），使用不同 NAME，评估分数融合的影响；BAPS-only 不使用这个选项。

## 6. 多随机种子汇总

```powershell
python src/scripts/summarize_paper_runs.py --reports results/paper --output results/paper_summary
```

输出 CSV/JSON 的均值和样本标准差（ddof=1）；单次运行标准差为 null。仅将数据内容、metadata、评价协议、源代码版本一致的报告合为一组，重复的设置/seed/协议会报错。相对 Baseline 的 AP 点数增益只按已知且相同的 seed 配对，保存在 JSON 的 `paired_gain_over_baseline`。seed 从检查点的训练参数读取，不从文件名猜测。旧检查点缺完整训练配置时，应核对原始实验记录后使用，不将文件名作为证据。计时数据不在多 seed 汇总中混合平均，请直接使用每份效率报告。

这些脚本为补充实验和农业数量/面积应用提供可核查数值，真实改进幅度需要完成实验后判断，尤其不能提前认定 BAPS-only 的 F1 高低是否正确。

## 本机验证记录

五组完整模型通过前向、正样本/空标注损失与梯度检查；指标与路径回归测试通过。另在本机已有 `yolov11` 环境（PyTorch 2.11.0+cpu、OpenCV 4.13.0）使用小型 n 规模模型和合成数据跑通 1 epoch 训练、检查点重载、原生验证、固定指标、THOP 与两种计时流程。这仅验证执行流程，不代表论文模型精度或 GPU 性能。

本机已有 `field` 环境（PyTorch 2.8.0+cpu）调用 OpenCV 仿射增强时出现 OpenMP 运行库冲突；未修改该环境依赖，也未用允许重复运行库的环境变量掩盖问题。正式实验请使用原实验可正常训练的环境；如使用本机 `field` 环境，应先解决此依赖冲突。
