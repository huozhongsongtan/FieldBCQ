# FieldBCQ 后处理实现

本目录包含 FieldAgent 最终五组协议和共享几何引擎，统一入口为 `src/scripts/run_postprocess.py`。安装和运行命令见[项目 README](../../README.md)。

`final_protocol.py` 固定五组主对照和仅选工具消融；`simple_sampling.py` 实现两种抽稀；`tool_router.py` 与 `routed_editor.py` 实现视觉选择、编辑及预览审核。`fieldagent/_engine/` 提供几何处理、评价和兼容依赖。

数据清单与服务配置由 `--inputs`、`--planner` 显式指定；输出由 `--output` 指定。核心候选参数、提示词、几何验收阈值和智能体算法沿用来源实现。
