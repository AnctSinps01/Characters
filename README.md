# 线条图变换实验

本仓库保留两条独立路线：原有的手写字生成模型位于 [`DiT-flow/`](DiT-flow/README.md)；通用线条图连续几何变形模型位于 [`twist/`](twist/README.md)。`twist` 已实现串联配准、几何约束和三类 loss。

## 目录

| 路径 | 用途 |
|---|---|
| [`DiT-flow/`](DiT-flow/README.md) | 原 Glyph2Hand 的模型、配置、CASIA 数据处理、训练入口、测试和设计文档 |
| [`twist/`](twist/README.md) | 通用线条图串联 Transformer 配准、几何约束与 loss |
| `fonts/` | 公共字体资源 |
| [`One-DM/`](One-DM/README.md) | 保留原位置的参考项目 |
| [`readme/`](readme/README.md) | 公共技术分析；原 DiT-flow 的 LaTeX 和 PDF 已移到 `DiT-flow/readme/` |
| [`essays/`](essays/README.md) | 公共论文资料 |
| [`requirements.txt`](requirements.txt) | 公共依赖，含 twist 距离场预处理使用的 SciPy |
| `data/`、`runs/` | 数据和实验结果的根目录，按用途或实现分子目录；当前检出的仓库不含这些数据与结果 |

现有 `data.py` 依赖字符标签、CASIA 和书写者采样，先随原实现保留。梯度诊断等可复用工具也先保持原来的调用关系；等 `twist` 确实需要时再提取到根目录的公共模块。

## 原实现的运行位置

原实现现在从 `DiT-flow/` 内运行，Python 包名仍为 `src`：

```bash
cd DiT-flow
python -m pip install -r ../requirements.txt
python -m src.train --help
python count_parameters.py
python -m unittest discover -s tests -v
```

完整数据准备和训练命令见 [`DiT-flow/TRAINING.md`](DiT-flow/TRAINING.md)。移动目录不改变 Python 源码、模型结构或检查点格式；现有数据和检查点通过调整命令中的相对路径继续使用。

## twist 的当前目标

输入两张近似同构的线条图 A、B，通过固定边界、受 Lipschitz 约束的小残差采样变换串联，生成可查询任意 `t ∈ [0,1]` 的轨迹。三类 loss 为软平衡 MSE、原始 A 的距离场匹配与形变正则，不需要真实变形场或中间帧标签。任务不依赖汉字标签；数据生成、完整训练入口和外观分支尚未实现。

完整规划见 [`twist/DESIGN.md`](twist/DESIGN.md)。
