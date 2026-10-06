# 线条图变换实验

本仓库保留两条独立路线：原有的手写字生成模型位于 [`DiT-flow/`](DiT-flow/README.md)；新的连续几何变形方案位于 [`twist/`](twist/README.md)。`twist` 当前只做设计，不包含模型或训练代码。

## 目录

| 路径 | 用途 |
|---|---|
| [`DiT-flow/`](DiT-flow/README.md) | 原 Glyph2Hand 的模型、配置、CASIA 数据处理、训练入口、测试和设计文档 |
| [`twist/`](twist/README.md) | 通用线条图连续变形的规划与后续独立实现 |
| `fonts/` | 公共字体资源 |
| [`One-DM/`](One-DM/README.md) | 保留原位置的参考项目 |
| [`readme/`](readme/README.md) | 公共技术分析；原 DiT-flow 的 LaTeX 和 PDF 已移到 `DiT-flow/readme/` |
| [`essays/`](essays/README.md) | 公共论文资料 |
| [`requirements.txt`](requirements.txt) | 现有公共依赖；twist 需要的额外依赖等方案验证后再确定 |
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

输入两张近似同构的线条图 A、B，生成一条合理的连续轨迹，并能查询任意 `t ∈ [0,1]` 的画面。主结构由平滑、可逆的空间变形场约束；小幅外观修正仅用于亮度和已确认的微小噪声。任务不依赖汉字标签，也不以少样本风格生成作为首版目标。

完整规划见 [`twist/DESIGN.md`](twist/DESIGN.md)。
