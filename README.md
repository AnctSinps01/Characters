# 线条图变换实验

本仓库保留两条独立路线：原有的手写字生成模型位于 [`DiT-flow/`](DiT-flow/README.md)；通用线条图连续几何变形模型位于 [`twist/`](twist/README.md)。`twist` 使用单个网络预测定常速度场，以 Scaling and Squaring 完成双向配准和插值。

## 目录

| 路径 | 用途 |
|---|---|
| [`DiT-flow/`](DiT-flow/README.md) | 原 Glyph2Hand 的模型、配置、训练入口、测试和设计文档 |
| [`twist/`](twist/README.md) | 通用线条图 Transformer 速度场配准、双向插值与 loss |
| [`glyphprep/`](glyphprep/prepare.py) | 公共 CASIA-HWDB 数据准备：`python -m glyphprep.prepare` 生成 DiT-flow 和 twist 共用的 `data/prepared/` |
| [`scripts/prepare_dataset.sh`](scripts/prepare_dataset.sh) | 一键下载、解压并准备数据 |
| `fonts/` | 公共字体资源 |
| [`One-DM/`](One-DM/README.md) | 保留原位置的参考项目 |
| [`readme/`](readme/README.md) | 公共技术分析；原 DiT-flow 的 LaTeX 和 PDF 已移到 `DiT-flow/readme/` |
| [`essays/`](essays/README.md) | 公共论文资料 |
| [`requirements.txt`](requirements.txt) | 公共依赖，含 twist 距离场预处理使用的 SciPy |
| `data/`、`runs/` | 数据和实验结果的根目录，按用途或实现分子目录；当前检出的仓库不含这些数据与结果 |

`data/prepared/` 是两条路线共用的数据集格式（`writers/*.npy`、`standards.npy`、`manifest.json`，版本 1），由 `glyphprep` 生成一次即可。`DiT-flow/src/data.py` 只负责加载已准备的数据并按书写者采样；梯度诊断等可复用工具仍保持原来的调用关系。

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

输入两张近似同构的线条图 A、B，通过单个网络内堆叠 SA / CA / FFN 预测定常速度场，只保留边界置零。Scaling and Squaring 积分得到 A 走 t、B 反向走 1−t 的两张变形图，再以 (1−t)、t 加权生成轨迹。损失固定为双向图像相似度、双向距离场 Chamfer 和温和的速度梯度平滑，不需要真实变形场或中间帧标签。任务不依赖汉字标签；全数据集同字手写图配对训练见 [`twist/train-all-pair/`](twist/train-all-pair/README.md)。

完整规划见 [`twist/DESIGN.md`](twist/DESIGN.md)。
