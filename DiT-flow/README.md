# Glyph2Hand

这是原有的 DiT + flow matching 手写字生成实现。模型、训练和测试均从本目录运行；公共字体、依赖和参考资料保留在仓库根目录。新的通用线条图变形路线见 [`../twist/`](../twist/README.md)。

项目说明已移至 [LaTeX 文档](readme/README.tex)，包含任务定义、模型架构图与未来可能方向。

当前生成主干只读取流状态、时间和参考风格；目标标准字仅用于构造加噪起点，不再作为额外图像通道传入主干。参考分支仍读取成对的手写字与标准字。

`readme/README.v1-with-diagrams.*` 是历史首版设计，不代表当前实现。

单张 RTX 3090 的环境准备、CASIA 数据处理、训练及断点续训见 [训练说明](TRAINING.md)。

- `src/prepare_data.py`：把服务器上的原始 GNT 转成按书写者组织的数据，并渲染标准字；使用 `python -m src.prepare_data` 运行。
- `src/train.py`：训练、EMA 验证、生成预览、检查点保存与恢复；使用 `python -m src.train` 运行。
- `src/config.py`：模型和训练参数，包括参考字数量范围（默认均匀采样 1–16 个）。

```bash
cd DiT-flow  # 从仓库根目录进入本实现
python -m pip install -r ../requirements.txt
python count_parameters.py
python -m unittest discover -s tests -v
```

Python 源码及包内导入保持原样。`DiT-flow` 是目录名，不作为 Python 导入包名使用。
