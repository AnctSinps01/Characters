# Glyph2Hand

项目说明已移至 [LaTeX 文档](readme/README.tex)，包含任务定义、模型架构图与未来可能方向。

当前生成主干只读取流状态、时间和参考风格；目标标准字仅用于构造加噪起点，不再作为额外图像通道传入主干。参考分支仍读取成对的手写字与标准字。

`readme/README.v1-with-diagrams.*` 是历史首版设计，不代表当前实现。

单张 RTX 3090 的环境准备、CASIA 数据处理、训练及断点续训见 [训练说明](TRAINING.md)。

- `src/prepare_data.py`：把服务器上的原始 GNT 转成按书写者组织的数据，并渲染标准字；使用 `python -m src.prepare_data` 运行。
- `src/train.py`：训练、EMA 验证、生成预览、检查点保存与恢复；使用 `python -m src.train` 运行。
- `src/config.py`：模型和训练参数，包括参考字数量范围（默认均匀采样 1–8 个）。
