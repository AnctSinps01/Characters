# 代码结构

本目录包含模型组件、数据处理和训练入口：

- `reference_encoder.py`：共享的逐组编码、每组 `Z` 个描述 token，以及顺序无关的 `M` 个风格 token 汇总。集合汇总由 `set_pool_depth` 个独立的 `CA → SA → FFN` 模块组成。
- `generator.py`：保留目标二维 patch 的条件 DiT；SA/FFN 使用 AdaLN-Zero，风格 CA 使用普通前置 LN 和残差。
- `model.py`：组合参考分支与速度预测主干。
- `flow.py`：构造标准字加噪声起点的 flow-matching 训练样本，并提供复用风格编码的 Euler 采样。
- `data.py`：GNT 读取、字图预处理、全部字图的 uint8 共享内存预加载、按书写者采样的参考与目标数据集。训练和验证及所有 worker 共用一份字图，采样时不再读取 NPY 文件。
- `prepare_data.py`：数据准备入口，从项目根目录运行 `python -m src.prepare_data`。
- `train.py`：训练入口，从项目根目录运行 `python -m src.train`。

模型和训练参数集中在 `config.py` 的 `ModelConfig` 与 `TrainingConfig`。新训练直接读取这些默认值，续训恢复检查点中的配置；仅运行方式开关 `preload_to_gpu` 始终读取当前 `config.py`。参考字数量在 `min_references` 到 `max_references` 之间均匀采样；数据准备默认至少保留 `max_references + 1` 个不同字符的书写者。数据准备与完整单卡训练流程的使用方式见 [训练说明](../TRAINING.md)。网络尺寸与训练超参数仍需通过真实数据实验验证。
