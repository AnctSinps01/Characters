# 代码结构

本目录包含模型组件、数据处理和训练入口：

本实现的运行根目录是 `DiT-flow/`，并非上一级仓库根目录；以下 `src` 模块命令都从 `DiT-flow/` 内运行。

- `reference_encoder.py`：共享的逐组编码、每组 `Z` 个描述 token，以及顺序无关的 `M` 个风格 token 汇总。集合汇总由 `set_pool_depth` 个独立的 `CA → SA → FFN` 模块组成。
- `generator.py`：仅对当前流状态 `state` 分块的条件 DiT，输入通道数为 `in_channels`；SA/FFN 使用 AdaLN-Zero，风格 CA 使用普通前置 LN 和残差。
- `model.py`：组合参考分支与速度预测主干。`forward(state, times, reference_handwriting, reference_standard, reference_mask=None)` 编码参考后调用 `predict_velocity(state, times, style)`；主干没有独立的目标标准字输入。
- `flow.py`：构造标准字加噪声起点的 flow-matching 训练样本，并提供复用风格编码的 Euler 采样。`target_standard` 仍用于构造起点，积分时主干只接收当前状态、时间和风格。
- `data.py`：GNT 读取、字图预处理、全部字图的 uint8 共享内存预加载、按书写者采样的参考与目标数据集。每个 episode 还带有稳定的 `author_index`，供作者风格损失组成同作者/异作者配对。训练和验证及所有 worker 共用一份字图，采样时不再读取 NPY 文件。
- `prepare_data.py`：数据准备入口，从 `DiT-flow/` 内运行 `python -m src.prepare_data`。
- `train.py`：训练入口，从 `DiT-flow/` 内运行 `python -m src.train`。
- `diagnostics.py`：裁剪前的梯度范数、参数/模块诊断及安全裁剪；范数归约溢出时以 float64 重算，真实非有限梯度交由训练器停止或重试。训练目录中的 `gradient_diagnostics.jsonl` 保存诊断。

模型和训练参数集中在 `config.py` 的 `ModelConfig` 与 `TrainingConfig`。新训练直接读取这些默认值，续训恢复检查点中的配置；仅运行方式开关 `preload_to_gpu` 始终读取当前 `config.py`。参考字数量在 `min_references` 到 `max_references` 之间均匀采样；数据准备默认至少保留 `max_references + 1` 个不同字符的书写者。`author_style_loss_weight` 控制作者风格损失相对 flow loss 的权重，默认值为 `0.05`；设为 `0` 可关闭。风格损失对固定 token 槽位使用归一化余弦相似度：同作者最小化 `1 - cos`，不同作者最小化 `cos²`，后者鼓励正交而不会把相反方向当作“足够不同”。一个 batch 中没有同作者重复样本时，同作者项自动为零。数据准备与完整单卡训练流程的使用方式见 [训练说明](../TRAINING.md)。网络尺寸与训练超参数仍需通过真实数据实验验证。
