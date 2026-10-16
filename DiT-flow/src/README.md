# 代码结构

本目录包含模型组件、数据处理和训练入口：

本实现的运行根目录是 `DiT-flow/`，并非上一级仓库根目录；以下 `src` 模块命令都从 `DiT-flow/` 内运行。

- `reference_encoder.py`：只读参考手写字，每字共享 patch/SA 编码后用一层 CA 和 FFN 浓缩为 `Z=description_tokens` 个描述；SA、CA 和 FFN 的残差使用初值为 `1e-4` 的静态 LayerScale；全部描述直接拼为 `N·Z` 个风格 token，不再浓缩集合。
- `content_encoder.py`：目标标准字的独立 patch/SA 编码器，结构与风格编码的浓缩前部分相同，深度由 `content_depth` 单独配置；SA 和 FFN 残差使用静态 LayerScale；保留全部空间 token。
- `generator.py`：当前状态分块后堆叠 `SA → Content CA → Style CA → FFN`。SA/FFN 使用 AdaLN-Zero，两类 CA 使用各自的普通 LN 和初值为 `1e-4` 的静态 LayerScale；Style CA 屏蔽补齐描述。
- `model.py`：`forward(state, times, target_standard, reference_handwriting, reference_mask=None)`。支持分别缓存内容与风格再调用 `predict_velocity(state, times, content, style, reference_mask=None)`。
- `flow.py`：纯高斯噪声到手写真值的 flow MSE，以及只编码一次内容/风格的 Euler 采样。
- `data.py`：GNT 读取、字图预处理、全部字图的 uint8 共享内存预加载、按书写者采样的参考与目标数据集。训练和验证及所有 worker 共用一份字图，采样时不再读取 NPY 文件。
- `prepare_data.py`：数据准备入口，从 `DiT-flow/` 内运行 `python -m src.prepare_data`。
- `train.py`：训练入口，从 `DiT-flow/` 内运行 `python -m src.train`。
- `../train-single/train.py`：单作者拟合入口，复用 `train.py`；自动选训练字符最多的作者，仅加载该作者手写图，拟合评估使用同一作者的字符池，选择记录在检查点中。
- `diagnostics.py`：裁剪前的梯度范数、参数/模块诊断及安全裁剪；范数归约溢出时以 float64 重算，真实非有限梯度交由训练器停止或重试。训练目录中的 `gradient_diagnostics.jsonl` 保存诊断。

模型和训练参数集中在 `config.py` 的两个 dataclass。完整默认值、张量接口、位置编码、初始化、LayerScale、flow MSE、采样和版本 5 检查点规则见 [主 README](../README.md)，操作见 [训练说明](../TRAINING.md)。每 batch 更新一次，验证和预览使用当前模型。新训练读取默认值，续训恢复保存配置；运行方式开关 `preload_to_gpu` 始终读取当前配置，设备、worker 和提前结束步可由命令行指定。
