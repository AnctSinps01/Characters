# 代码结构

本目录包含模型组件、数据处理和训练入口。模块命令的运行根目录是 `DiT-flow/`。

- `reference_encoder.py`：读取逐一配对的参考手写图和标准字形。两侧独立 patch 投影后分别加二维位置和类型嵌入，直接经过 `local_depth` 层双路联合注意力；两侧输出相减，补回二维位置，再用恰好一层 CA/FFN 浓缩为 `Z=description_tokens` 个描述。查询和浓缩输出不加位置；各对描述直接拼为 `N·Z` 个风格 token。
- `layers.py`：共享层定义。`JointAttentionBlock` 的两侧独立 LN/QKV/输出投影/FFN，拼接 Q/K/V 后以一次 SDPA 做双向联合注意力，再分别更新两侧。参考编码器与内容编码器的注意力/FFN 残差使用可学习静态 LayerScale，初值 `1e-4`。
- `content_encoder.py`：目标标准字的独立 patch/SA 编码器，深度由 `content_depth` 单独配置；输入加二维位置，保留全部空间 token，不浓缩，也不与参考标准分支共享参数。
- `generator.py`：当前状态独立 patch 投影并加二维位置，堆叠 `SA → Content CA → Style CA → FFN`。SA/FFN 使用 AdaLN-Zero，两类 CA 使用独立普通 LN 和初值 `1e-4` 的静态 LayerScale；Style CA 屏蔽补齐描述。
- `model.py`：`forward(state, times, target_standard, reference_handwriting, reference_standard, reference_mask=None)`。可分别调用 `encode_content(target_standard)` 与 `encode_style(reference_handwriting, reference_standard, reference_mask=None)`，再调用 `predict_velocity(state, times, content, style, reference_mask=None)`。
- `flow.py`：纯高斯噪声到手写真值的 flow MSE，以及只编码一次内容/风格的 Euler 采样；`sample_euler` 同样要求传入配对的 `reference_standard`。
- `data.py`：GNT 读取、字图预处理、uint8 字图共享预加载和按作者采样的 episode。目标和参考标准字均按全局字符标签从原有标准字库提取；参考手写/标准一起补白并使用同一有效位掩码。数据准备格式保持版本 `1`。
- `prepare_data.py`：数据准备入口，在 `DiT-flow/` 内运行 `python -m src.prepare_data`。
- `train.py`：共享训练入口，在 `DiT-flow/` 内运行 `python -m src.train`；全量、单作者、缩小数据集训练共用配对参考、训练、验证预览和续训路径。
- `../train-single/train.py`：单作者拟合入口，自动选字符最多的训练作者，由 `train-single/config.py` 限制固定字符子集。
- `../train-half/train.py`：按其目录 `config.py` 的 `DATASET_SIZE` 缩减作者和各作者字符池，用于缩小规模拟合。两种拟合模式均在所选训练池中评估，选择记录于检查点，续训恢复保存的选择。
- `diagnostics.py`：裁剪前梯度范数、参数/模块诊断和安全裁剪；范数归约溢出时以 float64 重算，真实非有限梯度交由训练器停止或重试。

模型和训练参数集中在 `config.py` 的两个 dataclass。完整默认值、张量接口、位置编码、初始化、损失、采样及版本 `6` 检查点规则见 [主 README](../README.md)，操作见 [训练说明](../TRAINING.md)。每 batch 更新一次，目标只有 flow MSE。新训练读取默认值，续训恢复保存设置；`preload_to_gpu` 始终读取当前配置，设备、worker 和提前结束步可由命令行指定。旧版本 `1..5` 模型不兼容，需使用新输出目录重新训练；已有预处理数据可继续使用。
