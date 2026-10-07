# 全数据集同字配对训练

A、B 来自两个不同书写者，字符标签相同。使用 `data/prepared/manifest.json` 与 `writers/*.npy` 中的全部手写图；标准字体图不参与此训练。

全部字图启动时以 uint8 载入内存，按原有书写者 train/val 划分配对。每轮打乱全部可配对源图，每张图各做一次 A，再从同一划分、同字的其他书写者中抽 B；新轮次重新抽取 B。不展开巨大的全部两两组合表。只有一份的字无法同字跨书写者配对，会跳过并打印数量。

A、B 各自在缩放到模型尺寸后，按每张图自身的最小、最大值计算 `2 * (pixel - min) / (max - min) - 1`。最暗为 −1、最亮为 +1，距离场仍以 `pixel <= 0` 为前景。恒定图（`max == min`）在配对前排除，写入 `excluded_images.json`；不把空白图当作笔迹。

## 云端运行

同步整个 `twist/` 和 `requirements.txt`，保留已有 `data/prepared/`。从仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python twist/train-all-pair/train.py --device cuda --output runs/twist-all-minmax
```

默认从零训练到累计 100,000 步。数据路径默认由脚本所在仓库确定，其他位置用 `--data /实际路径/data/prepared`。上面的输出目录为 `runs/twist-all-minmax/`。

继续训练：

```bash
python twist/train-all-pair/train.py --device cuda --resume runs/twist-all-minmax/last.pt
```

也可以直接训练到指定累计步数：

```bash
python twist/train-all-pair/train.py --device cuda --steps 10000 --output runs/twist/train-all-pair/ten-thousand
```

## 配置

[settings.py](settings.py) 设置 batch size、学习率、步数、验证/保存间隔、精度、缓存、预览和 GIF 帧数。[../config.py](../config.py) 设置模型结构与三类 loss 比例。所有字段支持同名命令行参数，学习率用 `--lr`。

- 默认 batch=32、lr=1e-4；`precision=auto` 在支持的 CUDA 显卡上使用 BF16，其余使用 FP32。几何采样和 loss 保持 FP32。
- 全部原始字图以 uint8 缓存；`preload_to_gpu=True` 可让它们常驻训练显卡，需显存足够。
- writer 加载日志分别打印累计 `read`（读文件）、`copy`（复制像素）、`metadata`（标签与书写者索引）时间；全部加载后单独打印同字配对索引耗时。这些时间也保存到 `run.json` 的 `data_loading_seconds`。CPU 像素复制采用 NumPy 直接写入 Tensor 的共享存储。
- 源图 A 的距离场按需准备，用 CPU LRU 缓存复用；默认最多 8,192 张，64×64 时约 128 MiB，满时淘汰旧项。距离场会依据实际输入尺寸计算。
- 默认固定选最多 512 对验证样本，在验证书写者之间配对，后续保持不变。`val_batches=0` 在首次训练时启用全部可配对验证源图，每个源图对应一个固定 B。
- 默认每 1,000 步验证并保存；第 0 步、短运行结束、训练结束也验证和保存。普通日志每 50 步记录更新后的训练批次。
- 默认预览 4 对验证图，尽量选择不同字符；只保存轨迹 GIF。帧数由 `animation_frames` 控制，包含 t=0、1。

需要临时看 GIF 对应的输入时，单独运行：

```bash
python twist/train-all-pair/show_inputs.py --run runs/twist-all-minmax --output inputs.png && rm -- twist/train-all-pair/show_inputs.py
```

在服务器仓库根目录生成 `inputs.png`，成功后删除临时脚本。每行对应一对 GIF 预览，左 A、右 B，行号对应 GIF 中的 `val_00`、`val_01` 等编号，并标注书写者与字符 Unicode 编码。脚本读取 `preview_pairs.json`，只加载这些图，使用该实验的缩放与归一化方式；可在训练期间运行。`--run` 指向当前实际训练输出目录；数据迁移后加 `--data /实际路径/data/prepared`。

## 输出与续训

| 文件 | 内容 |
|---|---|
| `run.json` | 配置、数据量、跳过数量、参数量与固定验证集的原图 MSE 基线 |
| `excluded_images.json` | 排除的恒定图的原始索引、字符、书写者、划分与像素范围 |
| `preview_pairs.json` | 固定预览对的字符与两位书写者 |
| `metrics.jsonl` | 训练批次/固定验证集合的三类 loss、形变子项、阶段 Lipschitz 上界；训练日志含裁剪前梯度范数 |
| `best.pt`、`last.pt` | 最优验证和最新保存的训练状态，含模型、优化器、轮次/游标、配对随机状态与固定验证对 |
| `previews/` | 轨迹 GIF，帧数由 `animation_frames` 控制 |
| `summary.json` | 最终步数、轮次和验证指标，以及完成/短运行结束/中断状态 |

检查点不包含全量字图和距离缓存。续训需要原预处理数据，迁移目录后用 `--data` 指向新位置；启动校验 manifest 指纹、归一化方式与可配对样本索引。轮次、剩余源图顺序及随机 B 的状态均恢复。`Ctrl+C` 等当前优化器更新完成后保存。

旧版固定按 0–255 缩放的全数据检查点不能接着此版续训，因为输入、损失基线与可配对样本可能变化。同步更新后使用新的输出目录从零开始，无需重做预处理数据。

续训使用保存的结构、loss 和训练配置，可覆盖累计步数、batch、运行间隔、精度、预加载位置、距离缓存容量、预览数与 GIF 帧数。固定验证对保持不变；例如改帧数加 `--animation-frames 61`。新训练使用新输出目录；从较早检查点恢复也应指定新目录。

日志中的 `step` 是累计优化器更新次数，`epoch` 是可配对训练源图的遍历进度；每轮每张图作为 A 一次。`train` 是更新后重新评估当前 batch 的总 loss，`mse`、`distance`、`def` 是乘系数前的三项。默认 `train = mse + distance + 1e-4 * def`；`def` 是弯曲能量加 `0.01 * 位移平方均值`。`grad` 是本次反向传播中裁剪前的全模型梯度 L2 范数，默认裁剪上限为 1。

纯连续形变无法消除所有真实手写字的拓扑与噪声差异，先比较验证 MSE 与原图基线，并检查 GIF 的端点和中间轨迹。

本地检查：

```bash
python -m unittest discover -s twist/train-all-pair -p "test_*.py" -v
```
