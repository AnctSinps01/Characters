# 全数据集同字配对训练

A、B 来自两个不同书写者，字符标签相同。使用 `data/prepared/manifest.json` 与 `writers/*.npy` 中的全部手写图；标准字体图不参与此训练。

全部字图启动时以 uint8 载入内存，按原有书写者 train/val 划分配对。每轮打乱全部可配对源图，每张图各做一次 A，再从同一划分、同字的其他书写者中抽 B；新轮次重新抽取 B。不展开巨大的全部两两组合表。只有一份的字无法同字跨书写者配对，会跳过并打印数量。

A、B 各自在缩放到模型尺寸后，按每张图自身的最小、最大值计算 `2 * (pixel - min) / (max - min) - 1`。最暗为 −1、最亮为 +1，距离场仍以 `pixel <= 0` 为前景。恒定图（`max == min`）在配对前排除，写入 `excluded_images.json`；不把空白图当作笔迹。

## 云端运行

同步整个 `twist/` 和 `requirements.txt`，保留已有 `data/prepared/`。从仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python twist/train-all-pair/train.py --device cuda --output runs/twist-svf-pixels
```

默认从零训练到累计 100,000 步。数据路径默认由脚本所在仓库确定，其他位置用 `--data /实际路径/data/prepared`。上面的输出目录为 `runs/twist-svf-pixels/`。

继续训练：

```bash
python twist/train-all-pair/train.py --device cuda --resume runs/twist-svf-pixels/last.pt
```

也可以直接训练到指定累计步数：

```bash
python twist/train-all-pair/train.py --device cuda --steps 10000 --output runs/twist/train-all-pair/ten-thousand
```

## 配置

[settings.py](settings.py) 设置 batch size、学习率、步数、验证/保存间隔、精度、缓存、预览和 GIF 帧数。[../config.py](../config.py) 设置模型结构、Scaling and Squaring 精度与 loss 权重。所有字段支持同名命令行参数，学习率用 `--lr`。

- 单个网络内默认堆叠 8 个 SA / CA / FFN 双流块，深度由 `num_blocks` 控制。每个 token 直接预测对应 patch 所有像素的二维速度；默认 256→128 解码 64 个二维向量，拼成 `[B,2,64,64]`。`squaring_steps=7` 先将速度缩放为 1/128，再自复合 7 次；命令行参数为 `--num-blocks`、`--squaring-steps`。
- 默认 batch=32、lr=1e-4；`precision=auto` 在支持的 CUDA 显卡上使用 BF16，其余使用 FP32。几何采样和 loss 保持 FP32。
- 全部原始字图以 uint8 缓存；`preload_to_gpu=True` 可让它们常驻训练显卡，需显存足够。
- writer 加载日志分别打印累计 `read`（读文件）、`copy`（复制像素）、`metadata`（标签与书写者索引）时间；全部加载后单独打印同字配对索引耗时。这些时间也保存到 `run.json` 的 `data_loading_seconds`。CPU 像素复制采用 NumPy 直接写入 Tensor 的共享存储。
- 固定使用双向图像相似度、双向距离场和速度平滑三项 loss。原始 A、B 的距离场按需预计算并共用 CPU LRU 缓存；同图跨配对、跨方向复用。默认最多 8,192 张，64×64 时约 128 MiB，满时淘汰旧项。
- 默认选最多 512 对固定验证样本，在验证书写者之间配对。每次启动按当前 `seed`、`val_batches` 与 batch 生成；设置相同则配对相同。`val_batches=0` 启用全部可配对验证源图，每个源图对应一个固定 B。
- 默认每 1,000 步验证并保存；第 0 步、短运行结束、训练结束也验证和保存。普通日志每 50 步记录更新后的训练批次。
- 默认预览 4 对验证图，尽量选择不同字符；只保存轨迹 GIF。帧数由 `animation_frames` 控制，首尾为原始 A、B。中间使用 `(1-t)*A(exp(tv)) + t*B(exp(-(1-t)v))`。

需要临时看 GIF 对应的输入时，单独运行：

```bash
python twist/train-all-pair/show_inputs.py --run runs/twist-svf-pixels --output inputs.png && rm -- twist/train-all-pair/show_inputs.py
```

在服务器仓库根目录生成 `inputs.png`，成功后删除临时脚本。每行对应一对 GIF 预览，左 A、右 B，行号对应 GIF 中的 `val_00`、`val_01` 等编号，并标注书写者与字符 Unicode 编码。脚本读取 `preview_pairs.json`，只加载这些图，使用该实验的缩放与归一化方式；可在训练期间运行。`--run` 指向当前实际训练输出目录；数据迁移后加 `--data /实际路径/data/prepared`。

## 输出与续训

| 文件 | 内容 |
|---|---|
| `run.json` | 配置、数据量、跳过数量、参数量与固定验证集的原图双向相似度与距离场基线 |
| `excluded_images.json` | 排除的恒定图的原始索引、字符、书写者、划分与像素范围 |
| `preview_pairs.json` | 固定预览对的字符与两位书写者 |
| `metrics.jsonl` | 训练批次/固定验证集的双向相似度、双向距离场与速度平滑，及各方向分项；最大速度、双向最小 Jacobian 与非正行列式比例；训练日志含裁剪前梯度范数 |
| `best.pt`、`last.pt` | 最优验证和最新保存的训练状态，含模型、优化器、轮次/游标、配对随机状态与固定验证对 |
| `previews/` | 轨迹 GIF，帧数由 `animation_frames` 控制 |
| `summary.json` | 最终步数、轮次和验证指标，以及完成/短运行结束/中断状态 |

检查点不包含全量字图和距离缓存。续训需要原预处理数据，迁移目录后用 `--data` 指向新位置；启动校验 manifest 指纹、归一化方式与可配对样本索引。轮次、剩余源图顺序及随机 B 的状态均恢复。`Ctrl+C` 等当前优化器更新完成后保存。

当前逐像素速度场检查点使用 version=3。旧粗网格速度场（version=2）、串联位移模型与旧归一化检查点无法直接续训；同步更新后使用新的输出目录从零开始，无需重做预处理数据。version=3 检查点可按上述命令正常续训。

每次启动或续训都从当前 `settings.py` 读取训练设置，从当前 `config.py` 读取 loss 设置；命令行优先。只有模型结构配置从检查点恢复。检查点不保存训练/loss 配置，Adam 也只保存动量与参数索引，学习率等优化器选项使用本次设置。旧检查点中的训练/loss 配置与优化器选项会被忽略。

修改文件后用同一条 `--resume` 命令即可生效；临时用命令行指定的参数，下次启动仍需指定或写进配置文件。进度、当前轮剩余源图顺序和随机状态会恢复；`seed` 用于新启动的随机初始化及后续轮次打乱，不覆盖已保存的当前轮顺序。旧版没有保存当前轮顺序的检查点，首次迁移时请保留原 `seed`。

每次续训重算当前模型、已有 `best.pt` 和原图基线的 loss，按当前验证集和 loss 比较 best。当前参数写入 `run.json`，每条指标记录包含当时的 `loss_config`；不同 loss 定义或权重下的总 loss 不能直接比较。新训练使用新输出目录；从较早检查点恢复也应指定新目录。

```bash
python twist/train-all-pair/train.py --device cuda --resume runs/twist-svf-pixels/last.pt --similarity-weight 1 --distance-weight 1 --smoothness-weight 0.002
```

这将速度平滑权重从默认 `0.001` 提高为 `0.002`。需要保留原实验的 best 和日志时，另加 `--output runs/twist-svf-pixels-reweighted`，使用新的空目录。

双向配准分别比较未混合的 A(exp(v)) 与 B、B(exp(-v)) 与 A。每个方向都使用参考图的 Sobel 灰度梯度平方加权，逐图归一化，默认 `mse_uniform_ratio=0.2`，即 80% 边缘加权、20% 全图误差；两方向直接相加。首尾混合帧本身等于原始 A、B，因此不用于端点配准损失。

双向距离场损失使用 D_A(exp(v))² 和 D_B(exp(-v))²，分别由 B、A 的软墨迹覆盖率归一化加权后相加。黑色权重 1、白色 0，灰色保留抗锯齿覆盖程度；不进行骨架细化。前景权重和原始距离场均停止梯度，坐标采样保留梯度。

速度平滑为未缩放 v 的一阶梯度平方，按原图归一化像素间距计算。日志 `similarity`、`distance`、`smoothness` 为三项损失的原值，`forward_similarity`、`reverse_similarity`、`forward_distance`、`reverse_distance` 是双向分项。默认 `total = similarity + distance + 0.001*smoothness`；三项权重通过 `--similarity-weight`、`--distance-weight`、`--smoothness-weight` 调整。

`velocity_max` 为最大速度模长，`jacobian_min` 为正反向映射中最小的离散行列式，`folding_fraction` 为非正行列式的网格单元比例，用于观察离散积分质量。速度场内部不做裁剪；`grad` 是优化器更新前全模型梯度 L2 范数，训练梯度裁剪仍由 `gradient_clip` 控制。

`similarity_vs_identity` 为双向相似度相对于不变形基线的比值。

日志中的 `step` 为累计优化器更新次数，`epoch` 为可配对训练源图的遍历进度；每轮每张图作为 A 一次。`train` 是更新后重新评估当前 batch 的总损失。

纯连续形变无法消除所有真实手写字的拓扑与噪声差异，先比较验证相似度和距离场与原图基线，并检查 GIF 的端点和中间轨迹。

本地检查：

```bash
python -m unittest discover -s twist/train-all-pair -p "test_*.py" -v
```
