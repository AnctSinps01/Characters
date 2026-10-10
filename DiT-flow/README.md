# Glyph2Hand：DiT + flow matching 手写字生成

根据同一书写者的若干「手写字 / 同字标准字」参考对，将目标标准字生成该书写者风格的手写图。本说明对照当前 `src/` 实现，记录模型、数据、损失、训练、推理与恢复行为；默认参数是实验设置，不代表已验证的生成质量。

架构图和公式见 [README.tex](readme/README.tex) / [README.pdf](readme/README.pdf)，服务器操作步骤见 [TRAINING.md](TRAINING.md)，源码索引见 [src/README.md](src/README.md)。`readme/README.v1-with-diagrams.*` 是历史设计，不代表当前实现。通用线条图变形路线见 [../twist/](../twist/README.md)。

## 运行位置与入口

从仓库根目录进入 `DiT-flow/` 后运行以下命令；Python 包名是 `src`，`DiT-flow` 不是导入包名。依赖和字体位于仓库根目录。

```bash
cd DiT-flow
python -m pip install -r ../requirements.txt
python -m src.prepare_data --help
python -m src.train --help
python count_parameters.py
python -m unittest discover -s tests -v
```

依赖为 PyTorch、NumPy、Pillow 和 fontTools，版本范围见 `../requirements.txt`；其中 SciPy 供 twist 使用。本训练器仅支持 CPU 或单张 CUDA GPU，没有分布式训练入口。服务器使用 CUDA 版 PyTorch；默认 BF16 要求 GPU 支持 BF16。CPU 自动使用 FP32。

## 模型接口与张量约定

设批量为 `B`、参考槽数为 `N`、图像边长为 `n`、块边长为 `p`、特征维为 `d`、每组描述数为 `Z`、风格 token 数为 `M`。空间 token 数 `L=(n/p)²`。PyTorch 使用通道在前的布局：

| 输入 / 输出 | 形状与含义 |
|---|---|
| `reference_handwriting`、`reference_standard` | `[B,N,C,n,n]`，按槽位成对，同一 episode 来自同一作者 |
| `reference_mask` | 可选 `[B,N]`；`True` 表示有效参考，省略时全部有效，每个样本至少有一个有效参考 |
| `state` / 预测速度 | `[B,C,n,n]`，当前流状态 / 像素速度 |
| `times` | `[B]`，连续流时间，训练采样范围为 `[0,1)` |
| `style` | `[B,M,d]`，固定数量、固定槽位的风格 token |

默认 `C=1`。模型层可按 `in_channels` 构造多通道版本，但现有数据准备、训练器和预览只支持单通道灰度字图。训练字图取值为 `[-1,1]`，白色为 `+1`，黑色为 `-1`；加噪状态和预测速度不受此范围约束。

`Glyph2Hand.forward(state, times, reference_handwriting, reference_standard, reference_mask=None)` 先编码参考，再预测速度。也可以分别调用 `encode_style(...)` 和 `predict_velocity(state, times, style)`，在多个目标和积分步之间复用风格。风格编码不依赖目标、时间或当前状态；训练中梯度保持连通，参数更新后需重算缓存。

目标标准字 **只用于构造流起点**，没有额外的干净目标图通道传入速度网络。参考分支仍同时读取手写字和同字标准字。

## 参考编码器

1. 将每个参考对沿通道拼为 `2C` 通道，将 `[B,N,...]` 合并为 `[B*N,...]`。使用 `Conv2d(kernel_size=p, stride=p, bias=True)` 分块，相当于把每个 `2Cp²` 维块投影为 `d` 维，再加固定二维正弦余弦位置编码。
2. 每组独立经过 `local_depth` 个 `SA → FFN` 块。每个子层使用前置 LayerNorm 和残差；所有组共享这些参数，此时没有组间注意力。
3. 共享的可学习 `Q_Z:[Z,d]` 通过一个 `CA → FFN` 块读取每组 `L` 个 token，得到 `[B*N,Z,d]` 描述。Q 和 memory 分别归一化，CA 与 FFN 各有残差。压缩后没有描述自注意力，也没有原始空间 token 的旁路。
4. 将描述整理为 `[B,NZ,d]`。另一个独立的可学习 `Q_M:[M,d]` 初始化风格 token，经过 `set_pool_depth` 个 `CA → SA → FFN` 汇总块；每层独立参数、普通前置 LN 与残差。每层 CA 读取同一份描述，描述本身不被更新；SA 仅在 `M` 个风格 token 间进行。输出没有额外的末尾 LN。

`Q_Z`、`Q_M` 独立，均以标准差 `0.02` 的正态分布初始化。`Z` 和 `M` 不必相等，也不等于特征维度或预设笔画种类数。内容与风格的严格解耦、槽位的可解释性都没有由结构保证。

汇总不加参考组序号或拼接序列位置。同步重排参考手写、标准字和掩码不会改变风格结果，允许浮点误差。掩码扩展到每组全部 `Z` 个描述，在每层集合 CA 中使用取反后的 `key_padding_mask`。补齐槽仍参与逐组编码，因此计算成本按槽数计算；应填有限的字图数值，当前数据集填白色 `+1`。

全部注意力使用 `nn.MultiheadAttention(batch_first=True, dropout=0.0)`，Q/K/V 投影彼此独立、有偏置，不绑定 K/V；调用时 `need_weights=False`。FFN 为 `Linear(d,int(d*mlp_ratio)) → GELU → Linear(...,d)`。普通 LN 使用 PyTorch 默认可学习仿射和 `eps=1e-5`。

## 速度预测主干

当前状态经独立的 `C` 通道 patch 投影和固定二维位置编码，保持 `L` 个空间 token。连续时间编码使用

```text
f_k = exp(-log(10000) * k / (time_frequency_size/2))
e_raw(t) = concat(cos(t*time_scale*f_k), sin(t*time_scale*f_k))
e(t) = Linear → SiLU → Linear(e_raw(t))
```

二维位置编码按行、列各分配 `d/2` 维，分别使用 `sin → cos` 拼接、周期基数 `10000`。位置和时间频率是固定、非持久化 buffer，不是学习参数，也不写入 `state_dict`。`time_frequency_size` 默认是定义配置类时的 `256`；单独改 `hidden_size` 不会自动改它。

每个主干块依次为 `SA → 风格 CA → FFN`：

- SA、FFN 使用各自的无仿射 LN（`eps=1e-6`），时间调制为 `(1+scale)*LN(x)+shift`。每块通过 `SiLU → Linear(d,6d)` 产生 `scale_sa,shift_sa,gate_sa,scale_ffn,shift_ffn,gate_ffn`，在 token 维广播；残差乘逐通道 gate，不使用 sigmoid。
- CA 使用普通 Q/memory LN 和普通残差，读取同一份 `style`；没有独立的时间调制或零门控。它的查询已受前面 SA 的时间调制影响。
- 输出头另用 `SiLU → Linear(d,2d)` 产生 `scale,shift`，调制无仿射 LN 后，共享 `Linear(d,p²C)` 输出每块速度，再按二维位置拼回整图。没有整图压成一个 token 的步骤。

各块的 `6d` 调制、输出头的 `2d` 调制及最终速度投影的权重和偏置均零初始化，其余层使用 PyTorch 默认初始化。SA/FFN 初始残差为零，CA 仍有普通残差，因而整个 block 初始不保证恒等；最终速度投影为零，因此模型初始速度输出为零。

## flow、作者风格损失与采样

`build_flow_matching_batch` 对每个样本采样独立高斯噪声和均匀时间，也允许显式传入 `noise`、`times`：

```text
x0 = target_standard + sigma * noise
xt = (1-t)*x0 + t*target_handwriting
u  = target_handwriting - x0
L_flow = mean((predicted_velocity-u)²)
```

同一 `x0` 用于路径和监督速度。均方误差平均全部样本、通道和像素，训练器以 FP32 计算此项。它是标准字上的加性噪声，目标手写真值不加入参考集合；路径不是已知的自然字形变形轨迹。

训练总损失为 `L_train = L_flow + author_style_loss_weight * L_author`，默认权重 `0.05`，设 `0` 关闭。`author_style_loss` 将风格 `[B,M,d]` 按每个固定槽位沿 `d` 维以 FP32 L2 归一化（`eps=1e-8`），比较同槽余弦：

```text
L_same = mean(1-cos(S_i[m],S_j[m]))，i != j 且作者相同
L_diff = mean(cos(S_i[m],S_j[m])²)，作者不同
L_author = L_same + L_diff
```

两个配对类型分别对全部配对及槽位求平均，再相加；实现包含对称有序配对、排除自配对。缺少某种配对时该项为可微零，`B=1` 时整个作者项为零。函数还支持独立的 `same_author_weight`、`different_author_weight`，默认都为 `1`，训练器使用默认值。异作者项鼓励对应槽位正交，反向量不能作为零损失解；这并不保证完全解耦。作者配对只在每个微批次内计算，梯度累积不会扩大配对集合，采样器也不保证同作者重复出现。归一化使用 FP32，但相似度的 `einsum` 仍受当前 autocast 上下文影响。

训练每个微批次只编码一次风格，供速度预测和作者项复用。EMA 模型在添加训练专用作者权重之前复制，因此 `val_loss_ema` **只统计 flow MSE**；它与含作者项的 `train_loss` 口径不同。

`sample_euler` 在 `no_grad` 下编码一次参考，初始化 `x0`，用 `J=steps` 个等长步积分：`t_j=j/J`、`x←x+v(x,t_j,S)/J`（最后一次网络时间为 `(J-1)/J`）。默认噪声强度取 `model.config.sigma`，允许显式提供噪声和覆盖 `sigma`；覆盖只改变起点，没有噪声强度条件输入或效果保证，正常推理应与训练分布一致。训练状态和积分中途不裁剪，只在返回时裁剪 `[-1,1]`；函数返回归一化张量，灰度映射由预览/调用方完成。`sigma=0` 是合法的确定性对照。

## 当前默认配置

新训练读取 `src/config.py` 的两个 dataclass，不读取额外配置文件，也不提供模型/学习率的训练命令行覆盖。

| `ModelConfig` 字段 | 默认值 | 含义 |
|---|---:|---|
| `image_size` / `patch_size` / `in_channels` | `64` / `8` / `1` | 灰度图与块；网格 `8×8`，`L=64` |
| `hidden_size` / `num_heads` / `mlp_ratio` | `256` / `8` / `4.0` | 每头维度 `32`，FFN 内宽 `1024` |
| `local_depth` / `set_pool_depth` / `generator_depth` | `4` / `2` / `8` | 逐组 / 集合 / 主干深度 |
| `description_tokens` / `style_tokens` | `32` / `32` | `Z` / `M` |
| `time_frequency_size` / `time_scale` | `256` / `1000.0` | 时间 Fourier 特征维度 / 放大倍数 |
| `sigma` | `0.1` | 标准字加噪声强度，允许 `0` |

模型要求图像边长可被块边长整除，`hidden_size` 可被头数和 `4` 整除，时间特征维为正偶数，集合深度至少 `1`；逐组和主干深度允许 `0`。默认可学习参数总数 **17,984,320**：参考编码器 **6,106,624**，主干 **11,877,696**。`count_parameters.py` 支持所有模型字段的 kebab-case 参数（如 `--hidden-size 128 --time-frequency-size 128`），仅用于统计，不改变训练配置。

| `TrainingConfig` 字段 | 默认值 | 含义 |
|---|---:|---|
| `steps` / `batch_size` / `accumulation_steps` | `100000` / `64` / `1` | 优化器步数 / 微批量 / 累积次数 |
| `min_references` / `max_references` | `1` / `16` | 每个 episode 均匀整数采样参考数量，端点均包含 |
| `learning_rate` / `min_lr_ratio` / `warmup_steps` | `1e-4` / `0.1` / `1000` | 基础 LR / 最低比例 / 线性预热步数 |
| `weight_decay` / `gradient_clip` / `ema_decay` | `0.01` / `1.0` / `0.999` | AdamW 衰减 / 梯度 L2 阈值 / EMA 衰减 |
| `author_style_loss_weight` | `0.05` | 作者项相对 flow MSE 的权重 |
| `precision` / `preload_to_gpu` / `workers` | `bf16` / `False` / `8` | CUDA 精度 / 字图常驻显存 / 训练加载 worker 数 |
| `val_batches` / `validate_every` | `16` / `1000` | 验证微批次数 / 验证间隔 |
| `save_every` / `log_every` | `500` / `50` | `last.pt` 保存间隔 / 日志间隔 |
| `sample_steps` / `seed` | `32` / `42` | 预览 Euler 步数 / 种子 |

有效批量为 `batch_size * accumulation_steps`（默认 `64`）；每次累积将微批损失除以累积次数。计数项须为正整数，`0<=warmup_steps<steps`、`workers>=0`、`min_references<=max_references`；LR 和裁剪阈值有限且为正，权重衰减及作者权重有限且非负，最低 LR 比例在 `[0,1]`，EMA 衰减在 `[0,1)`，精度仅接受 `fp32/fp16/bf16`，预加载开关须为布尔值。

## 数据准备与 episode

```bash
python -m src.prepare_data --gnt-root ../data/gnt --font ../fonts/WenJinMinchoP0-Regular.ttf --output ../data/prepared
```

训练说明使用文津宋体；代码不硬编码字体，实际字体由必填 `--font` 决定。LaTeX 架构图中的微软雅黑仅用于示意。

准备入口的必填参数为 `--gnt-root`、`--font`、`--output`。可选参数：`--font-index 0`、`--image-size 64`（模型默认值）、`--margin 4`、`--min-characters 17`（默认 `max_references+1`）、`--val-fraction 0.1`、`--seed 42`（训练默认种子）。修改模型图像尺寸或参考上限后，应检查预处理数据是否仍满足训练要求。

- 递归读取所有 `.gnt`（后缀不区分大小写），按文件排序；每个文件视为一个作者，作者 ID 为文件 stem，要求至少两个文件且 stem 不重复。入口不自动识别官方 Test 数据，只应放入训练集 GNT。
- GNT 的数值字段按小端解析（10 字节头，记录长度须等于 `10+width*height`）；字符直接按原始两个标签字节解码 GBK。截断、无效尺寸或非法标签会报错，不静默跳过损坏记录。
- 只保留单个基本 CJK 字符 `U+4E00..U+9FFF` 且字体 cmap 支持的字；每位作者只留同字首次成功处理的样本。空白字图、缺字、非 CJK、重复字和字符数不足的作者有跳过计数。
- 手写和标准字各自裁掉纯白背景，保持宽高比、LANCZOS 缩放至可用边长 `image_size-2*margin`，居中贴到白色方形。默认 `64×64`、边距 `4`；没有旋转、弹性变形或其他随机图像增强。标准字先以 `4*image_size` 字号渲染，再用同一流程归一化。
- 按作者划分 train/val，使用独立 `random.Random(seed)` 打乱作者；验证作者数为 `min(W-1,max(1,round(W*val_fraction)))`。两侧作者不重叠，但字符类别可以重叠；不是按字类别划分。
- 生成 uint8 的 `writers/*.npy`、`standards.npy` 和最后写入的完成标志 `manifest.json`（数据格式版本 `1`）。manifest 记录字符索引、作者文件/标签/划分、字体及像素文件 SHA-256、尺寸、边距、种子与跳过统计。输出必须新建或为空，准备中断后的非空目录不会直接重做覆盖。

训练启动把全部 train/val 字图以 uint8 预加载一份，逐作者读取 NPY，检查 dtype/shape 和标签索引；CPU 使用 `RawArray` 共享给 fork/spawn worker，训练与验证共用同一份字图，采样不再访问 NPY。占用字节为 `(手写图数+标准图数)*image_size²`，不含模型、浮点批次和加载器开销。`preload_to_gpu=True` 时加载到选定 CUDA 设备，并强制 `workers=0`。

每个 episode 的独立 RNG 为 `random.Random(f"{seed}:{index}")`：均匀选作者、均匀选参考数，再无放回抽 `N+1` 个不同字符，首个为目标、其余为参考。作者须有至少 `max_references+1` 个不同字符；按作者均匀采样并非按全部字图均匀采样。字图此时从 uint8 转 FP32：`pixel/127.5-1`，参考补到 `max_references` 个槽位并附布尔掩码和稳定的 manifest 作者索引。目标字符不会出现在同 episode 的参考中。

episode 由种子和全局索引确定，不受 worker 数或预取进度影响。续训从 `saved_step*有效批量` 继续索引；验证使用固定索引与 `seed+1`。数据指纹是 **manifest 原始字节的 SHA-256**，连空白变化也会触发续训拒绝；加载器没有重新核验 manifest 中的 NPY/字体内容哈希，因此它不是对所有文件重新做完整性校验。

## 优化、验证、日志与退出

训练使用 AdamW，`betas=(0.9,0.95)`，全部模型参数进入同一组，未单独排除 bias/LN/查询的权重衰减。优化器步数从 `1` 开始：预热 LR 为 `base_lr*step/warmup_steps`，随后余弦下降，在 `steps` 达到 `base_lr*min_lr_ratio`。`--stop-after` 仅提前停在绝对步数，不重设此日程。

CUDA 可选 BF16/FP16/FP32，启用 TF32 matmul/cuDNN；CPU 固定 FP32。只有 FP16 启用 GradScaler。先 AMP 反缩放再计算全局梯度 L2 范数，安全裁剪后更新优化器；float32 范数归约非有限时以 float64 重算。真实非有限梯度不更新参数：FP16 降低 scaler，用同一数据、时间与噪声重试，最多 `16` 次尝试；BF16/FP32 立即报错停止。非有限训练损失立即停止，不做溢出重试；异常保留已成功保存的检查点，不保证另存故障当前步。

每次成功更新后 `ema ← ema_decay*ema+(1-ema_decay)*model`。每隔 `validate_every` 及计划结束步以 EMA 验证和生成预览；正常 `--stop-after 100` 也会验证、保存。验证在隔离的 PyTorch RNG 中用 `seed+10000` 固定时间/噪声，预览在全部验证损失之后采样，避免改变训练 RNG 或损失采样序列。验证 batch 数为 `val_batches`、worker 固定 `0`；loss 按样本数平均。更改精度/设备可能影响数值，代码不保证跨设备逐位一致。

| 输出 | 行为 |
|---|---|
| `run.json` | 模型/训练设置、manifest 指纹、设备、实际精度、torch 版本和当前预加载方式；续训会重写 |
| `metrics.jsonl` | 间隔、验证和结束步追加；`train_loss`、`lr`、裁剪前 `grad_norm`、`grad_clip_scale`、可选 `val_loss_ema`，以及带 `s` 的本次启动耗时字符串 |
| `peak_allocated`（CUDA 日志字段） | `torch.cuda.max_memory_allocated/1024³`，字符串标记 `GB`，实际为 GiB；不是 reserved memory 或本步单独峰值 |
| `gradient_diagnostics.jsonl` | 启动/续训首步、日志间隔和计划末步记录梯度；非有限损失/梯度和 float64 回退额外记录 |
| `previews/step_XXXXXXX.png` | 验证第一批最多四个样本，列出全部参考槽、目标标准字、真值和 EMA 生成图；补齐参考显示空白 |
| `best.pt` | EMA flow 验证损失严格降低时保存的完整训练状态 |
| `last.pt` | 保存间隔及正常结束/响应退出请求时的完整训练状态，可能落后于内存中的当前步 |

日志普通浮点固定八位小数，绝对值 `>=1e8` 或非零且 `<1e-7` 时用科学计数法；数字仍为 JSON number，耗时/显存为带单位字符串。诊断列出梯度最大的八个参数和直接所属模块（不是整个参考/主干递归汇总）、最大梯度元素、裁剪比例、AMP scale、尝试序号和异常参数名，Inf/NaN 以字符串保存；非有限损失事件只有相应损失/尝试信息。

训练循环响应 SIGINT/SIGTERM：完成当前优化器步后，记录并保存 `last.pt`；退出请求会跳过该步验证。进程强杀或一般运行异常不会触发这个保存流程。详细梯度和存储排查见 [TRAINING.md](TRAINING.md)。

## 检查点与续训规则

```bash
python -m src.train --data ../data/prepared --output ../runs/DiT-flow/first --device cuda --stop-after 100
python -m src.train --data ../data/prepared --output ../runs/DiT-flow/first --resume ../runs/DiT-flow/first/last.pt --device cuda
```

必填 `--data`、`--output`；可选 `--resume`、`--device`（默认 `auto`：有 CUDA 则 CUDA，否则 CPU；也支持 `cuda:N`）、`--workers`、`--stop-after`。训练文件也支持从仓库根目录执行 `python DiT-flow/src/train.py ...`，参数路径仍相对当前工作目录；推荐统一从 `DiT-flow/` 运行模块入口。

当前检查点版本 **2**，保存 `settings`、manifest 指纹、已完成步数、最优验证损失、当前模型、EMA、AdamW、GradScaler、CPU 和选定 CUDA 设备的 PyTorch RNG 状态。`best.pt` 也可续训，它包含当前模型和 EMA 两套权重；推理通常选用 EMA。版本 1 的旧双通道主干检查点在写输出前被明确拒绝，不能直接续训；预处理数据仍可复用。

续训恢复保存的模型与训练配置；修改当前 `config.py` 的架构、LR、种子等不会替换旧运行设置。例外是运行方式 `preload_to_gpu` 始终读取当前配置，不写入检查点 `settings`，而单独记在 `run.json`；`--device`、`--workers`、`--stop-after` 也可在续训时指定。必须满足保存步数 `< stop-after/总步数`，不能靠 `--stop-after` 延长已保存的总日程。

新训练拒绝非空输出，只有配置/指纹一致、仅含 `run.json` 和空 `previews/` 的未开始运行可重试。续训允许新输出目录；非空旧目录须有一致的 `run.json`，且其中 `last.pt` 不能比所选恢复点新。从较早的 `best.pt` 回退时应选择新输出目录。

保存先写完整 `.pt.tmp`，flush/fsync 后替换对应 `.pt`；识别为 I/O 的错误最多尝试三次，间隔 `1`、`2` 秒，不重复训练，失败临时文件尝试清理。普通序列化错误直接抛出。先保存刷新后的 `best.pt`，再按条件保存 `last.pt`；两者不是一笔整体事务，后者失败时前者可能已经更新。需要额外一份完整检查点的空间，持续失败会报错停止。

## 计算规模与已有验证

参考注意力主项为 `O(local_depth*N*L²*d + N*Z*L*d + set_pool_depth*(M*N*Z+M²)*d)`，没有全部描述的全局 SA；主干 `J` 步注意力为 `O(J*generator_depth*(L²+L*M)*d)`，还需计入投影/FFN、反向和批次成本。可复用风格减少重复参考编码，当前采样仍每步重算主干各层的风格 LN/K/V 投影；没有 K/V cache。代码使用 PyTorch MHA，具体注意力后端取决于设备、版本、dtype 等，不保证始终使用 FlashAttention。

现有测试覆盖参考重排/掩码、作者损失、flow/采样、GNT 和真实字体预处理、共享内存及无文件采样、参考范围/目标排除、设备选择、日志格式、梯度溢出、保存失败以及小模型 CPU 中断续训与连续训练一致性。CUDA 设备选择测试使用模拟；这些测试不等于真实 3090 性能或生成质量验证。
