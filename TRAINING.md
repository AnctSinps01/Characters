# 训练操作

以下路径都相对于这个项目目录，所有命令都在项目根目录（有 `src`、`requirements.txt` 的这一层）运行。

## 1. 新建文件夹，放入文件

| 新建路径 | 放什么 |
|---|---|
| `fonts/` | 放入文津宋体标准版 [WenJinMinchoP0-Regular.ttf](https://github.com/takushun-wu/WenJinMincho/blob/main/ttf/WenJinMinchoP0-Regular.ttf)，打开链接后点击 Download raw file 下载 |
| `data/gnt/` | 放入下面两个训练集压缩包解压得到的 `.gnt` 文件，保留原文件名；不要放 Test 数据 |

训练集：[Part1](https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip)、[Part2](https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip)。数据可以等复制到服务器后再下载、解压到 `data/gnt/`。

## 2. 复制项目，安装依赖

把整个项目目录复制到 GPU 服务器。选择带 Python 3.10+、CUDA 版 PyTorch 2.3+ 的环境，进入复制后的项目根目录：

```bash
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))"
```

## 3. 处理数据（只运行一次）

```bash
python -m src.prepare_data --gnt-root data/gnt --font fonts/WenJinMinchoP0-Regular.ttf --output data/prepared
```

自动生成 `data/prepared/`：`writers/` 放手写图，`standards.npy` 放标准字图，`manifest.json` 放样本清单和训练/验证划分。首次运行时该目录应不存在或为空。

## 4. 先训练 100 步

模型和训练参数直接改 `src/config.py`。每个样本在 `min_references` 到 `max_references` 之间均匀随机抽参考字数。图像尺寸和参考数量上限请在处理数据前确定。

训练启动时自动把 `data/prepared/` 的全部字图加载到共享内存，打印实际占用 GiB；多个 worker 共用一份。要把字图改为常驻显存，将 `src/config.py` 中的 `preload_to_gpu` 改成 `True`；此时自动使用 `workers=0`。这个开关对旧检查点续训也生效。更新训练代码后无需重新 prepare。

```bash
python -m src.train --data data/prepared --output runs/first --device cuda --stop-after 100
```

自动生成 `runs/first/`，首次运行时该目录应不存在或为空：

| 文件 | 内容 |
|---|---|
| `last.pt` | 最新训练状态，用于续训 |
| `best.pt` | 验证损失最低的训练状态 |
| `metrics.jsonl` | 训练日志 |
| `gradient_diagnostics.jsonl` | 裁剪前梯度诊断：范数最大的参数/模块、最大梯度元素、裁剪比例与非有限值位置 |
| `run.json` | 本次配置 |
| `previews/step_0000100.png` | 第 100 步生成对照图，后续预览也放在 `previews/` |

## 5. 继续训练 / 中断后恢复

续训使用检查点保存的配置；修改 `src/config.py` 只影响新训练。

当前架构已移除生成主干的目标标准字提示通道，检查点版本为 2。版本 1 的旧检查点输入层尺寸不同，不能直接续训；请去掉 `--resume` 并使用新的输出目录重新训练。已有的预处理数据仍可复用，目标标准字继续用于构造流的加噪起点，参考字仍使用手写/标准字对。

```bash
python -m src.train --data data/prepared --output runs/first --resume runs/first/last.pt --device cuda
```

按 `Ctrl+C` 后等保存完成再退出。训练结果都在本项目的 `runs/first/`，退租前复制回来。

## 6. 排查梯度异常

`metrics.jsonl` 中的 `grad_norm` 是 AMP 反缩放后、裁剪前的全局 L2 范数；`grad_clip_scale` 是本步裁剪乘数。默认阈值为 1，原始范数 330 时裁剪乘数约为 0.00303；AdamW 的实际参数更新还取决于其历史状态。

启动/续训的第一步、每隔 `log_every` 步和计划结束步，会在 `gradient_diagnostics.jsonl` 记录梯度最大的 8 个参数及模块。真实非有限梯度、非有限损失或范数计算回退到 float64 时，无论日志间隔都会记录。每行包含步数和实际训练精度；梯度记录还包含 AMP 缩放倍率、尝试次数和裁剪信息。诊断中的 `"inf"` / `"nan"` 使用字符串保存，以保证文件仍是合法 JSON。

若 float32 范数计算得到 Inf/NaN，训练器先以 float64 重算，区分范数归约溢出与梯度元素本身异常。有限梯度按真实范数裁剪；真正非有限的梯度不会送入优化器。FP16 溢出仍会降低 AMP 缩放倍率并用相同数据、时间和噪声重试；BF16/FP32 遇到非有限梯度会记录参数位置后停止，保留已有检查点。这项处理不能保证网络本身稳定，持续增大的原始梯度仍应排查。

先查看异常前后 `top_parameters` 和 `top_modules` 的变化，确定问题集中在参考分支、时间调制、注意力还是输出层，再针对该处检查激活和归一化输入方差。参考分支的描述 token 与最终风格 token 在被下游注意力读取时已有 LayerNorm，仅在浓缩输出后重复添加归一化不能证明能够修复爆炸。

以上诊断不改变模型结构和检查点版本；当前版本 2 的检查点可直接续训，无需重新准备数据。
