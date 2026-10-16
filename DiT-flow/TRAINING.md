# 训练操作

所有命令都在 `DiT-flow/` 内（有 `src/` 的这一层）运行，先从仓库根目录执行 `cd DiT-flow`。以下路径相对于 `DiT-flow/`；公共依赖、字体、原始数据和预处理数据位于上一级仓库根目录，训练结果使用根目录下的 `runs/DiT-flow/`。

## 1. 新建文件夹，放入文件

| 新建路径 | 放什么 |
|---|---|
| `../fonts/` | 放入文津宋体标准版 [WenJinMinchoP0-Regular.ttf](https://github.com/takushun-wu/WenJinMincho/blob/main/ttf/WenJinMinchoP0-Regular.ttf)，打开链接后点击 Download raw file 下载 |
| `../data/gnt/` | 放入下面两个训练集压缩包解压得到的 `.gnt` 文件，保留原文件名；不要放 Test 数据 |

训练集：[Part1](https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip)、[Part2](https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip)。数据可以等复制到服务器后再下载、解压到 `../data/gnt/`。

## 2. 复制项目，安装依赖

把整个仓库目录复制到 GPU 服务器。选择带 Python 3.10+、CUDA 版 PyTorch 2.3+ 的环境，进入复制后的 `DiT-flow/`：

```bash
python -m pip install -r ../requirements.txt
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))"
```

## 3. 处理数据（只运行一次）

```bash
python -m src.prepare_data --gnt-root ../data/gnt --font ../fonts/WenJinMinchoP0-Regular.ttf --output ../data/prepared
```

自动生成 `../data/prepared/`：`writers/` 放手写图，`standards.npy` 放标准字图，`manifest.json` 放样本清单和训练/验证划分。首次运行时该目录应不存在或为空。

## 4. 先训练 100 步

模型和训练参数直接改 `src/config.py`。当前默认每样本 4 个参考，调整 `min_references`/`max_references` 后在范围内均匀整数抽样。图像尺寸和参考上限请在处理数据前确定。每 batch 更新一次；风格只读手写字，目标标准字经独立内容编码器提取 token，主干从纯高斯噪声生成。训练为 flow MSE 加作者损失（默认权重 `0.05`）：对有效参考的同描述槽取均值后，同作者用 `1-cosine`、异作者用 `cosine²`。只比较当前 batch，不缓存跨 batch 配对；完整配对数量与公式见 [README](README.md)。

训练启动时自动把 `../data/prepared/` 的全部字图加载到共享内存，打印实际占用 GiB；多个 worker 共用一份。要把字图改为常驻显存，将 `src/config.py` 中的 `preload_to_gpu` 改成 `True`；此时自动使用 `workers=0`。这个开关对旧检查点续训也生效。训练器兼容 PyTorch 2.3 的 `torch.cuda.amp.GradScaler` 和新版本的 `torch.amp.GradScaler`，更新训练代码后无需重新 prepare。

```bash
python -m src.train --data ../data/prepared --output ../runs/DiT-flow/first --device cuda --stop-after 100
```

自动生成 `../runs/DiT-flow/first/`，首次运行时该目录应不存在或为空：

| 文件 | 内容 |
|---|---|
| `last.pt` | 最近一次成功保存的完整训练状态，用于续训；保存间隔内可能落后于当前步 |
| `best.pt` | 当前模型双损失验证总目标最低时的完整训练状态 |
| `metrics.jsonl` | 训练日志 |
| `gradient_diagnostics.jsonl` | 裁剪前梯度诊断：范数最大的参数/模块、最大梯度元素、裁剪比例与非有限值位置 |
| `run.json` | 本次配置 |
| `previews/step_0000100.png` | 第 100 步生成对照图，后续预览也放在 `previews/` |

## 5. 继续训练 / 中断后恢复

续训使用检查点保存的配置；修改 `src/config.py` 中的模型和训练超参数只影响新训练。运行方式开关 `preload_to_gpu` 例外，始终读取当前配置；`--device`、`--workers` 和 `--stop-after` 也可以在续训时指定。`--stop-after` 是绝对优化器步数，不改变保存的总步数或学习率日程。

当前架构为纯噪声 flow、独立内容编码、未浓缩的多参考风格序列和双 CA 主干，检查点版本为 3。版本 1/2 权重结构不兼容，不能直接续训；请去掉 `--resume` 并使用新的输出目录重新训练。已有的灰度预处理数据只要尺寸与字符数量满足配置即可复用，目标标准字作为内容条件，参考字不再配标准字。

```bash
python -m src.train --data ../data/prepared --output ../runs/DiT-flow/first --resume ../runs/DiT-flow/first/last.pt --device cuda
```

在训练循环中按 `Ctrl+C` 后，会完成当前优化器步、跳过该步验证并保存 `last.pt`，等保存完成再退出。强杀进程或一般运行异常只保留此前成功保存的检查点。训练结果都在仓库根目录的 `runs/DiT-flow/first/`，退租前复制回来。

调整目录路径不改变数据格式，已有 `data/prepared/` 可复用。只有当前版本 3 的检查点支持 `--resume`；旧运行结果请保留，使用新的输出目录训练当前架构。

## 6. 排查梯度异常

`metrics.jsonl` 中的 `grad_norm` 是 AMP 反缩放后、裁剪前的全局 L2 范数；`grad_clip_scale` 是本步裁剪乘数。默认阈值为 1，原始范数 330 时裁剪乘数约为 0.00303；AdamW 的实际参数更新还取决于其历史状态。

启动/续训的第一步、每隔 `log_every` 步和计划结束步，会在 `gradient_diagnostics.jsonl` 记录梯度最大的 8 个参数及模块。真实非有限梯度、非有限损失或范数计算回退到 float64 时，无论日志间隔都会记录。每行包含步数和实际训练精度；梯度记录还包含 AMP 缩放倍率、尝试次数和裁剪信息。诊断中的 `"inf"` / `"nan"` 使用字符串保存，以保证文件仍是合法 JSON。

若 float32 范数计算得到 Inf/NaN，训练器先以 float64 重算，区分范数归约溢出与梯度元素本身异常。有限梯度按真实范数裁剪；真正非有限的梯度不会送入优化器。FP16 溢出仍会降低 AMP 缩放倍率并用相同数据、时间和噪声重试；BF16/FP32 遇到非有限梯度会记录参数位置后停止，保留已有检查点。这项处理不能保证网络本身稳定，持续增大的原始梯度仍应排查。

先查看异常前后 `top_parameters` 和 `top_modules` 的变化，确定问题集中在风格分支、内容分支、时间调制、注意力还是输出层，再针对该处检查激活和归一化输入方差。风格描述和内容 token 在被主干 CA 读取时已有 LayerNorm，仅重复添加归一化不能证明能够修复爆炸。

这些诊断逻辑保留在当前架构中。版本 3 的检查点可续训，版本 1/2 不兼容；无需仅因启用诊断而重新准备数据。

## 7. 检查点写入失败

如果训练/验证指标已打印，随后在 `torch.save` 出现 `Input/output error`、`iostream error` 或 `PytorchStreamWriter failed`，失败点是检查点写入；这段堆栈本身不表示梯度爆炸。固定在保存步数复现时，优先检查输出盘的空间、用户/平台配额、inode 和挂载状态。在云主机的 `DiT-flow/` 内运行（将目录换成实际运行目录）：

```bash
df -h ../runs/DiT-flow/first
df -i ../runs/DiT-flow/first
ls -lh ../runs/DiT-flow/first/*.pt*
du -sh ../runs/DiT-flow/first
```

`df` 的文件系统剩余空间不一定反映用户或平台的单独配额，还需检查平台存储用量。保存时先写完整的 `last.pt.tmp` 或 `best.pt.tmp`，再替换旧文件，因此除了已有检查点，还需要至少一份完整检查点的额外空间并留有余量。训练进程已退出时，可清理确认是失败保存遗留的 `.pt.tmp`；保留可加载的 `.pt`。

检查点写入现在使用 Python 文件流，完成后 flush/fsync，再替换旧文件。遇到文件 I/O 错误最多尝试 3 次，间隔 1、2 秒；重试使用当前内存中的同一步训练状态，不重新训练。失败的临时文件会尝试清理；持续失败会明确报错并停止，报告路径、底层错误和清理后的文件系统剩余空间，不会假装保存成功。这只能缓解短暂故障，不能解决磁盘满、配额耗尽或持续存储故障。

当前顺序先保存刷新后的 `best.pt`，再按保存间隔或结束条件保存 `last.pt`。如果后者失败，前者可能已经保存了当前步；可先查看两个检查点的 `step`，从可加载且步数更新的检查点续训。`best.pt` 同样包含完整优化器和随机数状态，支持 `--resume`；若选用的检查点早于输出目录中现有 `last.pt`，必须换新的 `--output` 目录。完整实现约定和所有默认值见 [README](README.md)。
