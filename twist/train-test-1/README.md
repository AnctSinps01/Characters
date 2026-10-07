# 单字训练测试

默认读取一个字的标准图 A 和全部同字手写图 B，一次载入内存，再放到训练设备。按照 `manifest.json` 的书写者 `train/val` 划分训练和验证。只选一个字，默认「中」。

数据沿用 [DiT-flow/TRAINING.md](../../DiT-flow/TRAINING.md) 的预处理结果：仓库根目录 `data/prepared/{manifest.json,standards.npy,writers/}`。无需重新处理 GNT。已有图像尺寸不同会先缩放到模型尺寸。

新训练在缩放后，对每张 A、B 分别使用 `2 * (pixel - min) / (max - min) - 1`，阈值 0 判断前景。恒定的手写图跳过并记录在 `run.json` 的 `glyphs.excluded_images`；恒定标准图无法作为 A，会报错。旧版单字检查点续训保留其中已缓存的输入与距离场，若要使用逐图 min–max 归一化，请用新目录从零训练。

## 云主机运行

复制仓库的 `twist/` 和 `requirements.txt`；保留云主机已有的 `data/prepared/`。下面从仓库根目录运行，CUDA 版 PyTorch 需已安装：

```bash
python -m pip install -r requirements.txt
python twist/train-test-1/train.py --char 中 --steps 200 --device cuda
```

默认数据路径相对于脚本所在仓库确定，默认输出为 `runs/twist/train-test-1/first/`。其他位置的数据用 `--data /实际路径/data/prepared` 指定。脚本也可以从其他工作目录通过绝对路径启动。

满意后从第 200 步继续到累计 1,000 步：

```bash
python twist/train-test-1/train.py --resume runs/twist/train-test-1/first/last.pt --steps 1000 --device cuda
```

换字请用新的输出目录，例如 `--char 永 --output runs/twist/train-test-1/yong`。启动会打印实际样本数、内存用量、参数量和不做变形的 MSE 基线。

## 两种排查模式

已知平滑形变：读取标准字 A，用固定边界、受 Lipschitz 约束的平滑场生成 B，反复拟合同一对。真值坐标仅作诊断，训练仍使用同样三类 loss：

```bash
python twist/train-test-1/train.py --mode synthetic --char 中 --steps 200 --device cuda --output runs/twist/train-test-1/synthetic
```

单对真实样本：`run.json` 的 `glyphs.train_ids` 列出书写者 ID，指定其中一个即可。仍缓存该字全部样本，训练只用指定的一对，不计算验证损失：

```bash
python twist/train-test-1/train.py --char 中 --writer 书写者ID --steps 200 --device cuda --output runs/twist/train-test-1/one-pair
```

## 配置与结果

- [settings.py](settings.py)：步数、batch size、学习率、日志/预览间隔、精度、合成形变强度。
- [../config.py](../config.py)：模型结构与三类 loss 比例。默认完整模型约 14.82M 参数；所有配置也可用同名命令行参数覆盖，例如 `--token-dim 128 --num-heads 4 --batch-size 8`。
- 默认 FP32。支持 BF16 的 CUDA GPU 可用 `--precision bf16`；续训恢复模型/loss/训练配置与抽样状态，允许调整目标步数、batch、日志/预览间隔和精度。

| 输出 | 内容 |
|---|---|
| `inputs.png`、`glyphs.pt` | 选中字的输入预览和全部缓存；续训无需再读取原始数据 |
| `run.json` | 配置、样本清单、train/val 数量和初始基线 |
| `metrics.jsonl` | 三类 loss、形变子项、各阶段最大 Lipschitz 上界；`scope=batch` 为抽样批次，`scope=full` 为完整 train/val 评估 |
| `best.pt`、`last.pt` | 最佳已保存评估步和最新保存步的模型/优化器/数据/随机状态；有验证集时按验证总 loss 选择 best |
| `previews/*_compare.png` | A、B、端点输出、叠加误差；红色为输出，青色为 B，重合为黑色 |
| `previews/*_stages.png` | 各串联阶段输出 |
| `previews/*_trajectory.png` | 固定展示 t=0、0.25、0.5、0.75、1 的静态横图 |
| `previews/*_trajectory.gif` | 自动保存的过渡动画；总帧数由 `settings.py` 的 `animation_frames` 控制，默认 21 帧，包含两个端点 |
| `previews/*_grid.png` | 反向采样网格 |
| `summary.json` | 最终指标与完成/中断状态 |

第 0 步、每隔 100 步和结束时评估完整集合、保存检查点和预览；普通日志每 10 步记录更新后的抽样批次。预览固定展示 train/val 各自第一个样本，具体 ID 见 `run.json`。`Ctrl+C` 等当前更新结束后保存。新实验拒绝覆盖非空目录；从较早检查点恢复请指定新的输出目录。

GIF 帧数也可运行时用 `--animation-frames 61` 覆盖。续训恢复检查点中的配置，要改变旧实验的帧数也使用这个参数。

先看端点叠加图和验证 MSE 是否改善，再看中间轨迹。真实手写字可能有连接关系变化，不能保证纯连续形变达到零误差。此实验只验证单字配准，不验证跨字泛化。

本地检查：

```bash
python -m unittest discover -s twist/train-test-1 -p "test_*.py" -v
```
