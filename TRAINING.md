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
| `run.json` | 本次配置 |
| `previews/step_0000100.png` | 第 100 步生成对照图，后续预览也放在 `previews/` |

## 5. 继续训练 / 中断后恢复

续训使用检查点保存的配置；修改 `src/config.py` 只影响新训练。

```bash
python -m src.train --data data/prepared --output runs/first --resume runs/first/last.pt --device cuda
```

按 `Ctrl+C` 后等保存完成再退出。训练结果都在本项目的 `runs/first/`，退租前复制回来。
