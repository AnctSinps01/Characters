# twist

单个 Transformer 配准网络：输入 A、B，层内堆叠 SA / CA / FFN 双流块，一次输出定常速度场。速度场只做边界置零，通过 Scaling and Squaring 积分，得到双向配准和任意 t∈[0,1] 的混合画面。白色为 +1，黑色为 −1，坐标为 [0,1]²。完整数学定义见 [DESIGN.md](DESIGN.md)。

## 文件

| 文件 | 用途 |
|---|---|
| [config.py](config.py) | 网络尺寸、num_blocks、squaring_steps、loss 权重 |
| [stage.py](stage.py) | 单个速度场预测网络 |
| [layers.py](layers.py) | 双流 SA / CA / FFN、patch 与位置编码 |
| [model.py](model.py) | 正反向配准及缓存速度的双向插值 |
| [geometry.py](geometry.py) | 边界置零、Scaling and Squaring、映射复合及 Jacobian |
| [losses.py](losses.py) | 双向图像相似度、双向距离场与速度一阶平滑 |
| [distance.py](distance.py) | 原始 A、B 的欧氏距离场预处理 |
| [checkpoint.py](checkpoint.py) | 训练状态与优化器动量恢复 |
| [preview.py](preview.py) | 双向插值轨迹 GIF |
| [generate_all.py](generate_all.py) | 加载训练检查点，为全部至少两张有效样本的汉字生成过渡与浏览总览 |
| [count_parameters.py](count_parameters.py) | 参数统计 |
| tests/ | 几何、数据流、梯度和混合精度验证 |
| [train-all-pair/](train-all-pair/README.md) | 全数据集同字手写图配对训练 |

## 使用

从仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m twist.count_parameters
python -m unittest discover -s twist/tests -v
```

A、B 为同形状浮点张量 [batch,1,image_size,image_size]，已归一化到 [-1,1]：

```python
from twist import ModelConfig, LossConfig, Twist, distance_fields, registration_loss

model = Twist(ModelConfig(num_blocks=8, squaring_steps=7))
config = LossConfig()
distance_a = distance_fields(A, threshold=config.distance_threshold)
distance_b = distance_fields(B, threshold=config.distance_threshold)
result = model(A, B)
losses = registration_loss(result, distance_a, distance_b, config)
losses.total.backward()

velocity = result.velocity          # [B,2,image_size,image_size]，默认 [B,2,64,64]
registered_a = result.image          # A(exp(v))，未混合配准端点
registered_b = result.reverse_image  # B(exp(-v))，未混合反向配准端点
middle = result.render(0.5)          # 复用速度，不再运行网络
image = middle.image                # (1-t)*A' + t*B'
```

RegistrationStage(model_config)(A,B) 可单独预测与原图同分辨率的速度场。每个 256 维 token 经 `Linear(256,128)` 输出对应 8×8 patch 的 64 个二维速度向量，按像素位置直接拼接为 64×64。原图最外圈像素速度置零，内部像素速度不做全局裁剪。网络中 A、B 只各编码一次，每个块为 A 的 SA→FFN 和 B 的 SA→CA→FFN；CA 用 B 作 Q、A 作 K/V。

`num_blocks` 控制单个网络内的堆叠深度；`squaring_steps=n` 表示先缩放为 t/2ⁿ，再自复合 n 次，默认 7（128 倍小步复合）。中间查询支持标量或 [batch] 时间张量。A 沿 exp(tv) 采样，B 沿 exp(-(1-t)v) 采样，混合后严格以原始 A、B 为首尾。连续指数关系成立，离散积分和插值有数值误差。

总损失有且只有三项，默认 `similarity + distance + 0.001*smoothness`：

- `similarity`：正向 A(exp(v)) 与 B、反向 B(exp(-v)) 与 A 的梯度加权 MSE 相加。默认 80% 参考图边缘误差、20% 普通全图误差。
- `distance`：双向距离场 Chamfer，相加而非平均。用 B 的软墨迹权重采样 D_A(exp(v))²，再用 A 的软墨迹权重采样 D_B(exp(-v))²；两侧分别按前景质量加 ε 归一化。黑色权重 1、白色 0、灰色保留墨迹覆盖程度。
- `smoothness`：未缩放速度 v 的一阶空间梯度平方，使用原图归一化像素间距。

权重分别由 `similarity_weight`、`distance_weight`、`smoothness_weight` 控制。距离场与前景权重停止梯度，坐标采样保留梯度；原始 A、B 的距离场应预计算并缓存。有限的平滑损失有助于控制空间变化，连续流的微分同胚结论仍依赖速度正则性，离散积分也有误差，详见 [DESIGN.md](DESIGN.md)。

## 默认规模与训练

64×64 输入，8×8 patch，token_dim=256，8 个头，8 个双流块，mlp_ratio=4；共 14,799,744 参数（约 14.800M）。积分次数和查询 t 不增加参数量。

```bash
python -m twist.count_parameters --token-dim 128 --num-heads 4 --num-blocks 12
python twist/train-all-pair/train.py --device cuda --num-blocks 8 --squaring-steps 7 --output runs/twist-svf-pixels
python -m unittest discover -s twist/train-all-pair -p "test_*.py" -v
```

训练设置见 [train-all-pair](train-all-pair/README.md)。逐像素速度场检查点为 version=3；旧粗网格速度场与串联位移检查点需新建实验从零训练，已准备的数据仍可使用。SciPy 用于双向距离场预处理，推理和反向传播由 PyTorch 执行。旧生成模型见 [DiT-flow](../DiT-flow/README.md)。

## 生成全部汉字的过渡

训练完成后，从仓库根目录运行 [generate_all.py](generate_all.py)，将检查点路径替换成自己的 `best.pt` 或 `last.pt`：

```bash
python twist/generate_all.py --checkpoint runs/twist-svf-audited/best.pt --device auto --output runs/twist-svf-audited/all-transitions
```

默认遍历 prepared 数据中全部汉字，**每个至少有两张有效样本的汉字随机选一对不同书写者的图，生成一份 GIF**。覆盖 train/val 合计样本，允许跨划分配对；不是生成全部两两组合。恒定图和缩放后恒定图不计入有效样本。配对由 `--seed` 固定。模型结构和图像分辨率从 version=3 检查点恢复，推理无需距离场；数据按 writer 扫描和按需读取，不加载全部字图到内存。

| 输出 | 内容 |
|---|---|
| `index.html` | 浏览器直接打开的总览，可搜索汉字、Unicode 和书写者；图片延迟加载 |
| `transitions/uXXXX.gif` | 左 A、中间过渡、右 B；默认 21 帧，首尾为原图 |
| `registration/uXXXX.png` | 第一行 A、A(exp(v))、B；第二行 B、B(exp(-v))、A，查看未混合配准效果 |
| `generation.json` | 模型/生成配置、全部配对的字符与原图位置、有效样本数量及不足两张的字符 |
| `summary.json` | 已生成字符数、完成/中断状态与本次耗时 |

默认 `--batch-size 16 --frames 21 --display-size 256 --seed 42 --split all`。`--display-size` 只控制显示面板大小，不改变模型分辨率。`--device auto` 有 CUDA 时使用 CUDA，否则 CPU；精度自动选择 BF16/FP32，也可用 `--precision fp32`。数据不在默认 `data/prepared/` 时添加 `--data /实际路径/data/prepared`；只看验证字符可用 `--split val`。

中断后使用同一指令加 `--resume`，跳过已完成的汉字：

```bash
python twist/generate_all.py --checkpoint runs/twist-svf-audited/best.pt --device auto --output runs/twist-svf-audited/all-transitions --resume
```

续跑会校验检查点内容、manifest 和生成设置，避免混合不同实验；可以调小 `--batch-size`。需要换模型、随机种子、帧数或显示尺寸时使用新的输出目录。可用 `best.pt` 查看最佳验证模型，用 `last.pt` 查看最后保存的模型。

脚本检查：

```bash
python -m unittest discover -s twist/tests -p "test_generate_all.py" -v
```
