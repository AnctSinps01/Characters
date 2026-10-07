# twist

串联 Transformer 配准模型：输入 A、B，预测受固定边界与 Lipschitz 限制的小位移，输出累计变形结果及任意 t∈[0,1] 的画面。白色为 +1、黑色为 −1；坐标为 [0,1]²。

CA 用 B 做 Q、当前 A 做 K/V，修改 B 的查询 token 并解码二维位移。后续阶段读取累计变形后的 A，始终从原始 A 采样。具体定义见 [DESIGN.md](DESIGN.md)。

## 文件

| 文件 | 用途 |
|---|---|
| [config.py](config.py) | 模型尺寸、阶段数、每阶段块数、几何上界、三类 loss 比例 |
| [stage.py](stage.py) | 单个配准阶段 |
| [layers.py](layers.py) | 双流注意力、patch 与位置编码 |
| [model.py](model.py) | 串联阶段与缓存轨迹 |
| [geometry.py](geometry.py) | 边界、Lipschitz 限制、连续场求值与反向采样 |
| [distance.py](distance.py) | 原始图的距离场准备；计算一次后缓存 |
| [losses.py](losses.py) | 软平衡 MSE、距离场和形变正则 |
| [preview.py](preview.py) | 训练入口共用的端点、阶段、轨迹 GIF 与采样网格预览 |
| [count_parameters.py](count_parameters.py) | 参数统计 |
| tests/ | 几何约束、数据流、loss 梯度与混合精度验证 |
| [train-test-1/](train-test-1/README.md) | 单字训练：缓存该字全部样本，train/val 评估、续训和轨迹预览 |
| [train-all-pair/](train-all-pair/README.md) | 全数据集训练：两个书写者的同字手写图配对、轮次遍历和固定验证集 |

## 使用

以下命令从仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m twist.count_parameters
python -m unittest discover -s twist/tests -v
```

A、B 是同形状的浮点张量 [batch, 1, image_size, image_size]，已归一化到 [-1,1]：

```python
from twist import ModelConfig, LossConfig, Twist, distance_fields, registration_loss

model_config = ModelConfig()
loss_config = LossConfig()
model = Twist(model_config)

# 每张原始 A 准备一次，实际训练可由数据集缓存并传入。
distance_a = distance_fields(A, threshold=loss_config.distance_threshold)
result = model(A, B)
losses = registration_loss(result, B, distance_a, loss_config)
losses.total.backward()

endpoint = result.image
all_displacements = result.displacements
all_stage_images = result.stage_images
middle = result.render(0.5).image  # 复用已经预测的位移，不再运行注意力层。
```

单阶段可以独立使用：RegistrationStage(model_config)(current_A, B) 返回受限控制网格 [batch, 2, patch_grid_size+2, patch_grid_size+2]。外圈固定为零，内部每个 token 对应一个二维控制点。轨迹查询支持标量 t 或 [batch] 时间张量。

MSE 根据 B 的软墨迹权重自动平衡前景/背景；三项比例为 mse_weight、distance_weight、deformation_weight。形变项内部的位移占比由 displacement_ratio 调整。默认比例是实验起点，尚未完成真实数据训练。

## 默认规模

64×64 输入，8×8 patch，token_dim=256，8 个头；每阶段 2 个双流块，共 4 个独立阶段。

- 单阶段：3,705,090 参数，约 3.705M。
- 完整模型：14,820,360 参数，约 14.820M。
- share_stage_weights=True：执行 4 阶段，共用 3,705,090 参数。

查询不同 t 不增加模型参数。统计其他设置，例如：

```bash
python -m twist.count_parameters --token-dim 128 --num-heads 4 --num-stages 3
python -m twist.count_parameters --share-stage-weights
```

单字实验见 [train-test-1](train-test-1/README.md)，全数据集同字配对训练见 [train-all-pair](train-all-pair/README.md)。通用数据生成器和亮度/去噪分支尚未包含。SciPy 用于距离场预处理；推理与反向传播由 PyTorch 执行。公共资源与依赖位于上一级，旧实现见 [DiT-flow](../DiT-flow/README.md)。
