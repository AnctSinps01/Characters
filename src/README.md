# 代码结构

本目录把设计文档中的网络拆成四层：

- `reference_encoder.py`：共享的逐组编码、每组 `Z` 个描述 token，以及顺序无关的 `M` 个风格 token 汇总。集合汇总由 `set_pool_depth` 个独立的 `CA → SA → FFN` 模块组成。
- `generator.py`：保留目标二维 patch 的条件 DiT；SA/FFN 使用 AdaLN-Zero，风格 CA 使用普通前置 LN 和残差。
- `model.py`：组合参考分支与速度预测主干。
- `flow.py`：构造标准字加噪声起点的 flow-matching 训练样本，并提供复用风格编码的 Euler 采样。

所有尚未由实验确定的尺寸都集中在 `config.py`。当前代码只实现模型与数学路径，不预设数据集格式、训练框架、日志系统或检查点协议。
