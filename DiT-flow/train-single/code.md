单作者子集大小只在本目录的 `config.py` 设置：`SUBSET_SIZE = 16`。新训练实际使用 `min(SUBSET_SIZE, 所选作者字数)` 个字；先按原规则选择训练字数最多的作者，再用训练种子固定打乱字池、选取子集。目标字、参考字和拟合预览全部来自同一份子集，episode 内字符仍不重复。默认 4 个参考字，子集至少需要 5 个字；小于 `max_references + 1` 会在写输出前报错。

续训恢复检查点保存的子集；修改此配置只影响新训练。缩小规模重新实验时使用新的输出目录，例如把下面的 `runs/DiT-single-author` 改为 `runs/DiT-single-author-subset16`。旧的未限制子集的单作者检查点仍按原来的作者全集续训。模型、batch 大小和优化器仍使用 `src/config.py` 的设置；训练目标是 flow MSE。加载器仍预加载所选作者手写图和标准字库，只有参与训练与评估的字符池被限制。

## 训练：

```bash
python DiT-flow/train-single/train.py --data data/prepared --output runs/DiT-single-author --device cuda --workers 0
```

## 续训：

```bash
python DiT-flow/train-single/train.py --data data/prepared --output runs/DiT-single-author --resume runs/DiT-single-author/last.pt --device cuda --workers 0
```
