`config.py` 中的 `DATASET_SIZE` 用来计算缩放比例。脚本读取完整训练池的图像总数，计算 `DATASET_SIZE / 完整训练图像数`，再按这个比例缩减作者数量和每位作者的字符池；因此它控制的是作者维度和字符维度的共同规模，最终选中的图像数会因两次缩减、取整以及每个 episode 至少需要 `max_references + 1` 个不同字符而与配置值有出入。作者按训练字符数从多到少稳定选择，字符子集由独立种子固定生成。训练和预览评估使用同一批选中的作者与字符池。

## 训练

```bash
python DiT-flow/train-half/train.py --data data/prepared --output runs/DiT-v2-half --device cuda --workers 0
```

## 续训

```bash
python DiT-flow/train-half/train.py --data data/prepared --output runs/DiT-v2-half --resume runs/DiT-half/last.pt --device cuda --workers 0
```

`DATASET_SIZE` 只影响新实验。续训会从 checkpoint 恢复已保存的作者和字符选择；如果要用新的规模，请使用新的输出目录。
