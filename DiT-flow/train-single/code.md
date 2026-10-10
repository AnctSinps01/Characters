## 训练：

```bash
python DiT-flow/train-single/train.py --data data/prepared --output runs/DiT-single-author --device cuda --workers 0
```

## 续训：

```bash
python DiT-flow/train-single/train.py --data data/prepared --output runs/DiT-single-author --resume runs/DiT-single-author/last.pt --device cuda --workers 0
```
