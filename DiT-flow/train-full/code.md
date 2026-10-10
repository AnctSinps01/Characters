## 训练：

```bash
python DiT-flow/src/train.py --data data/prepared --output runs/DiT-full --device cuda
```

## 续训：

```bash
python DiT-flow/src/train.py --data data/prepared --output runs/DiT-full --resume runs/DiT-full/last.pt --device cuda
```
