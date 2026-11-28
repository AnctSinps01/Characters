"""Print parameter counts and relative compute for DiT size variants."""

from VAE.flow.dit import DiTConfig, DiTFlow

VARIANTS = [
    ("current", 128, 24, 512),
    ("A deeper", 128, 48, 512),
    ("B wider", 192, 24, 768),
    ("C much wider", 256, 24, 1024),
    ("D mid+deeper", 160, 32, 640),
    ("E ffn only", 128, 24, 2048),
    ("F wide+deeper", 192, 36, 768),
]


def main():
    base = None
    rows = []
    for name, d, depth, ffn in VARIANTS:
        model = DiTFlow(DiTConfig(d_model=d, depth=depth, ffn=ffn))
        params = sum(p.numel() for p in model.parameters())
        compute = depth * (8 * d * d + 2 * d * ffn)
        if base is None:
            base = (params, compute)
        rows.append((name, d, depth, ffn, params, params / base[0], compute / base[1]))
    header = f"{'variant':14s} {'d':>4s} {'layers':>6s} {'ffn':>5s} {'params':>12s} {'x params':>9s} {'x compute':>9s}"
    print(header)
    for name, d, depth, ffn, params, pr, cr in rows:
        print(f"{name:14s} {d:>4d} {depth:>6d} {ffn:>5d} {params:>12,} {pr:>8.1f}x {cr:>8.1f}x")


if __name__ == "__main__":
    main()
