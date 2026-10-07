"""单字输入预览；通用 GIF 和形变预览复用 twist.preview。"""
from pathlib import Path
from twist.preview import pixels, save_preview, strip, tile


def save_inputs(glyphs, output: Path):
    items = [tile(glyphs["source"], "A: standard")]
    for split in ("train", "val"):
        for index in glyphs[f"{split}_indices"][:3].tolist():
            items.append(tile(glyphs["targets"][index:index+1], f"{split}: {glyphs['target_ids'][index]}"))
    strip(items).save(output / "inputs.png")


