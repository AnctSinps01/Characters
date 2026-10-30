"""CASIA-HWDB GNT parsing for the shared prepared dataset."""
from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image


def iter_gnt(path: Path):
    """Read raw label bytes as GBK; numeric fields alone are little-endian."""
    with path.open("rb") as stream:
        file_size = path.stat().st_size
        while stream.tell() < file_size:
            offset = stream.tell()
            header = stream.read(10)
            if len(header) != 10:
                raise ValueError(f"{path}:{offset}: truncated GNT header")
            size, label, width, height = struct.unpack("<I2sHH", header)
            if not width or not height or size != 10 + width * height:
                raise ValueError(f"{path}:{offset}: invalid GNT dimensions/record size")
            if size > file_size - offset:
                raise ValueError(f"{path}:{offset}: truncated GNT bitmap")
            bitmap = stream.read(width * height)
            try:
                character = label.decode("gbk").rstrip("\x00")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}:{offset}: invalid GBK label {label.hex()}") from exc
            yield character, Image.frombytes("L", (width, height), bitmap)
