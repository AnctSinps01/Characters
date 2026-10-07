"""Preloaded handwriting, same-character partners, and resumable shuffled epochs."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from twist import distance_fields


class PreparedPairs:
    def __init__(self, root: Path, *, image_size: int, threshold: float,
                 storage_device="cpu", distance_cache_size=8192, verbose=True):
        self.root = Path(root).resolve()
        raw = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        if self.manifest.get("version") != 1:
            raise ValueError("expected prepared manifest version 1")
        self.characters = self.manifest["characters"]
        self.writers = self.manifest["writers"]
        self.prepared_size = self.manifest["image_size"]
        self.image_size, self.threshold = image_size, threshold
        self.storage_device = torch.device(storage_device)
        self.cache_size = distance_cache_size
        self.distance_cache = OrderedDict()
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("characters must be nonempty and unique")
        if any(not isinstance(char, str) or len(char) != 1 for char in self.characters):
            raise ValueError("each character label must contain one character")
        if not self.writers or len({w["id"] for w in self.writers}) != len(self.writers):
            raise ValueError("writers must be nonempty with unique IDs")
        if type(self.prepared_size) is not int or self.prepared_size < 2:
            raise ValueError("invalid prepared image_size")
        if image_size < 2 or distance_cache_size < 1:
            raise ValueError("invalid image size or distance cache size")
        self.count = sum(len(w["labels"]) for w in self.writers)
        if not self.count:
            raise ValueError("prepared dataset has no handwriting")
        self.pixel_bytes = self.count * self.prepared_size**2
        if verbose:
            print(f"Preloading {self.count:,} handwriting images: {self.pixel_bytes/1024**3:.3f} GiB uint8 "
                  f"on {self.storage_device}; distance cache <= {distance_cache_size*image_size**2*4/1024**2:.1f} MiB CPU", flush=True)
        self.pixels = torch.empty((self.count, self.prepared_size, self.prepared_size),
                                  dtype=torch.uint8, device=self.storage_device)
        self.labels = torch.empty(self.count, dtype=torch.long)
        self.writer_numbers = torch.empty(self.count, dtype=torch.long)
        split_records = {"train": [], "val": []}
        start = 0
        for number, writer in enumerate(self.writers):
            labels = writer["labels"]
            if writer["split"] not in split_records:
                raise ValueError(f"invalid split for writer {writer['id']}")
            if len(set(labels)) != len(labels) or any(type(label) is not int or not 0 <= label < len(self.characters) for label in labels):
                raise ValueError(f"writer {writer['id']} has invalid or duplicate labels")
            count = len(labels)
            images = np.load(self.root / writer["file"], allow_pickle=False)
            if images.dtype != np.uint8 or images.shape != (count, self.prepared_size, self.prepared_size):
                raise ValueError(f"invalid prepared pixels for {writer['id']}: {images.dtype} {images.shape}")
            self.pixels[start:start+count].copy_(torch.from_numpy(images))
            del images
            self.labels[start:start+count] = torch.tensor(labels, dtype=torch.long)
            self.writer_numbers[start:start+count] = number
            split_records[writer["split"]].append(torch.arange(start, start+count))
            start += count
            if verbose and ((number + 1) % 20 == 0 or number + 1 == len(self.writers)):
                print(f"Loaded {number+1}/{len(self.writers)} writers", flush=True)
        self.group_start = torch.full((self.count,), -1, dtype=torch.long)
        self.group_count = torch.zeros(self.count, dtype=torch.long)
        self.group_rank = torch.zeros(self.count, dtype=torch.long)
        groups, table_offset = [], 0
        self.records, self.skipped = {}, {}
        for split, chunks in split_records.items():
            records = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)
            records = records[torch.argsort(self.labels[records], stable=True)]
            labels = self.labels[records]
            _, counts = torch.unique_consecutive(labels, return_counts=True)
            usable, position = [], 0
            for count in counts.tolist():
                group = records[position:position+count]
                position += count
                # 每个 writer 的标签唯一，因此排除自己即可保证是另一个 writer。
                if count < 2:
                    continue
                self.group_start[group] = table_offset
                self.group_count[group] = count
                self.group_rank[group] = torch.arange(count)
                groups.append(group)
                usable.append(group)
                table_offset += count
            self.records[split] = torch.cat(usable) if usable else torch.empty(0, dtype=torch.long)
            self.skipped[split] = len(records) - len(self.records[split])
        self.partners = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)
        if not len(self.records["train"]) or not len(self.records["val"]):
            raise ValueError("train and val must each contain a character written by at least two different writers")
        if verbose:
            print(f"Eligible source images: train={len(self.records['train']):,}, val={len(self.records['val']):,}; "
                  f"skipped without same-split partner: {self.skipped}", flush=True)

    def match(self, sources, generator):
        sources = torch.as_tensor(sources, dtype=torch.long, device="cpu")
        counts = self.group_count[sources]
        if not bool((counts >= 2).all()):
            raise ValueError("source has no same-character partner in its split")
        choices = (torch.rand(len(sources), generator=generator, dtype=torch.float64) * (counts-1)).long()
        choices += (choices >= self.group_rank[sources]).long()
        targets = self.partners[self.group_start[sources] + choices]
        return torch.stack((sources, targets), dim=1)

    def fixed_validation(self, count, seed):
        generator = torch.Generator().manual_seed(seed)
        candidates = self.records["val"]
        selected = candidates[torch.randperm(len(candidates), generator=generator)]
        if count:
            selected = selected[:count]
        return self.match(selected, generator)

    def normalized(self, indices, device):
        images = self.pixels[indices.to(self.storage_device)].to(device=device, dtype=torch.float32).unsqueeze(1) / 127.5 - 1
        if self.image_size != self.prepared_size:
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)
        return images

    def batch(self, pairs, device):
        pairs = torch.as_tensor(pairs, dtype=torch.long, device="cpu")
        source = self.normalized(pairs[:, 0], device)
        target = self.normalized(pairs[:, 1], device)
        # 只为缺失的源图准备距离场，重复来源去重；缓存始终在 CPU，避免占用训练显存。
        values, missing, positions = {}, [], []
        for position, source_id in enumerate(pairs[:, 0].tolist()):
            if source_id in values:
                continue
            if source_id in self.distance_cache:
                values[source_id] = self.distance_cache[source_id]
                self.distance_cache.move_to_end(source_id)
            elif source_id not in missing:
                missing.append(source_id)
                positions.append(position)
        if missing:
            fields = distance_fields(source[positions].detach().cpu(), threshold=self.threshold)
            for source_id, field in zip(missing, fields):
                field = field.clone()  # 不保留整个 batch 的底层存储，缓存上限对应实际像素存储。
                values[source_id] = field
                self.distance_cache[source_id] = field
                if len(self.distance_cache) > self.cache_size:
                    self.distance_cache.popitem(last=False)
        distance = torch.stack([values[source_id] for source_id in pairs[:, 0].tolist()]).to(device)
        return source, target, distance

    def describe(self, pair):
        source, target = map(int, pair)
        label = int(self.labels[source])
        return {"character": self.characters[label], "label": label, "source_index": source, "target_index": target,
                "source_writer": self.writers[int(self.writer_numbers[source])]["id"],
                "target_writer": self.writers[int(self.writer_numbers[target])]["id"]}


class ShuffledPairSampler:
    def __init__(self, data: PreparedPairs, seed, state=None):
        self.data, self.seed = data, seed
        self.epoch = state["epoch"] if state else 0
        self.offset = state["offset"] if state else 0
        self.generator = torch.Generator().manual_seed(seed ^ 0x45D9F3B)
        if state:
            self.generator.set_state(state["partner_rng"])
        self._shuffle()
        if not 0 <= self.offset <= len(self.order):
            raise ValueError("invalid resumed sampler offset")

    def _shuffle(self):
        records = self.data.records["train"]
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % 2**63)
        self.order = records[torch.randperm(len(records), generator=generator)]

    def next(self, batch_size):
        if self.offset == len(self.order):
            self.epoch += 1
            self.offset = 0
            self._shuffle()
        sources = self.order[self.offset:self.offset+batch_size]
        self.offset += len(sources)
        return self.data.match(sources, self.generator)

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset, "partner_rng": self.generator.get_state()}

    @property
    def epoch_progress(self):
        return self.epoch + self.offset / len(self.order)
