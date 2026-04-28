"""One-time index over a feature cache directory.

Scans .pt files, extracts (path, y, length, src), and caches the result
to <cache_dir>/_index.pt so subsequent runs start instantly.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm.auto import tqdm


class ShardMeta(NamedTuple):
    path: str
    y: int
    length: int
    src: str


def _read_meta(path_str: str) -> ShardMeta:
    p = Path(path_str)
    d = torch.load(p, weights_only=True, map_location="cpu")
    return ShardMeta(
        path=str(p),
        y=int(d["y"]),
        length=int(d["length"]),
        src=str(d.get("src", "")),
    )


def build_index(cache_dir: Path, num_workers: int = 16) -> list[ShardMeta]:
    """Scan cache_dir for .pt shards and return their metadata.

    Cached at <cache_dir>/_index.pt. Delete that file to force a rescan
    (e.g., after re-extracting features).
    """
    cache_dir = Path(cache_dir)
    index_file = cache_dir / "_index.pt"

    if index_file.exists():
        records = torch.load(index_file, weights_only=False)
        return [ShardMeta(**r) for r in records]

    files = [str(p) for p in cache_dir.glob("*.pt") if p.name != "_index.pt"]
    files.sort()

    metas: list[ShardMeta] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_read_meta, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(files),
                        desc=f"index {cache_dir.name}"):
            metas.append(fut.result())

    metas.sort(key=lambda m: m.path)  # deterministic order
    torch.save([m._asdict() for m in metas], index_file)
    return metas
