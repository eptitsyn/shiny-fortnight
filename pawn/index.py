"""One-time index over a feature cache directory.

Scans .pt files (recursively), extracts a small per-shard record
(path, label-as-y, length, src, split, full meta dict), and caches the
result to <cache_dir>/_index_v2.pt so subsequent runs start instantly.

Bumping to _v2 because shards now carry a generic `meta` dict alongside
the legacy `src` field; the new index has an extra column. Old `_index.pt`
files (if any) are simply ignored.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

INDEX_FILENAME = "_index_v2.pt"


@dataclass
class ShardMeta:
    path: str
    y: int
    length: int
    src: str = ""
    split: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _read_meta(path_str: str) -> ShardMeta:
    p = Path(path_str)
    d = torch.load(p, weights_only=True, map_location="cpu")
    return ShardMeta(
        path=str(p),
        y=int(d["y"]),
        length=int(d["length"]),
        src=str(d.get("src", "")),
        split=str(d.get("split", "")),
        meta=dict(d.get("meta", {})),
    )


def build_index(cache_dir: Path, num_workers: int = 16) -> list[ShardMeta]:
    """Scan cache_dir for .pt shards (recursively) and return their metadata.

    Cached at <cache_dir>/_index_v2.pt. Delete that file to force a rescan
    (e.g., after re-extracting features).
    """
    cache_dir = Path(cache_dir)
    index_file = cache_dir / INDEX_FILENAME

    if index_file.exists():
        records = torch.load(index_file, weights_only=False)
        return [ShardMeta(**r) for r in records]

    files = [
        str(p)
        for p in cache_dir.rglob("*.pt")
        if not p.name.startswith("_index")
    ]
    files.sort()

    metas: list[ShardMeta] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_read_meta, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(files),
                        desc=f"index {cache_dir.name}"):
            metas.append(fut.result())

    metas.sort(key=lambda m: m.path)  # deterministic order
    torch.save([asdict(m) for m in metas], index_file)
    return metas
