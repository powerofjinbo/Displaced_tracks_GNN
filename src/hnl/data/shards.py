"""Reader for samples stored as tar shards of Zenodo-format npz events (docs/hnl_points_spec.md section 5).

The HNL points produced in this project (``hnl_lrt/generated/hnl_<m>_<ctau>_v1/``) are stored as one uncompressed tar
per production chunk, ``events_<first>_<last>.tar`` (six-digit, zero-padded, inclusive), whose members are
``<event>.npz``: the exact bytes ``np.savez`` writes for a file, with the keys and dtypes of ``hnl.data.npz.NPZ_KEYS``.
Truth sidecars live next to the sample in ``<dir>_truth/truth_<first>.npz``.

This module only adds readers; ``hnl.data.npz`` and every existing sample are unchanged. The samples are listed in
``POINTS`` and are not added to ``hnl.data.npz.SAMPLES`` (code that iterates over SAMPLES does not see them).

    from hnl.data import shards
    ev = shards.load_event("hnl5_50", 17)              # dict of arrays, as hnl.data.npz.load_event
    for i, ev in shards.iter_events("hnl5_50"):        # shard by shard, one open tar at a time
        ...
    shards.extract("hnl5_50", "/scratch/dir")          # writes /scratch/dir/<dirname>/<i>.npz for file-based tools
    python -m hnl.data.shards list|check|extract ...
"""

from __future__ import annotations

import argparse
import functools
import io
import os
import re
import tarfile
from typing import Iterator

import numpy as np

from hnl.data.npz import GENERATED_ROOT, NPZ_KEYS, Sample

SHARD_RE = re.compile(r"^events_(\d{6})_(\d{6})\.tar$")

# The HNL points of docs/hnl_points_spec.md (evaluation only). n_events is the planned size; labeled: the particle
# table carries the generator parent codes in column 12, as the labeled Zenodo HNL samples (|parent| = 50: N
# daughter, |parent| = 24 and |PDG| = 13: W muon; hnl.truth.labels.labels_from_table). The *_ref points repeat the
# Zenodo points hnl10_100 and hnl15_200 in our simulation chain (in-stack references, spec section 1).
POINTS: dict[str, Sample] = {
    s.key: s
    for s in (
        Sample("hnl5_50", "hnl_5_50_v1", 3000, "hnl", True, 5.0, 50.0, root=GENERATED_ROOT),
        Sample("hnl15_500", "hnl_15_500_v1", 3000, "hnl", True, 15.0, 500.0, root=GENERATED_ROOT),
        Sample("hnl20_200", "hnl_20_200_v1", 3000, "hnl", True, 20.0, 200.0, root=GENERATED_ROOT),
        Sample("hnl10_100_ref", "hnl_10_100_v1", 3000, "hnl", True, 10.0, 100.0, root=GENERATED_ROOT),
        Sample("hnl15_200_ref", "hnl_15_200_v1", 3000, "hnl", True, 15.0, 200.0, root=GENERATED_ROOT),
    )
}


def sample_dir(sample: str | Sample, root: str | None = None) -> str:
    s = POINTS[sample] if isinstance(sample, str) else sample
    return os.path.join(root if root is not None else (s.root or GENERATED_ROOT), s.dirname)


def shard_name(first: int, last: int) -> str:
    return f"events_{first:06d}_{last:06d}.tar"


@functools.lru_cache(maxsize=64)
def list_shards(directory: str) -> tuple[tuple[int, int, str], ...]:
    """(first, last, path) of every complete shard in a directory, sorted; overlapping ranges are an error."""
    out = []
    for name in os.listdir(directory):
        m = SHARD_RE.match(name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), os.path.join(directory, name)))
    out.sort()
    for (a0, a1, pa), (b0, _, pb) in zip(out, out[1:]):
        if b0 <= a1:
            raise ValueError(f"overlapping shards {pa} and {pb}")
    return tuple(out)


def shard_for(directory: str, index: int) -> str:
    for first, last, path in list_shards(directory):
        if first <= index <= last:
            return path
    raise IndexError(f"{directory}: no shard holds event {index}")


@functools.lru_cache(maxsize=8)
def _members(path: str) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(path, "r:") as tf:
        return {ti.name: ti for ti in tf.getmembers() if ti.isfile()}


def _decode(data: bytes, where: str, check: bool) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as f:
        ev = {k: f[k] for k in f.files}
    if check:
        got = {k: str(v.dtype) for k, v in ev.items()}
        if got != NPZ_KEYS:
            raise ValueError(f"{where}: unexpected keys/dtypes {got}")
    return ev


def load_event_from(directory: str, index: int, check: bool = True) -> dict[str, np.ndarray]:
    """One event of a tar-sharded sample directory, as a dict of arrays (same as hnl.data.npz.load_event)."""
    path = shard_for(directory, index)
    name = f"{index}.npz"
    ti = _members(path).get(name)
    if ti is None:
        raise KeyError(f"{path}: no member {name}")
    with open(path, "rb") as f:
        f.seek(ti.offset_data)
        data = f.read(ti.size)
    return _decode(data, f"{path}:{name}", check)


def load_event(sample: str, index: int, root: str | None = None, check: bool = True) -> dict[str, np.ndarray]:
    s = POINTS[sample]
    if not 0 <= index < s.n_events:
        raise IndexError(f"{sample}: event {index} outside 0..{s.n_events - 1}")
    return load_event_from(sample_dir(s, root), index, check)


def iter_events_from(directory: str, indices=None, check: bool = True) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """(index, event) in increasing index order, reading each needed shard once, sequentially."""
    want = None if indices is None else set(int(i) for i in indices)
    for first, last, path in list_shards(directory):
        if want is not None and not any(first <= i <= last for i in want):
            continue
        with tarfile.open(path, "r:") as tf:
            items = []
            for ti in tf:
                if not ti.isfile() or not ti.name.endswith(".npz"):
                    continue
                i = int(ti.name[:-4])
                if want is None or i in want:
                    items.append((i, tf.extractfile(ti).read()))
        for i, data in sorted(items):
            yield i, _decode(data, f"{path}:{i}.npz", check)


def iter_events(sample: str, indices=None, root: str | None = None, check: bool = True):
    return iter_events_from(sample_dir(sample, root), indices, check)


def event_indices_from(directory: str) -> np.ndarray:
    idx = []
    for _, _, path in list_shards(directory):
        idx += [int(n[:-4]) for n in _members(path) if n.endswith(".npz")]
    return np.array(sorted(idx), dtype=np.int64)


def extract(sample: str, dest: str, root: str | None = None) -> str:
    """Write every event as <dest>/<dirname>/<i>.npz (byte-identical to the tar members) for file-based tools; members
    are written by name only (no paths, links or devices are taken from the archive). Returns the directory."""
    s = POINTS[sample]
    out = os.path.join(dest, s.dirname)
    os.makedirs(out, exist_ok=True)
    for _, _, path in list_shards(sample_dir(s, root)):
        with tarfile.open(path, "r:") as tf:
            for ti in tf:
                if not ti.isfile() or not re.fullmatch(r"\d+\.npz", ti.name):
                    raise ValueError(f"{path}: unexpected member {ti.name!r}")
                tmp = os.path.join(out, f".{ti.name}.tmp")
                with open(tmp, "wb") as f:
                    f.write(tf.extractfile(ti).read())
                os.replace(tmp, os.path.join(out, ti.name))
    return out


def check(directory: str) -> dict:
    """Shard ranges contiguous from 0, one member per event of each range, keys and dtypes as NPZ_KEYS."""
    shards = list_shards(directory)
    expect = 0
    n = 0
    for first, last, path in shards:
        if first != expect:
            raise ValueError(f"{directory}: gap before {path}")
        names = sorted(_members(path))
        want = sorted(f"{i}.npz" for i in range(first, last + 1))
        if names != want:
            raise ValueError(f"{path}: members do not match its range")
        for i, _ in iter_events_from(directory, [first, last]):
            n += 1
        expect = last + 1
    return {"directory": directory, "shards": len(shards), "events": expect, "events_checked": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("directory")
    p = sub.add_parser("check")
    p.add_argument("directory")
    p = sub.add_parser("extract")
    p.add_argument("sample", choices=sorted(POINTS))
    p.add_argument("dest")
    p.add_argument("--root")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for first, last, path in list_shards(a.directory):
            print(first, last, path)
    elif a.cmd == "check":
        print(check(a.directory))
    else:
        print(extract(a.sample, a.dest, a.root))


if __name__ == "__main__":
    main()
