"""Blinding of the H-val overlay (addendum H, docs/preregistration_pileup.md section 10, "Blinding on H-val").

Until configs/trackml/overlay_version_h_v1.json (overlay_hash and the provenance manifest; H-T0.5 step 4) is committed,
this module is the only evaluation of overlay ids with k = 0 (H-val, ids 50,000,000 + host), and
slurm/overlay/guard_h.sh, which every model and evaluation job on an overlay id calls, refuses any other.

Blinded counts (`blind_inputs`, `blind_counts`). Gun particles are those with vertex field (particle_id >> 52) in
1001-1010, selected by this field alone. A track with half or more gun hits (2 n_gun >= |T|) is removed: its hits
become unassigned (label -1). Gun hits take particle id 0, the id of hits of no particle, so a host track keeps its
gun hits in |T| and its host matching is the unblinded one. Gun rows leave the particles table. On these inputs
hnl.trackml.evaluate.event_counts sees no gun particle and no track with half or more gun hits:
- its fake rate n_fake_ge5 / n_tracks_ge5 is V2's host-only fake rate, over the tracks with at least 5 hits and more
  than half host hits (noise hits are host hits);
- its per-target rows are host targets only, and no count refers to a gun particle;
- its TrackML score is the unblinded one: gun hits carry weight 0, they still count in the hit totals of a host
  track, and a track with half or more gun hits is good for no host particle.

Sealed directory (`sealed_dir`: RUNS_H/sealed/<set>_<condition>, for H-val in C-full RUNS_H/sealed/hval_full):
  scores/                 edge scores with truth labels (hnl.trackml.infer score; the model job writes them here)
  tracks/b<K>of<N>.parquet raw track labels of every hit, one row group per event and configuration (`evaluate`)
  counts/b<K>of<N>/       blinded per-event counts, host per-target rows and timing of one block (`evaluate`)
  per_event.parquet, per_target.parquet, configs.json, blind.json      blinded, pooled over the blocks (`merge`)
  MANIFEST.sha256, SEALED.json                                         (`seal`)
`seal` writes the sha256 of every file (sha256sum format, sorted relative paths) and makes the files and directories
read-only; the sha256 of MANIFEST.sha256 is recorded in RESULTS.md as one line, `Addendum H sealed H-val manifest
sha256: <64 hex>` (`results_expect` reads it from RESULTS.md at HEAD; decision D4). Before the commit of step 4 only
this module reads the directory: `v2` reads the blinded per-event counts after `verify`. After the commit, the step-5
job opens it through `open_sealed`, which checks the committed record and the manifest against the sha256 of that line
first: `sealed_per_target` (called by hnl.overlay.h1) recomputes the unblinded per-target rows of one frozen
configuration from the raw track labels and checks that the same labels give the sealed blinded counts.

Grants (hnl.overlay.grant): `evaluate` and `v2` run on overlay ids only under a grant of kind blind (H-val) or dry
(pilot) in C-full (`evaluate` writes only under the grant's out, the sealed directory), `sealed_per_target` under one
of kind eval (H-val, which also needs the launch token of `guard_h.sh --launch hval`) or dry (pilot).

Launch (`write_launch`, `check_launch`; slurm/overlay/guard_h.sh --launch hval|htest; decision D3). The launch record
RUNS_H/launch/<set>.json holds the token and pins the sha256 of every file the decisive run depends on: the files the
guard names, every tracked file of slurm/overlay, the configs and the addendum (`LAUNCH_FILES`), and the hnl modules
that the entry points of `LAUNCH_ENTRY_MODULES` import, directly or not (`import_closure`); for H-test also every file
that ledger entry R15 names and the sha256 of the entry's text, for H-val the manifest sha256 of RESULTS.md and every
committed H-val record of decision D7 (results/pileup_hp/h_t05_*, `HT05_RECORDS`; the guard refuses the H-val launch
without one). The guard accepts the set's jobs only if the token matches and every pin equals the current file. The
decisive outputs (H1 on H-test; the step-5 evaluation of H-val; each per working point) are written once
(`write_once`): the run that writes one records a pending record RUNS_H/launch/<set>.pending.<WP>.json, places the
complete output exclusively, then records a consumed marker RUNS_H/launch/<set>.consumed.<WP>.json next to the launch
record, and later runs are refused (`once_refusal`, before any input is read); a run that stops before its pending
record may be repeated, and from the pending record on the working point is consumed: the next attempt records the
marker without writing anything, whatever output path it names (an empty or unreadable pending record counts the
same way).

R15 (`ledger_r15`): ledger entry R15 must name H-test and, each with at least 8 hex digits of its sha256, the H-test id
list, overlay_hash, split_h_v1.json, the version record, the addendum, the H1 code (h1.py, hstats.py) and every
committed H-T1 record (results/pileup_hp/h_t1_*, at least one); each prefix is compared with the current file.

V2 (`v2`): per frozen configuration, the host prompt-target efficiency, the fake rate over tracks with more than half
host hits and the TrackML score, pooled over the events, against the stored val outputs of the hosts
(runs/canto_l_v1/seed0_x1/grid/full/per_event.parquet, sha256 dd86b756..., job 6763371) at the same configuration.
Tolerances (addendum section 9.2): 0.05 points, 0.05 points and 0.001; a flag gate, decided at WP-A. Also reported:
the events whose host prompt-target count differs from the stored one (the host rows are unchanged, so none should).

Placeholders (`dry-scores`, `placeholder_scores`; code tests only): truth-made edge scores of the pilot set (k = 9) in
the format of hnl.trackml.infer score, in C-full into a scratch sealed directory (never under RUNS_H), in C-1GeV and
C-hs for hnl.overlay.tracks. No model runs on a pilot event: the dry configurations have no learned component (cc and
mwslayer), and `evaluate` refuses mwslmerge on k = 9. Table root (`overlay_root`): every production set is read from
the production overlay root; the pilot may be read from a scratch root (--root), as in the integration dry run.

    python -m hnl.overlay.blind evaluate --set hval --sealed DIR [--block K --n-blocks N] [--workers W] [--root R]
    python -m hnl.overlay.blind merge --sealed DIR
    python -m hnl.overlay.blind seal --sealed DIR
    python -m hnl.overlay.blind verify --sealed DIR [--expect MANIFEST_SHA256 | --results [--repo R]]
    python -m hnl.overlay.blind v2 --sealed DIR --out JSON
    python -m hnl.overlay.blind dry-scores --set pilot --sealed DIR [--sigma 2.0] [--root R]   (code tests only)
  used by slurm/overlay/guard_h.sh:
    python -m hnl.overlay.blind ids --set SET
    python -m hnl.overlay.blind sealed-dir --set SET --cond COND
    python -m hnl.overlay.blind check-ids --set SET --ids ID,ID,...|@FILE [--split FILE]
    python -m hnl.overlay.blind check-version --set SET [--record FILE] [--meta FILE]
    python -m hnl.overlay.blind ledger-r15 --ledger FILE [--split FILE] [--record FILE] [--repo R]
    python -m hnl.overlay.blind write-launch --set hval|htest --runs RUNS --repo R [--files 'F F ...']
    python -m hnl.overlay.blind check-launch --set hval|htest --runs RUNS --repo R --token TOKEN
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from hnl.overlay import grant as G

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROD_REPO = "/work/submit/jinboz1/hnl/hnl-displaced"                         # = PROD of slurm/overlay/guard_h.sh
EXT = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml"           # = hnl.trackml.split.EXT
RUNS_H = os.path.join(EXT, "runs", "overlay_h_v1")
DERIVED_OVERLAY = os.path.join(EXT, "derived", "overlay_v1")                 # = hnl.overlay.writer.DERIVED_OVERLAY
SPLIT_H = os.path.join(REPO, "configs", "trackml", "split_h_v1.json")
SELECTION = os.path.join(REPO, "configs", "trackml", "selection_v1.json")
VERSION_RECORD = os.path.join(REPO, "configs", "trackml", "overlay_version_h_v1.json")
LEDGER = os.path.join(REPO, "docs", "feedback", "test_use_ledger.md")
RESULTS_PREFIX = "Addendum H sealed H-val manifest sha256:"
RESULTS_LINE = re.compile(r"^Addendum H sealed H-val manifest sha256: ([0-9a-f]{64})$")
# what ledger entry R15 must name with a sha256 prefix (decision D3), besides the H-test list and overlay_hash
R15_FILES = ("configs/trackml/split_h_v1.json", "configs/trackml/overlay_version_h_v1.json",
             "docs/preregistration_pileup.md", "src/hnl/overlay/h1.py", "src/hnl/overlay/hstats.py")
HT1_RECORDS = "results/pileup_hp/h_t1_"                                      # tracked files with this prefix
HT05_RECORDS = "results/pileup_hp/h_t05_"    # the H-val records of decision D7 (H-T0.5 step 4), pinned by the launch
# the launch pins (decision D3): these files, every tracked file of LAUNCH_DIRS and the import closure of these modules
LAUNCH_FILES = ("slurm/overlay/guard_h.sh", "configs/trackml/split_h_v1.json", "configs/trackml/selection_v1.json",
                "configs/trackml/overlay_version_h_v1.json", "configs/trackml/pipeline_v1.yaml",
                "configs/trackml/res_v1.json", "docs/preregistration_pileup.md")
LAUNCH_DIRS = ("slurm/overlay",)
LAUNCH_ENTRY_MODULES = ("hnl.overlay.blind", "hnl.overlay.grant", "hnl.overlay.h1", "hnl.overlay.hstats",
                        "hnl.overlay.tracks", "hnl.overlay.split_h", "hnl.overlay.conditions", "hnl.overlay.v1",
                        "hnl.overlay.chain", "hnl.trackml.graphs", "hnl.trackml.infer", "hnl.trackml.trackbuild",
                        "hnl.trackml.evaluate", "hnl.trackml.freeze", "hnl.trackml.displaced_diag")

OVERLAY_FIRST_ID = 50_000_000
GUN_VERTEX = (1001, 1010)                    # = hnl.overlay.writer.GUN_VERTEX (vertex field 1000 + LLP index)
CONDITIONS_H = ("full", "1gev", "hs")
MANIFEST, SEALED_JSON = "MANIFEST.sha256", "SEALED.json"
PLACEHOLDER = "truth"                        # sidecar value of truth-made placeholder scores
PLACEHOLDER_MEANS = (2.2, -4.6)              # logit means of true / fake edges (as hnl.trackml.infer --oracle)
VAL_PER_EVENT_SHA256 = "dd86b75680ec665f34c56ae2a4ac68b2674ff4cd8aec3d499bb34944c016cde7"
V2_TOL = {"prompt_eff_points": 0.05, "fake_rate_points": 0.05, "trackml_score": 0.001}
# code tests on k = 9: builders without a learned component (WP-R-P3 and the mwslayer family's val choice)
DRY_CONFIGS = [{"name": "WP-R-P3", "family": "cc", "params": "0.8"},
               {"name": "mwslayer-0.15 (dry)", "family": "mwslayer", "params": "0.15"}]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ------------------------------------------------------------------------------------------------ gun particles
def vertex_field(pid) -> np.ndarray:
    return np.asarray(pid, dtype=np.int64) >> 52


def is_gun(pid) -> np.ndarray:
    """Gun particle ids (vertex field 1001-1010). Noise (0), host (vertex <= 1000) and closure ids (1024-2024) are not."""
    v = vertex_field(pid)
    return (v >= GUN_VERTEX[0]) & (v <= GUN_VERTEX[1])


def blind_labels(labels, gun_hit) -> np.ndarray:
    """Track labels with every track of half or more gun hits removed (its hits set to -1)."""
    labels = np.asarray(labels, dtype=np.int64)
    gun_hit = np.asarray(gun_hit, dtype=bool)
    out = labels.copy()
    on = labels >= 0
    if not on.any():
        return out
    tu, inv = np.unique(labels[on], return_inverse=True)
    n = np.bincount(inv, minlength=len(tu))
    g = np.bincount(inv, weights=gun_hit[on].astype(np.float64), minlength=len(tu)).astype(np.int64)
    drop = 2 * g >= n
    idx = np.flatnonzero(on)
    out[idx[drop[inv]]] = -1
    return out


def host_particles(parts: pd.DataFrame) -> pd.DataFrame:
    """The particles table without its gun rows."""
    return parts[~is_gun(parts.particle_id.to_numpy())].reset_index(drop=True)


def blind_inputs(pid, labels, parts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(pid, labels, particles) as event_counts sees them: gun hits with particle id 0, tracks with half or more gun
    hits removed, gun rows dropped from the particles table."""
    pid = np.asarray(pid, dtype=np.int64)
    gun_hit = is_gun(pid)
    return np.where(gun_hit, 0, pid), blind_labels(labels, gun_hit), host_particles(parts)


def blind_counts(pid, labels, parts: pd.DataFrame, hit_id=None, weight=None) -> tuple[dict, pd.DataFrame]:
    """hnl.trackml.evaluate.event_counts on the blinded inputs: host-only counts and host per-target rows."""
    from hnl.trackml.evaluate import event_counts
    p, lab, hp = blind_inputs(pid, labels, parts)
    c, rows = event_counts(p, lab, hp, hit_id, weight)
    if is_gun(rows.particle_id.to_numpy()).any():           # cannot happen: hp holds no gun row
        raise RuntimeError("a gun particle reached the blinded per-target rows")
    return c, rows


# ------------------------------------------------------------------------------------------------ sets and ids
def load_split(path: str = SPLIT_H) -> dict:
    with open(path) as fh:
        return json.load(fh)


def set_index(set_name: str, split: dict | None = None) -> int:
    split = split or load_split()
    if set_name not in split["sets"]:
        raise KeyError(f"no overlay set {set_name!r} in split_h_v1.json")
    return int(split["sets"][set_name]["k"])


def set_hosts(set_name: str, split: dict | None = None) -> list[int]:
    """Host ids of a set: H-val and closure the tp_v1 val list, pilot its five hosts, H-test the evaluated hosts of
    hnl.overlay.split_h.htest_hosts (the list of H-T1 with each dropped host replaced by its spare; never another
    spare; the list exists only after H-T1)."""
    from hnl.overlay.split_h import htest_hosts
    split = split or load_split()
    if set_name in ("hval", "closure"):
        hosts = split[set_name]
    elif set_name == "pilot":
        hosts = split["sets"]["pilot"]["hosts"]
    elif set_name == "htest":
        try:
            hosts = htest_hosts(split)
        except LookupError as e:
            raise ValueError(f"split_h_v1.json holds no H-test list (it is added at H-T1): {e}") from e
        if not hosts:
            raise ValueError("split_h_v1.json holds an empty H-test list")
    else:
        raise KeyError(f"no overlay set {set_name!r}")
    return sorted(int(h) for h in hosts)


def overlay_id(k: int, host: int) -> int:
    return OVERLAY_FIRST_ID + 1_000_000 * int(k) + int(host)


def set_ids(set_name: str, split: dict | None = None) -> list[int]:
    split = split or load_split()
    k = set_index(set_name, split)
    return [overlay_id(k, h) for h in set_hosts(set_name, split)]


def split_overlay_id(oid: int) -> tuple[int, int]:
    d = int(oid) - OVERLAY_FIRST_ID
    if d < 0:
        raise ValueError(f"{oid} is not an overlay id")
    return d // 1_000_000, d % 1_000_000


def check_ids(set_name: str, ids, split: dict | None = None) -> list[str]:
    """Problems (empty if none): ids that are not overlay ids of the set's hosts (for H-test: of its list, so never a
    spare), repeated ids, or an empty list."""
    try:
        allowed = set(set_ids(set_name, split))
    except (KeyError, ValueError) as e:
        return [str(e)]
    ids = [int(i) for i in ids]
    bad = []
    if not ids:
        bad.append("no ids given")
    if len(set(ids)) != len(ids):
        bad.append("repeated ids")
    out = sorted(set(ids) - allowed)
    if out:
        bad.append(f"{len(out)} ids are not in set {set_name}: {out[:5]}")
    return bad


def parse_ids(spec: str) -> list[int]:
    if spec.startswith("@"):
        with open(spec[1:]) as fh:
            spec = fh.read()
    return [int(x) for x in re.split(r"[\s,:]+", spec.strip()) if x]


# ------------------------------------------------------------------------------------------------ committed records
def git_committed(path: str, repo: str = REPO) -> bool:
    """True if the file is tracked and identical to its version in HEAD (as slurm/overlay/guard_h.sh checks it)."""
    rel = os.path.relpath(os.path.abspath(path), repo)
    if subprocess.run(["git", "-C", repo, "ls-files", "--error-unmatch", rel], capture_output=True).returncode:
        return False
    return subprocess.run(["git", "-C", repo, "diff", "--quiet", "HEAD", "--", rel]).returncode == 0


def record_overlay_hash(record: str = VERSION_RECORD) -> str:
    with open(record) as fh:
        oh = json.load(fh).get("overlay_hash")
    if not isinstance(oh, str) or not HEX64.match(oh):
        raise ValueError(f"{record}: no overlay_hash (64 hex digits)")
    return oh


def check_version(set_name: str, record: str = VERSION_RECORD, meta: str | None = None) -> list[str]:
    """Problems (empty if none): the record has no overlay_hash, or the set's stored meta carries another one."""
    meta = meta or os.path.join(DERIVED_OVERLAY, f"meta_{set_name}.json")
    try:
        oh = record_overlay_hash(record)
    except (OSError, ValueError) as e:
        return [str(e)]
    if not os.path.exists(meta):
        return [f"no meta file {meta}"]
    with open(meta) as fh:
        m = json.load(fh)
    if m.get("overlay_hash") != oh:
        return [f"{meta}: overlay_hash {str(m.get('overlay_hash'))[:12]}... differs from the record's {oh[:12]}..."]
    return []


def unblinded(repo: str = REPO, record: str | None = None) -> bool:
    """True once the overlay version record (step 4 of H-T0.5) is committed unchanged and holds an overlay_hash."""
    record = record or os.path.join(repo, os.path.relpath(VERSION_RECORD, REPO))
    if not os.path.exists(record) or not git_committed(record, repo):
        return False
    try:
        record_overlay_hash(record)
    except ValueError:
        return False
    return True


_PATH = re.compile(r"(?<![\w./-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:md|json|py|txt|csv|sh|sbatch|tsv|yaml))")


def ledger_entry(ledger: str, name: str = "R15") -> str:
    """The text of ledger entry `### <name>. ...` up to the next heading ('' if absent)."""
    out, on = [], False
    with open(ledger) as fh:
        for line in fh:
            if line.startswith(f"### {name}. "):
                on = True
            elif on and re.match(r"^#{2,3} ", line):
                break
            if on:
                out.append(line)
    return "".join(out)


_HEX = re.compile(r"(?<![0-9A-Za-z])([0-9a-fA-F]{8,64})(?![0-9A-Za-z])")


def git_tracked(repo: str, pathspec: str) -> list[str]:
    """Tracked files of the repository matching a git pathspec (repository-relative, sorted)."""
    p = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--", pathspec], capture_output=True)
    return sorted(x.decode() for x in p.stdout.split(b"\0") if x)


def ledger_r15(ledger: str = LEDGER, split: str = SPLIT_H, record: str = VERSION_RECORD,
               repo: str = REPO) -> tuple[list[str], list[str]]:
    """H-T2 rule (addendum section 10, decision D3): (problems, repo paths the entry names). Entry R15 must exist and
    name H-test and, each with at least 8 hex digits of its sha256: the H-test id list (the evaluated hosts,
    hnl.overlay.split_h.htest_sha256), overlay_hash, and the files of R15_FILES (split_h_v1.json, the version record
    with the manifest, the addendum, the H1 code h1.py and hstats.py) and every committed H-T1 record (tracked files
    results/pileup_hp/h_t1_*, at least one), each by its path. Every prefix is compared with the current file (a hex
    run of 8-64 digits anywhere in the entry counts, so a file is named with its prefix when some run of the entry is
    a prefix of its sha256). Every repo file the entry names must then be committed unchanged (the guard checks) and
    is pinned at the launch."""
    text = ledger_entry(ledger, "R15")
    if not text:
        return ["no ledger entry R15"], []
    hexes = {h.lower() for h in _HEX.findall(text)}
    prefixed = lambda sha: any(sha.startswith(h) for h in hexes)          # noqa: E731
    bad = []
    sp = load_split(split)
    try:
        ids = set_hosts("htest", sp)
        sha = hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()
        if not prefixed(sha):
            bad.append(f"R15 does not name the sha256 of the H-test list ({sha[:8]}...)")
    except ValueError as e:
        bad.append(str(e))
    try:
        oh = record_overlay_hash(record)
        if not prefixed(oh):
            bad.append(f"R15 does not name overlay_hash ({oh[:8]}...)")
    except (OSError, ValueError) as e:
        bad.append(str(e))
    if "H-test" not in text:
        bad.append("R15 does not name H-test")
    named = sorted({m for m in _PATH.findall(text) if os.path.isfile(os.path.join(repo, m))})
    ht1 = git_tracked(repo, HT1_RECORDS + "*")
    if not ht1:
        bad.append(f"no H-T1 record ({HT1_RECORDS}*) is committed")
    for need in (*R15_FILES, *ht1):
        p = os.path.join(repo, need)
        if need not in named:
            bad.append(f"R15 does not name {need}")
        elif not prefixed(sha256_file(p)):
            bad.append(f"R15 does not name the sha256 of {need} (now {sha256_file(p)[:8]}...)")
    return bad, named


# ------------------------------------------------------------------------------------------------ RESULTS.md (D4)
def results_expect(repo: str = REPO) -> str:
    """The manifest sha256 of the sealed H-val directory as RESULTS.md at HEAD records it: exactly one line
    `Addendum H sealed H-val manifest sha256: <64 hex>` (decision D4). LookupError otherwise."""
    p = subprocess.run(["git", "-C", repo, "show", "HEAD:RESULTS.md"], capture_output=True, text=True)
    if p.returncode:
        raise LookupError(f"no committed RESULTS.md in {repo}")
    found = [m.group(1) for m in (RESULTS_LINE.match(x.strip()) for x in p.stdout.splitlines()) if m]
    if len(found) != 1:
        raise LookupError(f"RESULTS.md at HEAD holds {len(found)} lines '{RESULTS_PREFIX} <64 hex>', not one")
    return found[0]


# ------------------------------------------------------------------------------------------------ launch (D3)
def runs_root(repo: str = REPO) -> str:
    """The runs root of addendum H: RUNS_H for the production repository; in a copy of it (tests) HNL_H_RUNS may
    redirect it, as in slurm/overlay/guard_h.sh."""
    if os.path.realpath(repo) == os.path.realpath(PROD_REPO):
        return RUNS_H
    return os.environ.get("HNL_H_RUNS") or RUNS_H


def _module_file(name: str, src: str) -> str | None:
    p = os.path.join(src, *name.split("."))
    for f in (p + ".py", os.path.join(p, "__init__.py")):
        if os.path.isfile(f):
            return f
    return None


def import_closure(modules=LAUNCH_ENTRY_MODULES, repo: str = REPO) -> list[str]:
    """Repository-relative paths of the hnl modules (and their packages' __init__.py) that the given modules import,
    directly or through other hnl modules, at module level or inside functions (static, by the ast of each file).
    Modules absent from the repository are skipped."""
    import ast
    src = os.path.join(repo, "src")
    seen, todo, files = set(), list(modules), set()
    while todo:
        m = todo.pop()
        if m in seen or not m.startswith("hnl"):
            continue
        seen.add(m)
        parts = m.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        f = _module_file(m, src)
        if f is None:
            continue
        files.add(f)
        with open(f) as fh:
            tree = ast.parse(fh.read(), f)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    pkg = parts if f.endswith("__init__.py") else parts[:-1]
                    base = ".".join(pkg[:len(pkg) - node.level + 1] + [node.module])
                todo += [base] + [f"{base}.{a.name}" for a in node.names]
    return sorted(os.path.relpath(f, repo) for f in files)


def launch_pins(set_name: str, repo: str = REPO, files=()) -> dict:
    """{repo path: sha256 or None (absent)} of every file the decisive run of the set depends on (module docstring)."""
    paths = set(files) | set(LAUNCH_FILES) | set(import_closure(LAUNCH_ENTRY_MODULES, repo))
    for d in LAUNCH_DIRS:
        paths |= set(git_tracked(repo, d))
    if set_name == "htest":
        _, named = ledger_r15(os.path.join(repo, os.path.relpath(LEDGER, REPO)),
                              os.path.join(repo, os.path.relpath(SPLIT_H, REPO)),
                              os.path.join(repo, os.path.relpath(VERSION_RECORD, REPO)), repo)
        paths |= set(named)
    if set_name == "hval":                       # the H-val records of decision D7 (the step-5 runs' committed inputs)
        paths |= set(git_tracked(repo, HT05_RECORDS + "*"))
    out = {}
    for p in sorted(paths):
        f = os.path.join(repo, p)
        out[p] = sha256_file(f) if os.path.isfile(f) else None
    return out


def _r15_sha(repo: str) -> str:
    return hashlib.sha256(ledger_entry(os.path.join(repo, os.path.relpath(LEDGER, REPO)), "R15").encode()).hexdigest()


def launch_path(set_name: str, runs: str) -> str:
    return os.path.join(runs, "launch", f"{set_name}.json")


def write_launch(set_name: str, runs: str, repo: str = REPO, files=()) -> str:
    """Writes the launch record of the set exactly once (the guard has checked everything before) and returns its
    token. FileExistsError if the set was launched before."""
    if set_name not in ("hval", "htest"):
        raise ValueError("only hval and htest are launched")
    rec = {"set": set_name, "token": secrets.token_hex(16),
           "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head(repo),
           "user": os.environ.get("USER"), "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "pins": launch_pins(set_name, repo, files),
           "r15_sha256": _r15_sha(repo) if set_name == "htest" else None,
           "hval_manifest_sha256": results_expect(repo) if set_name == "hval" else None,
           "rule": "docs/preregistration_pileup.md section 10 and decision D3: the set's jobs run with this token "
                   "only while every pinned file is unchanged; the decisive outputs are written once"}
    path = launch_path(set_name, runs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec["token"]


def check_launch(set_name: str, token: str | None, runs: str, repo: str = REPO) -> list[str]:
    """Problems (empty if none): the set was not launched, the token is not the launch record's, or a pinned file, the
    R15 entry (H-test) or the RESULTS.md manifest line (H-val) differs from the launch."""
    path = launch_path(set_name, runs)
    if not os.path.exists(path):
        return [f"set {set_name} was not launched (bash slurm/overlay/guard_h.sh --launch {set_name})"]
    with open(path) as fh:
        rec = json.load(fh)
    if not token or token != rec.get("token"):
        return [f"HNL_H_LAUNCH does not carry the token of {path}"]
    pins = rec.get("pins")
    if not pins:
        return [f"{path} pins no file"]
    bad = []
    for p, h in sorted(pins.items()):
        f = os.path.join(repo, p)
        cur = sha256_file(f) if os.path.isfile(f) else None
        if cur != h:
            bad.append(f"{p} changed since the launch of {set_name} (sha256 now {str(cur)[:12]}, at the launch "
                       f"{str(h)[:12]})")
    if set_name == "htest" and _r15_sha(repo) != rec.get("r15_sha256"):
        bad.append("ledger entry R15 changed since the launch of htest")
    if set_name == "hval":
        try:
            if results_expect(repo) != rec.get("hval_manifest_sha256"):
                bad.append("the manifest sha256 line of RESULTS.md changed since the launch of hval")
        except LookupError as e:
            bad.append(str(e))
    return bad


def consumed_path(set_name: str, wp: str, runs: str) -> str:
    return os.path.join(runs, "launch", f"{set_name}.consumed.{wp}.json")


def _write_marker(marker: str, rec: dict) -> bool:
    """Creates the consumed marker exclusively (read-only); False if it exists already."""
    os.makedirs(os.path.dirname(os.path.abspath(marker)), exist_ok=True)
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1)
    return True


def pending_path(marker: str) -> str:
    """The pending record of a consumed marker: <set>.pending.<WP>.json next to <set>.consumed.<WP>.json."""
    d, b = os.path.split(marker)
    return os.path.join(d, b.replace(".consumed.", ".pending.", 1) if ".consumed." in b else b + ".pending")


def _output_sha(out: str) -> str | None:
    if not os.path.isfile(out):
        return None
    with open(out, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def once_refusal(out: str, marker: str, info: dict | None = None) -> None:
    """Raises PermissionError if the decisive output was written already, or may have been: its consumed marker exists;
    its pending record exists (a run that began placing its output, at whatever path the record names, and stopped
    before its marker; the working point counts as consumed whether or not that output is in place); or the output
    exists without either (a run that stopped after placing its output and before its marker). An empty or unreadable
    pending record (a run that stopped while writing it) counts as consumed as well, and the refusal says so. In the
    last three cases the marker is then recorded, so the state is the same as after a complete run. Called before a
    decisive run reads any input and again by `write_once`."""
    if os.path.exists(marker):
        raise PermissionError(f"{marker} exists: this output was written once already")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pend = pending_path(marker)
    if os.path.lexists(pend):
        try:
            with open(pend) as fh:
                rec = json.load(fh)
            if not isinstance(rec, dict):
                raise ValueError(f"not a JSON object ({type(rec).__name__})")
        except (OSError, ValueError) as e:      # empty or cut short: a run stopped while writing it; consumed alike
            rec = {"unreadable": f"{type(e).__name__}: {e}"[:300]}
        placed = rec.get("output") or out
        _write_marker(marker, {**(info or {}), "output": placed, "output_present": os.path.isfile(placed),
                               "output_sha256": _output_sha(placed), "pending": rec, "utc": now,
                               "recovered": "a pending record without its marker (a run that stopped while placing "
                                            "its output); marker recorded by a later run, which wrote nothing"})
        why = (f"an unreadable pending record ({rec['unreadable']}), counted as consumed" if "unreadable" in rec
               else f"a run began placing its output {placed} and stopped before its marker")
        raise PermissionError(f"{pend} exists ({why}; marker {marker} now recorded): this output was written once "
                              "already")
    if os.path.lexists(out):
        _write_marker(marker, {**(info or {}), "output": os.path.abspath(out), "output_sha256": _output_sha(out),
                               "utc": now,
                               "recovered": "the output was in place without its marker (a run that stopped between "
                                            "the two); marker recorded by a later run, which wrote nothing"})
        raise PermissionError(f"{out} exists (written by an earlier run that stopped before its marker; marker "
                              f"{marker} now recorded): this output was written once already")


def write_once(out: str, text: str, marker: str, info: dict) -> dict:
    """Writes a decisive output once (decision D3), crash-safe: refused if the consumed marker, the pending record or
    the output exists (`once_refusal`); otherwise the output is written completely to a temporary file, then the
    pending record (output path and sha256) is created exclusively, then the output is placed exclusively with a hard
    link (never over an existing file), and only then the consumed marker is created exclusively. A run that stops
    before its pending record leaves nothing and may be repeated; from the pending record on, the working point is
    consumed: the next attempt records the marker and writes nothing, whether or not the output was placed."""
    once_refusal(out, marker, info)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sha = hashlib.sha256(text.encode()).hexdigest()
    tmp = f"{out}.tmp{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        pend = {**info, "output": os.path.abspath(out), "output_sha256": sha,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "slurm_job": os.environ.get("SLURM_JOB_ID"), "state": "placing the output"}
        if not _write_marker(pending_path(marker), pend):
            raise PermissionError(f"{pending_path(marker)} exists: another run is placing or placed this output")
        try:
            os.link(tmp, out)                         # exclusive: FileExistsError if anything is at `out`
        except FileExistsError:
            raise PermissionError(f"{out} exists: this output was written once already") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    rec = {**info, "output": os.path.abspath(out), "output_sha256": sha,
           "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "grant": os.environ.get(G.ENV)}
    if not _write_marker(marker, rec):                # recorded meanwhile by `once_refusal` of a concurrent attempt
        with open(marker) as fh:
            if json.load(fh).get("output_sha256") != rec["output_sha256"]:
                raise PermissionError(f"{marker} appeared while {out} was written, for another output")
    return rec


# ------------------------------------------------------------------------------------------------ sealed directory
def sealed_dir(set_name: str, cond: str = "full", runs: str = RUNS_H) -> str:
    if cond not in CONDITIONS_H:
        raise ValueError(f"condition {cond!r} not in {CONDITIONS_H}")
    return os.path.join(runs, "sealed", f"{set_name}_{cond}")


def is_sealed(d: str) -> bool:
    return os.path.exists(os.path.join(d, MANIFEST))


def _under(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip("/") + "/")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_entries(d: str) -> list[tuple[str, str]]:
    """(sha256, relative path) of every regular file below d except the manifest and the seal record, sorted by path."""
    out = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d)
            if rel in (MANIFEST, SEALED_JSON) or not os.path.isfile(p) or os.path.islink(p):
                continue
            out.append((sha256_file(p), rel))
    return sorted(out, key=lambda x: x[1])


def _git_head(repo: str = REPO) -> str:
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


SEAL_RULES = {
    "hval": "docs/preregistration_pileup.md section 10: read only by hnl.overlay.blind before the commit of "
            "configs/trackml/overlay_version_h_v1.json, then by the step-5 job after open_sealed",
    "dry": "docs/preregistration_pileup.md section 10: truth-made placeholder outputs of the pilot set (kind dry), "
           "a code test of the blinding, the guard and the H1 code; no model output, no result",
    "plumb": "decision D6 (docs/preregistration_pileup.md section 10): outputs of the frozen model stages on the "
             "pilot overlay events, a plumbing test, sealed right after they were written; read only for files, "
             "schemas, row counts, provenance, timing and memory (hnl.overlay.chain check_outputs, plumb-record); no "
             "score value, no track building, no number of any kind computed from them"}


def seal(d: str, required=("per_event.parquet", "per_target.parquet", "configs.json", "blind.json"),
         rule: str | None = None) -> dict:
    """Write MANIFEST.sha256 and SEALED.json and make the directory read-only. Refuses a sealed or incomplete one.
    rule: the rule text the seal record carries; by default the H-val rule, or the dry rule for a directory whose
    blind.json names the pilot set (hnl.overlay.chain.finish passes the plumbing rule, decision D6)."""
    if is_sealed(d):
        raise PermissionError(f"{d} is sealed already")
    missing = [r for r in required if not os.path.exists(os.path.join(d, r))]
    if not glob.glob(os.path.join(d, "scores", "*.parquet")):
        missing.append("scores/*.parquet")
    if missing:
        raise FileNotFoundError(f"{d}: cannot seal without {missing}")
    ent = manifest_entries(d)
    text = "".join(f"{h}  {p}\n" for h, p in ent)
    with open(os.path.join(d, MANIFEST), "w") as fh:
        fh.write(text)
    rec = {"manifest_sha256": hashlib.sha256(text.encode()).hexdigest(), "n_files": len(ent),
           "bytes": int(sum(os.path.getsize(os.path.join(d, p)) for _, p in ent)), "directory": os.path.realpath(d),
           "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "rule": rule or SEAL_RULES[_seal_kind(d)]}
    with open(os.path.join(d, SEALED_JSON), "w") as fh:
        json.dump(rec, fh, indent=1)
    ro_file = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for root, dirs, files in os.walk(d, topdown=False):
        for f in files:
            os.chmod(os.path.join(root, f), ro_file)
        os.chmod(root, ro_file | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return rec


def _seal_kind(d: str) -> str:
    """hval, or dry for a directory whose blind.json names the pilot set (k = 9)."""
    try:
        with open(os.path.join(d, "blind.json")) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return "hval"
    return "dry" if isinstance(rec, dict) and rec.get("set") == "pilot" else "hval"


def unseal_permissions(d: str) -> None:
    """Restore write permission (removing a scratch dry-run directory in tests); the manifest stays."""
    for root, dirs, files in os.walk(d):
        os.chmod(root, 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)


def verify(d: str, expect: str | None = None) -> list[str]:
    """Problems (empty if none): no manifest, a file missing, added or changed, or a manifest sha256 other than the
    recorded one (`expect`, as RESULTS.md records it; a prefix of at least 8 hex digits is accepted)."""
    mp = os.path.join(d, MANIFEST)
    if not os.path.exists(mp):
        return [f"{d} is not sealed (no {MANIFEST})"]
    with open(mp, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    bad = []
    if expect is not None and (len(expect) < 8 or not sha.startswith(expect.lower())):
        bad.append(f"manifest sha256 {sha[:16]}... is not the recorded {expect}")
    want = {}
    for line in raw.decode().splitlines():
        h, p = line.split("  ", 1)
        want[p] = h
    have = dict((p, h) for h, p in manifest_entries(d))
    for p in sorted(set(want) - set(have)):
        bad.append(f"missing: {p}")
    for p in sorted(set(have) - set(want)):
        bad.append(f"added after sealing: {p}")
    for p in sorted(set(want) & set(have)):
        if want[p] != have[p]:
            bad.append(f"changed after sealing: {p}")
    return bad


def manifest_sha256(d: str) -> str:
    with open(os.path.join(d, MANIFEST), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def open_sealed(d: str, expect: str | None = None, repo: str = REPO, record: str | None = None) -> str:
    """The step-5 job's way into a sealed directory: only after the overlay version record is committed, and only if
    the manifest verifies against the sha256 that RESULTS.md at HEAD records (`results_expect`, decision D4). An
    `expect` given by the caller must agree with that line (it never replaces it). Returns d."""
    if not unblinded(repo, record):
        raise PermissionError("configs/trackml/overlay_version_h_v1.json is not committed: the sealed outputs are read "
                              "only by hnl.overlay.blind")
    try:
        rec = results_expect(repo)
    except LookupError as e:
        raise PermissionError(f"{e}: the sealed directory is opened only with the manifest sha256 that RESULTS.md "
                              "records") from None
    if expect is not None and (len(expect) < 8 or not rec.startswith(expect.lower())):
        raise PermissionError(f"--expect {expect[:16]}... is not the manifest sha256 that RESULTS.md records "
                              f"({rec[:16]}...)")
    bad = verify(d, rec)
    if bad:
        raise PermissionError(f"{d} does not verify: {bad[:5]}")
    return d


def read_labels(sealed: str, events, config: int) -> dict[int, np.ndarray]:
    """Raw track labels of the given events at one configuration index, from tracks/*.parquet of a sealed directory
    (one row group per event and configuration; rows 0..N-1 of the stored event)."""
    import pyarrow.parquet as pq
    want = {int(e) for e in events}
    out: dict[int, np.ndarray] = {}
    for p in sorted(glob.glob(os.path.join(sealed, "tracks", "*.parquet"))):
        t = pq.read_table(p, filters=[("config", "==", int(config))]).to_pandas()
        for e, g in t[t.event.isin(list(want))].groupby("event"):
            if int(e) in out:
                raise ValueError(f"{sealed}: event {e} appears twice in tracks/")
            g = g.sort_values("row")
            if not np.array_equal(g.row.to_numpy(), np.arange(len(g))):
                raise ValueError(f"{sealed}: the labels of event {e} are not rows 0..N-1")
            out[int(e)] = g.label.to_numpy(np.int64)
    missing = sorted(want - set(out))
    if missing:
        raise KeyError(f"{sealed}: no track labels for events {missing[:5]} at configuration {config}")
    return out


def sealed_per_target(sealed: str, expect: str | None = None, config: str = "WP-A", repo: str = REPO,
                      record: str | None = None, root: str | None = None, split: dict | None = None) -> pd.DataFrame:
    """The step-5 read of a sealed C-full directory (H-T0.5 step 5; addendum section 10): the unblinded per-target rows
    (event, config, particle_id, dm, sm_any, k_sm_ge5; gun targets included) of one frozen configuration (by name, as
    in configs.json), recomputed from the raw track labels with hnl.trackml.evaluate.event_counts on the stored events.
    H-val (k = 0; the production sealed directory): only through `open_sealed`, i.e. after the overlay version record
    is committed and with the manifest sha256 that RESULTS.md records (an `expect` given must agree with it), under a
    grant of kind eval in C-full. Pilot (k = 9; a scratch directory, code tests only): the manifest must verify against
    `expect` (required), the scores must be truth-made placeholders, and the grant is of kind dry. The blinded counts
    of every event are recomputed from the same labels and must equal the sealed per_event rows, so the labels are the
    ones the blinded tables came from. Nothing is written into the directory."""
    from hnl.trackml import events as E
    from hnl.trackml.evaluate import event_counts
    split = split or load_split()
    production = _under(sealed, RUNS_H)
    if production:
        open_sealed(sealed, expect, repo, record)
    else:
        if not expect:
            raise PermissionError(f"{sealed}: a scratch sealed directory is read only with its manifest sha256")
        bad = verify(sealed, expect)
        if bad:
            raise PermissionError(f"{sealed} does not verify: {bad[:5]}")
    with open(os.path.join(sealed, "blind.json")) as fh:
        rec = json.load(fh)
    k, set_name = int(rec["k"]), rec["set"]
    if k == 0 and not production:
        raise PermissionError("the raw H-val outputs are read from the production sealed directory only")
    if k == 9 and (production or not rec["scores"].get("placeholder")):
        raise PermissionError("a pilot directory holds truth-made placeholder outputs in a scratch directory only")
    if k not in (0, 9):
        raise PermissionError(f"no sealed C-full directory for set index {k}")
    root = overlay_root(k, root if root is not None else rec.get("root"))
    with open(os.path.join(sealed, "configs.json")) as fh:
        configs = json.load(fh)
    idx = [i for i, c in enumerate(configs) if c["name"] == config]
    if len(idx) != 1:
        raise KeyError(f"{sealed}: {len(idx)} configurations named {config!r} ({[c['name'] for c in configs]})")
    i = idx[0]
    events = set_ids(set_name, split)
    if len(events) != int(rec["events"]):
        raise ValueError(f"{sealed}: {rec['events']} events sealed, the set has {len(events)}")
    G.check(events, "dry" if k == 9 else "eval", "full")
    labels = read_labels(sealed, events, i)
    pe = pd.read_parquet(os.path.join(sealed, "per_event.parquet"))
    pe = pe[pe.config == i].set_index("event")
    rows = []
    for ev in E.iter_events(events, root=root):
        lab = labels[ev.event]
        if len(lab) != ev.n:
            raise ValueError(f"event {ev.event}: {len(lab)} labels for {ev.n} hits")
        hid = ev.full_rows + 1
        cb, _ = blind_counts(ev.pid, lab, ev.particles, hid, ev.weight)
        st = pe.loc[ev.event]
        for key, v in cb.items():
            s_v = st[key]
            same = (np.isclose(float(s_v), float(v), rtol=0, atol=1e-12) or (np.isnan(float(s_v)) and np.isnan(float(v)))
                    if isinstance(v, float) else int(s_v) == int(v))
            if not same:
                raise ValueError(f"event {ev.event}: blinded {key} {v} from the labels differs from the sealed {s_v}")
        _, r = event_counts(ev.pid, lab, ev.particles, hid, ev.weight)
        r.insert(0, "config", np.int16(i))
        r.insert(0, "event", np.int32(ev.event))
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------ configurations
def frozen_configs(cond: str = "full", selection: str = SELECTION) -> list[dict]:
    from hnl.trackml.freeze import frozen_configs as fc
    return [{"name": c["name"], "family": c["family"], "params": str(c["params"]), "val_config": int(c["config"])}
            for c in fc(selection, cond)]


def config_tuple(c: dict) -> tuple:
    return (c["family"], *[float(x) for x in str(c["params"]).split(",") if x != ""])


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def scores_provenance(scores: str, k: int, events, selection: str = SELECTION, condition: str = "full") -> dict:
    """Checks the sidecars of the scored-edge files: the condition and every event present; for k = 9 truth-made
    placeholders only; otherwise the frozen classifier (sha256 of selection_v1.json) and no placeholder."""
    sides = {}
    for p in sorted(glob.glob(os.path.join(scores, "*.parquet"))):
        with open(p + ".json") as fh:
            sides[os.path.basename(p)] = json.load(fh)
    if not sides:
        raise FileNotFoundError(f"no scored-edge files in {scores}")
    have = {int(e) for s in sides.values() for e in s["events"]}
    missing = sorted({int(e) for e in events} - have)
    if missing:
        raise KeyError(f"no scores for events {missing[:5]}")
    want_sha = None
    if k != 9:
        with open(selection) as fh:
            want_sha = json.load(fh)["inputs"]["classifier_checkpoint"]["sha256"]
    for f, s in sides.items():
        if s.get("condition", "full") != condition:
            raise ValueError(f"{f}: scored in condition {s.get('condition')}, not {condition}")
        if k == 9 and s.get("placeholder") != PLACEHOLDER:
            raise PermissionError(f"{f}: k = 9 takes truth-made placeholder scores only; no model runs on a pilot event")
        if k != 9 and (s.get("placeholder") or s.get("ckpt_sha256") != want_sha):
            raise ValueError(f"{f}: not scored by the frozen classifier (sha256 {str(s.get('ckpt_sha256'))[:12]}...)")
    return {"files": sorted(sides), "placeholder": k == 9, "ckpt_sha256": want_sha, "condition": condition}


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate_event(ev, a, b, sc, cfgs: list[tuple], merge_model=None) -> tuple[list[dict], list[pd.DataFrame],
                                                                               list[np.ndarray]]:
    """Track labels of every configuration (the frozen builders, unchanged) and their blinded counts. Returns
    (per-config count dicts, per-config host per-target rows, per-config raw labels)."""
    from hnl.trackml import trackbuild as TB
    st = TB.EventState(ev, a, b, sc, merge_model)
    hid = ev.full_rows + 1
    counts, rows, labels = [], [], []
    for cfg in cfgs:
        t0 = time.time()
        lab = np.asarray(TB.build(cfg, st), dtype=np.int64)
        dt_build = time.time() - t0
        c, r = blind_counts(ev.pid, lab, ev.particles, hid, ev.weight)
        counts.append({**c, "seconds_build": round(dt_build, 2)})
        rows.append(r)
        labels.append(lab.astype(np.int32))
    return counts, rows, labels


def _evaluate_chunk(args):
    from hnl.trackml import events as E
    from hnl.trackml import trackbuild as TB
    events, scores, cfgs, merge_model_path, root = args
    model = None
    if merge_model_path:
        import joblib
        model = joblib.load(merge_model_path)
    sc_all = TB.read_scores(scores, events, "full", kinds=("blind", "dry"))    # the only reader that passes blind
    pe, pt, lab = [], [], []
    for ev in E.iter_events(events, root=root):
        a, b, sc = sc_all[ev.event]
        c, r, L = evaluate_event(ev, a, b, sc, cfgs, model)
        for i in range(len(cfgs)):
            pe.append({"event": ev.event, "config": i, **c[i]})
            ri = r[i].copy()
            ri.insert(0, "config", np.int16(i))
            ri.insert(0, "event", np.int32(ev.event))
            pt.append(ri)
            lab.append((ev.event, i, L[i]))
    return pe, pt, lab


def write_labels(path: str, labels: list[tuple[int, int, np.ndarray]]) -> None:
    """Raw track labels: one row group per (event, configuration); columns event, config, row, label."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([("event", pa.int32()), ("config", pa.int16()), ("row", pa.int32()), ("label", pa.int32())])
    tmp = path + ".tmp"
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=9,
                          column_encoding={"row": "DELTA_BINARY_PACKED"}, use_dictionary=["event", "config"]) as w:
        for e, i, L in sorted(labels, key=lambda x: (x[0], x[1])):
            n = len(L)
            t = pa.Table.from_arrays([pa.array(np.full(n, e, np.int32)), pa.array(np.full(n, i, np.int16)),
                                      pa.array(np.arange(n, dtype=np.int32)), pa.array(L.astype(np.int32))],
                                     schema=schema)
            w.write_table(t, row_group_size=max(n, 1))
    os.replace(tmp, path)


def block_name(block: int, n_blocks: int) -> str:
    return f"b{block:03d}of{n_blocks:03d}"


def overlay_root(k: int, root: str | None = None) -> str | None:
    """The tables to read the overlay events of set index k from, as hnl.trackml.events.iter_events takes them: None
    (the production root, hnl.trackml.events.table_root) for every production set (k = 0, 1, 2); the pilot (k = 9)
    may also be read from a scratch root (dry runs of the chain)."""
    if root is None or os.path.realpath(root) == os.path.realpath(DERIVED_OVERLAY):
        return None
    if int(k) != 9:
        raise PermissionError(f"set index {k}: overlay events are read from {DERIVED_OVERLAY} only, not {root}")
    return os.path.abspath(root)


def evaluate(set_name: str, sealed: str, block: int = 0, n_blocks: int = 1, workers: int = 1,
             configs: list[dict] | None = None, merge_model: str | None = None, split: dict | None = None,
             runs: str = RUNS_H, selection: str = SELECTION, root: str | None = None) -> dict:
    """The blinded evaluation of one block of a set's events in C-full (see the module docstring). root: the overlay
    tables (`overlay_root`: the production root, or for the pilot a scratch root)."""
    import concurrent.futures as cf
    split = split or load_split()
    k = set_index(set_name, split)
    if k not in (0, 9):
        raise PermissionError(f"hnl.overlay.blind evaluates H-val (k = 0) and pilot placeholders (k = 9), not k = {k}")
    root = overlay_root(k, root)
    if k == 0 and os.path.realpath(sealed) != os.path.realpath(sealed_dir(set_name, "full", runs)):
        raise PermissionError(f"raw H-val outputs go to the sealed directory {sealed_dir(set_name, 'full', runs)} only")
    if k == 9 and _under(sealed, RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    if is_sealed(sealed):
        raise PermissionError(f"{sealed} is sealed: nothing is written into it again")
    events = set_ids(set_name, split)
    per = -(-len(events) // n_blocks)
    events = events[block * per:(block + 1) * per]
    if not events:
        raise SystemExit(f"block {block} of {n_blocks} is empty")
    G.check(events, "blind" if k == 0 else "dry", "full", out=sealed)
    scores = os.path.join(sealed, "scores")
    prov = scores_provenance(scores, k, events, selection)
    if configs is None:
        configs = DRY_CONFIGS if k == 9 else frozen_configs("full", selection)
    cfgs = [config_tuple(c) for c in configs]
    if k == 9 and any(c[0] == "mwslmerge" for c in cfgs):
        raise PermissionError("no learned component runs on a pilot event (mwslmerge refused on k = 9)")
    if any(c[0] == "mwslmerge" for c in cfgs) and merge_model is None:
        with open(selection) as fh:
            mm = json.load(fh)["inputs"]["merge_model"]
        if sha256_file(mm["path"]) != mm["sha256"]:
            raise ValueError(f"{mm['path']}: sha256 differs from selection_v1.json")
        merge_model = mm["path"]
    name = block_name(block, n_blocks)
    cdir = os.path.join(sealed, "counts", name)
    if os.path.exists(os.path.join(cdir, "done")):
        raise SystemExit(f"{cdir} is done: refusing to repeat it")
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(sealed, "tracks"), exist_ok=True)
    t0 = time.time()
    chunks = [(events[i:i + 1], scores, cfgs, merge_model, root) for i in range(len(events))]
    with cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
        res = list(ex.map(_evaluate_chunk, chunks))
    pe = pd.DataFrame([r for x in res for r in x[0]])
    pt = pd.concat([r for x in res for r in x[1]], ignore_index=True)
    labs = [r for x in res for r in x[2]]
    write_labels(os.path.join(sealed, "tracks", name + ".parquet"), labs)
    timing = pe[["event", "config", "seconds_build"]]
    _write_parquet(pe.drop(columns=["seconds_build"]), os.path.join(cdir, "per_event.parquet"))
    _write_parquet(pt, os.path.join(cdir, "per_target.parquet"))
    _write_parquet(timing, os.path.join(cdir, "timing.parquet"))
    meta = {"set": set_name, "k": k, "block": block, "n_blocks": n_blocks, "events": [int(e) for e in events],
            "root": root or DERIVED_OVERLAY, "configs": configs, "merge_model": merge_model, "scores": prov,
            "elapsed_s": round(time.time() - t0, 1),
            "git": _git_head(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename}
    _write_json(os.path.join(cdir, "block.json"), meta)
    with open(os.path.join(cdir, "done"), "w") as fh:
        fh.write(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds") + "\n")
    return meta


def blocks_done(sealed: str) -> tuple[int, int]:
    """(blocks done, blocks expected) from the block directories present."""
    names = [os.path.basename(p) for p in glob.glob(os.path.join(sealed, "counts", "b*of*"))]
    if not names:
        return 0, 0
    ns = {int(n.split("of")[1]) for n in names}
    if len(ns) != 1:
        raise ValueError(f"{sealed}: blocks of different splits {sorted(ns)}")
    done = sum(os.path.exists(os.path.join(sealed, "counts", n, "done")) for n in names)
    return done, ns.pop()


def merge(sealed: str, split: dict | None = None) -> dict:
    """Pools the blocks: every event of the set exactly once for every configuration, the same configurations."""
    split = split or load_split()
    if is_sealed(sealed):
        raise PermissionError(f"{sealed} is sealed")
    done, n = blocks_done(sealed)
    if n == 0 or done != n:
        raise SystemExit(f"{sealed}: {done} of {n} blocks done")
    metas = [json.load(open(os.path.join(sealed, "counts", block_name(b, n), "block.json"))) for b in range(n)]
    if len({json.dumps(m["configs"], sort_keys=True) for m in metas}) != 1 or len({m["set"] for m in metas}) != 1 \
            or len({m.get("root", DERIVED_OVERLAY) for m in metas}) != 1:
        raise SystemExit("blocks differ in set, configurations or table root")
    set_name, configs = metas[0]["set"], metas[0]["configs"]
    root = metas[0].get("root", DERIVED_OVERLAY)
    want = set_ids(set_name, split)
    pe = pd.concat([pd.read_parquet(os.path.join(sealed, "counts", block_name(b, n), "per_event.parquet"))
                    for b in range(n)], ignore_index=True)
    pt = pd.concat([pd.read_parquet(os.path.join(sealed, "counts", block_name(b, n), "per_target.parquet"))
                    for b in range(n)], ignore_index=True)
    per = pe.groupby("config").event.agg(lambda e: sorted(int(x) for x in e))
    if len(per) != len(configs) or any(v != want for v in per):
        raise SystemExit(f"blocks do not hold every event of {set_name} exactly once for every configuration")
    if len(pt) and is_gun(pt.particle_id.to_numpy()).any():
        raise RuntimeError("a gun particle in the blinded per-target rows")
    _write_parquet(pe.sort_values(["config", "event"]).reset_index(drop=True), os.path.join(sealed, "per_event.parquet"))
    _write_parquet(pt.sort_values(["config", "event", "particle_id"]).reset_index(drop=True),
                   os.path.join(sealed, "per_target.parquet"))
    _write_json(os.path.join(sealed, "configs.json"), configs)
    meta_path = os.path.join(root, f"meta_{set_name}.json")
    om = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    rec = {"set": set_name, "k": metas[0]["k"], "condition": "full", "events": len(want), "root": root,
           "configs": configs,
           "blocks": [{k: m[k] for k in ("block", "elapsed_s", "git", "slurm_job", "host")} for m in metas],
           "scores": metas[0]["scores"], "merge_model": metas[0]["merge_model"],
           "overlay_meta": {"path": meta_path, "sha256": sha256_file(meta_path) if om else None,
                            "overlay_hash": om.get("overlay_hash"), "emulation_hash": om.get("emulation_hash")},
           "blinding": "event_counts on gun hits with particle id 0, tracks with half or more gun hits removed, gun "
                       "rows dropped (hnl.overlay.blind.blind_inputs); per_event and per_target hold host numbers only",
           "merged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head()}
    _write_json(os.path.join(sealed, "blind.json"), rec)
    return rec


# ------------------------------------------------------------------------------------------------ V2
def _pool(df: pd.DataFrame) -> dict:
    s = df.sum(numeric_only=True)
    r = lambda a, b: float(s[a] / s[b]) if s[b] else float("nan")
    sc = df.trackml_score.dropna() if "trackml_score" in df else pd.Series(dtype=float)
    return {"events": int(len(df)), "n_prompt_target": int(s["n_prompt_target"]),
            "prompt_eff": r("m_prompt_target", "n_prompt_target"), "target_eff": r("m_target", "n_target"),
            "n_tracks_ge5": int(s["n_tracks_ge5"]), "fake_rate": r("n_fake_ge5", "n_tracks_ge5"),
            "trackml_score": float(sc.mean()) if len(sc) else float("nan")}


def val_config_index(cfg: dict, val_configs: list) -> int:
    """The val grid index of a configuration: its frozen index, else the grid entry with the same tuple."""
    if "val_config" in cfg:
        return int(cfg["val_config"])
    t = config_tuple(cfg)
    for i, c in enumerate(val_configs):
        if c[0] == t[0] and len(c) == len(t) and np.allclose([float(x) for x in c[1:]], t[1:]):
            return i
    raise KeyError(f"configuration {cfg} is not in the val grid")


def v2_summary(per_event: pd.DataFrame, configs: list[dict], val_per_event: pd.DataFrame, val_configs: list,
               k: int) -> dict:
    """V2 numbers per configuration: overlay (blinded) against the stored val outputs of the same hosts."""
    rows = []
    for i, cfg in enumerate(configs):
        o = per_event[per_event.config == i].copy()
        o["host"] = o.event.astype(np.int64) - OVERLAY_FIRST_ID - 1_000_000 * k
        vi = val_config_index(cfg, val_configs)
        v = val_per_event[(val_per_event.config == vi) & val_per_event.event.isin(o.host)]
        if sorted(v.event) != sorted(o.host):
            raise KeyError(f"{cfg['name']}: stored val outputs lack hosts {sorted(set(o.host) - set(v.event))[:5]}")
        po, pv = _pool(o), _pool(v)
        j = o.merge(v[["event", "n_prompt_target"]].rename(columns={"event": "host", "n_prompt_target": "val_n"}),
                    on="host")
        d_eff = 100.0 * (po["prompt_eff"] - pv["prompt_eff"])
        d_fake = 100.0 * (po["fake_rate"] - pv["fake_rate"])
        d_score = po["trackml_score"] - pv["trackml_score"]
        ok = {"prompt_eff": abs(d_eff) <= V2_TOL["prompt_eff_points"],
              "fake_rate": abs(d_fake) <= V2_TOL["fake_rate_points"],
              "trackml_score": abs(d_score) <= V2_TOL["trackml_score"]}
        rows.append({"name": cfg["name"], "family": cfg["family"], "params": cfg["params"], "val_config": vi,
                     "overlay": po, "val": pv, "delta_prompt_eff_points": d_eff, "delta_fake_rate_points": d_fake,
                     "delta_trackml_score": d_score, "within": ok, "pass": all(ok.values()),
                     "events_host_prompt_targets_differ": [int(h) for h in j.host[j.n_prompt_target != j.val_n]]})
    return {"configs": rows}


def v2(sealed: str, out: str, val_grid: str | None = None, split: dict | None = None) -> dict:
    """The V2 summary of a sealed directory (verified first) against the stored val outputs; written to `out`, which
    must lie outside the sealed directory."""
    split = split or load_split()
    if _under(out, sealed):
        raise PermissionError("the V2 summary is written outside the sealed directory")
    bad = verify(sealed)
    if bad:
        raise PermissionError(f"{sealed} does not verify: {bad[:5]}")
    rec = json.load(open(os.path.join(sealed, "blind.json")))
    G.check(set_ids(rec["set"], split), "dry" if int(rec["k"]) == 9 else "blind", "full")
    configs = json.load(open(os.path.join(sealed, "configs.json")))
    pe = pd.read_parquet(os.path.join(sealed, "per_event.parquet"))
    if val_grid is None:
        with open(SELECTION) as fh:
            val_grid = os.path.join(json.load(fh)["run"], "grid", "full")
    vpath = os.path.join(val_grid, "per_event.parquet")
    vsha = sha256_file(vpath)
    if vsha != VAL_PER_EVENT_SHA256:
        raise ValueError(f"{vpath}: sha256 {vsha[:12]}... is not the stored val outputs' dd86b756...")
    vcfg = json.load(open(os.path.join(val_grid, "configs.json")))
    s = v2_summary(pe, configs, pd.read_parquet(vpath), vcfg, int(rec["k"]))
    dry = bool(rec["scores"].get("placeholder"))
    gate = s["configs"][0]
    res = {"title": ("DRY RUN (truth-made placeholder scores; code test, no result): " if dry else "")
           + f"V2, host change with the gun present ({rec['set']}, C-full)",
           "addendum": "docs/preregistration_pileup.md section 9.2 (V2, flag gate) and section 10 (blinding)",
           "set": rec["set"], "events": rec["events"], "sealed": os.path.realpath(sealed),
           "manifest_sha256": manifest_sha256(sealed), "overlay_meta": rec["overlay_meta"],
           "val_per_event": {"path": vpath, "sha256": vsha},
           "definitions": {"prompt_eff": "DM-matched host prompt targets / host prompt targets, pooled",
                           "fake_rate": "tracks with >= 5 hits and more than half host hits that DM-match no host "
                                        "particle / such tracks, pooled (blinded; the stored val value counts every "
                                        "track, all of which are host tracks there)",
                           "trackml_score": "mean over events (gun hit weights 0)"},
           "tolerances": V2_TOL, "gate": {"config": gate["name"], "pass": gate["pass"]}, **s,
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head()}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    _write_json(out, res)
    return res


# ------------------------------------------------------------------------------------------------ placeholders
def placeholder_edges(ev, sigma: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truth-made graph and scores of one event (no model): every true edge, and for each true edge its source joined
    to one of the four nearest hits (3-D) of its target; score = sigmoid(mu + N(0, sigma)), mu = 2.2 for true and
    -4.6 for fake edges (hnl.trackml.infer --oracle). Returns (a, b, score, y) with a < b."""
    from scipy.spatial import cKDTree
    from hnl.trackml.events import true_pair_mask
    te = ev.true_edges
    xyz = np.stack([ev.x, ev.y, ev.z], axis=1).astype(np.float64)
    a, b = te[0], te[1]
    if len(b):
        _, nn = cKDTree(xyz).query(xyz[b], k=min(5, ev.n))
        pick = nn[np.arange(len(b)), rng.integers(1, nn.shape[1], len(b))]
        a, b = np.concatenate([te[0], te[0]]), np.concatenate([te[1], pick])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]]).astype(np.int64), axis=1)
    y = true_pair_mask(ev.n, pairs, te)
    z = np.where(y, PLACEHOLDER_MEANS[0], PLACEHOLDER_MEANS[1]) + rng.normal(0.0, sigma, len(y))
    return pairs[0], pairs[1], (1.0 / (1.0 + np.exp(-z))).astype(np.float32), y


def placeholder_scores(set_name: str, out: str, condition: str = "full", sigma: float = 2.0,
                       split: dict | None = None, root: str | None = None) -> dict:
    """Truth-made placeholder scores (`placeholder_edges`) of the pilot set in one condition of addendum H (full, 1gev
    or hs, hnl.overlay.conditions.with_condition), in the format of hnl.trackml.infer score: hit indices are those of
    the event in that condition. Code tests only; never under the production runs root."""
    split = split or load_split()
    if set_index(set_name, split) != 9:
        raise PermissionError("placeholder scores are made for the pilot set (k = 9) only")
    if _under(out, RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    if condition not in CONDITIONS_H:
        raise ValueError(f"condition {condition!r} not in {CONDITIONS_H}")
    root = overlay_root(9, root)
    from hnl.overlay.conditions import with_condition          # after the refusals: these import the frozen chain
    from hnl.trackml import events as E
    from hnl.trackml.infer import write_scores
    os.makedirs(out, exist_ok=True)
    events = set_ids(set_name, split)
    meta = {"placeholder": PLACEHOLDER, "placeholder_rule": placeholder_edges.__doc__.split(" Returns")[0],
            "oracle_sigma": sigma, "oracle_means": list(PLACEHOLDER_MEANS), "ckpt": None, "ckpt_sha256": None,
            "condition": condition, "store": None, "layout": None, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename, "forward": None, "root": root or DERIVED_OVERLAY}
    stream = CONDITIONS_H.index(condition)
    sides = []
    for i in range(0, len(events), 50):
        rows = {}
        for ev in E.iter_events(events[i:i + 50], root=root):
            evc = with_condition(ev, condition)
            rng = np.random.default_rng([ev.event, 99] if stream == 0 else [ev.event, 99, stream])
            a, b, s, y = placeholder_edges(evc, sigma, rng)
            rows[ev.event] = {"a": a, "b": b, "score": s, "y": y}
        sides.append(write_scores(os.path.join(out, f"scores_{min(rows):05d}.parquet"), rows, meta))
    return {"files": len(sides), "events": len(events), "condition": condition,
            "edges": int(sum(sum(s["events"].values()) for s in sides))}


def dry_scores(set_name: str, sealed: str, sigma: float = 2.0, split: dict | None = None,
               root: str | None = None) -> dict:
    """Truth-made placeholder scores of the pilot set in C-full into a scratch sealed directory's scores/ (code tests
    of the blinded evaluation only)."""
    if _under(sealed, RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    return placeholder_scores(set_name, os.path.join(sealed, "scores"), "full", sigma, split, root)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--set", required=True)
    e.add_argument("--sealed", required=True)
    e.add_argument("--block", type=int, default=0)
    e.add_argument("--n-blocks", type=int, default=1)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--configs", default=None, help="JSON list of {name, family, params} (default: the frozen "
                   "configurations of C-full; on k = 9 the dry configurations)")
    e.add_argument("--root", default=None, help="overlay tables (pilot only: a scratch root; default the production "
                   "root)")
    for name in ("merge", "seal"):
        p = sub.add_parser(name)
        p.add_argument("--sealed", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--sealed", required=True)
    p.add_argument("--expect", default=None)
    p.add_argument("--results", action="store_true", help="expect the manifest sha256 that RESULTS.md at HEAD records")
    p.add_argument("--repo", default=REPO)
    p = sub.add_parser("v2")
    p.add_argument("--sealed", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("dry-scores")
    p.add_argument("--set", default="pilot")
    p.add_argument("--sealed", required=True)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--root", default=None)
    p = sub.add_parser("ids")
    p.add_argument("--set", required=True)
    p.add_argument("--split", default=SPLIT_H)
    p = sub.add_parser("sealed-dir")
    p.add_argument("--set", required=True)
    p.add_argument("--cond", default="full")
    p = sub.add_parser("check-ids")
    p.add_argument("--set", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--split", default=SPLIT_H)
    p = sub.add_parser("check-version")
    p.add_argument("--set", required=True)
    p.add_argument("--record", default=VERSION_RECORD)
    p.add_argument("--meta", default=None)
    p = sub.add_parser("ledger-r15")
    p.add_argument("--ledger", default=LEDGER)
    p.add_argument("--split", default=SPLIT_H)
    p.add_argument("--record", default=VERSION_RECORD)
    p.add_argument("--repo", default=REPO)
    for name in ("write-launch", "check-launch"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True, choices=["hval", "htest"])
        p.add_argument("--runs", required=True)
        p.add_argument("--repo", required=True)
        if name == "write-launch":
            p.add_argument("--files", default="", help="further repository files to pin (blank separated)")
        else:
            p.add_argument("--token", default="")
    a = ap.parse_args(argv)

    def fail(problems):
        for x in problems:
            print(f"blind: {x}", file=sys.stderr)
        raise SystemExit(1)

    if a.cmd == "evaluate":
        cfgs = json.loads(a.configs) if a.configs else None
        m = evaluate(a.set, a.sealed, a.block, a.n_blocks, a.workers, cfgs, root=a.root)
        print(json.dumps({k: m[k] for k in ("set", "block", "n_blocks", "elapsed_s")} | {"events": len(m["events"])}))
    elif a.cmd == "merge":
        r = merge(a.sealed)
        print(json.dumps({"set": r["set"], "events": r["events"], "configs": [c["name"] for c in r["configs"]]}))
    elif a.cmd == "seal":
        r = seal(a.sealed)
        print(f"sealed {r['directory']}: {r['n_files']} files, {r['bytes']} bytes; manifest sha256 "
              f"{r['manifest_sha256']} (record it in RESULTS.md)")
    elif a.cmd == "verify":
        expect = a.expect
        if a.results:
            try:
                expect = results_expect(a.repo)
            except LookupError as e:
                fail([str(e)])
            if a.expect and not expect.startswith(a.expect.lower()):
                fail([f"--expect {a.expect[:16]}... is not the manifest sha256 that RESULTS.md records"])
        bad = verify(a.sealed, expect)
        if bad:
            fail(bad)
        print(f"{a.sealed}: manifest verified (sha256 {manifest_sha256(a.sealed)}"
              + (", as RESULTS.md records it)" if a.results else ")"))
    elif a.cmd == "v2":
        r = v2(a.sealed, a.out)
        for c in r["configs"]:
            print(f"{c['name']:>22}: prompt eff {c['delta_prompt_eff_points']:+.4f} pt, fake {c['delta_fake_rate_points']:+.4f}"
                  f" pt, score {c['delta_trackml_score']:+.5f}: {'within' if c['pass'] else 'OUTSIDE'} tolerances")
        print(r["title"], "->", a.out)
    elif a.cmd == "dry-scores":
        print(json.dumps(dry_scores(a.set, a.sealed, a.sigma, root=a.root)))
    elif a.cmd == "ids":
        print(",".join(str(i) for i in set_ids(a.set, load_split(a.split))))
    elif a.cmd == "sealed-dir":
        print(sealed_dir(a.set, a.cond))
    elif a.cmd == "check-ids":
        bad = check_ids(a.set, parse_ids(a.ids), load_split(a.split))
        if bad:
            fail(bad)
    elif a.cmd == "check-version":
        bad = check_version(a.set, a.record, a.meta)
        if bad:
            fail(bad)
    elif a.cmd == "ledger-r15":
        bad, named = ledger_r15(a.ledger, a.split, a.record, a.repo)
        if bad:
            fail(bad)
        print("\n".join(named))
    elif a.cmd == "write-launch":
        try:
            print(write_launch(a.set, a.runs, a.repo, a.files.split()))
        except FileExistsError:
            fail([f"set {a.set} was launched before ({launch_path(a.set, a.runs)}): once only"])
    elif a.cmd == "check-launch":
        bad = check_launch(a.set, a.token or None, a.runs, a.repo)
        if bad:
            fail(bad)


if __name__ == "__main__":
    main()
