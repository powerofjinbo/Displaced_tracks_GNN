"""Grants of addendum H (docs/preregistration_pileup.md section 10): the Python side of slurm/overlay/guard_h.sh.

A job that runs a network or a learned component, or evaluates model outputs, on overlay events first asks the guard:

    G=$(bash slurm/overlay/guard_h.sh --set S --cond full|1gev|hs --kind K --ids ID,..|@FILE [--out DIR] --grant) \
        || exit 1; export "$G"

On success the guard writes a grant (`write`) and prints exactly one line, HNL_H_GRANT=<path>, on stdout. Grants of the
production sets lie under <runs root>/grants/, grants of the pilot set (k = 9) under the --out directory's parent.
Every entry point that runs the frozen model or a learned component, or evaluates outputs, on given events calls
`check(ids, kind, cond, out=...)` once its event list is known: the frozen chain's CLIs (hnl.trackml.graphs, infer,
trackbuild, displaced_diag, evaluate), its development CLIs (g0, probe, support_gpu, chunked, g3check), the choke points
hnl.trackml.events.Embedder.__call__ (any embedding network on an event) and hnl.trackml.trackbuild.read_scores (any
read of stored edge scores), and hnl.overlay.tracks grid, h1 run, v1 run, blind evaluate, v2 and sealed_per_target.
`check` does nothing, and calls no git, when every id is below 50,000,000 (no overlay id). Otherwise it raises
PermissionError unless HNL_H_GRANT names a readable grant (several may be joined by ':'; one must cover the request)
- whose kind is the requested one (a caller may accept several kinds),
- whose condition is the requested one, when the caller names one,
- whose ids include every requested id (and whose ids_sha256 is that of its ids),
- whose code state is the current one (decision E1, narrowed in the fourth code round): the git blob hashes at HEAD of
  exactly the code an overlay job runs (`bound_paths`): every file of src/hnl/overlay, src/hnl/trackml and
  slurm/overlay, every other src/hnl module in the static import closure of the modules of hnl.overlay and
  hnl.trackml (their tests excepted), and the files under configs that these modules name (a path literal, or
  os.path.join of literals; a directory names every file below it), together with the blob hash of the addendum at
  HEAD (None while it is not committed). A commit that changes no bound file and leaves the addendum as it was
  (paper, docs, results, DECISIONS, src/hnl/eval outside the closure, slurm/addendum_g*) keeps the grant valid; a
  commit that changes, adds or removes a bound file, or adds a module or a configs file to the bound set, stops it,
- while no bound file has an uncommitted change in the working tree (and no untracked python file lies under
  src/hnl/overlay or src/hnl/trackml or at a bound path, and no untracked file under src would be imported in place of
  a bound module: a package directory or a compiled module of its name, `shadow_paths`): a later process of the job
  would run it,
- whose slurm_job_id is the running job's: a grant made inside a Slurm job is refused in another job and outside Slurm
  (SLURM_JOB_ID unset), and a grant made outside Slurm is refused inside a job,
- whose out directory holds the path the caller writes, when the caller names one and the grant has an out,
- and, for H-test ids (k = 2), and for H-val ids (k = 0) under a grant of kind eval or of kind model in C-1GeV or C-hs
  (the runs after the overlay_hash commit), which carries the token of the set's launch record RUNS/launch/<set>.json,
  with every file pinned at the launch unchanged (hnl.overlay.blind.check_launch).
On success it writes one line naming the grant to stderr, so every job log records its grant.

The guard and the grants stop accidental runs. A deliberate bypass through direct Python calls stays possible.

Kinds: model (a network or learned component of the frozen chain on the events: graph, event store, edge scores),
plumb (the same on the pilot set only, outputs in a scratch directory, from which no number of any kind is computed;
decision D6), blind (hnl.overlay.blind), eval (any other evaluation of model outputs: track building with
hnl.overlay.tracks or hnl.trackml.trackbuild, matching, per-target tables, V1, the H1 code), dry (truth-made
placeholder outputs of the pilot set).

Grant JSON: set, cond, kind, ids (sorted ints), ids_sha256 (sha256 of the comma-joined sorted ids), code ({files: {repo
path: git blob hash at HEAD, or null for a bound path absent at HEAD}, addendum: blob hash of
docs/preregistration_pileup.md at HEAD or null}), commit (HEAD when granted; provenance only, not checked),
slurm_job_id (SLURM_JOB_ID or null), out, created_utc, launch (token or null).

    python -m hnl.overlay.grant write --set S --cond C --kind K --ids ID,..|@FILE --dir DIR [--out O] [--launch T]
        [--repo R]                                        (called by slurm/overlay/guard_h.sh --grant; prints the path)
    python -m hnl.overlay.grant check --ids ID,..|@FILE --kind K [--cond C] [--out PATH]
    python -m hnl.overlay.grant state [--repo R] [--json]   (the bound code: digest, file count, working-tree changes)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ENV = "HNL_H_GRANT"
OVERLAY_FIRST_ID = 50_000_000
KINDS = ("model", "plumb", "blind", "eval", "dry")
CONDS = ("full", "1gev", "hs")
FIELDS = ("set", "cond", "kind", "ids", "ids_sha256", "code", "commit", "slurm_job_id", "out", "created_utc", "launch")
TOKEN = re.compile(r"^[0-9a-f]{32}$")
ADDENDUM = "docs/preregistration_pileup.md"
# decision E1 (narrowed in the fourth code round): what a grant binds, computed from the code at HEAD (`bound_paths`)
BOUND_TREES = ("src/hnl/overlay", "src/hnl/trackml", "slurm/overlay")    # every tracked file below these
ENTRY_TREES = ("src/hnl/overlay/", "src/hnl/trackml/")                   # their modules seed the import closure
CONFIGS = "configs"
CONFIG_LITERAL = re.compile(r"^configs(/[A-Za-z0-9_.-]+)+/?$")
LISTED = ("src/hnl", "slurm/overlay", CONFIGS, ADDENDUM)                 # what `git ls-tree HEAD` lists for it
PILOT_K = 9
PLACEHOLDER = "truth"                                # = hnl.overlay.blind.PLACEHOLDER (sidecar of truth-made scores)


def parse_ids(spec) -> list[int]:
    """Ids from a comma (or blank) separated string, '@FILE', or any iterable of ints."""
    if isinstance(spec, str):
        if spec.startswith("@"):
            with open(spec[1:]) as fh:
                spec = fh.read()
        return [int(x) for x in re.split(r"[\s,:]+", spec.strip()) if x]
    return [int(x) for x in spec]


def ids_sha256(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(int(x) for x in ids)).encode()).hexdigest()


def set_index(oid: int) -> int:
    """k of an overlay id (50,000,000 + 1,000,000 k + host)."""
    return (int(oid) - OVERLAY_FIRST_ID) // 1_000_000


def is_overlay(oid: int) -> bool:
    return int(oid) >= OVERLAY_FIRST_ID


def git_head(repo: str | None = None) -> str:
    repo = repo or REPO
    p = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"git rev-parse HEAD failed in {repo}: {p.stderr.strip()}")
    return p.stdout.strip()


def _git(repo: str, *args, data: bytes | None = None) -> bytes:
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, input=data)
    if p.returncode:
        raise RuntimeError(f"git {args[0]} failed in {repo}: {p.stderr.decode().strip()}")
    return p.stdout


def head_listing(repo: str | None = None) -> dict[str, str]:
    """{repo path: blob hash} at HEAD of every file under src/hnl, slurm/overlay and configs, and of the addendum."""
    repo = repo or REPO
    have = {}
    for entry in _git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", *LISTED).split(b"\0"):
        if entry:
            meta, path = entry.decode().split("\t", 1)
            if meta.split()[1] == "blob":
                have[path] = meta.split()[2]
    return have


_TEXT: dict[str, str] = {}                    # blob hash -> text (blobs of HEAD read once per process)
_SCAN: dict[tuple, tuple] = {}                # (blob, module, package) -> (modules, possible modules, configs paths)


def _read_blobs(repo: str, blobs) -> None:
    need = [b for b in dict.fromkeys(blobs) if b not in _TEXT]
    if not need:
        return
    raw = _git(repo, "cat-file", "--batch", data="".join(b + "\n" for b in need).encode())
    i = 0
    for b in need:
        nl = raw.index(b"\n", i)
        head = raw[i:nl].decode().split()
        if len(head) != 3 or head[0] != b:
            raise RuntimeError(f"git cat-file --batch: unexpected header {head} for {b}")
        size = int(head[2])
        _TEXT[b] = raw[nl + 1:nl + 1 + size].decode("utf-8", "replace")
        i = nl + 1 + size + 1


def scan_source(text: str, module: str, package: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(modules imported for certain, names that may be modules, configs paths named) of one module's source: `import
    X` and the X of `from X import ...` (relative imports resolved), each `from X import Y` as X.Y, and the configs path
    literals of the code (a string 'configs/...', or os.path.join with the literal 'configs' followed by literals; a
    non-literal argument ends the path, which then names a directory). Docstrings and other prose never match."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [], []
    parts = module.split(".")
    pkg = parts if package else parts[:-1]
    sure, maybe, cfg = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            sure += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(pkg[:len(pkg) - node.level + 1] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                sure.append(base)
                maybe += [f"{base}.{a.name}" for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            for i, a in enumerate(node.args):
                if isinstance(a, ast.Constant) and a.value == CONFIGS:
                    seg = [CONFIGS]
                    for b in node.args[i + 1:]:
                        if not (isinstance(b, ast.Constant) and isinstance(b.value, str)):
                            break
                        seg.append(b.value.strip("/"))
                    cfg.append("/".join(seg))
                    break
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and CONFIG_LITERAL.match(node.value):
            cfg.append(node.value.rstrip("/"))
    return sure, maybe, cfg


def module_paths(listing) -> dict[str, str]:
    """{module name: repo path} of the python files under src/hnl (a package by its __init__.py)."""
    out = {}
    for p in listing:
        if p.startswith("src/hnl/") and p.endswith(".py"):
            name = p[len("src/"):-len(".py")].replace("/", ".")
            out[name[:-len(".__init__")] if name.endswith(".__init__") else name] = p
    return out


def bound_paths(repo: str | None = None, listing: dict | None = None) -> dict[str, str | None]:
    """{repo path: blob hash at HEAD, or None for a bound path absent at HEAD}: exactly the code an overlay job runs
    (decision E1, narrowed in the fourth code round), computed from the code at HEAD:
      - every file of src/hnl/overlay, src/hnl/trackml and slurm/overlay;
      - every other src/hnl module in the static import closure (module level and inside functions) of the modules of
        hnl.overlay and hnl.trackml (not their tests), with the __init__.py of each package on the way; a module
        imported by name that is absent at HEAD is bound as absent;
      - the files under configs that these modules name in their code (`scan_source`); a directory binds every file
        below it, and a named path absent at HEAD is bound as absent.
    The addendum is bound separately (`code_state`)."""
    repo = repo or REPO
    listing = head_listing(repo) if listing is None else listing
    mods = module_paths(listing)
    _read_blobs(repo, [listing[p] for p in mods.values()])
    out = {p: h for p, h in listing.items() if any(p.startswith(t + "/") for t in BOUND_TREES)}
    seen, cfg = set(), set()
    todo = sorted(m for m, p in mods.items() if p.startswith(ENTRY_TREES) and "/tests/" not in p)
    sure_all = set()
    while todo:
        m = todo.pop()
        if m in seen or not (m == "hnl" or m.startswith("hnl.")):
            continue
        seen.add(m)
        parts = m.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        p = mods.get(m)
        if p is None:
            continue
        out[p] = listing[p]
        key = (listing[p], m, p.endswith("__init__.py"))
        if key not in _SCAN:
            _SCAN[key] = scan_source(_TEXT[listing[p]], m, key[2])
        sure, maybe, c = _SCAN[key]
        sure_all.update(x for x in sure if x == "hnl" or x.startswith("hnl."))
        todo += sure + maybe
        cfg.update(c)
    for m in sorted(sure_all - set(mods)):                  # a module imported for certain, absent at HEAD
        out.setdefault(f"src/{m.replace('.', '/')}.py", None)
    for c in sorted(cfg):
        if c in listing:
            out[c] = listing[c]
            continue
        below = {p: h for p, h in listing.items() if p.startswith(c + "/")}
        if below:
            out.update(below)
        else:
            out.setdefault(c, None)
    return dict(sorted(out.items()))


def code_state(repo: str | None = None) -> dict:
    """{files: `bound_paths`, addendum: blob hash of the addendum at HEAD (None while it is not committed)}: what a
    grant binds (decision E1), from one `git ls-tree HEAD` (and the HEAD blobs of src/hnl, read once per process)."""
    repo = repo or REPO
    listing = head_listing(repo)
    return {"files": bound_paths(repo, listing), "addendum": listing.get(ADDENDUM)}


def digest(state: dict) -> str:
    """sha256 of the bound files' paths and blob hashes and of the addendum blob (for logs and records)."""
    text = "".join(f"{p} {h}\n" for p, h in sorted((state.get("files") or {}).items()))
    return hashlib.sha256((text + f"{ADDENDUM} {state.get('addendum')}\n").encode()).hexdigest()


def bound_modules(files) -> set[str]:
    """Module names of the bound python files under src (a package by its __init__.py), including those bound as
    absent (a module the closure imports by name that is not at HEAD)."""
    out = set()
    for p in files:
        if p.startswith("src/") and p.endswith(".py"):
            name = p[len("src/"):-len(".py")].replace("/", ".")
            out.add(name[:-len(".__init__")] if name.endswith(".__init__") else name)
    return out


def shadow_paths(files, repo: str | None = None) -> list[str]:
    """Untracked files under src (ignored ones included, __pycache__ excepted) that python could import in place of a
    bound module: a file named like a bound module with any import suffix (.py, .pyc or an extension module, which
    python tries before .py), or the __init__ of a package directory of that name (python prefers a package to a
    module file). A new module of any other name is not run by an overlay job (the closure is static)."""
    import importlib.machinery as M
    repo = repo or REPO
    mods = bound_modules(files)
    suffixes = sorted(M.all_suffixes(), key=len, reverse=True)
    bad = []
    for e in _git(repo, "ls-files", "--others", "-z", "--", "src").split(b"\0"):
        p = e.decode()
        if not p.startswith("src/") or "/__pycache__/" in p:
            continue
        suf = next((x for x in suffixes if p.endswith(x)), None)
        if suf is None:
            continue
        name = p[len("src/"):-len(suf)].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-len(".__init__")]
        if name in mods:
            bad.append(p)
    return bad


def worktree_changes(files, repo: str | None = None) -> list[str]:
    """Bound paths whose file in the working tree is not that of HEAD (a tracked change, staged or not, or an untracked
    file at a bound path), untracked python files under src/hnl/overlay and src/hnl/trackml, and untracked files that
    python could import in place of a bound module (`shadow_paths`): what a later process of a running job would run
    instead of the bound code."""
    repo = repo or REPO
    extra = sorted(p for p in files if not any(p.startswith(t + "/") for t in BOUND_TREES))
    raw = _git(repo, "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
               *BOUND_TREES, *extra)
    items, bad, i = raw.split(b"\0"), [], 0
    while i < len(items):
        e = items[i].decode()
        i += 1
        if len(e) < 4:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":
            i += 1                                          # the source path of a rename or copy
        if xy == "!!":
            continue
        if xy == "??" and not (path in files or (path.endswith(".py") and path.startswith(ENTRY_TREES))):
            continue
        bad.append(path)
    return sorted(set(bad) | set(shadow_paths(files, repo)))


def write(set_name: str, cond: str, kind: str, ids, directory: str, out: str | None = None,
          launch: str | None = None, repo: str | None = None) -> str:
    """Writes a new read-only grant into `directory` (never overwriting one) and returns its absolute path."""
    repo = repo or REPO
    if kind not in KINDS:
        raise ValueError(f"no kind {kind!r}")
    if cond not in CONDS:
        raise ValueError(f"no condition {cond!r}")
    ids = sorted({int(i) for i in parse_ids(ids)})
    if not ids:
        raise ValueError("a grant needs ids")
    if launch is not None and not TOKEN.match(launch):
        raise ValueError("a launch token is 32 hex digits")
    now = dt.datetime.now(dt.timezone.utc)
    job = os.environ.get("SLURM_JOB_ID") or None
    rec = {"set": set_name, "cond": cond, "kind": kind, "ids": ids, "ids_sha256": ids_sha256(ids),
           "code": code_state(repo), "commit": git_head(repo), "slurm_job_id": job,
           "out": os.path.abspath(out) if out else None, "created_utc": now.isoformat(timespec="seconds"),
           "launch": launch or None}
    os.makedirs(directory, exist_ok=True)
    name = f"{set_name}_{cond}_{kind}_{now:%Y%m%dT%H%M%SZ}_{job or 'nojob'}_{os.getpid()}_{secrets.token_hex(4)}.json"
    path = os.path.abspath(os.path.join(directory, name))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1)
    return path


def _under(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip("/") + "/")


def launched_sets(g: dict, ids) -> list[str]:
    """The sets whose launch record must accept this grant for these ids: htest for any H-test id; hval for H-val ids
    under a grant of kind eval, or of kind model in C-1GeV or C-hs (the runs that follow `guard_h.sh --launch hval`)."""
    ks = {set_index(i) for i in ids if is_overlay(i)}
    out = []
    if 0 in ks and (g["kind"] == "eval" or (g["kind"] == "model" and g["cond"] in ("1gev", "hs"))):
        out.append("hval")
    if 2 in ks:
        out.append("htest")
    return out


def launch_problems(set_name: str, token: str, repo: str | None = None) -> list[str]:
    """hnl.overlay.blind.check_launch for the grant's token: the token of RUNS/launch/<set>.json, every pinned file
    unchanged."""
    from hnl.overlay import blind as B                   # blind imports this module: imported when needed
    repo = repo or REPO
    runs = B.runs_root(repo)
    return [f"refused by the launch record {B.launch_path(set_name, runs)}: {p}"
            for p in B.check_launch(set_name, token, runs, repo)]


def problems(path: str, ids: list[int], kinds: tuple, cond: str | None, state: dict | None = None,
             out: str | None = None, repo: str | None = None) -> list[str]:
    """Why the grant at `path` does not cover the request (empty if it does)."""
    try:
        with open(path) as fh:
            g = json.load(fh)
    except (OSError, ValueError) as e:
        return [f"unreadable grant ({e})"]
    if not isinstance(g, dict) or any(k not in g for k in FIELDS):
        return [f"not a grant (fields {sorted(g) if isinstance(g, dict) else type(g).__name__})"]
    bad = []
    if g["kind"] not in kinds:
        bad.append(f"kind {g['kind']}, not {' or '.join(kinds)}")
    if cond is not None and g["cond"] != cond:
        bad.append(f"condition {g['cond']}, not {cond}")
    try:
        have = sorted(int(i) for i in g["ids"])
    except (TypeError, ValueError):
        return bad + ["ids are not a list of integers"]
    if ids_sha256(have) != g["ids_sha256"]:
        bad.append("ids_sha256 is not that of its ids")
    missing = sorted(set(ids) - set(have))
    if missing:
        bad.append(f"{len(missing)} requested ids not granted (first {missing[:3]})")
    state = state if state is not None else code_state(repo)
    code = g["code"] if isinstance(g["code"], dict) else {}
    files = code.get("files")
    if not isinstance(files, dict):
        bad.append("the grant binds no code files (not a grant of the current code binding): the code at HEAD changed "
                   "since the grant")
    else:
        now = state.get("files") or {}
        changed = sorted(p for p in set(files) | set(now) if files.get(p, "absent") != now.get(p, "absent"))
        if changed:
            bad.append(f"{len(changed)} bound code file(s) at HEAD changed since the grant: "
                       f"{', '.join(changed[:4])}{' ...' if len(changed) > 4 else ''}")
    if code.get("addendum", "absent") != state.get("addendum"):
        bad.append(f"the addendum {ADDENDUM} at HEAD changed since the grant ({str(code.get('addendum'))[:12]} -> "
                   f"{str(state.get('addendum'))[:12]})")
    dirty = state.get("worktree") or []
    if dirty:
        bad.append(f"uncommitted change to {len(dirty)} bound code file(s) in the working tree: "
                   f"{', '.join(dirty[:4])}{' ...' if len(dirty) > 4 else ''}")
    job = os.environ.get("SLURM_JOB_ID")
    if job and str(g["slurm_job_id"]) != job:
        bad.append(f"granted to Slurm job {g['slurm_job_id']}, this is job {job}")
    elif not job and g["slurm_job_id"]:
        bad.append(f"granted to Slurm job {g['slurm_job_id']}, and this process runs outside Slurm (SLURM_JOB_ID "
                   "is not set)")
    if out is not None and g["out"] and not _under(out, g["out"]):
        bad.append(f"writes {out}, outside the grant's out {g['out']}")
    for s in launched_sets(g, ids):
        if not (isinstance(g["launch"], str) and TOKEN.match(g["launch"])):
            bad.append(f"{'H-test' if s == 'htest' else 'H-val'} ids need a grant made under the launch token of "
                       f"`guard_h.sh --launch {s}`")
        elif not bad:                                     # (hashing the pins is skipped for a grant refused already)
            bad += launch_problems(s, g["launch"], repo)
    return bad


def check(ids, kind, cond: str | None = None, out: str | None = None) -> None:
    """None if no requested id is an overlay id or a grant of HNL_H_GRANT covers the request; PermissionError
    otherwise (module docstring). kind: one kind or a tuple of the kinds the caller accepts; out: a path the caller
    writes (it must lie under the grant's out, when the grant names one)."""
    ids = sorted({int(i) for i in parse_ids(ids)})
    over = [i for i in ids if is_overlay(i)]
    if not over:
        return
    kinds = (kind,) if isinstance(kind, str) else tuple(kind)
    what = f"{len(over)} overlay ids (first {over[0]}), kind {' or '.join(kinds)}" + (f", condition {cond}" if cond
                                                                                         else "")
    spec = os.environ.get(ENV, "")
    if not spec:
        raise PermissionError(f"{what}: no grant ({ENV} is not set). Jobs on overlay events run only after "
                              "`G=$(bash slurm/overlay/guard_h.sh ... --grant) || exit 1; export \"$G\"`")
    state = code_state()
    state["worktree"] = worktree_changes(state["files"])
    why = []
    for path in [p for p in spec.split(os.pathsep) if p]:
        bad = problems(path, ids, kinds, cond, state, out)
        if not bad:
            with open(path) as fh:
                g = json.load(fh)
            print(f"hnl.overlay.grant: {len(ids)} ids under {path} (set {g['set']}, kind {g['kind']}, condition "
                  f"{g['cond']}, bound code {digest(state)[:12]} ({len(state['files'])} files), job "
                  f"{g['slurm_job_id']}, launch {'yes' if g['launch'] else 'no'})", file=sys.stderr, flush=True)
            return
        why.append(f"{path}: {'; '.join(bad)}")
    raise PermissionError(f"{what}: no grant of {ENV} covers the request: " + " | ".join(why))


def check_tables(paths, kind, cond: str | None = None, out: str | None = None) -> None:
    """`check` on the events of stored tables (the `event` column of parquet files, or of <dir>/per_event.parquet for
    a grid directory); paths that do not exist are skipped."""
    import pyarrow.parquet as pq
    ev = set()
    for p in paths:
        if not p:
            continue
        f = os.path.join(p, "per_event.parquet") if os.path.isdir(p) else p
        if os.path.exists(f):
            ev |= {int(e) for e in pq.read_table(f, columns=["event"]).column("event").to_pylist()}
    check(sorted(ev), kind, cond, out)


def check_store(paths, kind, cond: str | None = None, out: str | None = None) -> None:
    """`check` on the events of event-store files (hnl.trackml.classifier.save_event_graph names them
    event<eid>.pyg[.gz]), from the file names alone (no file is read)."""
    from hnl.trackml.events import event_from_eid
    check([event_from_eid(int(os.path.basename(p)[5:].split(".")[0])) for p in paths], kind, cond, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--set", required=True)
    w.add_argument("--cond", required=True, choices=CONDS)
    w.add_argument("--kind", required=True, choices=KINDS)
    w.add_argument("--ids", required=True)
    w.add_argument("--dir", required=True)
    w.add_argument("--out", default=None)
    w.add_argument("--launch", default=None)
    w.add_argument("--repo", default=REPO)
    c = sub.add_parser("check")
    c.add_argument("--ids", required=True)
    c.add_argument("--kind", required=True)
    c.add_argument("--cond", default=None)
    c.add_argument("--out", default=None)
    st = sub.add_parser("state")
    st.add_argument("--repo", default=REPO)
    st.add_argument("--json", action="store_true", help="print every bound path with its blob hash")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        print(write(a.set, a.cond, a.kind, a.ids, a.dir, a.out, a.launch or None, a.repo))
    elif a.cmd == "state":
        s = code_state(a.repo)
        dirty = worktree_changes(s["files"], a.repo)
        if a.json:
            print(json.dumps({**s, "digest": digest(s), "worktree": dirty}, indent=1))
        else:
            absent = sum(h is None for h in s["files"].values())
            wt = "clean" if not dirty else "changed: " + " ".join(dirty[:5])
            print(f"bound code {digest(s)[:12]}: {len(s['files'])} files ({absent} absent at HEAD), addendum "
                  f"{str(s['addendum'])[:12]}, working tree {wt}")
    else:
        try:
            check(a.ids, tuple(a.kind.split(",")), a.cond, a.out)
        except PermissionError as e:
            print(f"grant: {e}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
