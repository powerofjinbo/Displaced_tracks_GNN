"""Model stages of the frozen chain on overlay ids (hnl.overlay.chain, slurm/overlay/chain*.sbatch, grid.sbatch;
addendum H section 10). Request rules, grants, freeze verify, the C-hs and C-1GeV filters at the model stages and the
provenance of the scores, on synthetic events with stand-ins for the networks (no checkpoint is loaded and no model
output of an overlay event is read). The grant module is replaced by a recording stub."""

import glob
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import blind as B
from hnl.overlay import chain as C
from hnl.overlay import conditions as OC
from hnl.trackml import events as E

REPO = B.REPO
needs_split = pytest.mark.skipif(not os.path.exists(B.SPLIT_H), reason="needs configs/trackml/split_h_v1.json")


def pid_of(vertex, k):
    return (int(vertex) << 52) | (int(k) << 36)


PARTS = [(pid_of(1, 1), 5.0), (pid_of(1, 2) | (1 << 24), 0.6), (pid_of(2, 1), 3.0), (pid_of(37, 4), 0.4),
         (pid_of(1001, 1), 4.0), (pid_of(1009, 2), 0.8)]


def synthetic(event: int, particles: bool = True, closure: bool = False) -> E.Event:
    """Hard scatter (vertex 1: a primary at 5 GeV and a 0.6 GeV secondary), pileup (vertex 2 at 3 GeV, vertex 37 at
    0.4 GeV), gun (1001 at 4 GeV, 1009 at 0.8 GeV), noise; six hits per particle along a line."""
    parts = PARTS + ([(pid_of(1025, 3) | (1 << 62), 2.0)] if closure else [])
    rng = np.random.default_rng(event % 1000)
    pid, rank, xyz = [], [], []
    for i, (p, _) in enumerate(parts):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for j in range(6):
            pid.append(p)
            rank.append(j)
            xyz.append(d * (40.0 + 60.0 * j) + rng.normal(0, 0.5, 3))
    for _ in range(8):
        pid.append(0)
        rank.append(255)
        xyz.append(rng.normal(0, 300, 3))
    xyz = np.asarray(xyz, np.float32)
    n = len(pid)
    s = pd.DataFrame({"row": np.arange(n), "pid": pid, "rank": rank})
    p = pd.DataFrame({"particle_id": np.array([q for q, _ in parts], np.int64),
                      "pt": np.array([t for _, t in parts], np.float32)})
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return E.Event(event=int(event), x=x, y=y, z=z, r=np.hypot(x, y), phi=np.arctan2(y, x),
                   volume=np.full(n, 8), layer=np.asarray(rank, np.int64) % 7, module=np.arange(n),
                   pid=np.asarray(pid, np.int64), rank=np.asarray(rank, np.int64), ncells=np.ones(n),
                   charge=np.ones(n), weight=np.zeros(n), true_edges=E._true_edges(s),
                   particles=p if particles else None)


class Recorder:
    def __init__(self, raise_with=None):
        self.calls = []
        self.outs = []
        self.raise_with = raise_with

    def __call__(self, ids, kind, cond=None, out=None, **kw):
        self.calls.append((sorted(int(i) for i in ids), kind, cond))
        self.outs.append(out)
        if self.raise_with:
            raise PermissionError(self.raise_with)


@pytest.fixture
def grant(monkeypatch):
    """hnl.overlay.grant.check replaced by a recorder (every caller looks it up on the module)."""
    from hnl.overlay import grant as G
    rec = Recorder()
    monkeypatch.setattr(G, "check", rec)
    monkeypatch.delenv("HNL_H_GRANT", raising=False)
    return rec


class StubEmbedder:
    seen: list = []

    def __init__(self, ckpt=None, device="cpu", **kw):
        self.ckpt = ckpt

    def __call__(self, ev):
        import torch
        StubEmbedder.seen.append((ev.event, ev.n, ev.pid.copy(), ev.condition))
        x = np.stack([ev.x, ev.y, ev.z], axis=1).astype(np.float32) / 1000.0
        return torch.from_numpy(x)


class StubSupport:
    def __init__(self, model_path, backend, device):
        self.backend = "torch"

    def wide(self, emb, r_wide):
        return E.radius_graph(emb, r_wide, None), 0.0

    def select(self, ev, cand, ops, fmap, max_triplets):
        return {op: cand[:, :0] for op in ops}, {"n_trip_cand": 0, "n_chunks": 0}


class StubModel:
    hparams = {"node_scales": [1000.0, 3.14, 1000.0]}


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """Synthetic overlay tables, an embedding of the hit positions, a learned support that adds nothing, a classifier
    returning zero logits, the backends recorded as on the GPU, and frozen inputs in a scratch selection."""
    import torch
    from hnl.trackml import chunked as CH
    from hnl.trackml import convert
    from hnl.trackml import graphs as GR
    from hnl.trackml import infer as INF

    closure = set()

    def iter_events(events, chunk=25, root=None, particles=True):
        for e in events:
            yield synthetic(int(e), particles, closure=int(e) in closure)

    StubEmbedder.seen = []
    monkeypatch.setattr(E, "iter_events", iter_events)
    monkeypatch.setattr(E, "Embedder", StubEmbedder)
    monkeypatch.setattr(GR, "Support", StubSupport)
    monkeypatch.setattr(GR, "_backend", lambda device: "frnn_cuda")
    monkeypatch.setattr(convert, "storage_check", lambda nbytes: {"projected": int(nbytes)})
    monkeypatch.setattr(INF, "load_model", lambda ckpt, device: StubModel())
    monkeypatch.setattr(CH, "forward", lambda model, d, *a, **k: torch.zeros(d.edge_index.shape[1]))
    gc = tmp_path / "graph_choice.json"
    gc.write_text(json.dumps({"ml_ckpt": "stub-ml", "ltf_model": "stub-ltf", "field_map": None, "r_star": 0.12,
                              "knn": 40, "r_wide": 0.15, "tau": 0.05, "k": 3}))
    ckpt = tmp_path / "classifier.ckpt"
    ckpt.write_bytes(b"stand-in classifier")
    fi = {"graph_choice": str(gc), "classifier": str(ckpt),
          "classifier_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(), "run": str(tmp_path / "run")}
    monkeypatch.setattr(C, "preflight", lambda selection=None: fi)
    monkeypatch.setattr(C, "frozen_inputs", lambda selection=None: fi)
    monkeypatch.setattr(C, "_score_threads", lambda: C.SCORE_CPUS)        # the threads of chain_score.sbatch
    return {"fi": fi, "closure": closure, "tmp": tmp_path}


# ------------------------------------------------------------------------------------------------ layout and requests
@needs_split
def test_paths_and_blocks(tmp_path):
    runs = str(tmp_path / "runs")
    assert C.paths("hval", "full", runs)["out"] == B.sealed_dir("hval", "full", runs)
    assert C.paths("hval", "full", runs)["scores"] == os.path.join(B.sealed_dir("hval", "full", runs), "scores")
    assert C.paths("hval", "full", runs)["grid"] is None
    assert C.paths("closure", "1gev", runs)["grid"] == os.path.join(runs, "grid", "closure_1gev")
    assert C.paths("htest", "hs", runs)["out"] == os.path.join(runs, "model", "htest_hs")
    with pytest.raises(PermissionError):
        C.paths("closure", "hs", runs)
    with pytest.raises(ValueError):
        C.paths("pilot", "full", runs)
    blocks = [C.block_ids("hval", k, 3) for k in range(3)]
    assert [len(b) for b in blocks] == [50, 50, 50] and sum(blocks, []) == B.set_ids("hval")
    with pytest.raises(ValueError):
        C.block_ids("hval", 0, 2)                     # 75 events per block: more than one scored-edge file
    with pytest.raises(ValueError):
        C.block_ids("hval", 3, 3)
    assert C.block_ids("pilot", 0, 1) == B.set_ids("pilot")


@needs_split
def test_request_rules(tmp_path, monkeypatch):
    runs = str(tmp_path / "runs")
    cl, hv = B.set_ids("closure")[:50], B.set_ids("hval")[:50]
    # closure: C-full and C-1GeV only, into the default directory
    with pytest.raises(PermissionError, match="C-full and C-1GeV only"):
        C.check_request("closure", "hs", C.paths("closure", "full", runs)["out"], cl, runs=runs)
    assert C.check_request("closure", "1gev", C.paths("closure", "1gev", runs)["out"], cl, runs=runs) == (1, None)
    with pytest.raises(PermissionError, match="go to"):
        C.check_request("closure", "full", str(tmp_path / "elsewhere"), cl, runs=runs)
    with pytest.raises(PermissionError, match="not in set closure"):
        C.check_request("closure", "full", C.paths("closure", "full", runs)["out"], B.set_ids("pilot"), runs=runs)
    with pytest.raises(PermissionError):                                     # a production set reads production tables
        C.check_request("closure", "full", C.paths("closure", "full", runs)["out"], cl, runs=runs,
                        root=str(tmp_path / "root"))
    # H-val before the overlay_hash commit: C-full only, into the sealed directory only
    monkeypatch.setattr(B, "unblinded", lambda repo=REPO, record=None: False)
    sealed = B.sealed_dir("hval", "full", runs)
    assert C.check_request("hval", "full", sealed, hv, runs=runs) == (0, None)
    with pytest.raises(PermissionError, match="sealed directory"):
        C.check_request("hval", "full", os.path.join(runs, "model", "hval_full"), hv, runs=runs)
    for c in ("1gev", "hs"):
        with pytest.raises(PermissionError, match="C-full only"):
            C.check_request("hval", c, C.paths("hval", c, runs)["out"], hv, runs=runs)
    os.makedirs(os.path.join(sealed, "counts"))
    with pytest.raises(PermissionError, match="blinded evaluation"):
        C.check_request("hval", "full", sealed, hv, runs=runs)
    os.rmdir(os.path.join(sealed, "counts"))
    open(os.path.join(sealed, B.MANIFEST), "w").write("")
    with pytest.raises(PermissionError, match="sealed"):
        C.check_request("hval", "full", sealed, hv, runs=runs)
    os.remove(os.path.join(sealed, B.MANIFEST))
    # after the commit: C-1GeV and C-hs, never C-full again
    monkeypatch.setattr(B, "unblinded", lambda repo=REPO, record=None: True)
    with pytest.raises(PermissionError, match="not repeated"):
        C.check_request("hval", "full", sealed, hv, runs=runs)
    assert C.check_request("hval", "hs", C.paths("hval", "hs", runs)["out"], hv, runs=runs) == (0, None)
    # pilot: plumbing outputs outside the TrackML data root and the runs root, one block
    pl = B.set_ids("pilot")
    assert C.check_request("pilot", "hs", str(tmp_path / "plumb" / "pilot_hs"), pl, runs=runs) == (9, None)
    for bad in (os.path.join(B.EXT, "scratch_plumb"), os.path.join(runs, "plumb")):
        with pytest.raises(PermissionError, match="outside the TrackML data root"):
            C.check_request("pilot", "full", bad, pl, runs=runs)
    with pytest.raises(PermissionError, match="one block"):
        C.check_request("pilot", "full", str(tmp_path / "p"), pl, n_blocks=2, runs=runs)
    with pytest.raises(PermissionError):
        C.check_request("pilot", "full", str(tmp_path / "p"), pl[:2] + B.set_ids("closure")[:1], runs=runs)
    # H-test: no id passes before H-T1 writes its list
    try:
        B.set_ids("htest")
    except ValueError:
        with pytest.raises(PermissionError):
            C.check_request("htest", "full", C.paths("htest", "full", runs)["out"], [52_021_289], runs=runs)


def test_kinds():
    assert C.kind_of("pilot") == "plumb"
    assert {C.kind_of(s) for s in ("hval", "closure", "htest")} == {"model"}


# ------------------------------------------------------------------------------------------------ freeze verify, grants
def test_freeze_verify_failure_stops_the_job(tmp_path, monkeypatch, grant):
    from hnl.trackml import freeze
    calls = []
    monkeypatch.setattr(freeze, "verify", lambda path, run: calls.append(run) or
                        ["classifier_checkpoint: sha256 of x differs from the frozen value"])
    out = tmp_path / "plumb" / "pilot_full"
    with pytest.raises(SystemExit, match="freeze verify"):
        C.run("pilot", "full", out=str(out), device="cpu")
    assert calls and not out.exists() and grant.calls == []                 # stopped before the grant and any write
    with pytest.raises(SystemExit):
        C.main(["preflight"])
    monkeypatch.setattr(freeze, "verify", lambda path, run: [])
    monkeypatch.setenv("HNL_TML_CONFIG", str(tmp_path / "other.yaml"))
    with pytest.raises(SystemExit, match="HNL_TML_CONFIG"):
        C.preflight()


SCRIPTS = ("chain.sbatch", "chain_score.sbatch", "grid.sbatch")


@pytest.mark.parametrize("name", SCRIPTS)
def test_job_scripts_verify_then_guard_then_run(name):
    """Each job script stops on an error, runs freeze verify (chain preflight) first and stops if it fails, then the
    guard with --grant, exports the grant, and only then a stage; no script lets HNL_TML_CONFIG / HNL_TML_SPLIT
    through."""
    text = open(os.path.join(REPO, "slurm", "overlay", name)).read()
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    idx = lambda pat: next(i for i, ln in enumerate(body) if re.search(pat, ln))
    assert "set -euo pipefail" in body and "unset HNL_TML_CONFIG HNL_TML_SPLIT" in body
    pre, guard, export = idx(r"hnl\.overlay\.chain preflight"), idx(r"guard_h\.sh .*--grant\) \|\| exit 1"), \
        idx(r'^export "\$G"$')
    assert "|| {" in body[pre] and "exit 1" in body[pre]
    stage = idx(r"hnl\.overlay\.(chain (run|dry)|tracks grid)")
    assert pre < guard < export < stage


def test_grant_is_checked_with_the_kind_of_the_set(stubs, grant):
    out = stubs["tmp"] / "plumb" / "pilot_hs"
    if not os.path.exists(B.SPLIT_H):
        pytest.skip("needs split_h_v1.json")
    grant.raise_with = "no grant"
    with pytest.raises(PermissionError, match="no grant"):
        C.run("pilot", "hs", out=str(out), device="cpu")
    assert grant.calls == [(B.set_ids("pilot"), "plumb", "hs")] and not out.exists()


def test_frozen_clis_keep_their_condition_choices():
    from hnl.trackml import graphs as GR
    from hnl.trackml import infer as INF
    for argv in (["additions", "--events", "1", "--graph-choice", "g", "--out", "o", "--condition", "hs"],
                 ["stage", "--events", "1", "--graph-choice", "g", "--additions", "a", "--dest", "d", "--condition",
                  "hs"]):
        with pytest.raises(SystemExit):
            GR.main(argv)
    with pytest.raises(SystemExit):
        INF.main(["score", "--store", "s", "--out", "o", "--condition", "hs"])


# ------------------------------------------------------------------------------------------------ model stages
@needs_split
@pytest.mark.parametrize("cond", ["full", "1gev", "hs"])
def test_model_stages_apply_the_condition_before_graph_building(stubs, grant, cond):
    """additions -> store -> score on synthetic events: the network sees only the hits of the condition (C-hs: vertex
    fields 1 and 1001-1010; C-1GeV: particles with pT > 1 GeV), the store's rows and particle ids are those of the
    condition, and the scores carry the frozen classifier's sha256, the condition and the whole-event chunked
    forward."""
    fi, out = stubs["fi"], str(stubs["tmp"] / f"out_{cond}")
    ids = [59_021_289, 59_021_332]
    m = C.Meter()
    C.stage_additions(ids, cond, out, fi, device="cpu", meter=m)
    assert [r["event"] for r in m.rows] == ids
    side = json.load(open(glob_one(os.path.join(out, "additions", "*.parquet.json"))))
    assert side["condition"] == cond and sorted(int(e) for e in side["events"]) == ids
    full = synthetic(ids[0])
    want = C.expected_rows(full, cond)
    seen = [s for s in StubEmbedder.seen if s[0] == ids[0]]
    assert seen and all(n == len(want) for _, n, _, _ in seen)               # the network saw the filtered event
    C.stage_store(ids, cond, out, "b000of001", fi, device="cpu", meter=C.Meter())
    from hnl.trackml import classifier as CL
    store = C.stored_events(C.store_dir(out, "b000of001"))
    assert sorted(store) == ids
    g = CL.load_event_graph(store[ids[0]])
    assert np.array_equal(g["rows"], want) and np.array_equal(g["pid"], full.pid[want])
    v = OC.vertex_field(g["pid"])
    if cond == "hs":
        assert set(np.unique(v)) == {1, 1001, 1009} and (g["pid"] != 0).all()
        assert len(want) == 4 * 6
    elif cond == "1gev":
        assert set(np.unique(g["pid"])) == {pid_of(1, 1), pid_of(2, 1), pid_of(1001, 1)}
    else:
        assert len(g["rows"]) == full.n
    C.stage_score(ids, cond, out, "b000of001", fi, meter=C.Meter())
    sides = C.score_sidecars(os.path.join(out, "scores"))
    (s,) = sides.values()
    assert s["condition"] == cond and s["ckpt_sha256"] == fi["classifier_sha256"] and s["forward"] == "chunked"
    assert s["layout"]["n_sectors"] == 1 and s["device"] == "cpu"
    assert C.check_score_sidecars(os.path.join(out, "scores"), ids, cond, fi) == []
    assert C.check_score_sidecars(os.path.join(out, "scores"), ids, cond, {**fi, "classifier_sha256": "0" * 64})
    other = "full" if cond != "full" else "1gev"
    assert C.check_score_sidecars(os.path.join(out, "scores"), ids, other, fi)
    with pytest.raises(RuntimeError, match="twice"):
        C.stage_score(ids, cond, out, "b000of001", fi)                        # no event is scored twice
    # every stage asked for the grant of its condition (the frozen functions call hnl.overlay.grant.check), each with
    # the directory it writes, under OUT (the grant's out)
    assert {c[2] for c in grant.calls} == {cond} and all(set(c[0]) <= set(ids) for c in grant.calls)
    assert grant.outs and all(o is not None and B._under(o, out) for o in grant.outs)
    # structure checks (ids of the pilot set not needed here): counts agree, the rows are the condition's
    res = C.check_outputs("pilot", cond, out, fi=fi, ids=ids)
    assert res["pass"], res["problems"]
    assert all(r["score_rows"] == r["edges"] for r in res["events"].values())
    wrong = "hs" if cond != "hs" else "1gev"
    assert not C.check_outputs("pilot", wrong, out, fi=fi, ids=ids)["pass"]


def copy_plain(src, dst):
    """Copy of a directory tree with fresh permissions (the source is sealed, read-only)."""
    for d, _, files in os.walk(src):
        os.makedirs(os.path.join(dst, os.path.relpath(d, src)), exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(d, f), os.path.join(dst, os.path.relpath(d, src), f))


def glob_one(pattern):
    import glob
    (p,) = glob.glob(pattern)
    return p


def test_hs_needs_the_overlay_condition_and_refuses_closure_events(stubs, grant):
    """Called without hnl.overlay.chain, the frozen stage has no C-hs (its condition function refuses it); a closure
    event (vertex fields 1024-2024) is refused in C-hs at the model stage."""
    from hnl.trackml import config as CFG
    from hnl.trackml import graphs as GR
    import argparse
    fi, out = stubs["fi"], str(stubs["tmp"] / "frozen_hs")
    a = argparse.Namespace(events="59021289", graph_choice=fi["graph_choice"], out=out, per_file=50, condition="hs",
                           support_backend=None, device="cpu", allow_test=False)
    with pytest.raises(ValueError):
        GR.cmd_additions(a, CFG.load(CFG.DEFAULT))
    stubs["closure"].add(51_021_289)
    with pytest.raises(ValueError, match="closure"):
        C.stage_additions([51_021_289], "hs", str(stubs["tmp"] / "cl_hs"), fi, device="cpu")
    # the patches end with the stage
    assert E.with_condition.__module__ == "hnl.trackml.events"


@needs_split
def test_pilot_plumbing_run_seal_and_record(stubs, grant):
    """The pilot path end to end with the stand-ins: every stage, the done marker, the seal right after the scores, the
    structure checks written outside the sealed directory, no further write, and the plumbing record."""
    base = stubs["tmp"] / "plumb"
    sel = stubs["tmp"] / "selection.json"
    sel.write_text(json.dumps({"run": str(stubs["tmp"] / "run"), "inputs": {}}))
    try:
        for cond in ("full", "1gev", "hs"):
            out = str(base / f"pilot_{cond}")
            r = C.run("pilot", cond, out=out, device="cpu")
            assert r["done"] and set(r["stages"]) == set(C.STAGES)
            assert C.run("pilot", cond, out=out, device="cpu")["skipped"]          # a done block is not repeated
            assert os.path.isdir(C.store_dir(out, "b000of001"))                   # kept on the pilot
            for st in C.STAGES:
                rec = json.load(open(os.path.join(out, "chain", f"{st}_b000of001.json")))
                assert rec["kind"] == "plumb" and [e["event"] for e in rec["per_event"]] == B.set_ids("pilot")
                assert all(e["seconds"] >= 0 and e["peak_rss_gb"] for e in rec["per_event"])
            with pytest.raises(PermissionError):
                C.finish("pilot", cond, out, os.path.join(out, "check.json"))      # the record stays outside
            res = C.finish("pilot", cond, out, str(base / "checks" / f"check_{cond}.json"))
            assert res["pass"], res["problems"]
            assert B.is_sealed(out) and B.verify(out, res["sealed"]["manifest_sha256"]) == []
            assert json.load(open(os.path.join(out, B.SEALED_JSON)))["rule"] == B.SEAL_RULES["plumb"]   # D6 rule
            with pytest.raises(PermissionError, match="sealed"):
                C.run("pilot", cond, out=out, device="cpu")
        kinds = {k if isinstance(k, str) else tuple(k) for _, k, _ in grant.calls}
        assert kinds == {"plumb", ("model", "plumb")}                               # chain and the frozen functions
        assert all(o is not None and B._under(o, str(base)) for o in grant.outs)   # every write under its OUT
        # without a CPU score job the CPU placement takes the GPU job's score time, flagged as not measured there
        rec = C.plumb_record(str(base), str(base / "plumb_check_v1.json"), selection=str(sel))
        assert not any(r["scores_on_cpu_nodes"]["score_time_measured_on_cpu_node"]
                       for r in rec["extrapolation"]["rows"])
        # the CPU score job of decision E2: the score stage again on a copy of the GPU job's additions and store
        for cond in ("full", "1gev"):
            src, cpu = base / f"pilot_{cond}", base / f"pilot_{cond}_cpuscore"
            for d in ("additions", "store"):
                copy_plain(src / d, cpu / d)
            r = C.run("pilot", cond, out=str(cpu), stages=["score"], device="cpu")
            assert r["done"] and r["stages"] == ["score"]
            res = C.finish("pilot", cond, str(cpu), str(base / "checks" / f"check_{cond}_cpuscore.json"))
            assert res["pass"], res["problems"]
        rec = C.plumb_record(str(base), str(base / "plumb_check_v1.json"), selection=str(sel))
        assert all(rec["conditions"][c]["checks_pass"] for c in ("full", "1gev", "hs"))
        assert rec["conditions"]["full"]["cpu_score_job"]["checks_pass"]
        assert all(rec["conditions"][c]["seal_now"]["verified_again"] and rec["conditions"][c]["seal_now"]["sealed"]
                   and not rec["conditions"][c]["seal_now"]["writable_paths"] for c in ("full", "1gev", "hs"))
        assert rec["conditions"]["1gev"]["cpu_score_job"]["seal_now"]["verified_again"]
        assert "cpu_score_job" not in rec["conditions"]["hs"]
        rows = rec["extrapolation"]["rows"]
        assert {(r["set"], r["condition"]) for r in rows} == {("hval", "full"), ("hval", "1gev"), ("hval", "hs"),
                                                             ("closure", "full"), ("closure", "1gev"),
                                                             ("htest", "full"), ("htest", "1gev"), ("htest", "hs")}
        assert {r["condition"]: r["scores_on_cpu_nodes"]["score_time_measured_on_cpu_node"] for r in rows} == \
            {"full": True, "1gev": True, "hs": False}
        # where no CPU score job ran, a chunked-vs-plain check record gives the CPU-node time (native host events)
        chk = base / "chunked_check.json"
        chk.write_text(json.dumps({"conditions": {"hs": {"events": [
            {"event": 21289, "seconds_chunked": 5.0, "threads": 6, "host": "cpu-node", "slurm_job": "1"},
            {"event": 21332, "seconds_chunked": 7.0, "threads": 6, "host": "cpu-node", "slurm_job": "2"}]},
            "full": {"events": [{"event": 21289, "seconds_chunked": 999.0, "threads": 6, "host": "cpu-node"}]}}}))
        rec2 = C.plumb_record(str(base), str(base / "plumb_check_v2.json"), selection=str(sel), cpu_timing=str(chk))
        hs = next(r for r in rec2["extrapolation"]["rows"] if r["condition"] == "hs")["scores_on_cpu_nodes"]
        assert hs["score_time_measured_on_cpu_node"] and hs["score_cpus_per_job"] == 6
        assert rec2["cost_inputs"]["hs"]["score_s_per_event_cpu_node"] == 6.0
        # in C-full both exist: the one measured at SCORE_CPUS threads is used (the CPU score job when it ran with
        # SCORE_CPUS threads, else the check), and the other is kept beside it
        job_threads = rec2["conditions"]["full"]["cpu_score_job"]["node"]["cpus"]
        full = rec2["cost_inputs"]["full"]
        assert (full["score_s_per_event_cpu_node"] == 999.0) == (job_threads != C.SCORE_CPUS)
        assert full["score_cpu_node_other"]["score_s_per_event_cpu_node"] != full["score_s_per_event_cpu_node"]
        assert rec["extrapolation"]["placement_chosen"] == C.PLACEMENT == "scores_on_cpu_nodes"   # decision E2
        assert set(rec["extrapolation"]["total"]) == {"scores_on_cpu_nodes", "scores_in_gpu_job", "grid_cpu_hours"}
        text = json.dumps(rec)
        for word in ("efficien", "fake_rate", "matched", "trackml_score", "dm"):
            assert f'"{word}' not in text                                          # no evaluation key of any kind
    finally:
        for d in glob.glob(str(base / "pilot_*")):
            B.unseal_permissions(d)


@needs_split
def test_production_block_drops_its_store_after_checked_scores(stubs, grant, tmp_path):
    """Decision E2: a production block runs additions and store in one job and the scores in a CPU job of their own;
    one job holding the scores with another stage is refused before anything runs."""
    runs = str(tmp_path / "runs")
    out = C.paths("closure", "1gev", runs)["out"]
    for stages in (C.STAGES, ("store", "score"), ("additions", "score")):
        with pytest.raises(PermissionError, match="decision E2"):
            C.run("closure", "1gev", block=1, n_blocks=3, stages=stages, device="cpu", runs=runs)
    assert not os.path.exists(out) and grant.calls == []
    r = C.run("closure", "1gev", block=1, n_blocks=3, stages=("additions", "store"), device="cpu", runs=runs)
    assert not r["done"] and os.path.isdir(C.store_dir(out, "b001of003"))
    r = C.run("closure", "1gev", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs)
    assert r["done"] and not os.path.exists(C.store_dir(out, "b001of003"))
    ids = C.block_ids("closure", 1, 3)
    assert sorted(C.scored_events(os.path.join(out, "scores"))) == ids
    assert {k for _, k, _ in grant.calls if isinstance(k, str)} == {"model"}
    assert C.run("closure", "1gev", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs)["skipped"]
    with pytest.raises(PermissionError):
        C.run("closure", "hs", block=1, n_blocks=3, stages=("additions", "store"), device="cpu", runs=runs)


@needs_split
def test_score_jobs_cut_each_block_into_parts_that_fit_the_time_limit():
    """Decision E2, the job layout: the GPU array makes blocks of up to 50 events; each block is scored by score_parts
    CPU jobs of at most SCORE_EVENTS_PER_JOB events (5 in C-full: 5 x 472 s plus the overhead, about 40 min, and
    under 1 h even at the slowest timed event, 632 s), the block's ids cut into equal slices in order."""
    assert C.SCORE_CPUS == 6 and C.SCORE_EVENTS_PER_JOB == {"full": 5, "1gev": 50, "hs": 50}
    assert 5 * 632.0 + 30 < C.HOUR_S and 5 * 472.46 + 29.64 < 0.75 * C.HOUR_S
    for set_name, cond, nb, parts in (("hval", "full", 3, 10), ("hval", "1gev", 3, 1), ("hval", "hs", 3, 1),
                                      ("closure", "full", 3, 10), ("closure", "full", 4, 8), ("pilot", "full", 1, 1)):
        assert C.score_parts(set_name, cond, nb) == parts
        for k in range(nb):
            got = [C.score_ids(set_name, cond, k, nb, j) for j in range(parts)]
            assert sum(got, []) == C.block_ids(set_name, k, nb)
            assert max(len(g) for g in got) <= C.SCORE_EVENTS_PER_JOB[cond]
        with pytest.raises(ValueError):
            C.score_ids(set_name, cond, 0, nb, parts)
    assert C.part_name(2, 3, 9, 10) == "b002of003p09of10" and C.part_name(2, 3, 0, 1) == "b002of003"
    lay = C.layout("hval", "full")
    assert (lay["n_blocks"], lay["score_parts"], lay["score_jobs"]) == (3, 10, 30)
    assert lay["gpu"] == "sbatch --array=0-2 slurm/overlay/chain.sbatch hval full 3"
    assert "--array=0-29 slurm/overlay/chain_score.sbatch hval full 3" in lay["score"]
    with pytest.raises(PermissionError):
        C.layout("closure", "hs")
    # the cost layout (extrapolate) counts exactly these jobs
    for n, gj, cond in ((150, 50, "full"), (600, 50, "full"), (150, 45, "full"), (150, 50, "1gev"), (7, 3, "full")):
        nb = -(-n // gj)
        per = -(-n // nb)
        s = C.score_parts_n(n, gj, cond)
        jobs, largest = 0, 0
        for b in range(nb):
            ids = list(range(n))[b * per:(b + 1) * per]
            q = -(-len(ids) // s)
            sl = [ids[j * q:(j + 1) * q] for j in range(s)]
            jobs += sum(1 for x in sl if x)
            largest = max([largest] + [len(x) for x in sl])
        assert C._score_layout(n, gj, cond) == (jobs, largest), (n, gj, cond)


@needs_split
def test_production_c_full_block_is_scored_in_parts(stubs, grant, tmp_path):
    """A C-full production block of 50 events is scored by 10 jobs of 5 events (score_ids); without --part, or with
    fewer threads than SCORE_CPUS, the score job is refused before any write; each part scores only its events into
    its own file; the block's store stays until every event of the block is scored and is then removed by the last
    part; a part run again is skipped."""
    runs = str(tmp_path / "runs")
    out = C.paths("closure", "full", runs)["out"]
    C.run("closure", "full", block=1, n_blocks=3, stages=("additions", "store"), device="cpu", runs=runs)
    store = C.store_dir(out, "b001of003")
    assert len(C.stored_events(store)) == 50
    n = len(grant.calls)
    with pytest.raises(PermissionError, match="give --part 0..9"):
        C.run("closure", "full", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs)
    with pytest.raises(ValueError, match="production sets"):
        C.run("closure", "full", block=1, n_blocks=3, stages=("additions", "store"), device="cpu", runs=runs, part=0)
    with pytest.raises(ValueError, match="part 10 not in 0..9"):
        C.run("closure", "full", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs, part=10)
    C._score_threads, keep = (lambda: 2), C._score_threads
    try:
        with pytest.raises(PermissionError, match="6 threads or more"):
            C.run("closure", "full", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs, part=0)
    finally:
        C._score_threads = keep
    assert len(grant.calls) == n and not os.path.exists(os.path.join(out, "scores"))
    ids = C.block_ids("closure", 1, 3)
    for j in range(10):
        r = C.run("closure", "full", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs, part=j)
        assert r["done"] and r["part"] == f"b001of003p{j:02d}of10"
        assert sorted(C.scored_events(os.path.join(out, "scores"))) == ids[:5 * (j + 1)]
        assert os.path.isdir(store) == (j < 9)
        assert os.path.exists(os.path.join(out, "chain", "b001of003.done")) == (j == 9)
        rec = json.load(open(os.path.join(out, "chain", f"score_b001of003p{j:02d}of10.json")))
        assert rec["ids"] == ids[5 * j:5 * j + 5] and rec["block_ids"] == ids
    assert grant.calls[-1][0] == ids[45:] and {k for _, k, _ in grant.calls[n:] if isinstance(k, str)} == {"model"}
    assert len(glob.glob(os.path.join(out, "scores", "*.parquet"))) == 10
    assert not glob.glob(os.path.join(out, "store", "*.removing.*"))
    assert C.run("closure", "full", block=1, n_blocks=3, stages=("score",), device="cpu", runs=runs, part=3)["skipped"]
    assert C.run("closure", "full", block=1, n_blocks=3, stages=("additions", "store"), device="cpu",
                 runs=runs)["skipped"]


def test_drop_store_leaves_a_block_with_unscored_events(tmp_path):
    """The store of a block is removed only when every event of the block is scored, by one caller only; a scored-edge
    file whose sidecar is not complete yet (another part's job writing it) counts as not scored."""
    out = str(tmp_path / "o")
    d = C.store_dir(out, "b000of001")
    os.makedirs(d)
    sc = os.path.join(out, "scores")
    os.makedirs(sc)
    open(os.path.join(sc, "scores_00001.parquet"), "w").write("x")
    json.dump({"events": {"1": 3}}, open(os.path.join(sc, "scores_00001.parquet.json"), "w"))
    open(os.path.join(sc, "scores_00002.parquet"), "w").write("x")             # its sidecar not written yet
    assert not C.drop_store(out, "b000of001", [1, 2]) and os.path.isdir(d)
    with pytest.raises(OSError):
        C.scored_events(sc)                                                    # strict: an incomplete file is an error
    open(os.path.join(sc, "scores_00002.parquet.json"), "w").write('{"events": {"2"')      # cut short
    assert C.scored_events(sc, strict=False) == {1: "scores_00001.parquet"}
    json.dump({"events": {"2": 3}}, open(os.path.join(sc, "scores_00002.parquet.json"), "w"))
    assert C.drop_store(out, "b000of001", [1, 2]) and not os.path.exists(d)
    assert not C.drop_store(out, "b000of001", [1, 2])


def _stages_block(text):
    """The STAGES lines of slurm/overlay/chain.sbatch (default and the E2 refusal), runnable on their own."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith('if [ "$SET" = pilot ]; then STAGES='))
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == "fi")
    return "\n".join(lines[i:j + 1])


@pytest.mark.parametrize("set_name,given,ok,stages", [("pilot", "", True, "additions,store,score"),
                                                      ("hval", "", True, "additions,store"),
                                                      ("htest", "", True, "additions,store"),
                                                      ("closure", "additions,store,score", False, None),
                                                      ("htest", "score", False, None),
                                                      ("pilot", "score", True, "score")])
def test_chain_job_places_production_scores_on_the_cpu_nodes(set_name, given, ok, stages):
    """slurm/overlay/chain.sbatch (decision E2): production sets default to STAGES=additions,store and refuse a STAGES
    that holds score (the scores run in chain_score.sbatch); the pilot plumbing run keeps all three stages.
    chain_score.sbatch asks for 6 CPUs, the configuration whose score time the cost estimate uses."""
    import subprocess
    text = open(os.path.join(REPO, "slurm", "overlay", "chain.sbatch")).read()
    env = {k: v for k, v in os.environ.items() if k != "STAGES"}
    if given:
        env["STAGES"] = given
    script = f'set -euo pipefail; SET={set_name}\n{_stages_block(text)}\necho "STAGES=$STAGES"'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    if ok:
        assert p.returncode == 0 and p.stdout.strip() == f"STAGES={stages}", (p.stdout, p.stderr)
    else:
        assert p.returncode != 0 and "refusing: decision E2" in p.stdout
        assert not [ln for ln in p.stdout.splitlines() if ln.startswith("STAGES=")]
    assert 'ARGS=(--stages "$STAGES")' in text
    heads = [ln.strip() for ln in open(os.path.join(REPO, "slurm", "overlay", "chain_score.sbatch"))
             if ln.startswith("#SBATCH")]
    assert f"#SBATCH -c {C.SCORE_CPUS}" in heads


def _score_task_block(text):
    """The lines of slurm/overlay/chain_score.sbatch that map an array task to a block and a part."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("NP=$("))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("BLK="))
    return "\n".join(lines[i:j + 1])


@pytest.mark.parametrize("task,ok,blk,part", [(0, True, 0, 0), (9, True, 0, 9), (10, True, 1, 0), (29, True, 2, 9),
                                              (30, False, None, None)])
def test_score_job_maps_its_task_to_a_block_and_a_part(task, ok, blk, part):
    """chain_score.sbatch: task K of the N x S array scores part K % S of block K / S (S from `chain score-parts`);
    a task beyond N x S is refused."""
    import subprocess
    text = open(os.path.join(REPO, "slurm", "overlay", "chain_score.sbatch")).read()
    stub = 'python() { echo 10; }'
    script = f'set -euo pipefail; {stub}\nSET=hval; COND=full; NB=3; K={task}\n{_score_task_block(text)}\n' \
             'echo "BLK=$BLK PART=$PART NP=$NP"'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if ok:
        assert p.returncode == 0 and p.stdout.strip() == f"BLK={blk} PART={part} NP=10", (p.stdout, p.stderr)
    else:
        assert p.returncode != 0 and "refusing: task 30" in p.stdout
    assert 'ARGS=(--stages score --part "$PART")' in text
    assert '--ids "$IDS"' in text and "chain score-ids" in text


# ------------------------------------------------------------------------------------------------ dry, meter, cost
@needs_split
def test_dry_placeholders_for_the_pilot_only(tmp_path, grant):
    with pytest.raises(PermissionError):
        C.dry("closure", "full", str(tmp_path / "x"))
    with pytest.raises(PermissionError):
        C.dry("pilot", "full", os.path.join(B.RUNS_H, "x"))
    assert grant.calls == []


def test_meter_marks_each_event():
    m = C.Meter()
    for e in (3, 1, 2):
        m.mark(e)
        _ = bytearray(1_000_000)
    m.close()
    s = m.summary()
    assert [r["event"] for r in s["per_event"]] == [3, 1, 2] and s["seconds_events"] <= s["seconds_total"] + 1e-6
    assert all(r["peak_rss_gb"] is None or r["peak_rss_gb"] > 0 for r in s["per_event"])


def test_meter_reads_cuda_peaks_through_torch_cuda(monkeypatch):
    """On a GPU node the meter resets and reads the peak CUDA memory through torch.cuda (a stand-in module here)."""
    import sys
    import types
    calls = []
    cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                                 max_memory_allocated=lambda: 2.5e9,
                                 reset_peak_memory_stats=lambda: calls.append("reset"))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    m = C.Meter()
    for e in (7, 8):
        m.mark(e)
    m.close()
    assert calls == ["reset", "reset"] and [r["peak_gpu_gb"] for r in m.rows] == [2.5, 2.5]


def test_extrapolate_arithmetic():
    """Both score placements; decision E2 chose the first: GPU jobs of up to 50 events and 75 % of the limit, each
    block scored by CPU jobs of at most SCORE_EVENTS_PER_JOB events at SCORE_CPUS threads (core-hours at SCORE_CPUS),
    at the CPU node's measured score time (else the GPU job's, flagged); or every stage in the GPU job."""
    per = {"full": {"gpu_s_per_event": 60.0, "gpu_overhead_s_per_job": 30.0, "score_s_per_event_gpu_job": 120.0,
                    "score_overhead_s_per_job_gpu_job": 10.0, "score_cpus_gpu_job": 16,
                    "score_s_per_event_cpu_node": 150.0, "score_overhead_s_per_job_cpu_node": 20.0,
                    "score_cpus_cpu_node": 6},
           "1gev": {"gpu_s_per_event": 6.0, "gpu_overhead_s_per_job": 30.0, "score_s_per_event_gpu_job": 3.0,
                    "score_overhead_s_per_job_gpu_job": 10.0, "score_cpus_gpu_job": 16,
                    "score_s_per_event_cpu_node": None}}
    r = C.extrapolate(per, plan={"hval": (150, ("full", "1gev"))}, grid={"full": {"mean_s": 255.0}})
    assert r["placement_chosen"] == "scores_on_cpu_nodes"                         # decision E2; both are given
    assert r["layout"]["score_cpus"] == C.SCORE_CPUS == 6
    full = next(x for x in r["rows"] if x["condition"] == "full")
    cpu, gpu = full["scores_on_cpu_nodes"], full["scores_in_gpu_job"]
    assert cpu["gpu_events_per_job"] == 45 and cpu["gpu_jobs"] == 4                 # 2700 s // 60 s, at most 50
    assert cpu["gpu_hours"] == pytest.approx((150 * 60 + 4 * 30) / 3600, abs=0.01)
    # blocks of 38, 38, 38, 36 events; 8 parts each (38 / 5), of 5 events (36: 5 x 7 + 1)
    assert cpu["score_parts_per_block"] == 8 and cpu["score_jobs"] == 32 and cpu["score_events_per_job_max"] == 5
    assert cpu["score_wall_hours"] == pytest.approx((150 * 150 + 32 * 20) / 3600, abs=0.01)
    assert cpu["score_cpu_core_hours"] == pytest.approx((150 * 150 + 32 * 20) / 3600 * 6, abs=0.1)
    assert cpu["longest_score_job_s"] == 5 * 150 + 20 and cpu["score_time_at_job_threads"]
    assert cpu["score_time_measured_on_cpu_node"]
    assert gpu["events_per_job"] == 15 and gpu["jobs"] == 10                          # 2700 s // 180 s
    assert gpu["gpu_hours"] == pytest.approx((150 * 180 + 10 * 40) / 3600, abs=0.01)
    assert full["grid_cpu_hours"] == pytest.approx(150 * 255 / 3600, abs=0.1)
    one = next(x for x in r["rows"] if x["condition"] == "1gev")
    assert one["scores_in_gpu_job"]["events_per_job"] == C.MAX_BLOCK and one["scores_in_gpu_job"]["jobs"] == 3
    oc = one["scores_on_cpu_nodes"]
    assert not oc["score_time_measured_on_cpu_node"] and not oc["score_time_at_job_threads"]  # GPU job's time, flagged
    assert oc["score_jobs"] == 3 and oc["score_cpu_core_hours"] == pytest.approx((150 * 3 + 3 * 10) / 3600 * 6, abs=0.1)
    assert r["total"]["scores_in_gpu_job"]["gpu_hours"] == pytest.approx(
        full["scores_in_gpu_job"]["gpu_hours"] + one["scores_in_gpu_job"]["gpu_hours"], abs=0.1)
    assert r["total"]["scores_on_cpu_nodes"]["longest_score_job_share_of_limit"] == pytest.approx(770 / 3600, abs=1e-3)


def test_cost_from_a_plumbing_record(tmp_path):
    """`chain cost` recomputes the cost of this layout from a plumbing record's cost inputs (JSON only) and gives the
    GPU hours against the 10.4 GPU-h estimate."""
    per = {"full": {"gpu_s_per_event": 48.04, "gpu_overhead_s_per_job": 40.87, "score_s_per_event_gpu_job": 317.76,
                    "score_overhead_s_per_job_gpu_job": 1.04, "score_cpus_gpu_job": 16,
                    "score_s_per_event_cpu_node": 472.46, "score_overhead_s_per_job_cpu_node": 29.64,
                    "score_cpus_cpu_node": 6}}
    rec = tmp_path / "plumb.json"
    rec.write_text(json.dumps({"git": "x", "cost_inputs": per, "grid_timing_native_test": {}}))
    out = tmp_path / "cost.json"
    r = C.cost(str(rec), str(out))
    assert json.load(open(out))["extrapolation"] == r["extrapolation"]
    rows = {(x["set"], x["condition"]): x["scores_on_cpu_nodes"] for x in r["extrapolation"]["rows"]}
    assert rows[("hval", "full")]["score_jobs"] == 30 and rows[("htest", "full")]["score_jobs"] == 120
    assert rows[("hval", "full")]["gpu_jobs"] == 3 and rows[("htest", "full")]["gpu_jobs"] == 12
    gh = r["extrapolation"]["total"]["scores_on_cpu_nodes"]["gpu_hours"]
    assert gh == pytest.approx(2.04 * 2 + 8.14, abs=0.05) and r["budget"]["ratio"] == round(gh / 10.4, 2)
    assert r["source_sha256"] == hashlib.sha256(rec.read_bytes()).hexdigest()


def test_cpu_timing_at_the_score_jobs_threads_comes_first():
    """plumb_record takes, for the estimate, the CPU-node score time measured at SCORE_CPUS threads (the threads of
    chain_score.sbatch), the pilot CPU job before the chunked-vs-plain check at equal threads, and keeps the other."""
    job16 = {"score_cpus_cpu_node": 16, "what": "job"}
    chk6 = {"score_cpus_cpu_node": 6, "what": "check"}
    job6 = {"score_cpus_cpu_node": 6, "what": "job6"}
    assert C._pick_cpu_timing([job16, chk6]) == (chk6, job16)
    assert C._pick_cpu_timing([job6, chk6]) == (job6, chk6)
    assert C._pick_cpu_timing([job16]) == (job16, None) and C._pick_cpu_timing([]) == (None, None)


@pytest.mark.skipif(not os.path.exists(json.load(open(B.SELECTION))["run"]) if os.path.exists(B.SELECTION) else True,
                    reason="needs the frozen run directory")
def test_preflight_on_the_frozen_selection(monkeypatch):
    monkeypatch.delenv("HNL_TML_CONFIG", raising=False)
    monkeypatch.delenv("HNL_TML_SPLIT", raising=False)
    fi = C.preflight()
    assert fi["classifier"].endswith(".ckpt") and len(fi["classifier_sha256"]) == 64


@needs_split
def test_with_the_real_grant_module(stubs, monkeypatch):
    """hnl.overlay.grant itself (no stub): a plumb grant of the pilot in C-hs with out = OUT lets the chain and the
    frozen stage functions run; a model grant, another condition, missing ids or another out directory are refused
    before any write."""
    from hnl.overlay import grant as G
    if not hasattr(G, "write"):
        pytest.skip("hnl.overlay.grant has no write()")
    ids = B.set_ids("pilot")
    gdir = str(stubs["tmp"] / "grants")

    def use(**kw):
        a = {"set_name": "pilot", "cond": "hs", "kind": "plumb", "ids": ids, "directory": gdir, "out": None}
        a.update(kw)
        monkeypatch.setenv("HNL_H_GRANT", G.write(a.pop("set_name"), a.pop("cond"), a.pop("kind"), a.pop("ids"),
                                                  a.pop("directory"), out=a.pop("out")))

    out = str(stubs["tmp"] / "plumb" / "pilot_hs")
    other = str(stubs["tmp"] / "plumb" / "other")
    for bad in ({"kind": "model"}, {"cond": "1gev"}, {"ids": ids[:3]}, {"out": other}):
        use(**bad)
        refused = stubs["tmp"] / f"refused_{len(os.listdir(gdir))}"
        with pytest.raises(PermissionError):
            C.run("pilot", "hs", out=str(refused) if "out" not in bad else out, device="cpu")
        assert not refused.exists() and not os.path.exists(out)
    use(out=out)                                       # the guard's grant: out = OUT, code state of HEAD (decision E1)
    assert C.run("pilot", "hs", out=out, device="cpu")["done"]
    rec = json.load(open(os.path.join(out, "chain", "score_b000of001.json")))
    assert rec["grant"]["path"] == os.environ["HNL_H_GRANT"] and rec["grant"]["content"]["kind"] == "plumb"
    assert rec["grant"]["content"]["code"] == G.code_state() == rec["code"]
    assert "git_head" not in rec["grant"]["content"] and rec["grant"]["out"] == os.path.realpath(out)
