"""Grants of addendum H (hnl.overlay.grant; decisions D1, E1): the check every entry point makes on overlay ids, what
a grant binds (the blob hashes of the code an overlay job runs and the addendum, not the commit), the working tree, the
Slurm job, the launch token and the out directory, and the bypass
paths that it closes: each CLI's main (frozen chain, development tools, addendum H entry points) and the choke points
hnl.trackml.events.Embedder.__call__ and hnl.trackml.trackbuild.read_scores refuse an overlay id without a grant before
any model, table or score is read. Synthetic grants, tables and repository copies only; no model runs and no model
output is read."""

import json
import os
import re
import stat
import subprocess

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import grant as G

HV = [50_021_117, 50_021_300]                     # H-val style ids (k = 0)
needs_split_file = pytest.mark.skipif(not os.path.exists(os.path.join(G.REPO, "configs/trackml/split_h_v1.json")),
                                      reason="needs configs/trackml/split_h_v1.json")
PILOT = [59_021_289, 59_021_332]
TOK = "ab" * 16


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(G.ENV, raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return monkeypatch


def grant(tmp_path, monkeypatch, *a, **k):
    p = G.write(*a, directory=str(tmp_path / "grants"), **k)
    monkeypatch.setenv(G.ENV, p)
    return p


def rewrite(path, **fields):
    """A copy of a grant with some fields changed (the grant itself is read-only)."""
    g = json.load(open(path))
    g.update(fields)
    q = path + f".{len(fields)}.json"
    json.dump(g, open(q, "w"))
    return q


def test_no_overlay_id_needs_no_grant(clean_env):
    assert G.check([1000, 21117, 49_999_999], "model") is None
    assert G.check([], "eval") is None and G.check("1000,1001", "eval", "full") is None
    with pytest.raises(PermissionError, match="no grant"):
        G.check([1000, 50_000_000], "eval")                                # one overlay id is enough


def test_grant_fields_and_file(tmp_path, clean_env):
    clean_env.setenv("SLURM_JOB_ID", "123")
    p = G.write("hval", "full", "model", "50021300,50021117", str(tmp_path / "g"), out=str(tmp_path / "o"))
    g = json.load(open(p))
    assert list(g) == list(G.FIELDS) and g["ids"] == HV and g["ids_sha256"] == G.ids_sha256(HV)
    assert g["code"] == G.code_state() and set(g["code"]) == {"files", "addendum"}
    files = g["code"]["files"]                                                 # blob hashes of HEAD (decision E1)
    assert all(h is None or len(h) == 40 for h in files.values())
    for f in ("src/hnl/overlay/grant.py", "src/hnl/trackml/infer.py", "slurm/overlay/guard_h.sh",
              "configs/trackml/selection_v1.json", "src/hnl/eval/matching.py", "src/hnl/__init__.py"):
        assert files.get(f), f
    assert not [f for f in files if f.startswith(("src/hnl/plots/", "slurm/addendum_g", "configs/addendum_g", "docs/",
                                                  "paper/", "results/", "tests/"))]
    assert "src/hnl/eval/addendum_g_t3.py" not in files and "src/hnl/eval/addendum_g.py" not in files
    assert g["commit"] == G.git_head() and g["slurm_job_id"] == "123" and g["launch"] is None
    assert g["out"] == str(tmp_path / "o") and os.path.dirname(p) == str(tmp_path / "g")
    assert not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)       # read-only
    with pytest.raises(ValueError):
        G.write("hval", "nonoise", "model", HV, str(tmp_path))
    with pytest.raises(ValueError):
        G.write("hval", "full", "train", HV, str(tmp_path))
    with pytest.raises(ValueError):
        G.write("htest", "full", "model", [52_300_001], str(tmp_path), launch="short")


def test_check_accepts_a_covering_grant(tmp_path, clean_env, capsys):
    grant(tmp_path, clean_env, "hval", "full", "model", HV)
    assert G.check(HV[:1], "model", "full") is None
    assert G.check(HV, ("model", "plumb")) is None                         # any accepted kind, condition not named
    assert "hnl.overlay.grant: 2 ids under" in capsys.readouterr().err   # the job log records the grant


def test_check_refusals(tmp_path, clean_env):
    p = grant(tmp_path, clean_env, "hval", "full", "model", HV)
    cases = [(lambda: G.check(HV, "eval", "full"), "kind model, not eval"),
             (lambda: G.check(HV, "model", "1gev"), "condition full, not 1gev"),
             (lambda: G.check([50_021_118], "model"), "requested ids not granted"),
             (lambda: G.check(HV + [1000], "model"), "requested ids not granted")]
    for f, msg in cases:
        with pytest.raises(PermissionError, match=msg):
            f()
    code = json.load(open(p))["code"]
    files = code["files"]
    less = {k: v for k, v in files.items() if k != "src/hnl/trackml/events.py"}
    for fields, msg in (({"code": {**code, "files": {**files, "src/hnl/overlay/grant.py": "0" * 40}}},
                         "1 bound code file.s. at HEAD changed since the grant: src/hnl/overlay/grant.py"),
                        ({"code": {**code, "files": {**files, "configs/trackml/split_h_v1.json": "0" * 40}}},
                         "changed since the grant: configs/trackml/split_h_v1.json"),
                        ({"code": {**code, "files": less}}, "changed since the grant: src/hnl/trackml/events.py"),
                        ({"code": {**code, "files": {**files, "src/hnl/eval/new.py": "0" * 40}}},
                         "changed since the grant: src/hnl/eval/new.py"),
                        ({"code": {**code, "addendum": "0" * 40}}, "addendum docs/preregistration_pileup.md at HEAD"),
                        ({"code": {"src": "0" * 40, "slurm": "0" * 40, "configs": "0" * 40, "addendum": None}},
                         "binds no code files"),                          # a grant of the round-3 format
                        ({"commit": "0" * 40}, None),                     # the commit is provenance only
                        ({"ids": HV + [50_021_118]}, "ids_sha256"), ({"kind": "plumb"}, "kind plumb")):
        clean_env.setenv(G.ENV, rewrite(p, **fields))
        if msg is None:
            assert G.check(HV, "model") is None
            continue
        with pytest.raises(PermissionError, match=msg):
            G.check(HV, "model")
    clean_env.setenv(G.ENV, str(tmp_path / "missing.json"))
    with pytest.raises(PermissionError, match="unreadable grant"):
        G.check(HV, "model")
    q = tmp_path / "not_a_grant.json"
    q.write_text(json.dumps({"set": "hval"}))
    clean_env.setenv(G.ENV, str(q))
    with pytest.raises(PermissionError, match="not a grant"):
        G.check(HV, "model")


def test_check_ties_the_grant_to_its_slurm_job(tmp_path, clean_env):
    """A grant made inside a Slurm job holds in that job only: refused in another job and outside Slurm (a grant path
    from a job log is no key on the login node); a grant made outside Slurm is refused inside any job."""
    clean_env.setenv("SLURM_JOB_ID", "7001")
    grant(tmp_path, clean_env, "closure", "1gev", "eval", [51_021_117])
    assert G.check([51_021_117], "eval", "1gev") is None
    clean_env.setenv("SLURM_JOB_ID", "7002")
    with pytest.raises(PermissionError, match="granted to Slurm job 7001, this is job 7002"):
        G.check([51_021_117], "eval", "1gev")
    clean_env.delenv("SLURM_JOB_ID")
    with pytest.raises(PermissionError, match="granted to Slurm job 7001, and this process runs outside Slurm"):
        G.check([51_021_117], "eval")
    grant(tmp_path, clean_env, "closure", "1gev", "eval", [51_021_117])    # made outside Slurm
    assert G.check([51_021_117], "eval") is None
    clean_env.setenv("SLURM_JOB_ID", "7003")
    with pytest.raises(PermissionError, match="granted to Slurm job None, this is job 7003"):
        G.check([51_021_117], "eval")


def launch_record(runs, set_name, token=TOK, pins=None):
    """A launch record as hnl.overlay.blind.write_launch writes it, pinning only the given files (repo path: sha256)."""
    from hnl.overlay import blind as B
    os.makedirs(os.path.join(runs, "launch"), exist_ok=True)
    pins = pins or {"configs/trackml/split_h_v1.json": B.sha256_file(os.path.join(G.REPO,
                                                                                "configs/trackml/split_h_v1.json"))}
    json.dump({"set": set_name, "token": token, "pins": pins, "r15_sha256": B._r15_sha(G.REPO),
               "hval_manifest_sha256": None}, open(B.launch_path(set_name, runs), "w"))


@needs_split_file
def test_launched_sets_need_the_token_of_their_launch_record(tmp_path, clean_env):
    """grant.check compares the grant's launch token with RUNS/launch/<set>.json and runs check_launch (every pinned
    file unchanged): H-test always; H-val for kind eval and for kind model in C-1GeV / C-hs; not for the blinded
    evaluation or the C-full model run of H-val."""
    from hnl.overlay import blind as B
    runs = str(tmp_path / "runs")
    clean_env.setattr(B, "runs_root", lambda repo=None: runs)
    clean_env.setattr(B, "results_expect", lambda repo=None: None)
    grant(tmp_path, clean_env, "htest", "full", "eval", [52_300_001])
    with pytest.raises(PermissionError, match="launch token"):
        G.check([52_300_001], "eval")
    grant(tmp_path, clean_env, "htest", "full", "eval", [52_300_001], launch=TOK)
    with pytest.raises(PermissionError, match="refused by the launch record .*was not launched"):
        G.check([52_300_001], "eval", "full")
    launch_record(runs, "htest", token="cd" * 16)
    with pytest.raises(PermissionError, match="does not carry the token"):
        G.check([52_300_001], "eval", "full")
    os.remove(B.launch_path("htest", runs))
    launch_record(runs, "htest")
    assert G.check([52_300_001], "eval", "full") is None
    os.remove(B.launch_path("htest", runs))
    launch_record(runs, "htest", pins={"configs/trackml/split_h_v1.json": "0" * 64})
    with pytest.raises(PermissionError, match="split_h_v1.json changed since the launch of htest"):
        G.check([52_300_001], "eval", "full")
    # H-val: blind and the C-full model run need no launch; eval and the model in C-1GeV / C-hs do
    for kind, cond in (("blind", "full"), ("model", "full")):
        grant(tmp_path, clean_env, "hval", cond, kind, HV)
        assert G.check(HV, kind, cond) is None
    for kind, cond in (("eval", "full"), ("eval", "hs"), ("model", "1gev"), ("model", "hs")):
        grant(tmp_path, clean_env, "hval", cond, kind, HV)
        with pytest.raises(PermissionError, match="H-val ids need a grant made under the launch token"):
            G.check(HV, kind, cond)
        grant(tmp_path, clean_env, "hval", cond, kind, HV, launch=TOK)
        with pytest.raises(PermissionError, match="launch/hval.json"):
            G.check(HV, kind, cond)
    launch_record(runs, "hval")
    assert G.check(HV, "model", "hs") is None
    grant(tmp_path, clean_env, "hval", "hs", "eval", HV, launch=TOK)
    assert G.check(HV, "eval", "hs") is None
    clean_env.setattr(B, "check_launch", lambda *a, **k: ["x.py changed since the launch of hval"])
    with pytest.raises(PermissionError, match="x.py changed since the launch of hval"):
        G.check(HV, "eval", "hs")


def test_out_is_enforced_when_the_grant_names_one(tmp_path, clean_env):
    grant(tmp_path, clean_env, "closure", "full", "model", [51_021_117], out=str(tmp_path / "o"))
    assert G.check([51_021_117], "model", out=str(tmp_path / "o")) is None
    assert G.check([51_021_117], "model", out=str(tmp_path / "o" / "scores" / "x.parquet")) is None
    assert G.check([51_021_117], "model") is None                          # a caller that writes nothing
    for bad in (tmp_path / "other", tmp_path / "o2", tmp_path):
        with pytest.raises(PermissionError, match="outside the grant's out"):
            G.check([51_021_117], "model", out=str(bad))
    grant(tmp_path, clean_env, "closure", "full", "model", [51_021_117])   # a grant without out: any path
    assert G.check([51_021_117], "model", out=str(tmp_path / "other")) is None
    # the frozen CLIs that write pass their output path
    from hnl.trackml import trackbuild as TB
    grant(tmp_path, clean_env, "closure", "full", "eval", [51_021_117], out=str(tmp_path / "o"))
    with pytest.raises(PermissionError, match="outside the grant's out"):
        TB.main(["grid", "--scores", str(tmp_path), "--events", "51021117", "--out", str(tmp_path / "g"),
                 "--builders", "cc"])
    assert not os.path.exists(tmp_path / "g")


def test_several_grants_one_must_cover(tmp_path, clean_env):
    a = G.write("pilot", "full", "dry", PILOT, str(tmp_path / "g"))
    b = G.write("pilot", "hs", "dry", PILOT, str(tmp_path / "g"))
    clean_env.setenv(G.ENV, os.pathsep.join([a, b]))
    assert G.check(PILOT, "dry", "full") is None and G.check(PILOT, "dry", "hs") is None
    with pytest.raises(PermissionError, match="condition full, not 1gev.*condition hs, not 1gev"):
        G.check(PILOT, "dry", "1gev")


def test_check_tables(tmp_path, clean_env):
    d = tmp_path / "grid"
    d.mkdir()
    pd.DataFrame({"event": np.array([21117, 21300], np.int32), "config": 0}).to_parquet(d / "per_event.parquet")
    G.check_tables([str(d), None, str(tmp_path / "absent")], "eval")       # TrackML ids: nothing to grant
    CL = [51_021_117, 51_021_300]                                           # closure ids (no launch needed)
    pd.DataFrame({"event": np.array(CL, np.int32), "config": 0}).to_parquet(d / "per_event.parquet")
    with pytest.raises(PermissionError):
        G.check_tables([str(d)], "eval")
    grant(tmp_path, clean_env, "closure", "1gev", "eval", CL)
    G.check_tables([str(d / "per_event.parquet")], "eval", "1gev")
    pd.DataFrame({"event": np.array(HV, np.int32), "config": 0}).to_parquet(d / "per_event.parquet")
    grant(tmp_path, clean_env, "hval", "1gev", "eval", HV)                 # H-val eval: only under the launch token
    with pytest.raises(PermissionError, match="launch token"):
        G.check_tables([str(d)], "eval", "1gev")


def test_cli(tmp_path, clean_env, capsys):
    G.main(["write", "--set", "pilot", "--cond", "hs", "--kind", "plumb", "--ids", ",".join(map(str, PILOT)),
            "--dir", str(tmp_path)])
    p = capsys.readouterr().out.strip()
    assert json.load(open(p))["kind"] == "plumb"
    with pytest.raises(SystemExit):
        G.main(["check", "--ids", "59021289", "--kind", "plumb"])
    clean_env.setenv(G.ENV, p)
    G.main(["check", "--ids", "59021289", "--kind", "model,plumb", "--cond", "hs"])


# ------------------------------------------------------------------------------------------------ bypass paths (R1)
OV = "50021117"


def _refused(f, *args):
    with pytest.raises(PermissionError, match="no grant"):
        f(list(args))


def test_frozen_graph_and_store_clis_refuse_overlay_ids(tmp_path, clean_env):
    from hnl.trackml import graphs
    t = str(tmp_path)
    _refused(graphs.main, "additions", "--events", OV, "--graph-choice", t + "/none.json", "--out", t + "/a")
    _refused(graphs.main, "additions", "--events", OV, "--graph-choice", t + "/none.json", "--out", t + "/a",
             "--condition", "1gev")
    _refused(graphs.main, "stage", "--events", OV, "--graph-choice", t + "/none.json", "--additions", t,
             "--dest", t + "/s")
    _refused(graphs.main, "fs", "--events", OV, "--dest", t + "/fs")
    _refused(graphs.main, "radii", "--events", OV, "--ml-ckpt", t + "/none.ckpt", "--out", t + "/r.json")
    # ltf-data and scan check the grant before they read the radii or load a network
    _refused(graphs.main, "ltf-data", "--events", OV, "--ml-ckpt", t + "/none.ckpt", "--radii", t + "/none.json",
             "--out", t + "/l")
    _refused(graphs.main, "scan", "--events", OV, "--ml-ckpt", t + "/none.ckpt", "--radii", t + "/none.json",
             "--ltf-model", t + "/none.pt", "--out", t + "/sc")
    assert not os.path.exists(t + "/a") and not os.path.exists(t + "/s") and not os.path.exists(t + "/sc")


def test_frozen_score_cli_refuses_overlay_events_of_a_store(tmp_path, clean_env):
    from hnl.trackml import infer
    store = tmp_path / "store" / "hval"
    store.mkdir(parents=True)
    (store / "event950021117.pyg.gz").write_bytes(b"")                     # an overlay event's file name only
    _refused(infer.main, "score", "--store", str(store), "--oracle", "1.0", "--out", str(tmp_path / "scores"))
    _refused(infer.main, "score", "--store", str(store), "--oracle", "1.0", "--out", str(tmp_path / "scores"),
             "--condition", "1gev", "--block", "0")
    # with a checkpoint: the grant is checked before the network is loaded (score and g2)
    _refused(infer.main, "score", "--store", str(store), "--ckpt", str(tmp_path / "none.ckpt"), "--out",
             str(tmp_path / "scores"))
    _refused(infer.main, "g2", "--store", str(store), "--ckpt", str(tmp_path / "none.ckpt"), "--out",
             str(tmp_path / "g2"))
    assert not os.path.exists(tmp_path / "scores") and not os.path.exists(tmp_path / "g2")


def test_frozen_trackbuild_cli_refuses_overlay_ids(tmp_path, clean_env):
    from hnl.trackml import trackbuild as TB
    t = str(tmp_path)
    _refused(TB.main, "grid", "--scores", t, "--events", OV, "--out", t + "/g", "--builders", "cc")
    _refused(TB.main, "grid", "--scores", t, "--events", OV, "--out", t + "/g", "--builders", "cc",
             "--condition", "1gev", "--block", "0", "--block-size", "1")
    _refused(TB.main, "grid-merge", "--blocks", t, "--events", OV, "--out", t + "/g")
    _refused(TB.main, "timing", "--scores", t, "--events", OV, "--out", t + "/g", "--builders", "cc")
    _refused(TB.main, "lmerge-data", "--scores", t, "--events", OV, "--out", t + "/l.npz")
    _refused(TB.main, "merge-check", "--scores", t, "--events", OV, "--out", t + "/g")
    _refused(TB.main, "merge-profile", "--scores", t, "--events", OV, "--out", t + "/g")
    assert not os.path.exists(t + "/g")
    d = tmp_path / "grid"
    d.mkdir()
    pd.DataFrame({"event": np.array([int(OV)], np.int32), "config": 0}).to_parquet(d / "per_event.parquet")
    _refused(TB.main, "select", "--grid", str(d), "--out", t + "/sel.json")


def test_frozen_diag_and_evaluate_clis_refuse_overlay_ids(tmp_path, clean_env):
    from hnl.trackml import displaced_diag as DD
    from hnl.trackml import evaluate as EV
    t = str(tmp_path)
    _refused(DD.main, "run", "--events", OV, "--scores", t, "--configs", t + "/c.json", "--merge-model", t + "/m",
             "--out", t + "/d")
    diag = pd.DataFrame({"event": np.array([int(OV)], np.int32), "particle_id": 1})
    diag.to_parquet(tmp_path / "diag_full_b0.parquet")
    _refused(DD.main, "summary", "--dir", t)
    d = tmp_path / "grid"
    d.mkdir()
    pd.DataFrame({"event": np.array([int(OV)], np.int32), "config": 0}).to_parquet(d / "per_event.parquet")
    (d / "configs.json").write_text(json.dumps([["cc", 0.8]]))
    _refused(EV.main, "report", "--grid", str(d), "--out", t + "/rep")
    assert not os.path.exists(t + "/rep")


needs_split = pytest.mark.skipif(not os.path.exists(os.path.join(G.REPO, "configs/trackml/split_h_v1.json")),
                                 reason="needs configs/trackml/split_h_v1.json")


@needs_split
def test_addendum_h_entry_points_refuse_overlay_ids(tmp_path, clean_env):
    from hnl.overlay import blind as B
    from hnl.overlay import h1 as H
    from hnl.overlay import tracks as TR
    t = str(tmp_path)
    _refused(TR.main, "grid", "--set", "pilot", "--cond", "hs", "--scores", t, "--out", t + "/g")
    _refused(TR.main, "grid", "--set", "closure", "--cond", "1gev", "--scores", t, "--out", t + "/g")
    _refused(B.main, "evaluate", "--set", "pilot", "--sealed", t + "/sealed")
    _refused(H.main, "run", "--set", "pilot", "--sealed", t, "--expect", "0" * 64, "--grid-1gev", t, "--grid-hs", t,
             "--out-json", t + "/h1.json")
    _refused(H.main, "run", "--set", "hval", "--sealed", t, "--grid-1gev", t, "--grid-hs", t,
             "--out-json", t + "/h1.json")
    # before H-T1 there is no H-test list at all (LookupError); after it, no grant (PermissionError)
    with pytest.raises((LookupError, PermissionError)):
        H.main(["run", "--set", "htest", "--grid-full", t, "--grid-1gev", t, "--grid-hs", t,
                "--out-json", t + "/h1.json"])
    assert not os.listdir(t) or set(os.listdir(t)) <= {"grants"}
    # a grant of another condition or kind does not help: the H1 run needs one for every condition it reads
    grant(tmp_path, clean_env, "pilot", "full", "dry", B.set_ids("pilot"), out=t)
    with pytest.raises(PermissionError, match="condition full, not 1gev"):
        H.main(["run", "--set", "pilot", "--sealed", t, "--expect", "0" * 64, "--grid-1gev", t, "--grid-hs", t,
                "--out-json", t + "/h1.json"])
    with pytest.raises(PermissionError, match="kind dry, not eval"):
        TR.main(["grid", "--set", "closure", "--cond", "full", "--scores", t, "--out", t + "/g"])
    # H-val with eval grants for its conditions but none made under the launch token: refused by the grant check
    gs = [G.write("hval", c, "eval", B.set_ids("hval"), t + "/grants") for c in ("full", "1gev", "hs")]
    clean_env.setenv(G.ENV, os.pathsep.join(gs))
    with pytest.raises(PermissionError, match="H-val ids need a grant made under the launch token"):
        H.main(["run", "--set", "hval", "--sealed", t, "--grid-1gev", t, "--grid-hs", t, "--out-json", t + "/h1.json"])
    assert not os.path.exists(t + "/h1.json")


@needs_split
def test_h1_run_needs_every_condition_before_any_other_check(tmp_path, clean_env):
    """A decisive run never goes without C-hs: hval and the pilot need the C-1GeV and C-hs grids, htest also C-full;
    a missing one stops the run before the grant, launch and marker checks (no grant is set here, so any later check
    would raise PermissionError instead)."""
    from hnl.overlay import h1 as H
    t = str(tmp_path)
    for args in (["--set", "hval", "--sealed", t, "--grid-1gev", t],
                 ["--set", "pilot", "--sealed", t, "--expect", "0" * 64, "--grid-1gev", t],
                 ["--set", "hval", "--sealed", t, "--grid-hs", t]):
        with pytest.raises(SystemExit):
            H.main(["run", *args, "--out-json", t + "/h1.json"])
    with pytest.raises(PermissionError, match="sealed directory"):          # hval in C-full never from a grid
        H.main(["run", "--set", "hval", "--sealed", t, "--grid-full", t, "--grid-1gev", t, "--grid-hs", t,
                "--out-json", t + "/h1.json"])
    assert not os.path.exists(t + "/h1.json")


# ------------------------------------------------------------------------------------------------ E1: code, not commit
def _repo(tmp_path):
    """A repository with src/, slurm/, configs/, the addendum and unrelated files (paper, results, DECISIONS)."""
    r = str(tmp_path / "repo")
    files = {"src/hnl/overlay/x.py": "x = 1\n", "slurm/overlay/x.sbatch": "#!/bin/bash\n",
             "configs/trackml/x.json": "{}\n", "docs/preregistration_pileup.md": "draft\n",
             "DECISIONS.md": "# Decisions\n", "paper/main.tex": "text\n", "results/x.csv": "a\n1\n"}
    for f, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
        open(os.path.join(r, f), "w").write(text)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", "-C", r, *a], check=True, capture_output=True, env=env)   # noqa: E731
    git("init", "-q")
    git("add", "-A", "--", ".", ":!docs/preregistration_pileup.md")         # the addendum not committed yet
    git("commit", "-qm", "copy")

    def commit(rel, text):
        p = os.path.join(r, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "a").write(text)
        git("add", rel)
        git("commit", "-qm", f"edit {rel}")
    return r, commit


VERSION_RECORD = "configs/trackml/overlay_version_h_v1.json"


def _without_version_record(r, git):
    """The version record has been tracked since H-T0.5 step 4 (8daa282); the tests below need it absent at the HEAD of
    their repository copy, as it was when they were written, so that writing it adds a bound file."""
    if os.path.exists(os.path.join(r, VERSION_RECORD)):
        git("rm", "-q", "--", VERSION_RECORD)
        git("commit", "-qm", "version record absent in this copy")


def _code_copy(tmp_path):
    """A repository holding the tracked files of src/hnl, slurm and configs as they are in this working tree, the
    addendum (not committed) and unrelated files; returns (path, commit(rel, text), git)."""
    import shutil
    r = str(tmp_path / "code_repo")
    tracked = subprocess.run(["git", "-C", G.REPO, "ls-files", "src/hnl", "slurm", "configs"], check=True,
                             capture_output=True, text=True).stdout.split()
    for f in tracked:
        if os.path.isfile(os.path.join(G.REPO, f)):
            os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
            shutil.copy(os.path.join(G.REPO, f), os.path.join(r, f))
    for f, text in {"docs/preregistration_pileup.md": "draft\n", "DECISIONS.md": "# Decisions\n",
                    "paper/main.tex": "text\n", "results/x.csv": "a\n1\n"}.items():
        os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
        open(os.path.join(r, f), "w").write(text)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", "-C", r, *a], check=True, capture_output=True, env=env)   # noqa: E731
    git("init", "-q")
    git("add", "-A", "--", ".", ":!docs/preregistration_pileup.md")
    git("commit", "-qm", "copy")

    def commit(rel, text, mode="a"):
        p = os.path.join(r, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, mode).write(text)
        git("add", rel)
        git("commit", "-qm", f"edit {rel}")
    return r, commit, git


def _closure_from_source(r):
    """The bound set computed independently from the files on disk: hnl.overlay.blind.import_closure (a separate AST
    walk) over every module of hnl.overlay and hnl.trackml outside their tests, every tracked file of src/hnl/overlay,
    src/hnl/trackml and slurm/overlay, and the configs paths named in those modules (a regex over their text for
    os.path.join with the literal "configs" and for whole string literals "configs/...")."""
    import re
    from hnl.overlay import blind as B
    src = os.path.join(r, "src")
    seeds = []
    for d in ("hnl/overlay", "hnl/trackml"):
        for root, dirs, fs in os.walk(os.path.join(src, d)):
            dirs[:] = [x for x in dirs if x not in ("tests", "__pycache__")]
            for f in fs:
                if f.endswith(".py"):
                    m = os.path.relpath(os.path.join(root, f), src)[:-3].replace("/", ".")
                    seeds.append(m[:-len(".__init__")] if m.endswith(".__init__") else m)
    closure = set(B.import_closure(seeds, r))
    tracked = subprocess.run(["git", "-C", r, "ls-files"], check=True, capture_output=True, text=True).stdout.split()
    files = {f for f in tracked if f.startswith(("src/hnl/overlay/", "src/hnl/trackml/", "slurm/overlay/"))} | closure
    named = set()
    for f in closure:
        text = open(os.path.join(r, f)).read()
        for m in re.finditer(r'os\.path\.join\([^()]*?"configs"((?:\s*,\s*"[^"]*")*)', text):
            named.add("/".join(["configs"] + re.findall(r'"([^"]*)"', m.group(1))))
        named |= {m.rstrip("/") for m in re.findall(r'"(configs(?:/[A-Za-z0-9_.-]+)+/?)"', text)}
    absent = set()
    for c in named:
        below = {f for f in tracked if f == c or f.startswith(c + "/")}
        files |= below
        if not below:
            absent.add(c)
    return files, absent


def test_bound_set_is_the_static_import_closure(tmp_path, clean_env):
    """Decision E1 (fourth code round): code_state binds exactly the files that the static import closure of the
    overlay and trackml modules computed from the source names, each with its blob hash at HEAD."""
    r, _, _ = _code_copy(tmp_path)
    files, absent = _closure_from_source(r)
    st = G.code_state(r)
    assert set(st["files"]) == files | absent
    assert {p for p, h in st["files"].items() if h is None} == absent
    present = sorted(files)
    blobs = subprocess.run(["git", "-C", r, "hash-object", "--", *present], check=True, capture_output=True,
                           text=True).stdout.split()
    assert dict(zip(present, blobs)) == {p: st["files"][p] for p in present}
    assert st["addendum"] is None
    outside = {f for f in st["files"] if not f.startswith(("src/hnl/overlay/", "src/hnl/trackml/", "slurm/overlay/"))}
    assert "src/hnl/eval/matching.py" in outside and "configs/trackml/pipeline_v1.yaml" in outside
    assert not [f for f in outside if f.startswith(("src/hnl/plots/", "slurm/addendum_g", "configs/addendum_g"))]


def test_grants_bind_the_code_an_overlay_job_runs(tmp_path, clean_env):
    """Decision E1: a commit outside the bound files (src/hnl/eval outside the closure, slurm/addendum_g_t3, paper,
    docs, results, DECISIONS, tests) keeps a running job's grant valid; a commit to any bound file, a new module or
    configs file in the bound set, a new import that adds a module to it, or the addendum, invalidates it. Both through
    problems() with the copy's state and through check() (as a job in the copy would run it)."""
    r, commit, git = _code_copy(tmp_path)
    _without_version_record(r, git)
    clean_env.setattr(G, "REPO", r)
    out = str(tmp_path / "o")
    p = G.write("pilot", "full", "plumb", PILOT, str(tmp_path / "g"), out=out)
    g = json.load(open(p))
    assert g["code"]["addendum"] is None and g["commit"] == G.git_head(r)    # plumb: the addendum need not be committed
    clean_env.setenv(G.ENV, p)
    assert G.check(PILOT, "plumb", "full") is None
    keep = ("src/hnl/eval/addendum_g_t3.py", "src/hnl/eval/new_t3_module.py", "slurm/addendum_g_t3/union_block.sbatch",
            "slurm/addendum_g_t3/new.sbatch", "configs/addendum_g/new.json", "src/hnl/plots/new_plot.py",
            "DECISIONS.md", "paper/main.tex", "results/x.csv", "results/pileup_hp/new.json", "RESULTS.md",
            "docs/other.md", "docs/feedback/test_use_ledger.md", "tests/test_x.py")
    for rel in keep:
        commit(rel, "# more\n")
        assert G.problems(p, PILOT, ("plumb",), "full", G.code_state(r)) == [], rel
    assert G.check(PILOT, "plumb", "full") is None and G.git_head(r) != g["commit"]
    bound = [f for f, h in G.code_state(r)["files"].items() if h is not None]
    assert len(bound) > 100
    for rel in bound:                                                          # every bound file, one commit each
        q = G.write("pilot", "full", "plumb", PILOT, str(tmp_path / "g"), out=out)
        commit(rel, "\n# changed\n")
        bad = G.problems(q, PILOT, ("plumb",), "full", G.code_state(r))
        assert len(bad) == 1 and "changed since the grant" in bad[0] and rel in bad[0], (rel, bad)
    with pytest.raises(PermissionError, match="bound code file.s. at HEAD changed since the grant"):
        G.check(PILOT, "plumb", "full")
    for rel, text, why in (("src/hnl/overlay/new_module.py", "x = 1\n", "src/hnl/overlay/new_module.py"),
                           ("src/hnl/trackml/infer.py", "\nimport hnl.eval.addendum_g_t3  # noqa\n",
                            "src/hnl/eval/addendum_g_t3.py"),
                           ("configs/trackml/overlay_version_h_v1.json", "{}\n", "overlay_version_h_v1.json"),
                           ("docs/preregistration_pileup.md", "committed\n", "addendum")):
        q = G.write("pilot", "full", "plumb", PILOT, str(tmp_path / "g"), out=out)
        clean_env.setenv(G.ENV, q)
        assert G.check(PILOT, "plumb", "full") is None
        commit(rel, text)
        with pytest.raises(PermissionError, match=why):
            G.check(PILOT, "plumb", "full")
    q = G.write("hval", "full", "blind", HV, str(tmp_path / "g"))
    assert json.load(open(q))["code"]["addendum"] == G.code_state(r)["addendum"] is not None   # committed: bound


def test_uncommitted_change_to_a_bound_file_stops_the_grant(tmp_path, clean_env):
    """grant.check also compares the working tree: an uncommitted (or staged) change to a bound file, an untracked
    python file under src/hnl/overlay or src/hnl/trackml, or an untracked file under src that python would import in
    place of a bound module (a shadow package or compiled module), stops a running job, since a later process of the
    job would run it; changes outside the bound files and untracked modules of other names do not."""
    r, commit, git = _code_copy(tmp_path)
    _without_version_record(r, git)
    clean_env.setattr(G, "REPO", r)
    clean_env.setenv(G.ENV, G.write("pilot", "hs", "plumb", PILOT, str(tmp_path / "g")))
    ok = lambda: G.check(PILOT, "plumb", "hs") is None                         # noqa: E731
    assert ok()
    for rel in ("src/hnl/eval/addendum_g_t3.py", "slurm/addendum_g_t3/union_block.sbatch", "DECISIONS.md"):
        open(os.path.join(r, rel), "a").write("# local edit\n")
        assert ok(), rel
    open(os.path.join(r, "slurm/overlay/.nfs000000001"), "w").write("x")     # not a module
    open(os.path.join(r, "src/hnl/eval/scratch.py"), "w").write("x = 1\n")   # a module outside the closure
    assert ok()
    for rel in ("src/hnl/overlay/blind.py", "src/hnl/eval/matching.py", "configs/trackml/selection_v1.json",
                "slurm/overlay/h1.sbatch"):
        open(os.path.join(r, rel), "a").write("\n")
        with pytest.raises(PermissionError, match=f"uncommitted change to 1 bound code file.s. in the working tree: "
                                                  f"{rel}"):
            G.check(PILOT, "plumb", "hs")
        git("add", rel)                                                        # staged is not committed
        with pytest.raises(PermissionError, match="uncommitted change"):
            G.check(PILOT, "plumb", "hs")
        git("reset", "-q", "--", rel)
        git("checkout", "--", rel)
        assert ok(), rel
    open(os.path.join(r, "src/hnl/trackml/new_module.py"), "w").write("y = 1\n")
    with pytest.raises(PermissionError, match="src/hnl/trackml/new_module.py"):
        G.check(PILOT, "plumb", "hs")
    os.remove(os.path.join(r, "src/hnl/trackml/new_module.py"))
    open(os.path.join(r, "configs/trackml/overlay_version_h_v1.json"), "w").write("{}")   # a bound path absent at HEAD
    with pytest.raises(PermissionError, match="overlay_version_h_v1.json"):
        G.check(PILOT, "plumb", "hs")
    os.remove(os.path.join(r, "configs/trackml/overlay_version_h_v1.json"))
    assert ok()
    # an untracked file that python would import in place of a bound module (a package directory or a compiled module
    # of its name), ignored by .gitignore or not; other untracked files under src stay allowed
    open(os.path.join(r, ".gitignore"), "w").write("__pycache__/\n*.so\n*.pyc\n")
    import sysconfig
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    for rel in ("src/hnl/eval/matching/__init__.py", f"src/hnl/eval/matching{ext}", "src/hnl/trackbuild/builders.so",
                "src/hnl/eval/matching/__init__.pyc"):
        os.makedirs(os.path.dirname(os.path.join(r, rel)), exist_ok=True)
        open(os.path.join(r, rel), "w").write("x = 1\n")
        assert rel in G.shadow_paths(G.code_state(r)["files"], r)
        with pytest.raises(PermissionError, match=f"in the working tree: {re.escape(rel)}"):
            G.check(PILOT, "plumb", "hs")
        os.remove(os.path.join(r, rel))
        assert ok(), rel
    for rel in ("src/hnl/eval/matching_notes.txt", "src/hnl/eval/__pycache__/matching.cpython-311.pyc",
                "src/hnl/eval/new_eval_module.py", "src/hnl/eval/new_pkg/__init__.py", "src/hnl/plots/matching.py"):
        os.makedirs(os.path.dirname(os.path.join(r, rel)), exist_ok=True)
        open(os.path.join(r, rel), "w").write("x = 1\n")
        assert ok(), rel


def test_imported_hnl_modules_lie_in_the_bound_set(tmp_path):
    """After importing every module of hnl.overlay and hnl.trackml (the overlay entry points and everything they reach
    at import time), every hnl module in sys.modules is a file of the bound set. Run in a fresh interpreter; the record
    is written to HNL_H_MODULES_RECORD when set (the test job keeps it)."""
    import sys
    code = r"""
import importlib, json, os, pkgutil, sys
import hnl.overlay, hnl.trackml
names = []
for pkg in (hnl.overlay, hnl.trackml):
    for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if m.name.split(".")[-1] != "tests":
            names.append(m.name)
failed = {}
for n in names:
    try:
        importlib.import_module(n)
    except Exception as e:                       # noqa: BLE001
        failed[n] = f"{type(e).__name__}: {e}"[:200]
src = os.path.realpath(os.path.join(os.path.dirname(hnl.__file__), ".."))
mods = {k: os.path.relpath(os.path.realpath(v.__file__), os.path.dirname(src))
        for k, v in sys.modules.items() if (k == "hnl" or k.startswith("hnl.")) and getattr(v, "__file__", None)}
print(json.dumps({"entry_modules": names, "failed": failed, "modules": mods}))
"""
    env = {k: v for k, v in os.environ.items() if k not in ("HNL_H_GRANT", "HNL_H_LAUNCH")}
    env["PYTHONPATH"] = os.path.join(G.REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    bound = G.code_state()["files"]
    outside = sorted(f for f in rec["modules"].values() if f not in bound)
    rec.update({"git": G.git_head(), "bound_digest": G.digest(G.code_state()), "outside_bound_set": outside,
                "n_modules": len(rec["modules"]), "n_bound": len(bound)})
    if os.environ.get("HNL_H_MODULES_RECORD"):
        with open(os.environ["HNL_H_MODULES_RECORD"], "w") as fh:
            json.dump(rec, fh, indent=1)
    assert not rec["failed"], rec["failed"]
    assert len(rec["modules"]) > 50 and outside == [], outside


@needs_split_file
def test_guard_grant_survives_unrelated_commits_in_a_repository_copy(tmp_path, clean_env):
    """The guard writes the grant of the copy (--repo); a docs-only commit keeps it, a src commit stops it."""
    import shutil
    import sys
    r, commit = _repo(tmp_path)
    for f in ("slurm/overlay/guard_h.sh", "src/hnl/overlay/blind.py", "src/hnl/overlay/grant.py",
              "slurm/overlay/blind.sbatch", "slurm/overlay/h1.sbatch", "src/hnl/overlay/split_h.py",
              "src/hnl/overlay/conditions.py", "src/hnl/overlay/tracks.py", "src/hnl/overlay/h1.py",
              "src/hnl/overlay/hstats.py", "src/hnl/overlay/v1.py", "src/hnl/trackml/evaluate.py",
              "src/hnl/trackml/trackbuild.py", "src/hnl/trackml/infer.py", "src/hnl/trackml/events.py",
              "configs/trackml/split_h_v1.json", "configs/trackml/selection_v1.json"):
        os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
        shutil.copy(os.path.join(G.REPO, f), os.path.join(r, f))
    subprocess.run(["git", "-C", r, "add", "-A", "--", ".", ":!docs/preregistration_pileup.md"], check=True)
    subprocess.run(["git", "-C", r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "code"], check=True)
    env = {**os.environ, "HNL_H_RUNS": str(tmp_path / "runs"), "HNL_H_DERIVED": str(tmp_path / "derived"),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
    env.pop("SLURM_JOB_ID", None)
    from hnl.overlay import blind as B
    ids = ",".join(str(i) for i in B.set_ids("pilot"))
    out = subprocess.run(["bash", os.path.join(r, "slurm/overlay/guard_h.sh"), "--set", "pilot", "--cond", "full",
                          "--kind", "plumb", "--ids", ids, "--out", str(tmp_path / "plumb" / "pilot_full"),
                          "--grant"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    p = out.stdout.strip().split("=", 1)[1]
    clean_env.setattr(G, "REPO", r)
    clean_env.setenv(G.ENV, p)
    P = B.set_ids("pilot")
    assert G.check(P, "plumb", "full", out=str(tmp_path / "plumb" / "pilot_full" / "scores")) is None
    commit("docs/notes.md", "a docs-only commit\n")
    commit("paper/main.tex", "and a paper commit\n")
    assert G.check(P, "plumb", "full") is None
    commit("src/hnl/eval/other.py", "y = 2\n")                         # outside the bound set (decision E1)
    assert G.check(P, "plumb", "full") is None
    commit("src/hnl/overlay/x.py", "y = 2\n")
    with pytest.raises(PermissionError, match="changed since the grant: src/hnl/overlay/x.py"):
        G.check(P, "plumb", "full")


# ------------------------------------------------------------------------------------------------ choke points
def _scores_dir(d, events, placeholder=None):
    """A scored-edge file in the format of hnl.trackml.infer score (one edge per event) with its sidecar."""
    os.makedirs(d, exist_ok=True)
    t = pd.DataFrame({"event": np.array(events, np.int32), "a": np.zeros(len(events), np.int32),
                      "b": np.ones(len(events), np.int32), "score": np.full(len(events), 0.9, np.float32),
                      "y": np.ones(len(events), bool)})
    t.to_parquet(os.path.join(d, "scores_00000.parquet"))
    side = {"events": {str(e): 1 for e in events}, "condition": "full", "placeholder": placeholder}
    json.dump(side, open(os.path.join(d, "scores_00000.parquet.json"), "w"))
    return d


def test_read_scores_choke_point(tmp_path, clean_env):
    """hnl.trackml.trackbuild.read_scores: native ids need nothing; overlay ids a grant of kind eval or dry (never model
    or plumb), and of kind blind only where hnl.overlay.blind asks for it; scores of pilot events only if truth-made
    placeholders (decision D6), even under a dry grant."""
    from hnl.overlay import blind as B
    from hnl.trackml import trackbuild as TB
    assert G.PLACEHOLDER == B.PLACEHOLDER and G.OVERLAY_FIRST_ID == B.OVERLAY_FIRST_ID
    nat = _scores_dir(str(tmp_path / "nat"), [21117, 21300])
    assert sorted(TB.read_scores(nat, [21117, 21300])) == [21117, 21300]
    real = _scores_dir(str(tmp_path / "plumb"), PILOT)                   # as the frozen classifier writes them
    with pytest.raises(PermissionError, match="no grant"):
        TB.read_scores(real, PILOT)
    for kind in ("plumb", "model", "blind"):
        grant(tmp_path, clean_env, "pilot", "full", kind, PILOT)
        with pytest.raises(PermissionError, match=f"kind {kind}, not eval or dry"):
            TB.read_scores(real, PILOT)
    grant(tmp_path, clean_env, "pilot", "full", "plumb", PILOT)
    with pytest.raises(PermissionError, match="kind plumb, not blind or dry"):
        TB.read_scores(real, PILOT, kinds=("blind", "dry"))                    # as hnl.overlay.blind reads them
    with pytest.raises(ValueError):
        TB.read_scores(real, PILOT, kinds=("model",))
    grant(tmp_path, clean_env, "pilot", "full", "dry", PILOT)
    with pytest.raises(PermissionError, match="truth-made placeholders only"):
        TB.read_scores(real, PILOT)
    ph = _scores_dir(str(tmp_path / "ph"), PILOT, placeholder=G.PLACEHOLDER)
    assert sorted(TB.read_scores(ph, PILOT)) == PILOT
    grant(tmp_path, clean_env, "closure", "full", "eval", [51_021_117])
    cl = _scores_dir(str(tmp_path / "cl"), [51_021_117])
    assert list(TB.read_scores(cl, [51_021_117])) == [51_021_117]           # production sets: the frozen scores
    hv = _scores_dir(str(tmp_path / "hv"), HV)
    grant(tmp_path, clean_env, "hval", "full", "blind", HV, out=str(tmp_path / "sealed"))
    with pytest.raises(PermissionError, match="kind blind, not eval or dry"):
        TB.read_scores(hv, HV)                                              # the blind grant opens blind alone
    assert sorted(TB.read_scores(hv, HV, kinds=("blind", "dry"))) == HV


def test_labels_rebuild_needs_an_eval_grant(tmp_path, clean_env):
    """hnl.trackml.labels rebuild (raw track labels and unblinded counts from stored scores) checks an eval grant with
    its output directory before any input is read: the blind grant of the sealed H-val directory does not open it."""
    from hnl.trackml import labels as LB
    hv = _scores_dir(str(tmp_path / "sealed" / "scores"), HV)
    argv = ["rebuild", "--events", ",".join(map(str, HV)), "--scores", hv, "--configs", str(tmp_path / "none.json"),
            "--out", str(tmp_path / "labels")]
    with pytest.raises(PermissionError, match="no grant"):
        LB.main(argv)
    grant(tmp_path, clean_env, "hval", "full", "blind", HV, out=str(tmp_path / "sealed"))
    with pytest.raises(PermissionError, match="kind blind, not eval"):
        LB.main(argv)
    grant(tmp_path, clean_env, "closure", "full", "eval", [51_021_117], out=str(tmp_path / "o"))
    with pytest.raises(PermissionError, match="outside the grant's out"):
        LB.main(["rebuild", "--events", "51021117", "--scores", hv, "--configs", str(tmp_path / "none.json"),
                 "--out", str(tmp_path / "labels")])
    with pytest.raises(ValueError):                                         # granted: on to the configurations
        LB.main(["rebuild", "--events", "51021117", "--scores", hv, "--configs", str(tmp_path / "none.json"),
                 "--out", str(tmp_path / "o" / "labels")])
    assert not os.path.exists(tmp_path / "labels") and not os.path.exists(tmp_path / "o" / "labels")


def test_embedder_choke_point(tmp_path, clean_env):
    """hnl.trackml.events.Embedder.__call__: an embedding network on an overlay event only under a grant of kind
    model or plumb; nothing is checked (and no git is called) for TrackML ids."""
    from types import SimpleNamespace
    from hnl.trackml import events as E
    emb = E.Embedder.__new__(E.Embedder)                                     # no network: the check comes first
    with pytest.raises(AttributeError):
        emb(SimpleNamespace(event=21117))                                    # native: straight to the network
    with pytest.raises(PermissionError, match="no grant"):
        emb(SimpleNamespace(event=PILOT[0]))
    grant(tmp_path, clean_env, "pilot", "hs", "dry", PILOT)
    with pytest.raises(PermissionError, match="kind dry, not model or plumb"):
        emb(SimpleNamespace(event=PILOT[0]))
    grant(tmp_path, clean_env, "pilot", "hs", "plumb", PILOT)
    with pytest.raises(AttributeError):                                      # granted: on to the network
        emb(SimpleNamespace(event=PILOT[0]))


def test_development_clis_refuse_overlay_ids(tmp_path, clean_env):
    """g0, probe, support_gpu, chunked and g3check: an overlay id without a grant is refused before any model, table or
    score is read; a plumb grant is not enough for them (they compute numbers from the outputs; decision D6)."""
    from hnl.trackml import chunked as CH
    from hnl.trackml import g0 as G0
    from hnl.trackml import g3check as G3
    from hnl.trackml import probe as PR
    from hnl.trackml import support_gpu as SG
    t = str(tmp_path)
    store = tmp_path / "store" / "valset"
    store.mkdir(parents=True)
    (store / "event959021289.pyg.gz").write_bytes(b"")                     # a pilot event's file name only
    calls = [(G0.main, ["base-check", "--events", OV, "--ml-ckpt", t + "/none", "--r", "0.1", "--out", t + "/o"]),
             (G0.main, ["sectors", "--store", str(store), "--out", t + "/o"]),
             (PR.main, ["graphs", "--events", OV, "--out", t + "/o"]),
             (PR.main, ["memory", "--store", str(tmp_path / "store"), "--out", t + "/o"]),
             (PR.main, ["check-support", "--event", OV, "--out", t + "/o"]),
             (SG.main, ["compare", "--events", OV, "--ml-ckpt", t + "/none", "--r-wide", "0.1", "--model", t + "/m",
                        "--out", t + "/o"]),
             (CH.main, ["check", "--store", str(store), "--scores", t, "--ckpt", t + "/none"]),
             (G3.main, ["reference", "--scores", t, "--event", OV, "--out", t + "/o"]),
             (G3.main, ["options", "--scores", t, "--event", OV, "--out", t + "/o"]),
             (G3.main, ["anycheck", "--scores", t, "--event", OV, "--out", t + "/o"]),
             (G3.main, ["timing", "--scores", t, "--events", OV, "--out", t + "/o"]),
             (G3.main, ["profile", "--scores", t, "--events", OV, "--out", t + "/o"])]
    for f, argv in calls:
        _refused(f, *argv)
    assert not os.path.exists(t + "/o")
    grant(tmp_path, clean_env, "pilot", "full", "plumb", PILOT, out=t + "/o")
    for f, argv in [(G0.main, ["sectors", "--store", str(store), "--out", t + "/o"]),
                    (CH.main, ["check", "--store", str(store), "--scores", t, "--ckpt", t + "/none"]),
                    (PR.main, ["memory", "--store", str(tmp_path / "store"), "--out", t + "/o"])]:
        with pytest.raises(PermissionError, match="kind plumb, not model"):
            f(argv)


@needs_split
def test_v1_run_needs_a_grant_for_each_closure_grid(tmp_path, clean_env):
    """hnl.overlay.v1 run: a kind-eval grant of the closure set for every condition given, whose out holds the output,
    before any grid or table is read."""
    from hnl.overlay import blind as B
    from hnl.overlay import v1 as V
    root = tmp_path / "root"
    root.mkdir()
    ev = B.set_ids("closure")
    json.dump({"events": [{"overlay_event": e} for e in ev]}, open(root / "sidecar_closure.json", "w"))
    t = str(tmp_path)
    argv = ["run", "--root", str(root), "--closure-full", t + "/gf", "--closure-1gev", t + "/g1",
            "--out-json", t + "/res/v1.json"]
    _refused(V.main, *argv)
    grant(tmp_path, clean_env, "closure", "full", "eval", ev, out=t + "/res/v1.json")
    with pytest.raises(PermissionError, match="condition full, not 1gev"):
        V.main(argv)
    a = G.write("closure", "full", "eval", ev, t + "/grants", out=t + "/res/v1.json")
    b = G.write("closure", "1gev", "eval", ev, t + "/grants", out=t + "/elsewhere.json")
    clean_env.setenv(G.ENV, os.pathsep.join([a, b]))
    with pytest.raises(PermissionError, match="outside the grant's out"):
        V.main(argv)
    b = G.write("closure", "1gev", "eval", ev, t + "/grants", out=t + "/res/v1.json")
    clean_env.setenv(G.ENV, os.pathsep.join([a, b]))
    with pytest.raises(PermissionError, match="production overlay root|overlay events are read from"):
        V.main(argv)                                                         # granted: on to the grid checks
    assert not os.path.exists(t + "/res")


@needs_split
def test_tracks_grid_checks_the_launch_on_hval_and_htest(tmp_path, clean_env):
    """hnl.overlay.tracks grid on H-val (after the overlay_hash commit) and H-test also runs check_launch with
    HNL_H_LAUNCH, as hnl.overlay.h1 run does, before any score is read."""
    from hnl.overlay import blind as B
    from hnl.overlay import tracks as TR
    runs = str(tmp_path / "runs")
    clean_env.setattr(B, "unblinded", lambda repo=None, record=None: True)
    clean_env.setattr(B, "runs_root", lambda repo=None: runs)
    clean_env.setattr(G, "check", lambda *a, **k: None)                   # the grant passes; the launch is next
    clean_env.setenv("HNL_H_LAUNCH", TOK)
    with pytest.raises(PermissionError, match="hval: refused by its launch record.*not launched"):
        TR.grid("hval", "1gev", str(tmp_path / "scores"), str(tmp_path / "g"))
    launch_record(runs, "hval")
    clean_env.setattr(B, "results_expect", lambda repo=None: None)
    clean_env.setenv("HNL_H_LAUNCH", "cd" * 16)
    with pytest.raises(PermissionError, match="does not carry the token"):
        TR.grid("hval", "hs", str(tmp_path / "scores"), str(tmp_path / "g"))
    clean_env.setenv("HNL_H_LAUNCH", TOK)
    with pytest.raises(FileNotFoundError, match="no scored-edge files"):  # launched: on to the scores
        TR.grid("hval", "hs", str(tmp_path / "scores"), str(tmp_path / "g"))
    assert not os.path.exists(tmp_path / "g")


# ------------------------------------------------------------------------------------------------ job scripts
SBATCH = os.path.join(G.REPO, "slurm", "overlay")


def _lines(name):
    return [ln.strip() for ln in open(os.path.join(SBATCH, name)) if ln.strip() and not ln.lstrip().startswith("#")]


def test_job_scripts_verify_the_frozen_inputs_before_the_guard():
    """blind.sbatch, h1.sbatch and v1.sbatch: hnl.overlay.chain preflight (freeze verify) first, then the guard with
    --grant, then the entry point; h1.sbatch and v1.sbatch take the grids of the grid jobs (hnl.overlay.chain paths)."""
    import re
    for name, entry in (("blind.sbatch", r"-m hnl\.overlay\.blind evaluate"), ("h1.sbatch", r"-m hnl\.overlay\.h1 run"),
                        ("v1.sbatch", r"-m hnl\.overlay\.v1 run")):
        ls = _lines(name)
        idx = lambda pat: min([i for i, x in enumerate(ls) if re.search(pat, x)] or [10 ** 6])   # noqa: E731
        pre, guard, run = idx(r"-m hnl\.overlay\.chain preflight"), idx(r"guard_h\.sh .*--grant\)"), idx(entry)
        assert pre < guard < run < 10 ** 6, (name, pre, guard, run)
        assert "exit 1" in ls[pre]
    for name in ("h1.sbatch", "v1.sbatch"):
        text = open(os.path.join(SBATCH, name)).read()
        assert "hnl.overlay.chain paths --set \"$SET\" --cond \"$1\" --key grid" in text


def _production_grid(name, tmp_path, cond, given, have_grid=True):
    """The production_grid function of a job script, run with a stand-in for `python -m hnl.overlay.chain paths`."""
    import re
    text = open(os.path.join(SBATCH, name)).read()
    fn = re.search(r"^production_grid\(\) \{.*?^\}", text, re.S | re.M).group(0)
    d = tmp_path / f"grid_{cond}"
    d.mkdir(exist_ok=True)
    if have_grid:
        (d / "grid.json").write_text("{}")
    elif (d / "grid.json").exists():
        (d / "grid.json").unlink()
    stub = f'python() {{ echo "{tmp_path}/grid_$8"; }}'         # python -I -m hnl.overlay.chain paths --set S --cond C
    p = subprocess.run(["bash", "-c", f'set -euo pipefail; {stub}\n{fn}\nSET=hval; X=$(production_grid {cond} '
                                      f'"{given}"); echo "GRID=$X"'], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def test_h1_and_v1_jobs_take_only_the_grid_jobs_grids(tmp_path):
    for name in ("h1.sbatch", "v1.sbatch"):
        code, out, err = _production_grid(name, tmp_path, "hs", "")
        assert code == 0 and out == f"GRID={tmp_path}/grid_hs", (out, err)
        code, out, err = _production_grid(name, tmp_path, "hs", f"{tmp_path}/grid_hs/")
        assert code == 0 and out == f"GRID={tmp_path}/grid_hs"
        code, out, err = _production_grid(name, tmp_path, "hs", f"{tmp_path}/elsewhere")
        assert code != 0 and "reads the grid of slurm/overlay/grid.sbatch" in err and "GRID=" not in out
        code, out, err = _production_grid(name, tmp_path, "1gev", "", have_grid=False)
        assert code != 0 and "no grid in 1gev" in err


CPU_EXCLUDE = ("submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,"
               "submit67,submit68,submit71,submit72,submit73")


def test_overlay_job_scripts_use_the_express_partition_only():
    """The production job scripts run torch or Polars code, which runs only on the express CPU node or the GPU node
    types: each asks for submit-gpu-express alone (no fallback partition); the CPU jobs exclude the A30 and GTX 1080
    nodes, and the GPU job asks for an A30, V100 or RTX 6000."""
    for name in ("grid.sbatch", "h1.sbatch", "blind.sbatch", "v1.sbatch", "chain.sbatch", "chain_score.sbatch"):
        heads = [ln.strip() for ln in open(os.path.join(SBATCH, name)) if ln.startswith("#SBATCH")]
        parts = [h for h in heads if h.startswith(("#SBATCH -p ", "#SBATCH --partition"))]
        assert parts == ["#SBATCH -p submit-gpu-express"], (name, parts)
        if name == "chain.sbatch":
            assert "#SBATCH --gres=gpu:1" in heads
            assert "#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000" in heads
        else:
            assert f"#SBATCH --exclude={CPU_EXCLUDE}" in heads, name
            assert not [h for h in heads if "--gres" in h], name


def _decisive(env_vars):
    """The decisive function of slurm/overlay/h1.sbatch, run with a stand-in for `python -m hnl.overlay.h1 out-path`."""
    import re
    text = open(os.path.join(SBATCH, "h1.sbatch")).read()
    fn = re.search(r"^decisive\(\) \{.*?^\}", text, re.S | re.M).group(0)
    stub = 'python() { echo "/repo/results/pileup_hp/h1_${6}_${8}_full-1gev-hs_v1.json"; }'
    env = {k: v for k, v in os.environ.items() if k not in ("OUT", "RESTRICTION", "FLAGS", "LOST_SHARE", "ROOT",
                                                            "RESTRICTION_RECORD")}
    env.update(env_vars)
    script = f'set -euo pipefail; {stub}\n{fn}\nSET=htest; WPN=WP-S; decisive; echo "OUT=$OUT"'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout + p.stderr).strip()


def test_h1_job_fixes_the_decisive_output_and_takes_no_free_inputs():
    """slurm/overlay/h1.sbatch on hval and htest: OUT is the fixed path of `hnl.overlay.h1 out-path` (another OUT is
    refused), RESTRICTION_RECORD is required, and RESTRICTION, FLAGS and LOST_SHARE are refused (decision D7)."""
    fixed = "/repo/results/pileup_hp/h1_htest_WP-S_full-1gev-hs_v1.json"
    rec = {"RESTRICTION_RECORD": "results/pileup_hp/h_t1_restriction_v1.json"}
    code, out = _decisive(rec)
    assert code == 0 and out.endswith(f"OUT={fixed}"), out
    code, out = _decisive({**rec, "OUT": fixed})
    assert code == 0 and out.endswith(f"OUT={fixed}"), out
    for extra, msg in (({"OUT": "/repo/results/pileup_hp/other.json"}, f"writes {fixed} only"),
                       ({"RESTRICTION": "central"}, "come from RESTRICTION_RECORD"), ({"FLAGS": "V2"}, "come from"),
                       ({"LOST_SHARE": "0.03"}, "come from"), ({"ROOT": "/tmp/x"}, "pilot dry runs only"),
                       ({"FLAGS": ""}, "come from"), ({"LOST_SHARE": "0"}, "come from"),
                       ({"RESTRICTION": ""}, "come from")):                     # set at all, even empty or zero
        code, out = _decisive({**rec, **extra})
        assert code != 0 and msg in out and "OUT=" not in out, (extra, out)
    code, out = _decisive({})
    assert code != 0 and "needs RESTRICTION_RECORD" in out
