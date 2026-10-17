"""Blinding of the H-val overlay (hnl.overlay.blind, slurm/overlay/guard_h.sh; addendum H section 10).

Synthetic events for the blinded counts, scratch directories for the seal, copies of the repository for the guard, and
the pilot set (k = 9) with truth-made placeholder scores for the whole chain. No model runs and no model output of an
overlay event is read."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import blind as B
from hnl.overlay import grant as GR

REPO = B.REPO
GUARD = os.path.join(REPO, "slurm", "overlay", "guard_h.sh")


def pid_of(vertex, particle):
    return (vertex << 52) | (particle << 36)


A, Bh = pid_of(1, 5), pid_of(1, 6)            # host particles
G, H, J = pid_of(1001, 1), pid_of(1009, 2), pid_of(1002, 3)   # gun particles (isolated 1001/1002, dense 1009)


def synthetic(variant="base"):
    """Hits: A x6, B x5, G x6, H x2, J x3, noise x5. Tracks:
    T0 = A x6 + G x2 (host majority), T1 = G x4 + B x1 (gun majority), T2 = B x4 + H x2 + noise (host majority),
    T3 = J x3 + noise x3 (exactly half gun: removed), T4 = one noise hit. Variant "short": T0 = A x4 + G x3 + noise x1
    (host majority, but A holds 4 of its 8 hits), T1 = G x3 + B x1, no T4."""
    pid = np.array([A] * 6 + [Bh] * 5 + [G] * 6 + [H] * 2 + [J] * 3 + [0] * 5, dtype=np.int64)
    ia, ib, ig, ih, ij, inz = (np.arange(0, 6), np.arange(6, 11), np.arange(11, 17), np.arange(17, 19),
                               np.arange(19, 22), np.arange(22, 27))
    lab = np.full(len(pid), -1, dtype=np.int64)
    lab[ia] = 0
    lab[ig[:2]] = 0
    lab[ig[2:]] = 1
    lab[ib[0]] = 1
    lab[ib[1:]] = 2
    lab[ih] = 2
    lab[inz[0]] = 2
    lab[ij] = 3
    lab[inz[1:4]] = 3
    lab[inz[4]] = 4
    if variant == "short":
        lab[ia[4:]] = -1
        lab[ig[2]] = 0
        lab[inz[4]] = 0
    parts = pd.DataFrame({"particle_id": [A, Bh, G, H, J], "pt": [2.0, 1.5, 3.0, 2.0, 0.5],
                          "eta": [0.1, -0.5, 1.0, 0.3, 2.0], "n_hits": [6, 5, 6, 2, 3],
                          "is_target": [True, True, True, False, False],
                          "is_prompt_target": [True, True, False, False, False]})
    w = np.where(B.is_gun(pid) | (pid == 0), 0.0, 1.0)
    return pid, lab, parts, w / w.sum()


def test_constants_agree_with_the_production_modules():
    from hnl.overlay import writer as W
    from hnl.trackml import split as SP
    assert B.GUN_VERTEX == W.GUN_VERTEX and B.EXT == SP.EXT and B.DERIVED_OVERLAY == W.DERIVED_OVERLAY
    assert B.OVERLAY_FIRST_ID == W.OVERLAY_FIRST_ID
    text = open(GUARD).read()
    assert f"RUNS_PROD={B.RUNS_H} " in text and f"DERIVED_PROD={B.DERIVED_OVERLAY}\n" in text
    assert 'if [ "$REPO" = "$PROD" ]' in text and "PROD=" + REPO in text   # no root override in the repository itself
    assert B.PROD_REPO == REPO and B.runs_root() == B.RUNS_H and GR.REPO == REPO
    assert B.sealed_dir("hval") == os.path.join(B.RUNS_H, "sealed", "hval_full")


def test_gun_field_selects_1001_to_1010_only():
    v = np.array([0, 1, 226, 1000, 1001, 1005, 1010, 1011, 1024, 2024], dtype=np.int64)
    assert B.is_gun(v << 52).tolist() == [False, False, False, False, True, True, True, False, False, False]


def test_blind_labels_remove_tracks_with_half_or_more_gun_hits():
    pid, lab, parts, _ = synthetic()
    out = B.blind_labels(lab, B.is_gun(pid))
    assert set(np.unique(out[out >= 0])) == {0, 2, 4}          # T1 (4 of 5 gun) and T3 (3 of 6) removed
    assert (out[lab == 1] == -1).all() and (out[lab == 3] == -1).all()
    assert (out[np.isin(lab, (0, 2, 4))] == lab[np.isin(lab, (0, 2, 4))]).all()
    p, l2, hp = B.blind_inputs(pid, lab, parts)
    assert not B.is_gun(p).any() and (p[B.is_gun(pid)] == 0).all() and (p[~B.is_gun(pid)] == pid[~B.is_gun(pid)]).all()
    assert sorted(hp.particle_id) == sorted([A, Bh]) and np.array_equal(l2, out)


def test_blind_counts_see_no_gun_particle_or_gun_track():
    from hnl.trackml.evaluate import event_counts
    pid, lab, parts, w = synthetic()
    hid = np.arange(len(pid)) + 1
    c, rows = B.blind_counts(pid, lab, parts, hid, w)
    assert sorted(rows.particle_id) == sorted([A, Bh]) and rows.dm.all()
    assert c["n_target"] == 2 and c["m_target"] == 2 and c["n_prompt_target"] == 2 and c["m_prompt_target"] == 2
    assert c["n_tracks"] == 3 and c["n_tracks_ge5"] == 2 and c["n_fake_ge5"] == 0        # T0, T2, T4
    assert c["dup_sum_ge3_all"] == 0 and c["n_cand_ge3"] == 2
    # unblinded, the gun target G is a matched target and T1 a track: both are absent from the blinded numbers
    cu, ru = event_counts(pid, lab, parts, hid, w)
    assert G in set(ru.particle_id) and cu["n_target"] == 3 and cu["n_tracks"] == 5
    # zeroing the gun ids changes nothing once the gun-majority tracks are gone; the score is the unblinded one
    lab_b = B.blind_labels(lab, B.is_gun(pid))
    ck, rk = event_counts(pid, lab_b, B.host_particles(parts), hid, w)
    assert ck == c and rk.equals(rows)
    assert c["trackml_score"] == pytest.approx(cu["trackml_score"], abs=1e-15)


def test_gun_hits_still_count_in_host_track_size():
    pid, lab, parts, w = synthetic("short")       # T0 = A x4 + G x3 + noise: 4 of 8 hits is not a majority
    c, rows = B.blind_counts(pid, lab, parts)
    assert not rows.set_index("particle_id").dm[A] and rows.set_index("particle_id").dm[Bh]
    assert c["n_tracks"] == 2 and c["n_tracks_ge5"] == 2 and c["n_fake_ge5"] == 1   # T0: 5 host hits of 8, a fake
    # with the gun hits dropped from the arrays instead, A would be matched (4 of 5): the blinding keeps them
    keep = ~B.is_gun(pid)
    c2, r2 = B.blind_counts(pid[keep], lab[keep], parts)
    assert r2.set_index("particle_id").dm[A]


def test_v2_counts_only_host_majority_tracks_as_fakes():
    pid, lab, parts, _ = synthetic()
    lab = lab.copy()
    lab[B.is_gun(pid) & (lab == 1)] = 5           # G's four hits alone: a pure gun track, not counted
    c, _ = B.blind_counts(pid, lab, parts)
    assert c["n_tracks"] == 4 and c["n_fake_ge5"] == 0           # T0, T1 (now B's one hit), T2, T4
    assert not any("gun" in k for k in c)


# ------------------------------------------------------------------------------------------------ sets and ids
needs_split = pytest.mark.skipif(not os.path.exists(B.SPLIT_H), reason="needs configs/trackml/split_h_v1.json")


@needs_split
def test_set_ids_and_check_ids():
    sp = B.load_split()
    hv = B.set_ids("hval", sp)
    assert len(hv) == 150 and hv[0] == 50_000_000 + sp["hval"][0]
    assert B.set_ids("pilot", sp) == [59_000_000 + h for h in (21289, 21332, 21610, 21753, 21835)]
    assert B.check_ids("hval", hv[:3], sp) == []
    assert B.check_ids("hval", [hv[0] + 1_000_000], sp)                  # a closure id
    assert B.check_ids("pilot", [hv[0]], sp) and B.check_ids("hval", [], sp) and B.check_ids("hval", hv[:1] * 2, sp)
    pre = {**sp, "htest": None, "htest_spares": None, "htest_dropped": None}
    assert "H-test" in B.check_ids("htest", [52_021_117], pre)[0]         # no list before H-T1
    fake = {**sp, "htest": [300001, 300002], "htest_spares": [300003, 300004], "htest_dropped": None}
    assert B.check_ids("htest", [52_300_001, 52_300_002], fake) == []
    assert B.check_ids("htest", [52_300_003], fake)                        # a spare or any other part03 id
    # a dropped host is replaced by the next spare: that spare passes, the dropped host and later spares do not
    drop = {**fake, "htest_dropped": [{"host": 300002, "reason": "x", "replacement": 300003}]}
    assert B.set_hosts("htest", drop) == [300001, 300003]
    assert B.check_ids("htest", [52_300_001, 52_300_003], drop) == []
    assert B.check_ids("htest", [52_300_002], drop) and B.check_ids("htest", [52_300_004], drop)
    assert B.check_ids("htest", [52_300_001], {**fake, "htest_dropped": [{"host": 300002, "reason": "x",
                                                                           "replacement": 300004}]})
    assert B.parse_ids("1,2:3 4") == [1, 2, 3, 4]


# ------------------------------------------------------------------------------------------------ sealed directory
def make_dir(d):
    os.makedirs(os.path.join(d, "scores"))
    for rel in ("scores/scores_0.parquet", "scores/scores_0.parquet.json", "per_event.parquet", "per_target.parquet",
                "configs.json", "blind.json", "tracks/b000of001.parquet"):
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        with open(os.path.join(d, rel), "w") as fh:
            fh.write(rel)
    return d


def test_seal_verify_and_tamper(tmp_path):
    d = make_dir(str(tmp_path / "sealed"))
    with pytest.raises(FileNotFoundError):
        B.seal(str(tmp_path))                                              # nothing to seal there
    rec = B.seal(d)
    try:
        assert B.is_sealed(d) and rec["n_files"] == 7 and B.verify(d) == []
        assert B.verify(d, rec["manifest_sha256"][:8]) == [] and B.verify(d, "0" * 8)
        assert not os.access(os.path.join(d, "per_event.parquet"), os.W_OK) or os.geteuid() == 0
        with pytest.raises(PermissionError):
            B.seal(d)
        with pytest.raises(PermissionError):
            B.merge(d)
        B.unseal_permissions(d)
        with open(os.path.join(d, "configs.json"), "a") as fh:
            fh.write("x")
        open(os.path.join(d, "tracks", "extra.parquet"), "w").close()
        os.remove(os.path.join(d, "blind.json"))
        bad = B.verify(d)
        assert any(b.startswith("changed after sealing: configs.json") for b in bad)
        assert "added after sealing: tracks/extra.parquet" in bad and "missing: blind.json" in bad
        assert B.verify(str(tmp_path)) and "not sealed" in B.verify(str(tmp_path))[0]
    finally:
        B.unseal_permissions(d)


@needs_split
def test_evaluate_refuses_other_sets_roots_and_sealed_dirs(tmp_path):
    with pytest.raises(PermissionError):
        B.evaluate("hval", str(tmp_path))                 # raw H-val outputs go to the production sealed directory only
    with pytest.raises(PermissionError):
        B.evaluate("closure", str(tmp_path))
    with pytest.raises(PermissionError):
        B.evaluate("pilot", os.path.join(B.RUNS_H, "sealed", "pilot_full"))
    with pytest.raises(PermissionError):
        B.dry_scores("hval", str(tmp_path))
    with pytest.raises(PermissionError):
        B.dry_scores("pilot", os.path.join(B.RUNS_H, "x"))
    d = make_dir(str(tmp_path / "s"))
    B.seal(d)
    try:
        with pytest.raises(PermissionError):
            B.evaluate("pilot", d)
        with pytest.raises(PermissionError):
            B.v2(d, os.path.join(d, "v2.json"))
    finally:
        B.unseal_permissions(d)


def test_placeholder_provenance_rules(tmp_path):
    s = tmp_path / "scores"
    s.mkdir()
    (s / "scores_1.parquet").write_bytes(b"")
    side = {"events": {"59021289": 10}, "condition": "full", "placeholder": "truth", "ckpt_sha256": None}
    (s / "scores_1.parquet.json").write_text(json.dumps(side))
    assert B.scores_provenance(str(s), 9, [59021289])["placeholder"]
    with pytest.raises(KeyError):
        B.scores_provenance(str(s), 9, [59021332])
    with pytest.raises(ValueError):
        B.scores_provenance(str(s), 0, [59021289])                       # placeholders never pass for k = 0
    (s / "scores_1.parquet.json").write_text(json.dumps({**side, "placeholder": None, "ckpt_sha256": "ab"}))
    with pytest.raises(PermissionError):
        B.scores_provenance(str(s), 9, [59021289])                       # k = 9: no model output at all
    sel = json.load(open(B.SELECTION))["inputs"]["classifier_checkpoint"]["sha256"]
    (s / "scores_1.parquet.json").write_text(json.dumps({**side, "placeholder": None, "ckpt_sha256": sel}))
    assert B.scores_provenance(str(s), 0, [59021289])["ckpt_sha256"] == sel


# ------------------------------------------------------------------------------------------------ repository copies
ENV_GIT = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, env={**os.environ, **ENV_GIT})


def head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def guard_code_list() -> list[str]:
    """The CODE list of slurm/overlay/guard_h.sh (every file it requires committed, and pins at a launch)."""
    import re
    text = open(GUARD).read()
    m = re.search(r'^CODE="([^"]*)"', text, re.M)
    return [f.replace("$SPLIT", "configs/trackml/split_h_v1.json") for f in m.group(1).split()]


GUARD_FILES = guard_code_list() + ["docs/preregistration_pileup.md", "docs/feedback/test_use_ledger.md"]


def test_guard_requires_the_addendum_h_evaluation_code():
    code = guard_code_list()
    for f in ("src/hnl/overlay/blind.py", "src/hnl/overlay/grant.py", "src/hnl/overlay/h1.py",
              "src/hnl/overlay/hstats.py", "src/hnl/overlay/v1.py", "src/hnl/overlay/conditions.py",
              "src/hnl/overlay/tracks.py", "src/hnl/overlay/split_h.py", "slurm/overlay/h1.sbatch",
              "slurm/overlay/blind.sbatch", "configs/trackml/split_h_v1.json", "configs/trackml/selection_v1.json"):
        assert f in code and os.path.exists(os.path.join(REPO, f)), f
    # the launch pins cover the evaluation path that the CODE list missed (review R4): the modules the entry points
    # import, directly or not
    pins = B.import_closure()
    for f in ("src/hnl/trackml/freeze.py", "src/hnl/trackml/convert.py", "src/hnl/trackml/config.py",
              "src/hnl/trackml/seqmerge.py", "src/hnl/trackbuild/builders.py",
              "src/hnl/trackbuild/learned_merge.py", "src/hnl/trackbuild/mutex_ws.py",
              "src/hnl/trackbuild/merge_fragments.py", "src/hnl/graphs/helix_union.py",
              "src/hnl/eval/trackbuild_val.py", "src/hnl/overlay/closure.py", "src/hnl/overlay/writer.py",
              "src/hnl/overlay/truth_checks.py", "src/hnl/overlay/digitise.py", "src/hnl/overlay/grant.py",
              "src/hnl/overlay/h1.py", "src/hnl/overlay/hstats.py", "src/hnl/__init__.py"):
        assert f in pins, f
    for f in B.R15_FILES[3:]:
        assert f in pins


@pytest.fixture
def repo_copy(tmp_path):
    r = str(tmp_path / "repo")
    for f in GUARD_FILES:
        os.makedirs(os.path.dirname(os.path.join(r, f)), exist_ok=True)
        if os.path.exists(os.path.join(REPO, f)):
            shutil.copy(os.path.join(REPO, f), os.path.join(r, f))
        else:                                       # the addendum before its commit: any text stands in
            open(os.path.join(r, f), "w").write("draft\n")
    subprocess.run(["git", "init", "-q", r], check=True)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "copy")
    runs, derived = str(tmp_path / "runs"), str(tmp_path / "derived")
    os.makedirs(derived)
    env = {**os.environ, "HNL_H_RUNS": runs, "HNL_H_DERIVED": derived,
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
    env.pop("HNL_H_LAUNCH", None)
    yield {"repo": r, "runs": runs, "derived": derived, "env": env, "tmp": str(tmp_path)}
    for d in (os.path.join(runs, "sealed", "hval_full"),):
        if os.path.exists(d):
            B.unseal_permissions(d)


def guard3(rc, *args, token=None):
    """(exit code, stdout, stderr) of the guard in the repository copy."""
    env = dict(rc["env"])
    if token:
        env["HNL_H_LAUNCH"] = token
    p = subprocess.run(["bash", os.path.join(rc["repo"], "slurm", "overlay", "guard_h.sh"), *args],
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def guard(rc, *args, token=None):
    code, out, err = guard3(rc, *args, token=token)
    return code, (out + err).strip()


def launch(rc, set_name):
    """The launch; stdout holds exactly the token line."""
    code, out, err = guard3(rc, "--launch", set_name)
    assert code == 0, err
    assert out.count("\n") == 1 and out.startswith("HNL_H_LAUNCH="), out
    return out.strip().split("=", 1)[1]


def ids(name, sp=None):
    return ",".join(str(i) for i in B.set_ids(name, sp))


def commit_file(rc, rel, text, msg="edit"):
    p = os.path.join(rc["repo"], rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)
    git(rc["repo"], "add", rel)
    git(rc["repo"], "commit", "-qm", msg)


def commit_results(rc, sha, extra=""):
    commit_file(rc, "RESULTS.md", f"# Results\n\n{extra}{B.RESULTS_PREFIX} {sha}\n", "results")


@needs_split
def test_guard_pilot_closure_and_code_state(repo_copy):
    rc = repo_copy
    scratch = os.path.join(rc["tmp"], "dry")
    ok = lambda *a: guard(rc, *a)[0] == 0                                                             # noqa: E731
    assert ok("--set", "pilot", "--cond", "full", "--kind", "dry", "--ids", ids("pilot"), "--out", scratch)
    assert ok("--set", "pilot", "--cond", "hs", "--kind", "plumb", "--ids", ids("pilot"), "--out", scratch)
    assert not ok("--set", "pilot", "--cond", "full", "--kind", "model", "--ids", ids("pilot"), "--out", scratch)
    assert not ok("--set", "pilot", "--cond", "full", "--kind", "eval", "--ids", ids("pilot"), "--out", scratch)
    for kind in ("dry", "plumb"):
        assert not ok("--set", "pilot", "--cond", "full", "--kind", kind, "--ids", ids("pilot"),
                      "--out", os.path.join(B.RUNS_H, "sealed", "pilot_full"))
        assert not ok("--set", "pilot", "--cond", "full", "--kind", kind, "--ids", ids("pilot"),
                      "--out", os.path.join(rc["runs"], "plumb"))
        assert not ok("--set", "pilot", "--cond", "full", "--kind", kind, "--ids", ids("pilot"),
                      "--out", os.path.join(B.DERIVED_OVERLAY, "x"))
        assert not ok("--set", "pilot", "--cond", "full", "--kind", kind, "--ids", ids("pilot"))
    assert not ok("--set", "pilot", "--cond", "full", "--kind", "dry", "--ids", ids("hval")[:8], "--out", scratch)
    for cond, want in (("full", True), ("1gev", True), ("hs", False)):
        assert ok("--set", "closure", "--cond", cond, "--kind", "model", "--ids", ids("closure")) == want
        assert ok("--set", "closure", "--cond", cond, "--kind", "eval", "--ids", ids("closure")) == want
    assert not ok("--set", "closure", "--cond", "full", "--kind", "blind", "--ids", ids("closure"))
    assert not ok("--set", "closure", "--cond", "full", "--kind", "plumb", "--ids", ids("closure"))
    assert not ok("--set", "closure", "--cond", "full", "--kind", "model", "--ids", ids("hval"))
    assert not ok("--set", "nope", "--cond", "full", "--kind", "model", "--ids", "1")
    # an uncommitted change to the blinding code or the guard stops every job; the addendum stops model jobs
    with open(os.path.join(rc["repo"], "src/hnl/overlay/blind.py"), "a") as fh:
        fh.write("# changed\n")
    rcode, msg = guard(rc, "--set", "closure", "--cond", "full", "--kind", "model", "--ids", ids("closure"))
    assert rcode and "uncommitted changes under src, slurm or configs" in msg and "src/hnl/overlay/blind.py" in msg
    assert not ok("--set", "pilot", "--cond", "full", "--kind", "dry", "--ids", ids("pilot"), "--out", scratch)
    git(rc["repo"], "checkout", "--", "src/hnl/overlay/blind.py")
    with open(os.path.join(rc["repo"], "docs/preregistration_pileup.md"), "a") as fh:
        fh.write("changed\n")
    assert not ok("--set", "closure", "--cond", "full", "--kind", "model", "--ids", ids("closure"))
    assert ok("--set", "pilot", "--cond", "full", "--kind", "dry", "--ids", ids("pilot"), "--out", scratch)
    assert ok("--set", "pilot", "--cond", "full", "--kind", "plumb", "--ids", ids("pilot"), "--out", scratch)


@needs_split
def test_guard_refuses_any_uncommitted_code(repo_copy):
    """Decision D2: any uncommitted change to a tracked file under src, slurm or configs, and any untracked .py file
    under src, stops every job, not only changes to the files of the CODE list."""
    rc = repo_copy
    r = rc["repo"]
    run = lambda: guard(rc, "--set", "closure", "--cond", "full", "--kind", "model", "--ids", ids("closure"))  # noqa
    assert run()[0] == 0
    commit_file(rc, "src/hnl/trackbuild/other.py", "x = 1\n")            # a module outside the CODE list
    assert run()[0] == 0
    with open(os.path.join(r, "src/hnl/trackbuild/other.py"), "a") as fh:
        fh.write("x = 2\n")
    code, msg = run()
    assert code and "uncommitted changes under src, slurm or configs" in msg and "src/hnl/trackbuild/other.py" in msg
    git(r, "add", "src/hnl/trackbuild/other.py")                            # staged is not committed
    assert run()[0] != 0
    git(r, "commit", "-qm", "other")
    assert run()[0] == 0
    open(os.path.join(r, "src/hnl/overlay/new_module.py"), "w").write("y = 1\n")
    code, msg = run()
    assert code and "untracked python files under src: src/hnl/overlay/new_module.py" in msg
    code, msg = guard(rc, "--set", "pilot", "--cond", "full", "--kind", "dry", "--ids", ids("pilot"),
                      "--out", os.path.join(rc["tmp"], "dry"))
    assert code and "untracked python files" in msg
    os.remove(os.path.join(r, "src/hnl/overlay/new_module.py"))
    open(os.path.join(r, "src/notes.txt"), "w").write("notes\n")          # an untracked file that is not a module
    open(os.path.join(r, "scratch.py"), "w").write("z = 1\n")             # and a module outside src
    assert run()[0] == 0
    with open(os.path.join(r, "configs/trackml/selection_v1.json"), "a") as fh:
        fh.write("\n")
    code, msg = run()
    assert code and "configs/trackml/selection_v1.json" in msg
    git(r, "checkout", "--", "configs/trackml/selection_v1.json")
    with open(os.path.join(r, "slurm/overlay/blind.sbatch"), "a") as fh:
        fh.write("\n")
    assert run()[0] != 0 and guard(rc, "--launch", "hval")[0] != 0


@needs_split
def test_guard_grant_mode(repo_copy):
    """--grant: one line HNL_H_GRANT=<path> on stdout and nothing else; the grant's fields; where grants go; plumb."""
    rc = repo_copy
    scratch = os.path.join(rc["tmp"], "dry", "pilot_full")
    P = B.set_ids("pilot")
    for kind in ("dry", "plumb"):
        code, out, err = guard3(rc, "--set", "pilot", "--cond", "hs", "--kind", kind, "--ids", ids("pilot"),
                                "--out", scratch, "--grant")
        assert code == 0, err
        lines = out.splitlines()
        assert len(lines) == 1 and lines[0].startswith("HNL_H_GRANT=") and "allowed" in err
        p = lines[0].split("=", 1)[1]
        assert os.path.isabs(p) and os.path.dirname(p) == os.path.join(rc["tmp"], "dry", "grants")
        g = json.load(open(p))
        assert list(g) == list(GR.FIELDS)
        assert (g["set"], g["cond"], g["kind"], g["ids"], g["launch"]) == ("pilot", "hs", kind, P, None)
        assert g["ids_sha256"] == GR.ids_sha256(P) and g["commit"] == head(rc["repo"])
        assert g["code"] == GR.code_state(rc["repo"]) and g["code"]["files"]          # the copy's code (E1)
        assert g["code"]["files"]["src/hnl/overlay/grant.py"] == GR.head_listing(rc["repo"])["src/hnl/overlay/grant.py"]
        assert g["slurm_job_id"] == (os.environ.get("SLURM_JOB_ID") or None) and g["out"] == scratch
    code, out, err = guard3(rc, "--set", "closure", "--cond", "1gev", "--kind", "model", "--ids", ids("closure"),
                            "--grant")
    assert code == 0, err
    p = out.strip().split("=", 1)[1]
    assert os.path.dirname(p) == os.path.join(rc["runs"], "grants") and json.load(open(p))["out"] is None
    # a refusal writes no grant and prints nothing on stdout
    n = len(os.listdir(os.path.join(rc["runs"], "grants")))
    for args in (("--set", "closure", "--cond", "hs", "--kind", "model", "--ids", ids("closure")),
                 ("--set", "closure", "--cond", "full", "--kind", "plumb", "--ids", ids("closure")),
                 ("--set", "hval", "--cond", "full", "--kind", "plumb", "--ids", ids("hval"), "--out", scratch),
                 ("--set", "pilot", "--cond", "full", "--kind", "eval", "--ids", ids("pilot"), "--out", scratch)):
        code, out, err = guard3(rc, *args, "--grant")
        assert code != 0 and out == "" and "guard_h: " in err, (args, out, err)
    assert len(os.listdir(os.path.join(rc["runs"], "grants"))) == n


def sealed_hval(rc):
    d = make_dir(os.path.join(rc["runs"], "sealed", "hval_full"))
    return B.seal(d)


def commit_record(rc, oh, meta_sets=("hval",), meta_oh=None):
    p = os.path.join(rc["repo"], "configs/trackml/overlay_version_h_v1.json")
    with open(p, "w") as fh:
        json.dump({"overlay_hash": oh}, fh)
    git(rc["repo"], "add", "configs/trackml/overlay_version_h_v1.json")
    git(rc["repo"], "commit", "-qm", "record")
    for s in meta_sets:
        with open(os.path.join(rc["derived"], f"meta_{s}.json"), "w") as fh:
            json.dump({"overlay_hash": meta_oh or oh}, fh)


HT1 = "results/pileup_hp/h_t1_v1.json"
HT05 = "results/pileup_hp/h_t05_restriction_v1.json"                  # an H-val record of decision D7


def r15_entry(rc, hosts, oh, skip=(), wrong=(), files=None):
    """A ledger entry R15 naming H-test, the sha256 prefixes of the list and overlay_hash, and each file with the first
    12 hex digits of its current sha256 (skip: named without a prefix; wrong: named with another prefix)."""
    r = rc["repo"]
    sha = __import__("hashlib").sha256(",".join(str(h) for h in hosts).encode()).hexdigest()
    parts = []
    for f in files if files is not None else (*B.R15_FILES, HT1):
        h = B.sha256_file(os.path.join(r, f))
        if f in wrong:
            h = ("f" if h[0] != "f" else "e") + h[1:]
        parts.append(f if f in skip else f"{f} (sha256 {h[:12]}...)")
    return (f"\n### R15. Addendum H step H-T2: H-test\n\n- H-test ids (sha256 {sha[:8]}...), overlay_hash {oh[:8]}..., "
            + ", ".join(parts) + ".\n")


def write_ledger(rc, entry, base=None):
    led = os.path.join(rc["repo"], "docs/feedback/test_use_ledger.md")
    base = open(led).read().split("\n### R15. ")[0] if base is None else base
    with open(led, "w") as fh:
        fh.write(base + entry)


@needs_split
def test_guard_hval_before_and_after_the_overlay_hash_commit(repo_copy):
    rc = repo_copy
    sealed = os.path.join(rc["runs"], "sealed", "hval_full")
    hv = ids("hval")
    run = lambda kind, cond, out=None, tok=None: guard(rc, "--set", "hval", "--cond", cond, "--kind", kind,  # noqa
                                                       "--ids", hv, *(["--out", out] if out else []), token=tok)[0] == 0
    # before: C-full model into the sealed directory and hnl.overlay.blind only
    assert run("model", "full", os.path.join(sealed, "scores"))
    assert not run("model", "full", os.path.join(rc["tmp"], "elsewhere"))
    assert not run("model", "full")
    assert not run("model", "1gev", os.path.join(sealed, "scores")) and not run("model", "hs", sealed)
    assert not run("eval", "full", sealed) and not run("dry", "full", sealed) and not run("plumb", "full", sealed)
    assert run("blind", "full", sealed) and not run("blind", "full", rc["tmp"])
    assert not guard(rc, "--set", "hval", "--cond", "full", "--kind", "model", "--ids", ids("closure"),
                     "--out", sealed)[0] == 0
    assert guard(rc, "--launch", "hval")[0] != 0                         # no record yet
    rec = sealed_hval(rc)
    assert not run("model", "full", os.path.join(sealed, "scores"))       # sealed: no further model output
    assert run("blind", "full", sealed)
    # a committed record whose overlay_hash the stored H-val set does not carry stops everything
    commit_record(rc, "a" * 64, meta_oh="b" * 64)
    assert not run("blind", "full", sealed) and not run("eval", "full", sealed)
    assert guard(rc, "--launch", "hval")[0] != 0
    commit_record(rc, "c" * 64)
    assert not run("eval", "full") and not run("model", "1gev")           # not launched
    # D4: the launch verifies the sealed directory against the one line of RESULTS.md at HEAD
    code, msg = guard(rc, "--launch", "hval")
    assert code and "RESULTS.md" in msg
    commit_results(rc, "0" * 64)
    code, msg = guard(rc, "--launch", "hval")
    assert code and "does not verify" in msg
    commit_results(rc, rec["manifest_sha256"], extra=f"{B.RESULTS_PREFIX} {'1' * 64}\n")   # two lines: refused
    assert guard(rc, "--launch", "hval")[0] != 0
    with open(os.path.join(rc["repo"], "RESULTS.md"), "w") as fh:                       # right, but not committed
        fh.write(f"{B.RESULTS_PREFIX} {rec['manifest_sha256']}\n")
    assert guard(rc, "--launch", "hval")[0] != 0
    commit_results(rc, rec["manifest_sha256"])
    # decision D7: the H-val record (results/pileup_hp/h_t05_*) committed before the launch, which pins it
    code, msg = guard(rc, "--launch", "hval")
    assert code and "no H-val record of decision D7" in msg
    d7 = os.path.join(rc["repo"], HT05)
    os.makedirs(os.path.dirname(d7), exist_ok=True)
    json.dump({"restriction": None, "flags": [], "lost_share": None}, open(d7, "w"))
    code, msg = guard(rc, "--launch", "hval")
    assert code and "no H-val record of decision D7" in msg                           # untracked: not committed
    git(rc["repo"], "add", HT05)
    with open(d7, "a") as fh:
        fh.write("\n")
    code, msg = guard(rc, "--launch", "hval")
    assert code and f"{HT05} is not committed as is" in msg                           # staged, then changed
    git(rc["repo"], "checkout", "--", HT05)
    git(rc["repo"], "commit", "-qm", "H-val record of decision D7")
    tok = launch(rc, "hval")
    lrec = json.load(open(os.path.join(rc["runs"], "launch", "hval.json")))
    assert len(tok) == 32 and lrec["token"] == tok and lrec["hval_manifest_sha256"] == rec["manifest_sha256"]
    assert {"src/hnl/overlay/h1.py", "src/hnl/overlay/grant.py", "slurm/overlay/guard_h.sh",
            "docs/preregistration_pileup.md", "configs/trackml/overlay_version_h_v1.json"} <= set(lrec["pins"])
    assert lrec["pins"][HT05] == B.sha256_file(d7)
    assert lrec["pins"]["src/hnl/overlay/h1.py"] == B.sha256_file(os.path.join(rc["repo"], "src/hnl/overlay/h1.py"))
    assert guard(rc, "--launch", "hval")[0] != 0                         # once
    assert run("eval", "full", tok=tok) and run("eval", "1gev", tok=tok) and run("eval", "hs", tok=tok)
    assert run("model", "1gev", tok=tok) and run("model", "hs", tok=tok)
    assert not run("model", "full", os.path.join(sealed, "scores"), tok=tok)
    assert not run("eval", "full", tok="0" * 32)
    assert run("blind", "full", sealed)
    # the grant of a launched job carries the token
    code, out, err = guard3(rc, "--set", "hval", "--cond", "1gev", "--kind", "eval", "--ids", hv, "--grant", token=tok)
    assert code == 0 and json.load(open(out.strip().split("=", 1)[1]))["launch"] == tok
    # D3: a committed change to a pinned file after the launch stops the set's jobs (review R3)
    h1 = open(os.path.join(rc["repo"], "src/hnl/overlay/h1.py")).read()
    commit_file(rc, "src/hnl/overlay/h1.py", h1 + "# changed after the launch\n")
    code, msg = guard(rc, "--set", "hval", "--cond", "1gev", "--kind", "eval", "--ids", hv, token=tok)
    assert code and "src/hnl/overlay/h1.py changed since the launch of hval" in msg
    commit_file(rc, "src/hnl/overlay/h1.py", h1)                          # the launched content again
    assert run("eval", "1gev", tok=tok)
    commit_results(rc, rec["manifest_sha256"], extra="more text\n")       # the line itself unchanged
    assert run("eval", "1gev", tok=tok)
    commit_results(rc, "2" * 64)
    code, msg = guard(rc, "--set", "hval", "--cond", "1gev", "--kind", "eval", "--ids", hv, token=tok)
    assert code and "manifest sha256 line of RESULTS.md changed" in msg


@needs_split
def test_guard_htest_needs_r15_list_and_launch(repo_copy):
    rc = repo_copy
    r = rc["repo"]
    sp_path = os.path.join(r, "configs/trackml/split_h_v1.json")
    sp = json.load(open(sp_path))
    run = lambda tok=None, idl=None, kind="model": guard(rc, "--set", "htest", "--cond", "full", "--kind", kind,  # noqa
                                                         "--ids", idl or "52300001,52300002", token=tok)
    assert run()[0] != 0                                                # no list, no record, no R15
    sp["htest"], sp["htest_spares"] = [300001, 300002], [300003, 300004]
    json.dump(sp, open(sp_path, "w"), indent=1)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "H-T1 list")
    commit_record(rc, "d" * 64, meta_sets=("htest",))
    code, msg = run()
    assert code and "R15" in msg
    led = os.path.join(r, "docs/feedback/test_use_ledger.md")
    lr15 = lambda: B.ledger_r15(led, sp_path, os.path.join(r, "configs/trackml/overlay_version_h_v1.json"), r)  # noqa
    # D3: R15 must name every prefix, each against the current file; the H-T1 records must exist
    write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64, files=B.R15_FILES))
    bad, _ = lr15()
    assert bad == [f"no H-T1 record ({B.HT1_RECORDS}*) is committed"]
    commit_file(rc, HT1, json.dumps({"dropped": []}), "H-T1 record")
    bad, _ = lr15()
    assert bad == [f"R15 does not name {HT1}"]
    for f in (*B.R15_FILES, HT1):
        write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64, skip=(f,)))
        assert lr15()[0] == [f"R15 does not name the sha256 of {f} (now {B.sha256_file(os.path.join(r, f))[:8]}...)"]
        write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64, wrong=(f,)))
        assert len(lr15()[0]) == 1 and f in lr15()[0][0]
    write_ledger(rc, r15_entry(rc, [300001, 300002], "e" * 64))
    assert lr15()[0] == ["R15 does not name overlay_hash (dddddddd...)"]
    write_ledger(rc, r15_entry(rc, [300001, 300003], "d" * 64))
    assert "does not name the sha256 of the H-test list" in lr15()[0][0]
    write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64, skip=("src/hnl/overlay/h1.py",)))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 without the prefix of the H1 code")
    code, msg = run()
    assert code and "R15 does not name the sha256 of src/hnl/overlay/h1.py" in msg
    assert guard(rc, "--launch", "htest")[0] != 0
    write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64))
    assert lr15()[0] == []
    assert run()[0] != 0                                                # R15 not committed
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15")
    assert run()[0] != 0                                                # not launched
    tok = launch(rc, "htest")
    lrec = json.load(open(os.path.join(rc["runs"], "launch", "htest.json")))
    assert HT1 in lrec["pins"] and lrec["r15_sha256"] and "src/hnl/overlay/hstats.py" in lrec["pins"]
    assert guard(rc, "--launch", "htest")[0] != 0
    assert run(tok)[0] == 0 and run(tok, kind="eval")[0] == 0
    assert run(tok, kind="blind")[0] != 0 and run(tok, kind="plumb")[0] != 0
    assert run(tok, idl="52300003")[0] != 0                             # never a spare or another id
    assert run(tok, idl="50021117")[0] != 0
    for cond in ("1gev", "hs"):
        assert guard(rc, "--set", "htest", "--cond", cond, "--kind", "model", "--ids", "52300001", token=tok)[0] == 0
    code, out, err = guard3(rc, "--set", "htest", "--cond", "full", "--kind", "eval", "--ids", "52300001", "--grant",
                            token=tok)
    assert code == 0 and json.load(open(out.strip().split("=", 1)[1]))["launch"] == tok
    with open(os.path.join(r, "docs/preregistration_pileup.md"), "a") as fh:   # a file R15 names, changed
        fh.write("x\n")
    assert run(tok)[0] != 0
    git(r, "checkout", "--", "docs/preregistration_pileup.md")
    # D3: committed changes after the launch: a pinned file (refused by the pin), the H1 code (refused by R15 too),
    # and the R15 entry itself
    for f in ("src/hnl/overlay/tracks.py", "src/hnl/overlay/h1.py", "slurm/overlay/h1.sbatch"):
        text = open(os.path.join(r, f)).read()
        commit_file(rc, f, text + "\n# committed after the launch\n")
        code, msg = run(tok, kind="eval")
        assert code and (f"{f} changed since the launch of htest" in msg or f"sha256 of {f}" in msg), msg
        commit_file(rc, f, text)
        assert run(tok, kind="eval")[0] == 0
    base = open(led).read()
    write_ledger(rc, "", base=base + "\n(later note)\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 text changed")
    code, msg = run(tok)
    assert code and "ledger entry R15 changed since the launch" in msg
    write_ledger(rc, "", base=base)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 back")
    assert run(tok)[0] == 0
    # the ledger reader: what R15 names
    bad, named = lr15()
    assert bad == [] and "src/hnl/overlay/h1.py" in named and "configs/trackml/split_h_v1.json" in named


def _expect_refusals(rc, cases, table, stage, token=None):
    for name, args, want in cases:
        code, msg = guard(rc, *args, token=token)
        table.append({"stage": stage, "case": name, "args": " ".join(args), "exit": code,
                      "message": msg.splitlines()[-1] if msg else ""})
        assert code != 0, f"{stage}: {name} was allowed"
        assert want in msg, f"{stage}: {name}: {msg!r} lacks {want!r}"


@needs_split
def test_guard_refuses_each_forbidden_case(repo_copy):
    """Every forbidden case of addendum H section 10 against the guard, each with the reason it gives, through the
    stages of the protocol (before the overlay_hash record, H-val sealed, record committed and launched, H-test
    recorded with R15 and launched). HNL_GUARD_TABLE=<json> writes the table of cases and messages."""
    rc = repo_copy
    r = rc["repo"]
    scratch = os.path.join(rc["tmp"], "dry")
    sealed = os.path.join(rc["runs"], "sealed", "hval_full")
    P, C, V = ids("pilot"), ids("closure"), ids("hval")
    a = lambda st, cond, kind, idl, *x: ["--set", st, "--cond", cond, "--kind", kind, "--ids", idl, *x]   # noqa: E731
    table = []
    ok = lambda *x: guard(rc, *x)[0] == 0                                                             # noqa: E731
    assert ok(*a("pilot", "full", "dry", P, "--out", scratch)) and ok(*a("closure", "1gev", "model", C))
    assert ok(*a("pilot", "full", "plumb", P, "--out", scratch))
    assert ok(*a("hval", "full", "model", V, "--out", os.path.join(sealed, "scores")))
    _expect_refusals(rc, [
        ("pilot: a model run", a("pilot", "full", "model", P, "--out", scratch), "kinds dry and plumb only"),
        ("pilot: an evaluation of model outputs", a("pilot", "hs", "eval", P, "--out", scratch),
         "kinds dry and plumb only"),
        ("pilot: the blinded evaluation", a("pilot", "full", "blind", P, "--out", scratch), "kinds dry and plumb only"),
        ("pilot: dry without an output directory", a("pilot", "full", "dry", P), "kind dry needs --out"),
        ("pilot: plumb without an output directory", a("pilot", "full", "plumb", P), "kind plumb needs --out"),
        ("pilot: dry into the production runs root", a("pilot", "full", "dry", P, "--out",
                                                         os.path.join(B.RUNS_H, "x")), "production runs root"),
        ("pilot: plumb into the production runs root", a("pilot", "full", "plumb", P, "--out",
                                                           os.path.join(B.RUNS_H, "x")), "production runs root"),
        ("pilot: plumb into the overlay tables", a("pilot", "full", "plumb", P, "--out",
                                                     os.path.join(B.DERIVED_OVERLAY, "x")), "overlay tables"),
        ("pilot: ids of another set", a("pilot", "full", "dry", V, "--out", scratch), "ids outside set pilot"),
        ("any set: a condition other than full, 1gev, hs", a("pilot", "nonoise", "dry", P, "--out", scratch),
         "no condition 'nonoise'"),
        ("any set: an unknown kind", a("closure", "full", "train", C), "no kind 'train'"),
        ("any set: repeated ids", a("closure", "full", "model", C.split(",")[0] + "," + C.split(",")[0]),
         "ids outside set closure"),
        ("any set: no ids", ["--set", "closure", "--cond", "full", "--kind", "model"], "are required"),
        ("any set: an unknown set", a("htrain", "full", "model", "53021117"), "ids outside set htrain"),
        ("closure: C-hs", a("closure", "hs", "model", C), "C-full and C-1GeV only"),
        ("closure: kind dry", a("closure", "full", "dry", C), "takes kinds model and eval"),
        ("closure: kind plumb", a("closure", "full", "plumb", C), "takes kinds model and eval"),
        ("closure: kind blind", a("closure", "full", "blind", C), "takes kinds model and eval"),
        ("closure: H-val ids", a("closure", "full", "model", V), "ids outside set closure"),
        ("H-val before the record: C-1GeV model", a("hval", "1gev", "model", V, "--out", sealed),
         "takes the C-full model run and hnl.overlay.blind only"),
        ("H-val before the record: C-hs model", a("hval", "hs", "model", V, "--out", sealed),
         "takes the C-full model run and hnl.overlay.blind only"),
        ("H-val before the record: an evaluation", a("hval", "full", "eval", V, "--out", sealed),
         "takes the C-full model run and hnl.overlay.blind only"),
        ("H-val: kind dry", a("hval", "full", "dry", V, "--out", sealed), "are for the pilot set"),
        ("H-val: kind plumb", a("hval", "full", "plumb", V, "--out", sealed), "are for the pilot set"),
        ("H-val before the record: C-full outputs outside the sealed directory",
         a("hval", "full", "model", V, "--out", scratch), "go to " + sealed + " only"),
        ("H-val: C-full model without --out", a("hval", "full", "model", V), "--out (the sealed directory)"),
        ("H-val: the blinded evaluation outside the sealed directory", a("hval", "full", "blind", V, "--out", scratch),
         "kind blind works in"),
        ("H-val: closure ids", a("hval", "full", "model", C, "--out", sealed), "ids outside set hval"),
        ("H-test before H-T1: no list", a("htest", "full", "model", "52300001"), "ids outside set htest"),
        ("launch of H-val before the record", ["--launch", "hval"], "no committed overlay_hash record"),
        ("launch of another set", ["--launch", "closure"], "--launch takes hval or htest"),
    ], table, "before the overlay_hash record")
    # uncommitted code (decision D2: anything under src, slurm or configs) or addendum
    for f in ("src/hnl/overlay/h1.py", "src/hnl/overlay/tracks.py", "src/hnl/overlay/split_h.py",
              "src/hnl/overlay/grant.py", "slurm/overlay/h1.sbatch", "configs/trackml/split_h_v1.json"):
        with open(os.path.join(r, f), "a") as fh:
            fh.write("\n")
        _expect_refusals(rc, [(f"{f} changed, not committed", a("pilot", "full", "dry", P, "--out", scratch),
                               f"uncommitted changes under src, slurm or configs:  M {f}")], table, "uncommitted code")
        git(r, "checkout", "--", f)
    open(os.path.join(r, "src/hnl/overlay/extra.py"), "w").write("\n")
    _expect_refusals(rc, [("an untracked module under src", a("closure", "full", "model", C),
                           "untracked python files under src: src/hnl/overlay/extra.py")], table, "uncommitted code")
    os.remove(os.path.join(r, "src/hnl/overlay/extra.py"))
    with open(os.path.join(r, "docs/preregistration_pileup.md"), "a") as fh:
        fh.write("changed\n")
    _expect_refusals(rc, [("addendum changed: a model run", a("closure", "full", "model", C),
                           "preregistration_pileup.md is not committed as is")], table, "uncommitted addendum")
    assert ok(*a("pilot", "full", "dry", P, "--out", scratch))           # placeholder code tests need no addendum
    git(r, "checkout", "--", "docs/preregistration_pileup.md")
    # H-val sealed
    rec = sealed_hval(rc)
    _expect_refusals(rc, [("H-val sealed: a further C-full model output",
                           a("hval", "full", "model", V, "--out", os.path.join(sealed, "scores")), "is sealed")],
                     table, "H-val sealed")
    # a record whose overlay_hash the stored H-val set does not carry
    commit_record(rc, "a" * 64, meta_oh="b" * 64)
    _expect_refusals(rc, [("record committed, H-val meta of another version: blinded evaluation",
                           a("hval", "full", "blind", V, "--out", sealed), "does not carry the overlay_hash"),
                          ("the same: launch", ["--launch", "hval"], "no committed overlay_hash record")],
                     table, "record of another version")
    commit_record(rc, "c" * 64)
    _expect_refusals(rc, [("H-val after the record: C-full model again",
                           a("hval", "full", "model", V, "--out", os.path.join(sealed, "scores")), "not repeated"),
                          ("H-val after the record: C-1GeV model before the launch", a("hval", "1gev", "model", V),
                           "was not launched"),
                          ("H-val after the record: step-5 evaluation before the launch", a("hval", "full", "eval", V),
                           "was not launched"),
                          ("H-val after the record: kind blind outside the sealed directory",
                           a("hval", "full", "blind", V, "--out", scratch), "kind blind works in"),
                          ("H-val launch without the manifest line of RESULTS.md", ["--launch", "hval"],
                           "no committed RESULTS.md")],
                     table, "record committed")
    commit_results(rc, "f" * 64)
    _expect_refusals(rc, [("H-val launch: RESULTS.md records another manifest sha256", ["--launch", "hval"],
                           "does not verify against the manifest sha256 that RESULTS.md records")],
                     table, "record committed")
    commit_results(rc, rec["manifest_sha256"])
    _expect_refusals(rc, [("H-val launch without a committed H-val record of decision D7", ["--launch", "hval"],
                           "no H-val record of decision D7")], table, "record committed")
    commit_file(rc, HT05, json.dumps({"restriction": None, "flags": [], "lost_share": None}), "D7 record")
    tok = launch(rc, "hval")
    _expect_refusals(rc, [("H-val: a second launch", ["--launch", "hval"], "once only")], table, "H-val launched")
    _expect_refusals(rc, [("H-val launched: another token", a("hval", "1gev", "model", V), "does not carry the token")],
                     table, "H-val launched", token="0" * 32)
    _expect_refusals(rc, [("H-val launched: C-full model again", a("hval", "full", "model", V, "--out", sealed),
                           "not repeated")], table, "H-val launched", token=tok)
    assert guard(rc, *a("hval", "hs", "model", V), token=tok)[0] == 0
    hs_py = open(os.path.join(r, "src/hnl/overlay/hstats.py")).read()
    commit_file(rc, "src/hnl/overlay/hstats.py", hs_py + "# after the launch\n")
    _expect_refusals(rc, [("H-val launched: the H1 code changed and committed after the launch",
                           a("hval", "full", "eval", V), "src/hnl/overlay/hstats.py changed since the launch of hval")],
                     table, "H-val launched", token=tok)
    commit_file(rc, "src/hnl/overlay/hstats.py", hs_py)
    # H-test
    sp_path = os.path.join(r, "configs/trackml/split_h_v1.json")
    sp = json.load(open(sp_path))
    sp["htest"], sp["htest_spares"] = [300001, 300002], [300003, 300004, 300005]
    sp["htest_dropped"] = [{"host": 300002, "reason": "no hard-scatter vertex", "replacement": 300003}]
    json.dump(sp, open(sp_path, "w"), indent=1)
    t1 = os.path.join(r, HT1)                                             # an H-T1 record that R15 names
    os.makedirs(os.path.dirname(t1), exist_ok=True)              # results/pileup_hp holds the D7 record
    json.dump({"dropped": sp["htest_dropped"]}, open(t1, "w"))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "H-T1 list")
    commit_record(rc, "d" * 64, meta_sets=("hval", "htest"))
    H = "52300001,52300003"
    _expect_refusals(rc, [("H-test: no ledger entry R15", a("htest", "full", "model", H), "no ledger entry R15")],
                     table, "H-T1 recorded")
    write_ledger(rc, r15_entry(rc, [300001, 300002], "d" * 64))           # the list before the drop
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 with the list before the drop")
    _expect_refusals(rc, [("H-test: R15 names the sha256 of another list", a("htest", "full", "model", H),
                           "does not name the sha256 of the H-test list")], table, "R15 committed")
    write_ledger(rc, r15_entry(rc, [300001, 300003], "d" * 64, wrong=("docs/preregistration_pileup.md",)))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 with a wrong prefix")
    _expect_refusals(rc, [("H-test: R15 names a wrong sha256 prefix of the addendum", a("htest", "full", "model", H),
                           "does not name the sha256 of docs/preregistration_pileup.md")], table, "R15 committed")
    write_ledger(rc, r15_entry(rc, [300001, 300003], "d" * 64, skip=("src/hnl/overlay/hstats.py",)))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15 without a prefix")
    _expect_refusals(rc, [("H-test: R15 names the H1 code without its sha256 prefix", a("htest", "full", "model", H),
                           "does not name the sha256 of src/hnl/overlay/hstats.py")], table, "R15 committed")
    write_ledger(rc, r15_entry(rc, [300001, 300003], "d" * 64))
    _expect_refusals(rc, [("H-test: R15 not committed", a("htest", "full", "model", H), "is not committed as is")],
                     table, "R15 written")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "R15")
    _expect_refusals(rc, [("H-test: before the launch", a("htest", "full", "model", H), "was not launched")],
                     table, "R15 committed")
    tok = launch(rc, "htest")
    _expect_refusals(rc, [("H-test: a second launch", ["--launch", "htest"], "once only")], table, "H-test launched")
    with open(t1, "a") as fh:
        fh.write("\n")
    _expect_refusals(rc, [("H-test: an H-T1 record named in R15 changed", a("htest", "full", "eval", H),
                           f"R15 does not name the sha256 of {HT1}")], table, "H-test launched", token=tok)
    git(r, "checkout", "--", HT1)
    tr_py = open(os.path.join(r, "src/hnl/overlay/tracks.py")).read()
    commit_file(rc, "src/hnl/overlay/tracks.py", tr_py + "# after the launch\n")
    _expect_refusals(rc, [("H-test: a pinned file changed and committed after the launch",
                           a("htest", "full", "eval", H),
                           "src/hnl/overlay/tracks.py changed since the launch of htest")],
                     table, "H-test launched", token=tok)
    commit_file(rc, "src/hnl/overlay/tracks.py", tr_py)
    _expect_refusals(rc, [
        ("H-test: the dropped host", a("htest", "full", "model", "52300002"), "ids outside set htest"),
        ("H-test: a spare that replaces no dropped host", a("htest", "full", "model", "52300004"),
         "ids outside set htest"),
        ("H-test: kind blind", a("htest", "full", "blind", H), "takes kinds model and eval"),
        ("H-test: kind dry", a("htest", "full", "dry", H), "takes kinds model and eval"),
        ("H-test: kind plumb", a("htest", "full", "plumb", H), "takes kinds model and eval"),
        ("H-test: another token", a("htest", "full", "model", H), "does not carry the token"),
    ], table, "H-test launched", token="1" * 32)
    for cond in ("full", "1gev", "hs"):
        assert guard(rc, *a("htest", cond, "model", H), token=tok)[0] == 0
    _expect_refusals(rc, [("H-test launched: the dropped host", a("htest", "1gev", "eval", "52300002"),
                           "ids outside set htest")], table, "H-test launched", token=tok)
    out_path = os.environ.get("HNL_GUARD_TABLE")
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(table, fh, indent=1)
    assert len(table) >= 60


def test_write_once_and_consumed_marker(tmp_path, monkeypatch):
    """Decision D3: the decisive output is written once; each working point has its own marker. Crash-safe order: the
    complete output in a temporary file, then a pending record (exclusive), then the output placed (exclusive), then the
    consumed marker. A run that stops before its pending record leaves nothing and may be repeated; from the pending
    record on the working point is consumed: the next attempt records the marker and writes nothing, whether the output
    was placed or not, and whatever output path it names."""
    runs = str(tmp_path / "runs")
    m = B.consumed_path("htest", "WP-A", runs)
    assert m == os.path.join(runs, "launch", "htest.consumed.WP-A.json")
    assert B.pending_path(m) == os.path.join(runs, "launch", "htest.pending.WP-A.json")
    out = str(tmp_path / "res" / "h1_htest_WP-A_full-1gev-hs_v1.json")
    sha = lambda b: __import__("hashlib").sha256(b).hexdigest()                 # noqa: E731

    def run(crash):
        if crash:
            raise RuntimeError("stopped before the output")
        return B.write_once(out, '{"delta": -0.1}', m, {"set": "htest", "wp": "WP-A"})

    with pytest.raises(RuntimeError):
        run(True)
    assert not os.path.exists(m) and not os.path.exists(out) and not os.path.exists(B.pending_path(m))
    # stopped before the pending record (here: while the temporary file is synced): nothing left, may be repeated
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(RuntimeError("stopped while syncing")))
    with pytest.raises(RuntimeError, match="syncing"):
        run(False)
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert not os.path.exists(B.pending_path(m)) and not os.path.exists(out) and not os.path.exists(m)
    assert not [f for f in os.listdir(tmp_path / "res") if ".tmp" in f]
    rec = run(False)
    assert json.load(open(out)) == {"delta": -0.1} and json.load(open(m))["output"] == out
    assert rec["output_sha256"] == sha(b'{"delta": -0.1}')
    assert json.load(open(B.pending_path(m)))["output"] == out
    with pytest.raises(PermissionError, match="written once already"):
        run(False)
    assert json.load(open(out)) == {"delta": -0.1} and not [f for f in os.listdir(tmp_path / "res") if ".tmp" in f]
    B.write_once(str(tmp_path / "res" / "wps.json"), "{}", B.consumed_path("htest", "WP-S", runs), {})
    assert sorted(os.listdir(os.path.join(runs, "launch"))) == ["htest.consumed.WP-A.json", "htest.consumed.WP-S.json",
                                                                "htest.pending.WP-A.json", "htest.pending.WP-S.json"]
    # a crash after the pending record and before the output is placed: consumed, even for another output path
    m2, out2 = B.consumed_path("htest", "WP-R-P2", runs), str(tmp_path / "res" / "p2.json")
    real_link = os.link
    monkeypatch.setattr(os, "link", lambda *a: (_ for _ in ()).throw(RuntimeError("stopped before the link")))
    with pytest.raises(RuntimeError, match="before the link"):
        B.write_once(out2, '{"delta": 2}', m2, {"set": "htest", "wp": "WP-R-P2"})
    monkeypatch.setattr(os, "link", real_link)
    assert os.path.exists(B.pending_path(m2)) and not os.path.exists(out2) and not os.path.exists(m2)
    other = str(tmp_path / "res" / "elsewhere.json")
    with pytest.raises(PermissionError, match="began placing its output"):
        B.once_refusal(other, m2, {"set": "htest", "wp": "WP-R-P2"})           # another path: refused all the same
    rec = json.load(open(m2))
    assert rec["output"] == out2 and rec["output_present"] is False and rec["recovered"] and rec["pending"]
    with pytest.raises(PermissionError, match="written once already"):
        B.write_once(other, "{}", m2, {})
    assert not os.path.exists(other) and not os.path.exists(out2)
    # a crash after the output is placed and before the marker: the next attempt records the marker, writes nothing
    m3, out3 = B.consumed_path("htest", "WP-R-P3", runs), str(tmp_path / "res" / "p3.json")
    real = B._write_marker
    calls = []

    def marker_crash(path, rec):
        calls.append(path)
        if path == m3:
            raise RuntimeError("stopped before the marker")
        return real(path, rec)
    monkeypatch.setattr(B, "_write_marker", marker_crash)
    with pytest.raises(RuntimeError, match="before the marker"):
        B.write_once(out3, '{"delta": 1}', m3, {"set": "htest", "wp": "WP-R-P3"})
    monkeypatch.setattr(B, "_write_marker", real)
    assert calls == [B.pending_path(m3), m3]                                    # the pending record came first
    assert os.path.exists(out3) and not os.path.exists(m3)
    with pytest.raises(PermissionError, match="began placing its output"):
        B.once_refusal(str(tmp_path / "res" / "p3_other.json"), m3, {"set": "htest", "wp": "WP-R-P3"})
    rec = json.load(open(m3))
    assert rec["output"] == out3 and rec["recovered"] and rec["output_present"]
    assert json.load(open(out3)) == {"delta": 1}
    assert rec["output_sha256"] == sha(b'{"delta": 1}')
    with pytest.raises(PermissionError, match="written once already"):
        B.write_once(out3, '{"delta": 2}', m3, {})
    assert json.load(open(out3)) == {"delta": 1}
    # an output in place without pending record or marker (written by other means): refused, kept, consumed
    m4, out4 = B.consumed_path("hval", "WP-A", runs), str(tmp_path / "res" / "taken.json")
    open(out4, "w").write("x")
    with pytest.raises(PermissionError, match="stopped before its marker"):
        B.write_once(out4, "{}", m4, {})
    assert open(out4).read() == "x" and json.load(open(m4))["recovered"]
    assert not [f for f in os.listdir(tmp_path / "res") if ".tmp" in f]
    # an empty, cut-short or non-object pending record (a run stopped while writing it): consumed, refused with a
    # PermissionError that says so, the marker recorded, nothing written
    for i, text in enumerate(("", '{"output": "/x', "[1, 2]")):
        m5 = B.consumed_path("hval", f"WP-X{i}", runs)
        out5 = str(tmp_path / "res" / f"x{i}.json")
        open(B.pending_path(m5), "w").write(text)
        with pytest.raises(PermissionError, match="unreadable pending record .*counted as consumed"):
            B.once_refusal(out5, m5, {"set": "hval", "wp": f"WP-X{i}"})
        assert json.load(open(m5))["pending"]["unreadable"] and not os.path.exists(out5)
        with pytest.raises(PermissionError, match="written once already"):
            B.write_once(out5, "{}", m5, {})
        assert not os.path.exists(out5)


def test_seal_records_the_rule_of_its_directory(tmp_path):
    """SEALED.json carries the rule of what was sealed: the H-val rule for the sealed H-val directory, the dry rule for
    a pilot placeholder directory, the plumbing rule (decision D6) when hnl.overlay.chain.finish passes it."""
    for set_name, rule in (("hval", B.SEAL_RULES["hval"]), ("pilot", B.SEAL_RULES["dry"])):
        d = make_dir(str(tmp_path / set_name))
        json.dump({"set": set_name}, open(os.path.join(d, "blind.json"), "w"))
        assert B.seal(d)["rule"] == rule == json.load(open(os.path.join(d, B.SEALED_JSON)))["rule"]
        B.unseal_permissions(d)
    d = str(tmp_path / "plumb")
    os.makedirs(os.path.join(d, "scores"))
    open(os.path.join(d, "scores", "scores_00000.parquet"), "w").write("x")
    assert B.seal(d, required=(), rule=B.SEAL_RULES["plumb"])["rule"].startswith("decision D6")
    B.unseal_permissions(d)
    assert "hnl.overlay.blind" in B.SEAL_RULES["hval"] and "placeholder" in B.SEAL_RULES["dry"]
    assert "no number of any kind" in B.SEAL_RULES["plumb"]


def test_load_version_uses_the_committed_file_only(tmp_path):
    """Review R7: writer.load_version never takes a version file that is on disk but not committed as is."""
    from hnl.overlay import writer as W
    r = str(tmp_path / "repo")
    os.makedirs(os.path.join(r, "configs", "trackml"))
    subprocess.run(["git", "init", "-q", r], check=True)
    p = os.path.join(r, "configs", "trackml", "overlay_version_h_v1.json")
    assert W.load_version(p, r) is None
    json.dump({"overlay_hash": "a" * 64, "manifest": {}}, open(p, "w"))
    with pytest.raises(RuntimeError, match="not committed as is"):
        W.load_version(p, r)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "record")
    assert W.load_version(p, r)["overlay_hash"] == "a" * 64
    json.dump({"overlay_hash": "b" * 64, "manifest": {}}, open(p, "w"))
    with pytest.raises(RuntimeError, match="not committed as is"):
        W.load_version(p, r)
    q = tmp_path / "outside.json"
    q.write_text(json.dumps({"overlay_hash": "c"}))
    assert W.load_version(str(q), r) == {"overlay_hash": "c"}


def test_sealed_per_target_rules(tmp_path, monkeypatch):
    """The step-5 read: H-val only from the production sealed directory and through open_sealed; a pilot directory
    only with placeholder scores; the manifest is checked before anything in the directory is read."""
    for name, rec, msg in (("h", {"set": "hval", "k": 0, "events": 150, "scores": {"placeholder": False}},
                            "production sealed directory"),
                           ("p", {"set": "pilot", "k": 9, "events": 5, "scores": {"placeholder": False}},
                            "placeholder"),
                           ("c", {"set": "closure", "k": 1, "events": 150, "scores": {"placeholder": False}},
                            "set index 1")):
        d = make_dir(str(tmp_path / name))
        with open(os.path.join(d, "blind.json"), "w") as fh:
            json.dump(rec, fh)
        sha = B.seal(d)["manifest_sha256"]
        try:
            with pytest.raises(PermissionError, match=msg):
                B.sealed_per_target(d, sha)
            with pytest.raises(PermissionError, match="does not verify"):
                B.sealed_per_target(d, "0" * 64)
        finally:
            B.unseal_permissions(d)
    d = str(tmp_path / "h")
    B.seal(d) if not B.is_sealed(d) else None
    monkeypatch.setattr(B, "RUNS_H", str(tmp_path))          # as if d were the production sealed directory
    try:
        with pytest.raises(PermissionError, match="overlay_version_h_v1.json is not committed"):
            B.sealed_per_target(d, B.manifest_sha256(d), repo=str(tmp_path))
    finally:
        B.unseal_permissions(d)


def test_unblinded_and_open_sealed(tmp_path):
    """open_sealed: the committed record, then the manifest sha256 of the one line of RESULTS.md at HEAD (decision D4);
    an --expect of the caller must agree with that line and never replaces it."""
    r = str(tmp_path / "repo")
    os.makedirs(os.path.join(r, "configs", "trackml"))
    subprocess.run(["git", "init", "-q", r], check=True)
    rec = os.path.join(r, "configs", "trackml", "overlay_version_h_v1.json")
    res = os.path.join(r, "RESULTS.md")
    d = make_dir(str(tmp_path / "sealed"))
    sha = B.seal(d)["manifest_sha256"]
    line = f"{B.RESULTS_PREFIX} {sha}\n"
    try:
        assert not B.unblinded(r)
        with pytest.raises(PermissionError, match="not committed"):
            B.open_sealed(d, sha, r)
        json.dump({"overlay_hash": "e" * 64}, open(rec, "w"))
        assert not B.unblinded(r)                                        # not committed
        git(r, "add", "-A")
        git(r, "commit", "-qm", "record")
        assert B.unblinded(r)
        with pytest.raises(PermissionError, match="no committed RESULTS.md"):
            B.open_sealed(d, sha, r)                                     # the caller's sha256 alone is not enough
        open(res, "w").write("# Results\n\n" + line)
        with pytest.raises(PermissionError, match="no committed RESULTS.md"):
            B.open_sealed(d, None, r)                                    # written, not committed
        git(r, "add", "-A")
        git(r, "commit", "-qm", "results")
        assert B.results_expect(r) == sha
        assert B.open_sealed(d, None, r) == d and B.open_sealed(d, sha, r) == d and B.open_sealed(d, sha[:8], r) == d
        with pytest.raises(PermissionError, match="is not the manifest sha256 that RESULTS.md records"):
            B.open_sealed(d, "0" * 64, r)                                # another manifest named by the caller
        B.main(["verify", "--sealed", d, "--results", "--repo", r])
        with pytest.raises(SystemExit):
            B.main(["verify", "--sealed", d, "--results", "--repo", r, "--expect", "0" * 64])
        open(res, "w").write("# Results\n\n" + line + f"  {B.RESULTS_PREFIX} {'1' * 64}  \n")
        git(r, "commit", "-qam", "two lines")
        with pytest.raises(LookupError, match="2 lines"):
            B.results_expect(r)
        with pytest.raises(PermissionError, match="2 lines"):
            B.open_sealed(d, sha, r)
        open(res, "w").write(f"{B.RESULTS_PREFIX} {'1' * 64}\n")
        git(r, "commit", "-qam", "another manifest")
        with pytest.raises(PermissionError, match="does not verify"):
            B.open_sealed(d, None, r)
        with pytest.raises(SystemExit):
            B.main(["verify", "--sealed", d, "--results", "--repo", r])
        open(res, "w").write(line)
        git(r, "commit", "-qam", "back")
        json.dump({"overlay_hash": "short"}, open(rec, "w"))
        assert not B.unblinded(r)
        with pytest.raises(PermissionError):
            B.open_sealed(d, None, r)
    finally:
        B.unseal_permissions(d)


# ------------------------------------------------------------------------------------------------ pilot dry run
PILOT_META = os.path.join(B.DERIVED_OVERLAY, "meta_pilot.json")
VAL_GRID = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/grid/full"


@pytest.mark.skipif(not (os.path.exists(PILOT_META) and os.path.exists(os.path.join(VAL_GRID, "per_event.parquet"))
                         and os.path.exists(B.SPLIT_H)), reason="needs the pilot overlay tables and the val grid")
def test_pilot_dry_run_end_to_end(tmp_path, monkeypatch):
    """Truth-made placeholder scores of the five pilot events through evaluate (two blocks), merge, seal, verify and
    V2; the stored outputs hold no gun particle; raw labels reproduce the blinded counts; the score is unblinded.
    Evaluate, V2 and the step-5 read refuse the pilot ids without a dry grant of C-full."""
    from hnl.trackml import events as E
    from hnl.trackml.evaluate import trackml_score
    d = str(tmp_path / "pilot_full")
    info = B.dry_scores("pilot", d, sigma=2.0)
    monkeypatch.delenv(GR.ENV, raising=False)
    with pytest.raises(PermissionError, match="no grant"):
        B.evaluate("pilot", d, block=0, n_blocks=2)
    monkeypatch.setenv(GR.ENV, GR.write("pilot", "hs", "dry", B.set_ids("pilot"), str(tmp_path / "g"), out=d))
    with pytest.raises(PermissionError, match="condition hs, not full"):
        B.evaluate("pilot", d, block=0, n_blocks=2)
    assert not os.path.exists(os.path.join(d, "counts"))
    monkeypatch.setenv(GR.ENV, GR.write("pilot", "full", "dry", B.set_ids("pilot"), str(tmp_path / "g"), out=d))
    assert info["events"] == 5 and info["edges"] > 0
    try:
        B.merge(d)
        raise AssertionError("merge without blocks")
    except SystemExit:
        pass
    m0 = B.evaluate("pilot", d, block=0, n_blocks=2, workers=2)
    assert len(m0["events"]) == 3 and B.blocks_done(d) == (1, 2)
    with pytest.raises(SystemExit):
        B.evaluate("pilot", d, block=0, n_blocks=2)                      # a done block is not repeated
    with pytest.raises(PermissionError):
        B.evaluate("pilot", d, block=1, n_blocks=2, configs=[{"name": "WP-A", "family": "mwslmerge", "params": "0.3"}])
    B.evaluate("pilot", d, block=1, n_blocks=2, workers=2)
    rec = B.merge(d)
    assert rec["events"] == 5 and rec["scores"]["placeholder"] and rec["overlay_meta"]["overlay_hash"]
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    pt = pd.read_parquet(os.path.join(d, "per_target.parquet"))
    assert len(pe) == 10 and not B.is_gun(pt.particle_id.to_numpy()).any()
    seal = B.seal(d)
    try:
        assert B.verify(d, seal["manifest_sha256"]) == []
        out = str(tmp_path / "v2_dry.json")
        res = B.v2(d, out, VAL_GRID)
        assert res["title"].startswith("DRY RUN") and res["manifest_sha256"] == seal["manifest_sha256"]
        assert [c["val_config"] for c in res["configs"]] == [15, 43]
        for c in res["configs"]:
            assert c["events_host_prompt_targets_differ"] == []          # host prompt targets = the stored val ones
            assert c["overlay"]["n_prompt_target"] == c["val"]["n_prompt_target"]
        assert json.load(open(out))["gate"]["config"] == "WP-R-P3"
        # the step-5 read: unblinded rows from the raw labels, gun targets included; host rows equal the blinded ones
        full = B.sealed_per_target(d, seal["manifest_sha256"], "WP-R-P3")
        assert B.is_gun(full.particle_id.to_numpy()).any() and set(full.config) == {0}
        host = full[~B.is_gun(full.particle_id.to_numpy())]
        blinded = pt[pt.config == 0]
        m = host.merge(blinded, on=["event", "config", "particle_id"], suffixes=("", "_b"))
        assert len(m) == len(blinded) == len(host) and (m.dm == m.dm_b).all()
        with pytest.raises(PermissionError):
            B.sealed_per_target(d, "0" * 64, "WP-R-P3")
        with pytest.raises(KeyError):
            B.sealed_per_target(d, seal["manifest_sha256"], "WP-A")      # not a configuration of the directory
        # raw labels in the sealed directory reproduce the blinded counts; blinding removed some gun tracks; the
        # blinded TrackML score equals the score of the raw labels
        lab = pd.read_parquet(os.path.join(d, "tracks", "b000of002.parquet"))
        evs = E.load_events(m0["events"])
        removed = 0
        for e, g in lab.groupby(["event", "config"]):
            ev = evs[e[0]]
            L = g.sort_values("row").label.to_numpy(np.int64)
            assert len(L) == ev.n
            c, rows = B.blind_counts(ev.pid, L, ev.particles, ev.full_rows + 1, ev.weight)
            row = pe[(pe.event == e[0]) & (pe.config == e[1])].iloc[0]
            assert all(row[k] == c[k] for k in ("n_tracks", "n_tracks_ge5", "n_fake_ge5", "m_prompt_target",
                                                "n_prompt_target", "m_target"))
            assert c["trackml_score"] == pytest.approx(trackml_score(ev.full_rows + 1, ev.pid, ev.weight, L),
                                                       abs=1e-12)
            gun = B.is_gun(ev.pid)
            removed += int(((B.blind_labels(L, gun) < 0) & (L >= 0)).sum())
            hp = B.host_particles(ev.particles)
            want = set(hp.particle_id[hp.is_target.astype(bool)])
            assert set(pt[(pt.event == e[0]) & (pt.config == e[1])].particle_id) == want
        assert removed > 0
    finally:
        B.unseal_permissions(d)
