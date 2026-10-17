"""hnl.overlay.d7record: the records of decision D7 of addendum H (docs/preregistration_pileup.md sections 6, 7, 9.3 and
10, step 4). The H-val record (results/pileup_hp/h_t05_d7_v1.json) and the H-T1 record
(results/pileup_hp/h_t1_d7_v1.json) from synthetic gate outputs in the formats of their modules: the mechanical rules
of section 9.3 (flags, the forward fallback, the descriptive state after the third pass), every refusal of the writer,
the read-back through hnl.overlay.h1.restriction_record, and the parsers on the real pilot gate outputs of the
integration runs (truth and truth-made placeholder outputs only; no model output of an overlay event is read)."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

from hnl.overlay import blind as B
from hnl.overlay import d7record as D7
from hnl.overlay import h1 as H

OH = D7.ADDENDUM_OVERLAY_HASH
OH2, OH3 = "2" * 64, "3" * 64
MAN = "ab" * 32
needs_split = pytest.mark.skipif(not os.path.exists(B.SPLIT_H), reason="needs configs/trackml/split_h_v1.json")
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t"}


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, env=ENV, capture_output=True)


def make_repo(path, manifest=MAN):
    """A git repository with RESULTS.md holding the sealed H-val manifest line (decision D4)."""
    r = str(path)
    os.makedirs(os.path.join(r, "results", "pileup_hp"), exist_ok=True)
    os.makedirs(os.path.join(r, "configs", "trackml"), exist_ok=True)
    with open(os.path.join(r, "RESULTS.md"), "w") as fh:
        fh.write("# results\n" + (f"{B.RESULTS_PREFIX} {manifest}\n" if manifest else ""))
    subprocess.run(["git", "init", "-q", r], check=True)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


def commit(repo, *paths, msg="c"):
    git(repo, "add", *paths)
    git(repo, "commit", "-qm", msg)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def v1a(delta=0.3, regions_ok=(True, True, True)):
    ok = dict(zip(D7.REGIONS, regions_ok))
    return {"pairs": 3276, "delta": delta, "sigma_paired": 0.2, "regions": {r: {"pairs": 1000, "delta": 0.2 if ok[r]
                                                                                 else 2.0} for r in D7.REGIONS},
            "rule": {"kind": "decisive", "limit": 1.0, "limit_region": 1.5, "regions_pass": ok,
                     "pass": abs(delta) <= 1.0 and all(regions_ok)}}


def set_v1(d, a=None, c=True, vd=True, b=True, dd=True):
    """Sets the V1 output's rules consistently (decisive_pass from V1 (a), V1 (c) and V1-Delta (a))."""
    v = d["v1"]
    if a is not None:
        v["v1"]["a"] = a
    v["v1"]["c"]["rule"]["pass"] = c
    v["v1_delta"]["a"]["rule"]["pass"] = vd
    v["v1_delta"]["decisive_pass"] = vd
    v["v1"]["b"]["rule"]["pass"] = b
    v["v1"]["d"]["rule"]["pass"] = dd
    v["v1"]["decisive_pass"] = v["v1"]["a"]["rule"]["pass"] and c
    v["decisive_pass"] = v["v1"]["a"]["rule"]["pass"] and c and vd


def hval_outputs(root, oh=OH):
    """Passing gate outputs of H-val and closure in the formats of their modules (the keys the record reads)."""
    hv, cl = B.set_ids("hval"), B.set_ids("closure")
    wg, wc = os.path.join(root, "work_hval"), os.path.join(root, "work_closure")
    brief = lambda kind, lim: {"pairs": 400, "delta": 0.5, "sigma_paired": 1.0,      # noqa: E731
                               "rule": {"kind": kind, "limit": lim, "pass": True}}
    return {
        "truth_checks": {"set": "hval", "root": root, "work": wg, "events": hv,
                         "version": {"overlay_hash": oh, "pass": True, "committed_version": False},
                         "format": {"host_root": "/x/derived/tp_v1"}, "pass": {k: True for k in D7.TRUTH_KEYS}},
        "closure_checks": {"set": "closure", "root": root, "work": wc, "events": cl, "overlay_hash": oh,
                           "version": {"overlay_hash": oh, "pass": True},
                           "pass": {k: True for k in D7.CLOSURE_KEYS}},
        "c1_confirm": {"set": "hval", "root": root, "work": wg, "events": hv, "outcome": "pass", "pass": True,
                       "blocks": False, "part1_matched_bins": {"outcome": "pass"}, "part2_outcome": "pass",
                       "part2_failed": []},
        "closure_gate": {"set": "closure", "root": root, "work": wc, "events": len(cl), "outcome": "pass",
                         "counted": 9, "failed": [], "pass": True},
        "v1": {"label": D7.V1_LABEL, "root": root, "events": len(cl),
               "grids": {c: {"set": "closure", "k": 1, "condition": c, "n_events": len(cl), "root": root,
                             "scores_provenance": {"placeholder": False}} for c in ("full", "1gev")},
               "v1": {"a": v1a(), "b": brief("flag", 3.0), "c": brief("decisive", 0.2), "d": brief("flag", 3.0),
                      "decisive_pass": True, "flags": []},
               "v1_delta": {"a": {"pairs": 3276, "difference": 0.1, "sigma_paired": 0.2,
                                  "rule": {"kind": "decisive", "limit": 0.5, "pass": True}}, "decisive_pass": True},
               "decisive_pass": True},
        "v2": {"title": "V2, host change with the gun present (hval, C-full)", "set": "hval", "events": len(hv),
               "manifest_sha256": MAN, "overlay_meta": {"path": os.path.join(root, "meta_hval.json"), "sha256": None,
                                                        "overlay_hash": oh},
               "gate": {"config": "WP-A", "pass": True},
               "configs": [{"name": "WP-A", "delta_prompt_eff_points": 0.01, "delta_fake_rate_points": -0.02,
                            "delta_trackml_score": 0.0001, "within": {"prompt_eff": True}, "pass": True}]},
        "gun_content": {"set": "hval", "root": root, "work": wg, "events": len(hv),
                        "nav_losses": {"gun_particles": 9000, "lost": 180, "pp_like": 2000, "pp_like_lost": 58,
                                       "pp_like_lost_share": 58 / 2000}},
    }


class Case:
    """A synthetic H-val case: a repository, an overlay root with the meta files and the gate outputs on disk."""

    def __init__(self, tmp, oh=OH):
        self.tmp = str(tmp)
        self.repo = make_repo(tmp / "repo")
        self.root = str(tmp / "root")
        self.runs = str(tmp / "runs")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.runs, exist_ok=True)
        self.oh = oh
        for s in ("hval", "closure"):
            self.meta(s, oh)
        self.d = hval_outputs(self.root, oh)

    def meta(self, s, oh):
        with open(os.path.join(self.root, f"meta_{s}.json"), "w") as fh:
            json.dump({"set": s, "overlay_hash": oh}, fh)

    def paths(self):
        self.d["v2"]["overlay_meta"]["sha256"] = self.d["v2"]["overlay_meta"]["sha256"] or sha(
            os.path.join(self.root, "meta_hval.json"))
        out = {}
        for k, v in self.d.items():
            p = os.path.join(self.tmp, "gates", f"{k}.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(v, fh)
            out[k] = p
        return out

    def record(self, passes=None, **kw):
        return D7.hval_record(self.paths(), passes or [self.oh], self.repo, self.root, runs=self.runs, **kw)


@needs_split
def test_hval_record_all_pass_and_read_back_by_h1(tmp_path):
    c = Case(tmp_path)
    rec = c.record()
    assert (rec["restriction"], rec["flags"], rec["descriptive"]) == (None, [], False)
    assert rec["lost_share"] == pytest.approx(0.029) and rec["set"] == "hval" and rec["overlay_hash"] == OH
    assert rec["passes"] == [OH] and rec["derivation"]["passes"] == 1 and not rec["derivation"]["fallback"]
    assert set(rec["gates"]) == set(D7.DECISIVE + D7.FLAG_GATES)
    assert rec["inputs"]["v2"]["sha256"] == sha(rec["inputs"]["v2"]["path"]) and rec["version_record"] is None
    assert rec["metas"]["hval"]["overlay_hash"] == OH
    path = D7.write_record(rec, c.repo, c.runs)
    assert os.path.relpath(path, c.repo) == "results/pileup_hp/h_t05_d7_v1.json" == D7.RECORD_PATHS["hval"]
    assert path.startswith(os.path.join(c.repo, B.HT05_RECORDS))
    with pytest.raises(PermissionError, match="not committed as is"):        # h1 reads it only once committed
        H.restriction_record(path, c.repo, "hval")
    commit(c.repo, D7.RECORD_PATHS["hval"])
    rr = H.restriction_record(path, c.repo, "hval")
    assert (rr["restriction"], rr["flags"], rr["descriptive"]) == (None, [], False)
    assert rr["lost_share"] == pytest.approx(0.029) and rr["path"] == D7.RECORD_PATHS["hval"]
    with pytest.raises(PermissionError, match="h_t1_"):                       # not an H-T1 record
        H.restriction_record(path, c.repo, "htest")
    w = json.load(open(path))
    assert w["code"]["module"] == "hnl.overlay.d7record" and w["schema"] == 1 and w["written_utc"]
    assert D7.read_hval_record(c.repo)["passes"] == [OH]
    with pytest.raises(D7.Refused, match="written once"):                     # never over an existing record
        D7.write_record(rec, c.repo, c.runs)


@needs_split
def test_flags_and_not_informative(tmp_path):
    c = Case(tmp_path)
    c.d["closure_gate"].update(outcome="fail")
    c.d["closure_gate"]["pass"] = False
    set_v1(c.d, b=False, dd=False)
    c.d["v2"]["gate"]["pass"] = False
    rec = c.record()
    assert rec["flags"] == ["closure truth gate", "V1 (b)", "V1 (d)", "V2"]
    assert rec["restriction"] is None and rec["descriptive"] is False
    c = Case(tmp_path / "b")
    c.d["closure_gate"].update(outcome="not informative", counted=0)
    c.d["closure_gate"]["pass"] = None
    c.d["c1_confirm"].update(outcome="not informative")
    c.d["c1_confirm"]["pass"] = None
    set_v1(c.d, dd=False)
    rec = c.record()
    assert rec["flags"] == ["V1 (d)"] and rec["descriptive"] is False and rec["restriction"] is None
    assert rec["derivation"]["not_informative"] == ["C1 confirmation", "closure truth gate"]


@needs_split
def test_forward_fallback_d12(tmp_path):
    cases = [((False, True, True), ["fwd+", "fwd-"]), ((True, False, True), ["central", "fwd-"]),
             ((False, True, False), ["fwd+"]), ((True, False, False), ["central"])]
    for i, (ok, keep) in enumerate(cases):
        c = Case(tmp_path / f"f{i}")
        set_v1(c.d, a=v1a(0.4, ok))
        rec = c.record()
        assert rec["restriction"] == keep and rec["descriptive"] is False, ok
        assert rec["derivation"]["fallback"] and rec["derivation"]["decisive_failed"] == ["V1 (a)"]
    # all three regions fail: no fallback; a change is due before the third pass, descriptive on it
    c = Case(tmp_path / "three")
    set_v1(c.d, a=v1a(0.4, (False, False, False)))
    with pytest.raises(D7.Refused, match=r"V1 \(a\) failed on pass 1 of 3 .* allows a change"):
        c.record()
    c = Case(tmp_path / "three3", oh=OH3)
    set_v1(c.d, a=v1a(0.4, (False, False, False)))
    rec = c.record(passes=[OH, OH2, OH3])
    assert rec["descriptive"] is True and rec["restriction"] is None and not rec["derivation"]["fallback"]
    # V1 (a) fails in all (|Delta| > 1.0 point): no fallback
    c = Case(tmp_path / "all")
    set_v1(c.d, a=v1a(1.2, (True, False, True)))
    with pytest.raises(D7.Refused, match="fallback does not apply"):
        c.record()
    # another decisive gate fails as well: no fallback
    for kw in ({"c": False}, {"vd": False}):
        c = Case(tmp_path / f"other{len(kw)}{list(kw)[0]}")
        set_v1(c.d, a=v1a(0.4, (False, True, True)), **kw)
        with pytest.raises(D7.Refused, match="fallback does not apply"):
            c.record()
    c = Case(tmp_path / "c1")
    set_v1(c.d, a=v1a(0.4, (False, True, True)))
    c.d["c1_confirm"].update(outcome="fail", blocks=True)
    c.d["c1_confirm"]["pass"] = False
    with pytest.raises(D7.Refused, match="C1 confirmation, V1 \\(a\\) failed"):
        c.record()


@needs_split
def test_decisive_failures_and_the_three_passes(tmp_path):
    def c1_fail(c):
        c.d["c1_confirm"].update(outcome="fail")
        c.d["c1_confirm"]["pass"] = False
    for n, passes, oh in ((1, [OH], OH), (2, [OH, OH2], OH2)):
        c = Case(tmp_path / f"p{n}", oh=oh)
        c1_fail(c)
        with pytest.raises(D7.Refused, match=f"failed on pass {n} of 3"):
            c.record(passes=passes)
    c = Case(tmp_path / "p3", oh=OH3)
    c1_fail(c)
    set_v1(c.d, b=False)
    rec = c.record(passes=[OH, OH2, OH3])
    assert rec["descriptive"] is True and rec["restriction"] is None and rec["flags"] == ["V1 (b)"]
    assert rec["derivation"]["decisive_failed"] == ["C1 confirmation"] and rec["derivation"]["passes"] == 3
    c = Case(tmp_path / "vd3", oh=OH3)
    set_v1(c.d, vd=False)
    assert c.record(passes=[OH, OH2, OH3])["descriptive"] is True
    # a failed per-set truth check is refused on every pass (the set does not go to the model)
    for key, check in (("truth_checks", "production"), ("closure_checks", "ids")):
        c = Case(tmp_path / f"t{key}", oh=OH3)
        c.d[key]["pass"][check] = False
        with pytest.raises(D7.Refused, match="does not go to the model"):
            c.record(passes=[OH, OH2, OH3])
    # the pass history
    c = Case(tmp_path / "hist", oh=OH2)
    for passes, msg in (([OH2], "first pass is the version of the addendum"), ([OH, OH3], "last pass"),
                        ([OH, OH, OH2], "twice"), ([OH, OH3, "4" * 64, OH2], "one to 3 passes"),
                        ([], "one to 3 passes"), ([OH, "xyz"], "64 hex")):
        with pytest.raises(D7.Refused, match=msg):
            D7.hval_record(c.paths(), passes, c.repo, c.root, runs=c.runs)
    assert c.record(passes=[OH, OH2])["passes"] == [OH, OH2]


def _edit(path_keys, value):
    def f(d):
        cur = d
        for k in path_keys[:-1]:
            cur = cur[k]
        if value is KeyError:
            del cur[path_keys[-1]]
        else:
            cur[path_keys[-1]] = value
    return f


@needs_split
def test_inconsistent_inputs_are_refused(tmp_path):
    hv = B.set_ids("hval")
    edits = [
        (_edit(("truth_checks", "set"), "pilot"), "an output of set 'pilot'"),
        (_edit(("truth_checks", "events"), hv[:-1]), "not every event of hval"),
        (_edit(("truth_checks", "pass", "version"), KeyError), "does not hold the checks"),
        (_edit(("closure_checks", "root"), "/elsewhere"), "not the overlay root"),
        (_edit(("closure_checks", "overlay_hash"), OH2), "different overlay hashes"),
        (_edit(("c1_confirm", "root"), "/elsewhere"), "not the overlay root"),
        (_edit(("c1_confirm", "work"), "/elsewhere"), "another work directory"),
        (_edit(("c1_confirm", "pass"), False), "does not agree with its outcome"),
        (_edit(("c1_confirm", "outcome"), "few bins"), "outcome 'few bins'"),
        (_edit(("closure_gate", "events"), 149), "149 events"),
        (_edit(("closure_gate", "work"), "/elsewhere"), "another work directory"),
        (_edit(("v1", "label"), "DRY RUN: placeholders"), "label"),
        (_edit(("v1", "grids", "1gev", "scores_provenance", "placeholder"), True), "placeholder scores"),
        (_edit(("v1", "grids", "1gev"), KeyError), "grids.1gev"),
        (_edit(("v1", "v1_delta"), KeyError), "v1_delta"),
        (_edit(("v1", "decisive_pass"), False), "decisive_pass does not agree"),
        (_edit(("v1", "v1", "a", "rule", "limit"), 2.0), "limits"),
        (_edit(("v1", "v1", "a", "rule", "pass"), False), r"V1 \(a\) pass False does not agree"),
        (_edit(("v1", "v1", "b", "rule", "kind"), "decisive"), "no flag rule"),
        (_edit(("v1", "events"), 5), "5 events"),
        (_edit(("v2", "title"), "DRY RUN (truth-made placeholder scores): V2"), "dry run"),
        (_edit(("v2", "gate", "config"), "WP-S"), "not WP-A"),
        (_edit(("v2", "manifest_sha256"), "cd" * 32), "not the one RESULTS.md records"),
        (_edit(("v2", "overlay_meta", "sha256"), "0" * 64), "another H-val meta file"),
        (_edit(("v2", "overlay_meta", "overlay_hash"), OH2), "different overlay hashes"),
        (_edit(("v2", "set"), "pilot"), "not hval"),
        (_edit(("gun_content", "nav_losses", "pp_like_lost_share"), None), "no navigator-loss share"),
        (_edit(("gun_content", "nav_losses", "pp_like_lost_share"), 0.5), "is not 58 / 2000"),
        (_edit(("gun_content", "events"), 5), "5 events"),
        (_edit(("gun_content", "work"), "/elsewhere"), "another work directory"),
    ]
    for i, (f, msg) in enumerate(edits):
        c = Case(tmp_path / f"e{i}")
        f(c.d)
        with pytest.raises(D7.Refused, match=msg):
            c.record()
    # the meta files of the stored sets carry the version
    c = Case(tmp_path / "meta")
    c.d["v2"]["overlay_meta"]["sha256"] = None
    c.meta("closure", OH2)
    with pytest.raises(D7.Refused, match="different overlay hashes"):
        c.record()
    c = Case(tmp_path / "nometa")
    os.remove(os.path.join(c.root, "meta_closure.json"))
    with pytest.raises(D7.Refused, match="no meta file"):
        c.record()
    # V2 is read only against the one manifest line of RESULTS.md at HEAD (decision D4)
    c = Case(tmp_path / "d4")
    with open(os.path.join(c.repo, "RESULTS.md"), "w") as fh:
        fh.write("# results\n")
    commit(c.repo, "RESULTS.md")
    with pytest.raises(D7.Refused, match="decision D4"):
        c.record()
    # files: required, existing, JSON objects, never under the sealed directories, never an H1 output
    c = Case(tmp_path / "files")
    p = c.paths()
    for k, v, msg in (("v2", None, "needs --v2"), ("v1", c.tmp + "/none.json", "no such file"),
                      ("gun_content", __file__, "not JSON"),
                      ("v2", os.path.join(c.runs, "sealed", "hval_full", "v2.json"), "blinding"),
                      ("v2", os.path.join(c.repo, "results", "pileup_hp", "h1_hval_WP-A_full-1gev-hs_v1.json"),
                       "output of the H1 code")):
        q = dict(p)
        q[k] = v
        with pytest.raises(D7.Refused, match=msg):
            D7.hval_record(q, [OH], c.repo, c.root, runs=c.runs)
    lst = os.path.join(c.tmp, "list.json")
    json.dump([1, 2], open(lst, "w"))
    with pytest.raises(D7.Refused, match="not a JSON object"):
        D7.hval_record({**p, "closure_gate": lst}, [OH], c.repo, c.root, runs=c.runs)


@needs_split
def test_version_record_on_disk_must_agree(tmp_path):
    c = Case(tmp_path)
    vp = os.path.join(c.repo, "configs", "trackml", "overlay_version_h_v1.json")
    json.dump({"overlay_hash": OH2}, open(vp, "w"))
    with pytest.raises(D7.Refused, match="version record holds overlay_hash"):
        c.record()
    json.dump({"overlay_hash": OH, "restriction": ["central"], "descriptive": False}, open(vp, "w"))
    with pytest.raises(D7.Refused, match="version record holds restriction"):
        c.record()
    json.dump({"overlay_hash": OH, "restriction": None, "descriptive": False}, open(vp, "w"))
    rec = c.record()
    assert rec["version_record"]["committed"] is False and rec["version_record"]["sha256"] == sha(vp)


@needs_split
def test_writer_cli(tmp_path, monkeypatch, capsys):
    c = Case(tmp_path)
    p = c.paths()
    monkeypatch.setattr(D7, "REPO", c.repo)
    monkeypatch.setattr(D7, "DERIVED_OVERLAY", c.root)
    monkeypatch.setenv("HNL_H_RUNS", c.runs)
    monkeypatch.setattr(D7, "code_clean", lambda repo=None: [])
    args = ["hval", "--passes", OH] + sum(([f"--{k.replace('_', '-')}", v] for k, v in p.items()), [])
    for drop in range(1, len(args), 2):                            # every input is required
        if args[drop].startswith("--"):
            with pytest.raises(SystemExit) as e:
                D7.main(args[:drop] + args[drop + 2:])
            assert e.value.code == 2
    monkeypatch.setattr(D7, "code_clean", lambda repo=None: ["src/hnl/overlay/d7record.py"])
    with pytest.raises(SystemExit) as e:
        D7.main(args)
    assert e.value.code == 1 and "uncommitted code" in capsys.readouterr().err
    monkeypatch.setattr(D7, "code_clean", lambda repo=None: [])
    os.makedirs(os.path.join(c.runs, "launch"))
    open(B.launch_path("hval", c.runs), "w").write("{}")
    with pytest.raises(SystemExit):
        D7.main(args)
    assert "was launched" in capsys.readouterr().err
    os.remove(B.launch_path("hval", c.runs))
    D7.main(args)
    out = json.loads(capsys.readouterr().out)
    assert out["wrote"] == D7.RECORD_PATHS["hval"] and out["restriction"] is None and out["flags"] == []
    assert os.path.exists(os.path.join(c.repo, D7.RECORD_PATHS["hval"]))
    with pytest.raises(SystemExit):
        D7.main(args)
    assert "written once" in capsys.readouterr().err
    D7.main(["path", "--set", "htest"])
    assert capsys.readouterr().out.strip() == "results/pileup_hp/h_t1_d7_v1.json"


def test_code_clean_names_uncommitted_modules(tmp_path):
    r = make_repo(tmp_path / "r")
    d = os.path.join(r, "src", "hnl", "overlay")
    os.makedirs(d)
    for m in ("d7record.py", "h1.py", "blind.py"):
        open(os.path.join(d, m), "w").write("x = 1\n")
    assert len(D7.code_clean(r)) == 3
    commit(r, "src")
    assert D7.code_clean(r) == []
    open(os.path.join(d, "h1.py"), "a").write("y = 2\n")
    assert D7.code_clean(r) == ["src/hnl/overlay/h1.py"]


# ------------------------------------------------------------------------------------------------ C1 flag decision
FLAG = "C1 confirmation (u bias, volumes 12 and 9)"
ROW9 = {"volume": 9, "quantity": "bias_u"}
DEC_REL = "configs/trackml/overlay_h_c1_flag_v1.json"
REAL_C1 = os.path.join(B.REPO, "results", "pileup_hp", "h_t05_truth_c1_confirm_hval_v1.json")
REAL_TRUTH = os.path.join(B.REPO, "results", "pileup_hp", "h_t05_truth_checks_hval_v1.json")
REAL_DECISION = os.path.join(B.REPO, DEC_REL)
BASE = "7dacde0"                       # the last commit of d7record.py before --c1-flag-decision


def decision(**kw):
    """The decision of DECISIONS 2026-10-16 in the schema of --c1-flag-decision (as overlay_h_c1_flag_v1.json)."""
    d = {"decision": "DECISIONS.md 2026-10-16: addendum H section 9.3 after the failed C1 confirmation (reading A)",
         "first_overlay_hash": OH, "flag": FLAG, "part1_bias_u_volumes": [12], "part2_failed": [dict(ROW9)]}
    d.update(kw)
    return d


def put_decision(c, d=None, text=None, commit_it=True):
    """Writes the decision file into the case's repository (committed unless commit_it is false); its absolute path."""
    p = os.path.join(c.repo, DEC_REL)
    with open(p, "w") as fh:
        fh.write(text if text is not None else json.dumps(decision() if d is None else d, indent=1) + "\n")
    if commit_it:
        commit(c.repo, DEC_REL)
    return p


def c1_fails(c, bins=(), vols=(12,), part2=(ROW9,)):
    """The C1 output fails in exactly these rows, in the format of hnl.overlay.c1_confirm (default: the two rows of
    pass 1, part 1 u bias of volume 12 and part 2 u bias of volume 9)."""
    x = c.d["c1_confirm"]
    x.update(outcome="fail", blocks=True, part2_outcome="fail" if part2 else "pass",
             part2_failed=[dict(r) for r in part2])
    x["pass"] = False
    x["part1_matched_bins"] = {"outcome": "fail" if bins or vols else "pass", "n_bins_failed": len(bins),
                               "failed_bins": [dict(b) for b in bins], "bias_failed_volumes": list(vols)}


@needs_split
def test_c1_flag_decision_carries_the_two_rows_as_a_flag(tmp_path):
    # either row or both: refused as today without the decision, a decided record with the flag first with it
    for i, (vols, part2) in enumerate((((12,), (ROW9,)), ((12,), ()), ((), (ROW9,)))):
        c = Case(tmp_path / f"a{i}")
        c1_fails(c, vols=vols, part2=part2)
        dp = put_decision(c)
        with pytest.raises(D7.Refused, match=r"C1 confirmation failed on pass 1 of 3"):
            c.record()
        rec = c.record(c1_flag_decision=dp)
        assert (rec["restriction"], rec["flags"], rec["descriptive"]) == (None, [FLAG], False)
        cf = rec["derivation"]["c1_flag"]
        assert cf["used"] and cf["covered_rows"] == {"part1_failed_bins": [], "part1_bias_u_volumes": list(vols),
                                                     "part2_failed": list(part2)}
    # with failed flag gates, written, committed and read back by hnl.overlay.h1 unchanged
    c = Case(tmp_path / "w")
    c1_fails(c)
    set_v1(c.d, b=False)
    c.d["v2"]["gate"]["pass"] = False
    dp = put_decision(c)
    rec = c.record(c1_flag_decision=DEC_REL)                                     # relative to the repository
    assert (rec["restriction"], rec["flags"], rec["descriptive"]) == (None, [FLAG, "V1 (b)", "V2"], False)
    cf = rec["derivation"]["c1_flag"]
    assert cf["decision"] == {"path": DEC_REL, "sha256": sha(dp), "decision": decision()["decision"],
                              "first_overlay_hash": OH} and cf["flag"] == FLAG
    assert rec["derivation"]["decisive_failed"] == [] and "carried as the flag" in rec["derivation"]["how"]
    assert rec["inputs"]["c1_flag_decision"] == {"path": DEC_REL, "sha256": sha(dp)}
    assert rec["gates"]["C1 confirmation"]["outcome"] == "fail"                 # the gate table keeps the failure
    path = D7.write_record(rec, c.repo, c.runs)
    commit(c.repo, D7.RECORD_PATHS["hval"])
    rr = H.restriction_record(path, c.repo, "hval")
    assert (rr["restriction"], rr["flags"], rr["descriptive"]) == (None, [FLAG, "V1 (b)", "V2"], False)
    assert D7.read_hval_record(c.repo)["flags"] == [FLAG, "V1 (b)", "V2"]
    # C1 counts as no failed decisive gate: the forward fallback applies when V1 (a) fails in one region only
    c = Case(tmp_path / "fb")
    c1_fails(c)
    set_v1(c.d, a=v1a(0.4, (True, False, True)))
    rec = c.record(c1_flag_decision=put_decision(c))
    assert rec["restriction"] == ["central", "fwd-"] and rec["flags"] == [FLAG] and rec["derivation"]["fallback"]
    # on a later pass the same rows stay a flag (the decision names the first pass)
    c = Case(tmp_path / "p2", oh=OH2)
    c1_fails(c)
    rec = c.record(passes=[OH, OH2], c1_flag_decision=put_decision(c))
    assert rec["flags"] == [FLAG] and rec["descriptive"] is False and rec["passes"] == [OH, OH2]
    # another decisive gate that fails is still refused before the third pass, descriptive on it, with the flag
    c = Case(tmp_path / "vc")
    c1_fails(c)
    set_v1(c.d, c=False)
    with pytest.raises(D7.Refused, match=r"decisive gate\(s\) V1 \(c\) failed on pass 1 of 3"):
        c.record(c1_flag_decision=put_decision(c))
    c = Case(tmp_path / "vc3", oh=OH3)
    c1_fails(c)
    set_v1(c.d, c=False)
    rec = c.record(passes=[OH, OH2, OH3], c1_flag_decision=put_decision(c))
    assert rec["descriptive"] is True and rec["flags"] == [FLAG] and rec["derivation"]["decisive_failed"] == ["V1 (c)"]


@needs_split
def test_c1_flag_decision_other_failed_rows_as_today(tmp_path):
    fbin = {"volume": 13, "quantity": "span0", "lo": 0.0, "hi": 0.2, "rel_diff": 0.12}
    extra = [{"vols": (12, 13)}, {"vols": (9,), "part2": ()}, {"bins": (fbin,)}, {"bins": (fbin,), "vols": ()},
             {"part2": (ROW9, {"volume": 9, "quantity": "span0"})},
             {"part2": (ROW9, {"volume": 12, "quantity": "bias_u"})},
             {"vols": (), "part2": ({"volume": 9, "quantity": "mad_u"},)}]
    for i, kw in enumerate(extra):
        c = Case(tmp_path / f"b{i}")
        c1_fails(c, **kw)
        dp = put_decision(c)
        with pytest.raises(D7.Refused, match=r"decisive gate\(s\) C1 confirmation failed on pass 1 of 3 .* does not "
                                             r"name"):
            c.record(c1_flag_decision=dp)
        c2 = Case(tmp_path / f"b{i}p2", oh=OH2)
        c1_fails(c2, **kw)
        with pytest.raises(D7.Refused, match=r"C1 confirmation failed on pass 2 of 3"):
            c2.record(passes=[OH, OH2], c1_flag_decision=put_decision(c2))
    # on the third pass H is descriptive, as without the decision; the decision is recorded as not used
    c = Case(tmp_path / "third", oh=OH3)
    c1_fails(c, vols=(12, 13))
    rec = c.record(passes=[OH, OH2, OH3], c1_flag_decision=put_decision(c))
    base = c.record(passes=[OH, OH2, OH3])
    assert (rec["restriction"], rec["flags"], rec["descriptive"]) == (base["restriction"], base["flags"],
                                                                      base["descriptive"]) == (None, [], True)
    cf = rec["derivation"]["c1_flag"]
    assert not cf["used"] and cf["uncovered_rows"] == [{"part": 1, "bias_u_volume": 13}]
    assert rec["derivation"]["decisive_failed"] == ["C1 confirmation"]
    # a C1 output whose failed rows do not agree with its outcomes is refused
    for i, f in enumerate((lambda x: x["part1_matched_bins"].update(outcome="pass"),
                           lambda x: x.update(part2_outcome="pass"),
                           lambda x: x["part1_matched_bins"].update(n_bins_failed=1),
                           lambda x: x["part1_matched_bins"].pop("bias_failed_volumes"),
                           lambda x: x.update(part2_failed=None))):
        c = Case(tmp_path / f"bad{i}")
        c1_fails(c)
        f(c.d["c1_confirm"])
        with pytest.raises(D7.Refused, match="C1 confirmation: "):
            c.record(c1_flag_decision=put_decision(c))


@needs_split
def test_c1_flag_decision_committed_as_is(tmp_path):
    c = Case(tmp_path)
    c1_fails(c)
    dp = put_decision(c, commit_it=False)
    with pytest.raises(D7.Refused, match="not committed as is"):                 # untracked
        c.record(c1_flag_decision=dp)
    commit(c.repo, DEC_REL)
    assert c.record(c1_flag_decision=dp)["flags"] == [FLAG]
    put_decision(c, decision(part1_bias_u_volumes=[12, 13]), commit_it=False)   # modified after the commit
    with pytest.raises(D7.Refused, match="not committed as is"):
        c.record(c1_flag_decision=dp)
    git(c.repo, "add", DEC_REL)                                                 # staged is not committed
    with pytest.raises(D7.Refused, match="not committed as is"):
        c.record(c1_flag_decision=dp)
    outside = os.path.join(c.tmp, "decision.json")                              # not a file of the repository
    json.dump(decision(), open(outside, "w"))
    for p in (outside, os.path.join(c.repo, "configs", "trackml", "none.json"), "../decision.json"):
        with pytest.raises(D7.Refused, match="no file of the repository"):
            c.record(c1_flag_decision=p)


@needs_split
def test_c1_flag_decision_names_the_first_pass(tmp_path):
    c = Case(tmp_path)
    c1_fails(c)
    with pytest.raises(D7.Refused, match="is not the first pass"):
        c.record(c1_flag_decision=put_decision(c, decision(first_overlay_hash=OH2)))
    c = Case(tmp_path / "p2", oh=OH2)                                           # on pass 2 the last pass is no first
    c1_fails(c)
    with pytest.raises(D7.Refused, match="is not the first pass"):
        c.record(passes=[OH, OH2], c1_flag_decision=put_decision(c, decision(first_overlay_hash=OH2)))


@needs_split
def test_c1_flag_decision_schema(tmp_path):
    """Exactly the five keys with their types, checked whatever the C1 outcome (here it passes)."""
    bad = [decision(extra=1), {k: v for k, v in decision().items() if k != "flag"},
           decision(decision=""), decision(decision=5), decision(flag=" "), decision(flag=["x"]),
           decision(flag="V2"), decision(flag="C1 confirmation"),
           decision(first_overlay_hash=OH[:12]), decision(first_overlay_hash=OH.upper()),
           decision(part1_bias_u_volumes=["12"]), decision(part1_bias_u_volumes=[12.0]),
           decision(part1_bias_u_volumes=[True]), decision(part1_bias_u_volumes=[12, 12]),
           decision(part1_bias_u_volumes=12), decision(part2_failed=[{"volume": "9", "quantity": "bias_u"}]),
           decision(part2_failed=[{**ROW9, "lo": 0.0}]), decision(part2_failed=[{"volume": 9}]),
           decision(part2_failed=[{"volume": 9, "quantity": ""}]), decision(part2_failed=[ROW9, ROW9]),
           decision(part2_failed=ROW9), decision(part1_bias_u_volumes=[], part2_failed=[]), [decision()]]
    for i, d in enumerate(bad):
        c = Case(tmp_path / f"s{i}")
        with pytest.raises(D7.Refused, match="--c1-flag-decision"):
            c.record(c1_flag_decision=put_decision(c, d))
    text = json.dumps(decision(), indent=1)
    for i, t in enumerate((text.replace('"flag":', '"flag": "x",\n "flag":'),           # a key twice
                           text.replace("[\n  12\n ]", "[NaN]"), "{", "")):
        c = Case(tmp_path / f"t{i}")
        with pytest.raises(D7.Refused, match="--c1-flag-decision"):
            c.record(c1_flag_decision=put_decision(c, text=t))


def c1_not_informative(c):
    c.d["c1_confirm"].update(outcome="not informative", part2_outcome="not informative")
    c.d["c1_confirm"]["pass"] = None


@needs_split
def test_c1_flag_decision_unused_when_c1_passes(tmp_path):
    for i, prep in enumerate((lambda c: None, c1_not_informative)):
        c = Case(tmp_path / f"u{i}")
        prep(c)
        dp = put_decision(c)
        base = c.record()
        rec = c.record(c1_flag_decision=dp)
        cf = rec["derivation"].pop("c1_flag")
        assert rec["inputs"].pop("c1_flag_decision") == {"path": DEC_REL, "sha256": sha(dp)}
        assert rec == base and rec["flags"] == []                               # otherwise the record without it
        assert cf["used"] is False and "not used" in cf["how"] and cf["decision"]["sha256"] == sha(dp)
        assert "covered_rows" not in cf and "uncovered_rows" not in cf


def base_module(tmp_path):
    """hnl.overlay.d7record as it was before --c1-flag-decision (commit BASE of the repository)."""
    p = subprocess.run(["git", "-C", B.REPO, "show", f"{BASE}:src/hnl/overlay/d7record.py"], capture_output=True,
                       text=True)
    if p.returncode:
        pytest.skip(f"needs commit {BASE} of the repository")
    tmp_path.mkdir(parents=True, exist_ok=True)
    f = tmp_path / "d7record_base.py"
    f.write_text(p.stdout)
    spec = importlib.util.spec_from_file_location("d7record_base", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@needs_split
def test_without_the_decision_the_record_is_as_before(tmp_path, monkeypatch):
    """No --c1-flag-decision: hval_record and derive on the same gate table give exactly what the code of BASE gives
    (the same JSON text, or the same refusal)."""
    old = base_module(tmp_path / "base")
    seen = []
    derive = D7.derive
    monkeypatch.setattr(D7, "derive", lambda g, n, c1=None: seen.append((copy.deepcopy(g), n, c1)) or derive(g, n, c1))

    def result(f):
        try:
            return json.dumps(f(), allow_nan=False)
        except Exception as e:                                                  # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    def flags(c):
        c.d["closure_gate"].update(outcome="fail")
        c.d["closure_gate"]["pass"] = False
        set_v1(c.d, b=False, dd=False)
        c.d["v2"]["gate"]["pass"] = False

    def ni(c):
        c.d["closure_gate"].update(outcome="not informative", counted=0)
        c.d["closure_gate"]["pass"] = None
        c1_not_informative(c)

    cases = [("pass", lambda c: None, [OH]), ("flags", flags, [OH]), ("ni", ni, [OH]),
             ("fallback", lambda c: set_v1(c.d, a=v1a(0.4, (False, True, True))), [OH]),
             ("c1", c1_fails, [OH]), ("c1p2", c1_fails, [OH, OH2]), ("c1p3", c1_fails, [OH, OH2, OH3]),
             ("c1x", lambda c: c1_fails(c, vols=(12, 13)), [OH, OH2, OH3]),
             ("c1fb", lambda c: (c1_fails(c), set_v1(c.d, a=v1a(0.4, (True, False, True)))), [OH]),
             ("v1a3", lambda c: set_v1(c.d, a=v1a(0.4, (False, False, False))), [OH]),
             ("truth", lambda c: c.d["truth_checks"]["pass"].update(format=False), [OH])]
    for name, prep, passes in cases:
        c = Case(tmp_path / name, oh=passes[-1])
        prep(c)
        put_decision(c)                                                         # committed, but not given
        p = c.paths()
        n = len(seen)
        new = result(lambda: D7.hval_record(p, passes, c.repo, c.root, runs=c.runs))
        assert new == result(lambda: old.hval_record(p, passes, c.repo, c.root, runs=c.runs)), name
        assert len(seen) == n + 1, name
        g, k, c1 = seen[-1]
        assert c1 is None and k == len(passes)
        assert result(lambda: derive(g, k)) == result(lambda: old.derive(g, k)) == result(lambda: derive(g, k, None))
        assert "c1_flag" not in new


@needs_split
def test_writer_cli_with_the_c1_flag_decision(tmp_path, monkeypatch, capsys):
    c = Case(tmp_path)
    c1_fails(c)
    c.d["v2"]["gate"]["pass"] = False
    put_decision(c)
    p = c.paths()
    monkeypatch.setattr(D7, "REPO", c.repo)
    monkeypatch.setattr(D7, "DERIVED_OVERLAY", c.root)
    monkeypatch.setenv("HNL_H_RUNS", c.runs)
    monkeypatch.setattr(D7, "code_clean", lambda repo=None: [])
    args = ["hval", "--passes", OH] + sum(([f"--{k.replace('_', '-')}", v] for k, v in p.items()), [])
    with pytest.raises(SystemExit) as e:                                        # without it: refused as today
        D7.main(args)
    assert e.value.code == 1 and "C1 confirmation failed on pass 1 of 3" in capsys.readouterr().err
    D7.main(args + ["--c1-flag-decision", DEC_REL])
    out = json.loads(capsys.readouterr().out)
    assert out["flags"] == [FLAG, "V2"] and out["restriction"] is None and out["descriptive"] is False
    w = json.load(open(os.path.join(c.repo, D7.RECORD_PATHS["hval"])))
    assert w["inputs"]["c1_flag_decision"]["path"] == DEC_REL and w["derivation"]["c1_flag"]["used"] is True


@needs_split
def test_htest_record_and_h1_carry_the_c1_flag(tmp_path):
    c, split, t, tp03 = htest_case(tmp_path, c1_flag=True)
    assert D7.read_hval_record(c.repo)["flags"] == [FLAG, "V2"]
    rec = D7.htest_record(write_inputs(c, t), c.repo, c.root, split, c.runs, host_root=tp03)
    assert rec["flags"] == [FLAG, "V2"] and rec["restriction"] == ["fwd+", "fwd-"] and rec["descriptive"] is False
    path = D7.write_record(rec, c.repo, c.runs)
    commit(c.repo, D7.RECORD_PATHS["htest"])
    rr = H.restriction_record(path, c.repo, "htest")
    assert rr["flags"] == [FLAG, "V2"] and rr["restriction"] == ["fwd+", "fwd-"]


@needs_split
@pytest.mark.skipif(not all(os.path.exists(p) for p in (REAL_C1, REAL_TRUTH, REAL_DECISION)),
                    reason="needs the committed pass-1 C1 record and the committed decision file")
def test_the_committed_decision_names_the_failed_rows_of_pass_1(tmp_path):
    """configs/trackml/overlay_h_c1_flag_v1.json against the committed C1 confirmation of pass 1
    (results/pileup_hp/h_t05_truth_c1_confirm_hval_v1.json, truth only): it names exactly the failed rows, its first
    pass is the pass-1 overlay_hash of the committed H-val truth checks, and C1 is carried as the flag."""
    real = json.load(open(REAL_C1))
    g = D7.c1_gate(real, "hval", B.set_ids("hval"), real["root"])
    assert g["outcome"] == "fail" and (g["part1"], g["part2"]) == ("fail", "fail")
    rows = D7.c1_failed_rows(real)
    assert rows == {"part1_failed_bins": [], "part1_bias_u_volumes": [12], "part2_failed": [ROW9]}
    r = make_repo(tmp_path / "r")
    shutil.copy(REAL_DECISION, os.path.join(r, DEC_REL))
    commit(r, DEC_REL)
    dec = D7.load_c1_flag_decision(DEC_REL, r, [OH])
    oh_pass1 = json.load(open(REAL_TRUTH))["version"]["overlay_hash"]
    assert dec["first_overlay_hash"] == oh_pass1 == D7.ADDENDUM_OVERLAY_HASH and dec["flag"] == FLAG
    assert (dec["part1_bias_u_volumes"], dec["part2_failed"]) == (rows["part1_bias_u_volumes"], rows["part2_failed"])
    assert dec["sha256"] == sha(REAL_DECISION)
    use = D7.c1_flag_use(g["outcome"], dec, rows)
    assert use["used"] and use["covered_rows"] == rows and use["flag"] == FLAG


# ------------------------------------------------------------------------------------------------ H-T1 record
def htest_case(tmp_path, restriction_ok=(False, True, True), flag_v2=True, c1_flag=False):
    """A repository with the committed H-val record (fallback and a V2 flag; with c1_flag also the C1 confirmation
    failed in the two rows of the committed decision and carried as a flag), the committed version record and the
    H-test truth outputs of a split with an H-test list."""
    c = Case(tmp_path)
    set_v1(c.d, a=v1a(0.4, restriction_ok))
    c.d["v2"]["gate"]["pass"] = not flag_v2
    if c1_flag:
        c1_fails(c)
    rec = c.record(c1_flag_decision=put_decision(c)) if c1_flag else c.record()
    D7.write_record(rec, c.repo, c.runs)
    vp = os.path.join(c.repo, "configs", "trackml", "overlay_version_h_v1.json")
    json.dump({"overlay_hash": OH, "restriction": rec["restriction"], "descriptive": False}, open(vp, "w"))
    commit(c.repo, D7.RECORD_PATHS["hval"], "configs/trackml/overlay_version_h_v1.json", msg="step 4")
    split = copy.deepcopy(B.load_split())
    split.update(htest=[3001, 3005, 3007], htest_spares=[3010, 3002], htest_dropped=[
        {"host": 3005, "reason": "no hard-scatter vertex", "replacement": 3010}])
    ids = B.set_ids("htest", split)
    assert ids == [52_003_001, 52_003_007, 52_003_010]
    work = os.path.join(c.root, "work_htest")
    tp03 = os.path.join(c.tmp, "derived", "tp03_v1")
    t = {"truth_checks": {"set": "htest", "root": c.root, "work": work, "events": ids,
                          "version": {"overlay_hash": OH, "pass": True, "committed_version": True},
                          "format": {"host_root": tp03}, "pass": {k: True for k in D7.TRUTH_KEYS}},
         "gun_content": {"set": "htest", "root": c.root, "work": work, "events": len(ids),
                         "nav_losses": {"gun_particles": 40, "lost": 1, "pp_like": 20, "pp_like_lost": 1,
                                        "pp_like_lost_share": 0.05}}}
    c.meta("htest", OH)
    return c, split, t, tp03


def write_inputs(c, t):
    out = {}
    for k, v in t.items():
        p = os.path.join(c.tmp, "ht1", f"{k}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(v, open(p, "w"))
        out[k] = p
    return out


@needs_split
def test_htest_record_from_the_committed_hval_record(tmp_path):
    c, split, t, tp03 = htest_case(tmp_path)
    rec = D7.htest_record(write_inputs(c, t), c.repo, c.root, split, c.runs, host_root=tp03)
    assert rec["set"] == "htest" and rec["restriction"] == ["fwd+", "fwd-"] and rec["flags"] == ["V2"]
    assert rec["descriptive"] is False and rec["lost_share"] == 0.05 and rec["passes"] == [OH]
    assert rec["hval_record"]["path"] == D7.RECORD_PATHS["hval"] and rec["version_record"]["committed"]
    path = D7.write_record(rec, c.repo, c.runs)
    assert os.path.relpath(path, c.repo) == "results/pileup_hp/h_t1_d7_v1.json"
    assert path.startswith(os.path.join(c.repo, B.HT1_RECORDS))
    commit(c.repo, D7.RECORD_PATHS["htest"])
    rr = H.restriction_record(path, c.repo, "htest")
    assert rr["restriction"] == ["fwd+", "fwd-"] and rr["flags"] == ["V2"] and rr["lost_share"] == 0.05
    with pytest.raises(PermissionError, match="h_t05_"):
        H.restriction_record(path, c.repo, "hval")
    hv = os.path.join(c.repo, D7.RECORD_PATHS["hval"])            # the H-val record holds H-val's own share
    assert H.restriction_record(hv, c.repo, "hval")["lost_share"] == pytest.approx(0.029)


@needs_split
def test_htest_record_refusals(tmp_path):
    def run(c, split, t, tp03):
        return D7.htest_record(write_inputs(c, t), c.repo, c.root, split, c.runs, host_root=tp03)
    edits = [(lambda c, s, t: t["truth_checks"]["pass"].update(weights=False), "H-T1 stops"),
             (lambda c, s, t: t["truth_checks"]["version"].update(committed_version=False), "committed version"),
             (lambda c, s, t: t["truth_checks"]["format"].update(host_root="/x/derived/tp_v1"), "hosts not read"),
             (lambda c, s, t: t["truth_checks"]["version"].update(overlay_hash=OH2), "another overlay_hash"),
             (lambda c, s, t: c.meta("htest", OH2), "another overlay_hash"),
             (lambda c, s, t: t["truth_checks"].update(events=t["truth_checks"]["events"][:2]), "not every event"),
             (lambda c, s, t: t["gun_content"].update(set="hval"), "an output of set 'hval'"),
             (lambda c, s, t: t["gun_content"].update(work="/elsewhere"), "another work directory"),
             (lambda c, s, t: s.update(htest=None), "no H-test list")]
    for i, (f, msg) in enumerate(edits):
        c, split, t, tp03 = htest_case(tmp_path / f"e{i}")
        f(c, split, t)
        with pytest.raises(D7.Refused, match=msg):
            run(c, split, t, tp03)
    # the H-val record and the version record committed as is, with one overlay_hash
    c, split, t, tp03 = htest_case(tmp_path / "hv")
    with open(os.path.join(c.repo, D7.RECORD_PATHS["hval"]), "a") as fh:
        fh.write("\n")
    with pytest.raises(D7.Refused, match="h_t05_d7_v1.json is not committed as is"):
        run(c, split, t, tp03)
    c, split, t, tp03 = htest_case(tmp_path / "vr")
    vp = os.path.join(c.repo, "configs", "trackml", "overlay_version_h_v1.json")
    json.dump({"overlay_hash": OH2}, open(vp, "w"))
    with pytest.raises(D7.Refused, match="is not committed as is"):
        run(c, split, t, tp03)
    commit(c.repo, "configs/trackml/overlay_version_h_v1.json")
    with pytest.raises(D7.Refused, match="version record holds overlay_hash"):
        run(c, split, t, tp03)
    c, split, t, tp03 = htest_case(tmp_path / "vr2")
    vp = os.path.join(c.repo, "configs", "trackml", "overlay_version_h_v1.json")
    json.dump({"overlay_hash": OH, "restriction": None}, open(vp, "w"))
    commit(c.repo, "configs/trackml/overlay_version_h_v1.json")
    with pytest.raises(D7.Refused, match="version record holds restriction"):
        run(c, split, t, tp03)
    # written before R15 and before the H-test launch only
    c, split, t, tp03 = htest_case(tmp_path / "r15")
    rec = run(c, split, t, tp03)
    led = os.path.join(c.repo, "docs", "feedback", "test_use_ledger.md")
    os.makedirs(os.path.dirname(led))
    with open(led, "w") as fh:
        fh.write("# ledger\n\n### R14. other\n\ntext\n")
    assert os.path.basename(D7.check_writable("htest", c.repo, c.runs)) == "h_t1_d7_v1.json"
    with open(led, "a") as fh:
        fh.write("\n### R15. Addendum H step H-T2: H-test\n\n- names\n")
    with pytest.raises(D7.Refused, match="R15 exists"):
        D7.write_record(rec, c.repo, c.runs)
    with open(led, "w") as fh:
        fh.write("# ledger\n")
    os.makedirs(os.path.join(c.runs, "launch"))
    open(B.launch_path("htest", c.runs), "w").write("{}")
    with pytest.raises(D7.Refused, match="was launched"):
        D7.write_record(rec, c.repo, c.runs)
    with pytest.raises(D7.Refused, match="no record of decision D7"):
        D7.check_writable("pilot", c.repo, c.runs)
    bad = {**rec, "lost_share": 1.5}
    os.remove(B.launch_path("htest", c.runs))
    with pytest.raises(D7.Refused, match="lost_share"):               # the value checks of hnl.overlay.h1
        D7.write_record(bad, c.repo, c.runs)
    assert not os.path.exists(os.path.join(c.repo, D7.RECORD_PATHS["htest"]))


# ------------------------------------------------------------------------------------------------ real formats
PILOT_OUT = "/work/submit/jinboz1/hnl/pileup_design/h_fix/gates/out"
PILOT_GUN = "/work/submit/jinboz1/hnl/pileup_design/h_integrate/out/gun_content_pilot.json"
PILOT_V2 = "/work/submit/jinboz1/hnl/pileup_design/h_fix/chain_b/dry/v2_pilot.json"


@needs_split
@pytest.mark.skipif(not os.path.isdir(PILOT_OUT) or not os.path.exists(PILOT_GUN) or not os.path.exists(PILOT_V2),
                    reason="needs the pilot gate outputs of the integration runs")
def test_parsers_on_the_real_pilot_gate_outputs():
    """The record reads the gate outputs the modules write: the pilot outputs of jobs 6821095 (truth checks, C1,
    closure checks and gate, V1 dry run on placeholder grids), 6820567 (gun content) and the wiring run's V2 on
    placeholder scores. The pilot closure set was made with the pilot base seed, so its version check fails and the
    record is refused on it (the set would not go to the model)."""
    ld = lambda p: D7.load_input("x", p)[0]                                    # noqa: E731
    root = "/work/submit/jinboz1/hnl/pileup_design/h_integrate/root"
    pilot = B.set_ids("pilot")
    closure = [B.overlay_id(1, h) for h in B.set_hosts("pilot")]
    tc = D7.truth_gate(ld(f"{PILOT_OUT}/truth_checks_pilot.json"), "truth checks", "pilot", pilot, root)
    assert tc["outcome"] == "pass" and tc["overlay_hash"] == OH
    cc = D7.truth_gate(ld(f"{PILOT_OUT}/closure_checks_pilot.json"), "closure checks", "closure", closure, root,
                       D7.CLOSURE_KEYS)
    assert cc["outcome"] == "fail" and cc["failed_checks"] == ["version"]
    c1 = D7.c1_gate(ld(f"{PILOT_OUT}/c1_confirm_pilot.json"), "pilot", pilot, root)
    assert c1["outcome"] == "pass" and c1["part1"] == "not informative"
    cg = D7.closure_gate(ld(f"{PILOT_OUT}/closure_gate_pilot.json"), closure, root)
    assert cg["outcome"] == "not informative"
    with pytest.raises(D7.Refused, match="label"):                             # a dry run is never a production V1
        D7.v1_gates(ld(f"{PILOT_OUT}/v1_dry_closure_pilot.json"), closure, root)
    v1 = D7.v1_gates(ld(f"{PILOT_OUT}/v1_dry_closure_pilot.json"), closure, root, dry=True)
    raw = ld(f"{PILOT_OUT}/v1_dry_closure_pilot.json")
    assert v1["V1 (a)"]["outcome"] == ("pass" if raw["v1"]["a"]["rule"]["pass"] else "fail")
    assert v1["V1-Delta (a)"]["outcome"] == ("pass" if raw["v1_delta"]["decisive_pass"] else "fail")
    assert [g for g in ("V1 (b)", "V1 (d)") if v1[g]["outcome"] == "fail"] == [
        f"V1 ({x})" for x in raw["v1"]["flags"]]
    assert set(v1["V1 (a)"]["regions"]) == set(D7.REGIONS)
    with pytest.raises(D7.Refused, match="not hval"):                          # a pilot V2 is never H-val's
        D7.v2_gate(ld(PILOT_V2), pilot, None, None)
    v2 = D7.v2_gate(ld(PILOT_V2), pilot, None, None, dry=True)
    assert v2["outcome"] == ("pass" if ld(PILOT_V2)["gate"]["pass"] else "fail")
    nl = D7.nav_losses(ld(PILOT_GUN), "pilot", pilot, root)
    assert (nl["pp_like"], nl["pp_like_lost"]) == (57, 2) and nl["pp_like_lost_share"] == pytest.approx(2 / 57)
    gates = {"per-set truth checks, H-val": tc, "per-set truth checks, closure": cc, "C1 confirmation": c1,
             "closure truth gate": cg, **v1, "V2": v2}
    with pytest.raises(D7.Refused, match="per-set truth checks, closure failed"):
        D7.derive(gates, 1)
