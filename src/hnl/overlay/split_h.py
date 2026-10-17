"""Sets, hash rules and production seeds of addendum H (docs/preregistration_pileup.md; docs/pileup_overlay_plan.md
sections 5.4, 6 and 9).

`make` writes configs/trackml/split_h_v1.json at H-T0: the overlay sets with their set index k and base seed, the
H-val and closure hosts (tp_v1 val), the pilot hosts, the H-test hash rule with its per-event and per-set checks and
the return rule, the Fatras event numbering and the excluded sets. training_part03 is not streamed at H-T0, so the
H-test ids are fixed by the rule alone; `htest` applies it at H-T1 to the list of complete part03 events that pass the
converter's data-quality rule, and the result is added to the file then.

Hash rule: key(id) = int(sha256("canto-trackml-overlay-v1:<id>").hexdigest()[:8], 16) with <id> the decimal event id
without leading zeros (23001, not the file name's 000023001); events in increasing key, ties by increasing id; the
first N_HTEST are H-test, the rest are spares in that order.

Fatras event numbers (`fatras_event_numbers`) are fixed per host before its set is simulated, so that a set split
over several jobs, or a dropped host, shifts no ACTS stream: H-val and closure = position of the host in the val list
(increasing id), 0-149; H-test = rank of the host in the hash order of all eligible part03 events (H-test hosts 0-599,
spares from 600; a spare that replaces a dropped host keeps its own rank); pilot = position in the pilot list, 0-4
(as the H-P dry run). Production numbers must lie in the range of the seed check (0 to N_EVENT_NEW - 1). The drivers
(hnl.overlay.gun, hnl.overlay.closure, slurm/overlay/*.sbatch) read the base seed (`base_seed`) and these numbers from
the committed split file.

`seeds` checks the production base seeds by enumeration, as for generic_iso_x1 (RESULTS.md, addendum G T1 deviation
13): every ACTS stream of a new set, (base + 13) + Cantor(algorithm, event) mod 2^32 for algorithms 1-16 and Fatras
events 0-999, against every ACTS stream recorded in this project (EARLIER) and against the other new sets. ACTS v15
seeds a std::mt19937 per (algorithm, event) with seed + Cantor(algorithm, event)
(Examples/Framework/src/Framework/RandomNumbers.cpp); algorithm numbers start at 1. The numpy streams of the overlay
(gun default_rng([base + 7, overlay id]), digitiser default_rng([base + 13, overlay id]), emulation
default_rng([base + 13, overlay id, 1])) carry the overlay id (>= 50,000,000, set index in the millions digit) as
their second word; the check confirms that no new entropy tuple equals one of the pilot set or of an earlier sample.

    python -m hnl.overlay.split_h make [--out configs/trackml/split_h_v1.json]
    python -m hnl.overlay.split_h fatras --set hval --hosts 21117,21124     (prints the Fatras event numbers)
    python -m hnl.overlay.split_h seeds [--out results/pileup_hp/h_seed_check_v1.json]
    python -m hnl.overlay.split_h htest --eligible FILE [--out FILE]      (H-T1 only; FILE = one id per line)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_TP = os.path.join(REPO, "configs", "trackml", "split_tp_v1.json")
OUT_SPLIT = os.path.join(REPO, "configs", "trackml", "split_h_v1.json")
OUT_SEEDS = os.path.join(REPO, "results", "pileup_hp", "h_seed_check_v1.json")

VERSION = "h_v1"
HTEST_PREFIX = "canto-trackml-overlay-v1"
PILOT_PREFIX = "canto-trackml-overlay-v1-pilot"
PROMPT_PREFIX = "canto-trackml-overlay-v1-prompt"
N_HTEST = 600
N_PILOT = 5
OVERLAY_FIRST_ID = 50_000_000
GUN_OFFSET, SIM_OFFSET = 7, 13

# set: (set index k, base seed). Production bases = 600,000,001 + 10,000,000 k, fixed at H-T0 and checked by `seeds`.
# The plan's v2 values (650001, 950001, 750001, 850001) coincide with the HNL-point bases 650001, 750001 and 850001
# (docs/hnl_points_spec.md section 5). Below about 2.9e8 the earlier ACTS streams of this project (generic_iso_x1 reaches
# 260014 + Cantor(16, 23999) = 288,656,133) are dense enough that a first try at 2,000,001-4,000,001 gave 4,376-6,797
# coincidences per set (pileup_design/hp_addendum/h_seed_check_try1.json, job 6820083).
SETS = {"hval": (0, 600_000_001), "closure": (1, 610_000_001), "htest": (2, 620_000_001), "pilot": (9, 990_101)}
PRODUCTION = ("hval", "closure", "htest")
N_ALG = 16                      # algorithm numbers 1..N_ALG (generic_iso_x1 used 1-12)
N_EVENT_NEW = 1000              # Fatras events 0..999 per set (H-val and closure 150, H-test 600)

# Every ACTS stream family recorded in this project: (name, base, offsets added to the base, first event, last event,
# source). Offsets (0, 13) cover the generation step and the simulation/digitisation step of a production base.
EARLIER = [
    ("ttbar validation (spec v1)", 150001, (0, 13), 0, 999, "docs/new_samples_spec.md s4, s8"),
    ("generic_iso_v1", 250001, (0, 13), 0, 7999, "docs/new_samples_spec.md s3-4"),
    ("generic_dense_v1", 350001, (0, 13), 0, 7999, "docs/new_samples_spec.md s3-4"),
    ("hss15_100_v1", 450001, (0, 13), 0, 1499, "docs/new_samples_spec.md s3-4"),
    ("hss40_300_v1", 550001, (0, 13), 0, 1499, "docs/new_samples_spec.md s3-4"),
    ("generic_iso_x1", 260001, (0, 13), 8000, 23999, "docs/new_samples_spec.md s9"),
    ("hnl5_50", 650001, (0, 13), 0, 2999, "docs/hnl_points_spec.md s5"),
    ("hnl15_500", 750001, (0, 13), 0, 2999, "docs/hnl_points_spec.md s5"),
    ("hnl20_200", 850001, (0, 13), 0, 2999, "docs/hnl_points_spec.md s5"),
    ("hnl10_100_ref", 1050001, (0, 13), 0, 2999, "docs/hnl_points_spec.md s5"),
    ("hnl15_200_ref", 1150001, (0, 13), 0, 2999, "docs/hnl_points_spec.md s5"),
    ("overlay pilot (H-P dry run, base 990101)", 990114, (0,), 0, 4, "results/pileup_hp/hp_summary.md s3.1"),
    ("overlay dry run and gun probe (base 990001)", 990014, (0,), 0, 599, "slurm/acts/hp_field_gates.sbatch"),
    ("C1 gun sample (base 910001)", 910014, (0,), 0, 1999, "results/pileup_hp/c1_v1.md"),
    ("F1 field probe", 790, (0,), 0, 999, "slurm/acts/hp_field_gates.sbatch"),
    ("emulation Fatras packs", 20261014, (0,), 0, 999, "pileup_design/hp_overlay/emul/fatras_out/fatras.json"),
    ("G0 smoke test", 4242, (0,), 0, 999, "pileup_design/hp_geometry/smoke_fatras.sbatch"),
    ("design-study probes", 1, (0,), 0, 999, "pileup_design/field_material/acts_probe.py"),
    ("design-study probes", 777, (0,), 0, 999, "pileup_design/hit_format/acts_planar.py"),
    ("ACTS default --rnd-seed", 1234567890, (0,), 0, 999, "src-v15 CommonOptions.cpp"),
    ("seeds in job logs", 20261005, (0, 13), 0, 999, "logs/*.out"),
    ("seeds in job logs", 20261006, (0, 13), 0, 999, "logs/*.out"),
]
# HNL pilot, planning and lifetime-check bases (results/hnl_points_closure_v1.md s7, results/hnl_points_lifetime_v1.json,
# job logs). The ranges are expanded at a step of 1,000, a superset of the recorded bases, with 4,000 events each and
# both offsets. Most of these runs drew from std::seed_seq or Pythia's own generator, not from ACTS streams; they are
# included anyway.
PILOT_RANGES = [(961001, 972001), (991001, 992001), (9710001, 9760001), (9810001, 9930001), (9960001, 10000001),
                (10110001, 10160001), (10210001, 10280001), (10410001, 10420001), (10510001, 10520001),
                (10610001, 10610001), (10700001, 10700001), (10710001, 10770001), (31000001, 31000001)]
PILOT_EXTRA = [10700002, 10230001]
PILOT_EVENTS = (0, 3999)


def key(prefix: str, *parts) -> int:
    return int(hashlib.sha256(":".join([prefix, *map(str, parts)]).encode()).hexdigest()[:8], 16)


def hash_order(ids, prefix: str = HTEST_PREFIX) -> list[int]:
    return sorted((int(i) for i in ids), key=lambda i: (key(prefix, i), i))


def htest_lists(eligible) -> dict:
    o = hash_order(eligible)
    return {"htest": sorted(o[:N_HTEST]), "htest_hash_order": o[:N_HTEST], "spares_hash_order": o[N_HTEST:]}


def _sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def htest_hosts(split: dict) -> list[int]:
    """The evaluated H-test hosts in increasing id: the htest list recorded at H-T1 after the drops of htest_dropped
    (one record {"host", "reason", "replacement"} per drop, in the order of the drops). A drop removes a host of the
    current list and gives a reason; its replacement must be the first spare in hash order (htest_spares) that has not
    joined the list before. No id leaves htest or htest_spares, so no rank and no Fatras event number changes. Raises
    LookupError before H-T1 and ValueError for a record that breaks the rule."""
    if split.get("htest") is None or split.get("htest_spares") is None:
        raise LookupError("the H-test hosts and spares are added to the split file at H-T1")
    htest, spares = [int(i) for i in split["htest"]], [int(i) for i in split["htest_spares"]]
    if len(set(htest)) != len(htest) or set(htest) & set(spares):
        raise ValueError("htest repeats an id or shares one with htest_spares")
    cur, joined = set(htest), set(htest)
    for d in split.get("htest_dropped") or []:
        h, r = int(d["host"]), int(d["replacement"])
        if not str(d.get("reason") or "").strip():
            raise ValueError(f"the drop of host {h} gives no reason")
        if h not in cur:
            raise ValueError(f"dropped host {h} is not in the current H-test list")
        nxt = next((s for s in spares if s not in joined), None)
        if r != nxt:
            raise ValueError(f"host {h} is replaced by {r}, not by the next spare in hash order ({nxt})")
        cur.remove(h)
        cur.add(r)
        joined.add(r)
    return sorted(cur)


def htest_sha256(split: dict) -> str:
    """sha256 of the evaluated H-test host ids (`htest_hosts`, increasing, comma-joined): the H-test list sha256 that
    ledger entry R15 names (first 8 hex digits at least)."""
    return _sha(htest_hosts(split))


def overlay_id(set_name: str, host: int) -> int:
    return OVERLAY_FIRST_ID + 1_000_000 * SETS[set_name][0] + int(host)


# ------------------------------------------------------------------------------------------------ reading the split
def load_split(path: str = OUT_SPLIT) -> dict:
    with open(path) as fh:
        return json.load(fh)


def split_sha256(path: str = OUT_SPLIT) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def base_seed(set_name: str, split: dict | None = None, path: str = OUT_SPLIT) -> int:
    """The base seed of a set as the split file records it (fixed at H-T0)."""
    split = load_split(path) if split is None else split
    if set_name not in split["sets"]:
        raise KeyError(f"no set {set_name!r} in the split file (sets: {sorted(split['sets'])})")
    return int(split["sets"][set_name]["base_seed"])


def parse_hosts(hosts) -> list[int]:
    """Host ids from a list or from a string separated by commas or colons; no id twice."""
    if isinstance(hosts, str):
        hosts = [x for x in hosts.replace(":", ",").split(",") if x.strip()]
    out = [int(h) for h in hosts]
    if len(set(out)) != len(out):
        raise ValueError("a host is listed twice")
    return out


def fatras_order(set_name: str, split: dict) -> list[int]:
    """The host order that numbers the Fatras events of a set (module docstring): position = Fatras event number."""
    if set_name in ("hval", "closure"):
        order = [int(i) for i in split[set_name]]
        if order != sorted(order):
            raise ValueError(f"the {set_name} list of the split file is not in increasing id")
        return order
    if set_name == "pilot":
        return [int(i) for i in split["sets"]["pilot"]["hosts"]]
    if set_name == "htest":
        if split.get("htest") is None or split.get("htest_spares") is None:
            raise LookupError("the H-test hosts and spares are added to the split file at H-T1; before that no H-test "
                              "host has a Fatras event number")
        htest, spares = [int(i) for i in split["htest"]], [int(i) for i in split["htest_spares"]]
        order = hash_order([*htest, *spares])
        if sorted(order[:N_HTEST]) != sorted(htest) or order[N_HTEST:] != spares:
            raise ValueError("htest and htest_spares of the split file do not follow the hash rule")
        return order
    raise KeyError(f"no Fatras numbering for set {set_name!r}")


def fatras_event_numbers(set_name: str, hosts, split: dict | None = None, path: str = OUT_SPLIT) -> dict[int, int]:
    """{host: Fatras event number} of the given hosts of a set by the rule of the split file (module docstring). Every
    host must belong to the set; production numbers must lie in the seed check's range 0..N_EVENT_NEW - 1."""
    split = load_split(path) if split is None else split
    hosts = parse_hosts(hosts)
    rank = {h: i for i, h in enumerate(fatras_order(set_name, split))}
    missing = [h for h in hosts if h not in rank]
    if missing:
        raise KeyError(f"{len(missing)} hosts are not in the {set_name} list of the split file, first {missing[:5]}")
    out = {h: rank[h] for h in hosts}
    if set_name in PRODUCTION:
        high = sorted(n for n in out.values() if n >= N_EVENT_NEW)
        if high:
            raise ValueError(f"Fatras event numbers {high[:5]} lie beyond the seed check (events 0-{N_EVENT_NEW - 1}); "
                             "extend `seeds` before simulating them")
    return out


def make_split(out: str = OUT_SPLIT) -> dict:
    with open(SPLIT_TP, "rb") as fh:
        raw = fh.read()
    tp = json.loads(raw)
    val = sorted(int(i) for i in tp["val"])
    pilot = sorted(hash_order(val, PILOT_PREFIX)[:N_PILOT])
    sets = {}
    for name, (k, base) in SETS.items():
        sets[name] = {"k": k, "overlay_ids": f"{OVERLAY_FIRST_ID} + 1000000 * {k} + host",
                      "base_seed": base, "gun_seed": base + GUN_OFFSET, "fatras_digitiser_emulation_seed": base + SIM_OFFSET}
    sets["hval"].update(hosts="tp_v1 val", n_hosts=len(val), role="development, gates V2 and V3, C1 confirmation; "
                        "val estimate of H1 after the overlay_hash commit; no claim")
    sets["closure"].update(hosts="tp_v1 val", n_hosts=len(val), role="gate V1: native val targets of the V1 "
                           "populations removed and simulated again by the overlay chain; no gun LLPs")
    sets["htest"].update(hosts="training_part03 by the H-test hash rule (ids added at H-T1)", n_hosts=N_HTEST,
                         role="H1 (H-T2), read once")
    sets["pilot"].update(hosts=pilot, n_hosts=len(pilot), role="H-P dry run only; not used for any result",
                         note="hosts = the first five tp_v1 val ids in increasing key('" + PILOT_PREFIX + "', id); "
                              "base seed 990101 (recorded at H-P), Fatras events 0-4. An earlier three-host dry run "
                              "(21117, 21124, 21127) used base 990001 in scratch. The V1 closure dry run (job 6820120, "
                              "results/pileup_hp/v1_machinery_v1.md) used the closure ids 51,021,289-51,021,835 of "
                              "these five hosts with the pilot base 990101 and Fatras events 0-4, so its numpy tuples "
                              "begin with 990114 and its ACTS streams equal those of the gun pilot; its tables were "
                              "written to pileup_design/hp_dryrun/closure_v1, not to the production root, and enter "
                              "no result")
    rec = {
        "version": VERSION,
        "addendum": "docs/preregistration_pileup.md (addendum H)",
        "plan": "docs/pileup_overlay_plan.md (design v2, bd7e357), sections 5.4, 6 and 9",
        "based_on": {"file": "configs/trackml/split_tp_v1.json", "sha256": hashlib.sha256(raw).hexdigest(),
                     "val_sha256": _sha(val)},
        "split_tp_v1_scope": ("split_tp_v1.json does not cover training_part03. Its rule text reads 'train = every "
                              "other event of the nine parts', but its scope field limits the rule to the two "
                              "streamed parts (training_part01, training_part02). training_part03 was never streamed "
                              "and no model has read any of its events; under addendum H its events are H-test hosts "
                              "and spares only."),
        "overlay_id_rule": "overlay id = 50,000,000 + 1,000,000 * k + host id (TrackML ids are below 1,000,000)",
        "sets": sets,
        "hval": val,
        "closure": val,
        "htest_rule": {
            "eligible": ("every complete training_part03 event (all four CSV members present and non-empty) that "
                         "passes the data-quality rule of hnl.trackml.convert.event_tables"),
            "key": "int(sha256('canto-trackml-overlay-v1:<id>').hexdigest()[:8], 16)",
            "id_format": ("<id> is the decimal event id without leading zeros (for example "
                          "'canto-trackml-overlay-v1:21117' for the member event000021117-hits.csv), not the "
                          "zero-padded file name"),
            "order": "increasing key, ties by increasing id",
            "htest": f"the first {N_HTEST} eligible events in that order",
            "spares": ("the remaining eligible events in that order; no model reads them. A host dropped by a "
                       "per-event check is replaced by the next spare in that order"),
            "record_at_h_t1": ("htest = the first 600 eligible ids in hash order, listed in increasing id; "
                               "htest_spares = the remaining eligible ids in hash order; htest_dropped = one record "
                               "{host, reason, replacement} per dropped host, in the order of the drops, the "
                               "replacement being the first spare in hash order that has not joined the list "
                               "before. Every eligible id stays in htest or htest_spares, so no rank, and no Fatras "
                               "event number, changes. The evaluated H-test hosts are htest after the replacements "
                               "(hnl.overlay.split_h.htest_hosts); the sha256 of their ids in increasing order, "
                               "comma-joined (hnl.overlay.split_h.htest_sha256), is the H-test list sha256 that ledger "
                               "entry R15 names, and slurm/overlay/guard_h.sh passes only their overlay ids"),
            "per_event_checks": {
                "rule": ("a host is dropped with its reason, and replaced by the next spare, only for a property of "
                         "the host event itself; where possible these are checked before the gun of that host is "
                         "simulated. The spare keeps its own Fatras event number (fatras_event_rule)"),
                "checks": ["no hard-scatter vertex by the rule of hnl.overlay.gun (hard_scatter_vertex)",
                           "host weights that do not sum to one within 1e-6 in the host's own stored rows",
                           ("host particles that break a precondition of the merge, such as a vertex field above "
                            "1000 or a repeated particle id")],
            },
            "per_set_checks": {
                "rule": "every other check is a check of the whole set and stops H-T1 when it fails",
                "checks": [("the per-set truth checks of addendum H section 9.2 (converter, weight sums after the "
                            "merge, containment, host rows), and any Fatras failure other than a logged navigator "
                            "loss"),
                           ("Version: overlay_hash and every value of the provenance manifest equal the committed "
                            "ones (configs/trackml/overlay_version_h_v1.json), and the base seed equals that of this "
                            "file"),
                           ("No earlier read: the part03 ids are disjoint from every id in split_tp_v1.json and "
                            "split_v1.json, and no part03 event has the same hits (a hash of x, y, z and module over "
                            "all hits) as a tp_v1 event"),
                           "the H-test overlay reads its hosts from derived/tp03_v1, named explicitly"],
            },
            "return_rule": ("after a per-set failure the cause is found on truth and logged in RESULTS.md. A fix that "
                            "touches neither overlay_hash nor a file of the manifest (a driver or a check) rebuilds "
                            "the H-test set and is not a pass. A fix that touches either one returns to H-T0.5 once, "
                            "if a pass remains: H-val and closure are made again with the fixed version, every val "
                            "gate runs again, the new overlay_hash and manifest are committed and marked 'made after "
                            "the H-val estimate', and the H-val estimate is computed again and reported beside the "
                            "first. If no pass remains, H becomes descriptive"),
            "code": "hnl.overlay.split_h.htest_lists",
            "fixed_at": "H-T0 (rule); ids at H-T1, before any model reads an overlay of a part03 event",
        },
        "fatras_event_rule": {
            "hval": "position of the host in the hval list of this file (increasing id): 0-149",
            "closure": "position of the host in the closure list of this file (the hval list, increasing id): 0-149",
            "htest": ("rank of the host in the hash order of all eligible training_part03 events (htest_rule), from "
                      f"0: the {N_HTEST} H-test hosts take 0-{N_HTEST - 1} and the spares {N_HTEST} onward"),
            "pilot": "position of the host in the pilot host list (increasing id): 0-4, the numbers of the H-P dry run",
            "fixed": ("before the set is simulated. A host keeps its number when a set is split over several jobs and "
                      "when another host is dropped; a spare that replaces a dropped host keeps its own rank"),
            "seed_check": (f"the seed check covers Fatras events 0-{N_EVENT_NEW - 1}; a host of rank {N_EVENT_NEW} or "
                           "more needs the check extended before it is simulated, and the code refuses it until then"),
            "streams": ("ACTS: (base + 13) + Cantor(algorithm, Fatras event number) mod 2^32, the event number being "
                        "that of the particle file event<N>-particles.csv. The numpy streams of the gun, the "
                        "digitiser and the emulation carry the overlay id, not the Fatras event number"),
            "code": ("hnl.overlay.split_h.fatras_event_numbers; the drivers pass the numbers explicitly "
                     "(hnl.overlay.gun, hnl.overlay.closure make, slurm/overlay/overlay_set.sbatch and "
                     "closure_set.sbatch)"),
        },
        "v1_prompt_subset_rule": ("the first 3,000 prompt gen-0 val targets (|d0| < 1 mm and r_v < 1 mm) in "
                                  "increasing int(sha256('canto-trackml-overlay-v1-prompt:<event>:<particle_id>')"
                                  ".hexdigest()[:8], 16), ties by event, then particle_id; event = tp_v1 host id, "
                                  "particle_id = the native id in decimal"),
        "htest": None,
        "htest_spares": None,
        "htest_dropped": None,
        "excluded": {
            "tp_v1 train (1,399 stored)": "no evaluation; emulation inputs from truth only",
            "tp_v1 test (300)": "read once for the model card; excluded",
            "T2 (accuracy phase, 300)": "read once for the model card; excluded",
            "training_part04 ... 09": "not used; training_part04 is the reserve for a later retraining study",
            "H-train": "none: no retraining arm in addendum H (DECISIONS 2026-10-14)",
        },
        "sha256": {"hval": _sha(val), "closure": _sha(val), "pilot": _sha(pilot)},
        "seed_check": os.path.relpath(OUT_SEEDS, REPO),
    }
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    return rec


# ------------------------------------------------------------------------------------------------ seed check
def cantor(a: np.ndarray, e: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    e = e.astype(np.int64)
    return (a + e) * (a + e + 1) // 2 + e


def cantor_set(e_lo: int, e_hi: int) -> np.ndarray:
    a, e = np.meshgrid(np.arange(1, N_ALG + 1), np.arange(e_lo, e_hi + 1), indexing="ij")
    return np.unique(cantor(a.ravel(), e.ravel()))


def families() -> list[dict]:
    fam = [{"name": n, "base": b, "offset": o, "events": [lo, hi], "source": s}
           for n, b, offs, lo, hi, s in EARLIER for o in offs]
    bases = sorted({b for lo, hi in PILOT_RANGES for b in range(lo, hi + 1, 1000)} | set(PILOT_EXTRA))
    for b in bases:
        for o in (0, SIM_OFFSET):
            fam.append({"name": "HNL pilot, planning or lifetime base (range expanded)", "base": b, "offset": o,
                        "events": list(PILOT_EVENTS), "source": "results/hnl_points_closure_v1.md s7"})
    return fam


def new_streams(name: str) -> np.ndarray:
    seed = SETS[name][1] + SIM_OFFSET
    a, e = np.meshgrid(np.arange(1, N_ALG + 1), np.arange(N_EVENT_NEW), indexing="ij")
    return np.unique((seed + cantor(a.ravel(), e.ravel())) % 2 ** 32)


def seed_check(out: str = OUT_SEEDS) -> dict:
    fam = families()
    csets: dict[tuple, np.ndarray] = {}
    new = {n: new_streams(n) for n in PRODUCTION}
    hits, per_set = [], {}
    for n, v in new.items():
        n_hit = 0
        for f in fam:
            lo, hi = f["events"]
            cs = csets.setdefault((lo, hi), cantor_set(lo, hi))
            seed = f["base"] + f["offset"]
            top = seed + int(cs[-1])
            # earlier values seed + c (mod 2^32); new value v coincides iff (v - seed) mod 2^32 is in cs
            d = (v.astype(np.int64) - seed) % 2 ** 32
            m = np.isin(d, cs, assume_unique=False)
            if top >= 2 ** 32:
                raise RuntimeError(f"family {f['name']} wraps at 2^32; check not written for that case")
            k = int(m.sum())
            if k:
                n_hit += k
                hits.append({"set": n, "family": f["name"], "base": f["base"], "offset": f["offset"],
                             "examples": [int(x) for x in v[m][:5]]})
        per_set[n] = {"streams": int(v.size), "range": [int(v[0]), int(v[-1])], "coincidences": n_hit}
    cross = {}
    names = list(PRODUCTION)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}/{b}"] = int(np.intersect1d(new[a], new[b]).size)
    pilot_v = np.unique((SETS["pilot"][1] + SIM_OFFSET + cantor(*[x.ravel() for x in np.meshgrid(
        np.arange(1, N_ALG + 1), np.arange(N_EVENT_NEW), indexing="ij")])) % 2 ** 32)
    for n in PRODUCTION:
        cross[f"{n}/pilot"] = int(np.intersect1d(new[n], pilot_v).size)
    # numpy entropy tuples: first words of the new sets against every first word used with an overlay id or an event
    first_words = {n: [SETS[n][1] + GUN_OFFSET, SETS[n][1] + SIM_OFFSET] for n in PRODUCTION}
    earlier_first = sorted({f["base"] + o for f in fam for o in (0, GUN_OFFSET, SIM_OFFSET)} |
                           {SETS["pilot"][1] + GUN_OFFSET, SETS["pilot"][1] + SIM_OFFSET, 990008, 990014, 910001})
    np_clash = {n: sorted(set(w) & set(earlier_first)) for n, w in first_words.items()}
    np_internal = len({w for ws in first_words.values() for w in ws}) == 2 * len(PRODUCTION)
    bases_used = sorted({f["base"] + f["offset"] for f in fam})
    rec = {
        "rule": ("ACTS v15: per (algorithm, event) a std::mt19937 seeded with (seed + Cantor(algorithm, event)) mod 2^32, "
                 "Cantor(a, e) = (a + e)(a + e + 1)/2 + e, algorithm numbers from 1 (RandomNumbers.cpp, Sequencer.cpp)"),
        "new_sets": {n: {"k": SETS[n][0], "base_seed": SETS[n][1], "gun_seed": SETS[n][1] + GUN_OFFSET,
                         "fatras_seed": SETS[n][1] + SIM_OFFSET, **per_set[n]} for n in PRODUCTION},
        "algorithms": [1, N_ALG], "new_events": [0, N_EVENT_NEW - 1],
        "earlier_families": len(fam), "earlier_streams": int(sum(N_ALG * (f["events"][1] - f["events"][0] + 1)
                                                                  for f in fam)),
        "coincidences_with_earlier": hits,
        "coincidences_between_new_sets_and_pilot": cross,
        "numpy_first_word_clashes": np_clash,
        "numpy_first_words_distinct_between_sets": np_internal,
        "numpy_note": ("the second word of every new numpy entropy tuple is an overlay id 50,000,000 + 1,000,000 k + "
                       "host with k in {0, 1, 2}; earlier tuples carry an event number below 24,000 or a pilot id "
                       "with k = 9, so no tuple can coincide even where a first word does"),
        "base_seeds_distinct": all(SETS[n][1] + o not in bases_used for n in PRODUCTION for o in (0, 7, 13)),
        "families": [{k_: f[k_] for k_ in ("name", "base", "offset", "events", "source")} for f in fam
                     if not f["name"].startswith("HNL pilot")],
        "pilot_ranges_expanded": {"ranges": PILOT_RANGES, "extra": PILOT_EXTRA, "step": 1000,
                                  "events": list(PILOT_EVENTS), "offsets": [0, SIM_OFFSET]},
    }
    rec["pass"] = (not hits and not any(cross.values()) and np_internal and rec["base_seeds_distinct"])
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1)
            fh.write("\n")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default=OUT_SPLIT)
    s = sub.add_parser("seeds")
    s.add_argument("--out", default=OUT_SEEDS)
    h = sub.add_parser("htest")
    h.add_argument("--eligible", required=True)
    h.add_argument("--out", default=None)
    f = sub.add_parser("fatras")
    f.add_argument("--set", required=True, choices=sorted(SETS))
    f.add_argument("--hosts", required=True, help="host ids separated by commas or colons")
    f.add_argument("--split", default=OUT_SPLIT)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        r = make_split(a.out)
        print(json.dumps({"sha256": r["sha256"], "pilot": r["sets"]["pilot"]["hosts"],
                          "sets": {k: v["base_seed"] for k, v in r["sets"].items()}}, indent=1))
    elif a.cmd == "fatras":
        split = load_split(a.split)
        fe = fatras_event_numbers(a.set, a.hosts, split)
        print(json.dumps({"set": a.set, "base_seed": base_seed(a.set, split),
                          "fatras_events": {str(k): v for k, v in fe.items()}}, indent=1))
    elif a.cmd == "seeds":
        r = seed_check(a.out)
        print(json.dumps({k: r[k] for k in ("new_sets", "coincidences_with_earlier",
                                             "coincidences_between_new_sets_and_pilot", "numpy_first_word_clashes",
                                             "base_seeds_distinct", "pass", "earlier_families", "earlier_streams")},
                         indent=1))
    else:
        with open(a.eligible) as fh:
            ids = [int(x) for x in fh.read().split()]
        r = htest_lists(ids)
        txt = json.dumps({**r, "n_eligible": len(ids), "sha256_htest": _sha(r["htest"])}, indent=1)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()
