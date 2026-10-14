"""Sets, hash rules and production seeds of addendum H (docs/preregistration_pileup.md; docs/pileup_overlay_plan.md
sections 5.4, 6 and 9).

`make` writes configs/trackml/split_h_v1.json at H-T0: the overlay sets with their set index k and base seed, the
H-val and closure hosts (tp_v1 val), the pilot hosts, the H-test hash rule and the excluded sets. training_part03 is
not streamed at H-T0, so the H-test ids are fixed by the rule alone; `htest` applies it at H-T1 to the list of complete
part03 events that pass the converter's data-quality rule, and the result is added to the file then.

Hash rule: key(id) = int(sha256("canto-trackml-overlay-v1:<id>").hexdigest()[:8], 16); events in increasing key,
ties by increasing id; the first N_HTEST are H-test, the rest are spares in that order.

`seeds` checks the production base seeds by enumeration, as for generic_iso_x1 (RESULTS.md, addendum G T1 deviation
13): every ACTS stream of a new set, (base + 13) + Cantor(algorithm, event) mod 2^32 for algorithms 1-16 and Fatras
events 0-999, against every ACTS stream recorded in this project (EARLIER) and against the other new sets. ACTS v15
seeds a std::mt19937 per (algorithm, event) with seed + Cantor(algorithm, event)
(Examples/Framework/src/Framework/RandomNumbers.cpp); algorithm numbers start at 1. The numpy streams of the overlay
(gun default_rng([base + 7, overlay id]), digitiser default_rng([base + 13, overlay id]), emulation
default_rng([base + 13, overlay id, 1])) carry the overlay id (>= 50,000,000, set index in the millions digit) as
their second word; the check confirms that no new entropy tuple equals one of the pilot set or of an earlier sample.

    python -m hnl.overlay.split_h make [--out configs/trackml/split_h_v1.json]
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


def overlay_id(set_name: str, host: int) -> int:
    return OVERLAY_FIRST_ID + 1_000_000 * SETS[set_name][0] + int(host)


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
                              "base seed 990101 (recorded at H-P). An earlier three-host dry run (21117, 21124, 21127) "
                              "used base 990001 in scratch")
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
            "order": "increasing key, ties by increasing id",
            "htest": f"the first {N_HTEST} eligible events in that order",
            "spares": ("the remaining eligible events in that order; no model reads them. A host whose overlay "
                       "fails a per-event check at H-T1 is dropped with its reason and replaced by the next spare"),
            "code": "hnl.overlay.split_h.htest_lists",
            "fixed_at": "H-T0 (rule); ids at H-T1, before any model reads an overlay of a part03 event",
        },
        "v1_prompt_subset_rule": ("the first 3,000 prompt gen-0 val targets (|d0| < 1 mm and r_v < 1 mm) in "
                                  "increasing int(sha256('canto-trackml-overlay-v1-prompt:<event>:<particle_id>')"
                                  ".hexdigest()[:8], 16), ties by event, then particle_id; event = tp_v1 host id, "
                                  "particle_id = the native id in decimal"),
        "htest": None,
        "htest_spares": None,
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
    a = ap.parse_args(argv)
    if a.cmd == "make":
        r = make_split(a.out)
        print(json.dumps({"sha256": r["sha256"], "pilot": r["sets"]["pilot"]["hosts"],
                          "sets": {k: v["base_seed"] for k, v in r["sets"].items()}}, indent=1))
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
