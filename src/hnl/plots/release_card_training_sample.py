"""Release card of the training-sample study: Canto-L and acorn, each trained on the HNL signal, on ttbar + HNL and on
ttbar, all six with the same track builder (MWS-layer + learned merging), on HNL validation and on the one-time test.

The cells are those of the full card (hnl.plots.release_card_full: the same sources, cross-checks and cell format),
restricted to these six columns and to the HNL rows of its validation band, its test band (with the per-sample rows
and the external sample's displaced-track row) and its training band (without the TrackML lines). The first place of
each row is taken among these six columns only (the user's marking rule, 2026-10-12: bold, lightly filled and framed
in orange, whichever column it is in; no column singled out, no 2-sigma shading).

Supersedes figures/release/canto_card_training_sample_{mwslmerge,mwslayermerge} (2026-10-07: validation only, one-seed
previews, the old marking).

    python -m hnl.plots.release_card_training_sample --out figures/release/canto_card_training_sample [--dump cells.csv]
"""

from __future__ import annotations

import argparse
import dataclasses

from hnl.plots import release_card_full as F

COLUMNS = [
    ("CLH", "Canto-L (HNL)", ("{n} · HNL signal", "learned merging {wp}"), 0),
    ("CLTH", "Canto-L\n(ttbar+HNL)", ("{n} · ttbar+HNL", "learned merging {wp}"), 0),
    ("CL", "Canto-L (ttbar)", ("{n} · ttbar", "learned merging"), 0),
    ("AH", "acorn (HNL)", ("{n} · HNL signal", "learned merging"), 1),
    ("ATH", "acorn\n(ttbar+HNL)", ("{n} · ttbar+HNL", "learned merging"), 1),
    ("A", "acorn (ttbar)", ("{n} · ttbar", "learned merging"), 1),
]
GROUPS = ["Canto-L", "acorn"]
TITLE = "Canto-L · training sample"
SUBTITLE = ("Canto-L and acorn trained on the HNL signal, on ttbar + HNL and on ttbar, all with the same track builder "
            "· HNL validation and one-time test")
# bands of the full card that this card reads
VAL, TEST, EXT, TRAIN = ("HNL samples · validation", "HNL samples · one-time blind test",
                         "Wang et al.'s evaluation sample · hnl15_100_external", "Training · edge classifier")


def _restrict(row: F.Row, keys: list) -> F.Row:
    missing = [k for k in keys if k not in row.cells]
    if missing:
        raise SystemExit(f"{row.title} ({' / '.join(row.subs)}): no cell for {missing}")
    return dataclasses.replace(row, cells={k: row.cells[k] for k in keys})


def build(S: F.Sources) -> tuple[list, dict]:
    full, ctx = F.build(S)
    keys = [c[0] for c in COLUMNS]
    bands: dict = {}
    cur = None
    for it in full:
        if isinstance(it, F.Band):
            cur = bands.setdefault(it.title, [])
        else:
            cur.append(it)
    for b in (VAL, TEST, EXT, TRAIN):
        if b not in bands:
            raise SystemExit(f"full card: no band {b!r}")
    if len(bands[VAL]) != len(F.HNL_ROWS) or len(bands[TEST]) != len(F.HNL_ROWS) + 2:
        raise SystemExit("full card: the HNL val / test bands changed")

    PS = S.PS
    ev = lambda split, sset: int(PS.df[(PS.df.split == split) & (PS.df.set == sset)].events.iloc[0])
    per_seed = lambda sset: int(PS.one(split="test", set=sset, model=F.PS_MODEL["CL"]).n_per_seed)
    fr = F._pct(ctx["fr_star"])
    items: list = []
    items.append(F.Band("HNL samples · validation",
                        f"pooled HNL val (hnl10_100, hnl15_200; {F._n(ev('val', 'hnl_val_pooled'))} events) and ttbar "
                        f"val · mean over the seeds in the header · every model at a track fake rate ≤ FR* = {fr}, its "
                        "builder setting chosen per seed on these events"))
    items += [_restrict(r, keys) for r in bands[VAL]]
    items.append(F.Band("HNL samples · one-time test",
                        f"pooled HNL test (hnl10_100_test, hnl15_200_test, hnl15_100_external; "
                        f"{F._n(ev('test', 'hnl_test_pooled'))} events) and ttbar_test · evaluated once at the settings "
                        "chosen on val · Canto-L (HNL) and Canto-L (ttbar+HNL): trained after the freeze, settings "
                        "committed before their test jobs (post-freeze)"))
    items += [_restrict(r, keys) for r in bands[TEST]]
    ext = bands[EXT][0]                                  # the external sample's displaced-track row
    if ext.title != "Displaced tracks":
        raise SystemExit(f"full card: first row of {EXT!r} is {ext.title!r}")
    sub = (f"hnl15_100_external only · {F._n(per_seed('hnl15_100_external'))} per seed · its mass and lifetime in no "
           "training sample",)
    items.append(dataclasses.replace(_restrict(ext, keys), subs=sub))
    items.append(F.Band("Training · edge classifier", "the HNL-trained and ttbar+HNL-trained models use the training "
                        "split of hnl10_100 and hnl15_200"))
    for r in bands[TRAIN]:
        r = dataclasses.replace(_restrict(r, keys), subs=tuple(x.replace(" (TrackML: one window)", "")
                                                                for x in r.subs))
        cells = {k: dataclasses.replace(c, lines=tuple(l for l in c.lines if not l.startswith("TrackML:")))
                 for k, c in r.cells.items()}
        items.append(dataclasses.replace(r, cells=cells))
    return items, ctx


def notes(ctx: dict) -> list[tuple[str, str]]:
    fr = F._pct(ctx["fr_star"])
    out = [
        ("Samples and definitions.",
         "ACTS generic detector simulation without pileup: HNL signal samples (m_N, cτ in the sample name) and ttbar. "
         "Track efficiency by double-majority matching: more than half of the track's hits come from the particle "
         "and more than half of the particle's hits are on the track. Targets: pT > 1 GeV, |η| < 4, at least 5 hits. "
         "Displaced tracks: HNL-daughter targets with |d0| ≥ 10 mm or r_v ≥ 100 mm. Fake-track rate: the fraction of "
         "tracks with at least 5 hits that match no particle."),
        ("Models and working points.",
         "Every column runs MWS-layer + learned merging, the frozen track builder of Canto-L, at a track fake rate "
         f"≤ FR* = {fr} on pooled HNL val (preregistered). Canto-L (ttbar): the frozen Canto-L, selected by rules R6 "
         "and R8 before the test (results/final_selection_v1.json). Canto-L (HNL) and Canto-L (ttbar+HNL): the same "
         "recipe with only the training sample changed, one seed each, the builder setting (header) chosen on val by "
         "the builder rule at FR*; trained after the freeze, evaluated once on test (post-freeze, descriptive). "
         "acorn (HNL), acorn (ttbar+HNL) and acorn (ttbar): acorn's interaction network trained on the same three "
         "samples, three seeds each, with the val-best builder of the full menu chosen per seed (MWS-layer + learned "
         "merging for every seed)."),
        ("Seeds and uncertainties.",
         "Means over the training seeds in the column header; k/n pooled over the seeds. Second line of a cell: the "
         "combined uncertainty σ = (σ_boot² + s²/n)^½ of the mean, in percentage points, with σ_boot from an event "
         f"bootstrap (B = {F._n(ctx['B_val'])} on val, {ctx['B_test']} on test) and s the standard deviation over "
         "the n seeds. The |d0| ≥ 300 mm rows show k/n instead (statistics limited). Canto-L (HNL) and Canto-L "
         "(ttbar+HNL): one seed; their σ takes the seed standard deviation of Canto-L (ttbar) in the same row as its "
         "seed term; on val only the displaced-track row has a σ in the source, the other val rows show k/n, or "
         "'1 seed, no σ' for the fake rate."),
        ("Shading.",
         "Framed in orange: the first place of each row, the highest value (the lowest for rates; cells showing the "
         "same value share it), in whichever column it is. A difference below 2σ, σ the two cells' uncertainties in "
         "quadrature, is not resolved. Paired differences on the same events, with smaller uncertainties: "
         "results/training_sample_paired_test_v1.csv (test) and results/training_sample_v1.md (val)."),
        ("Training.",
         "Edge classifier only (results/model_table_v1.md); the metric-learning graph stage is not included. One "
         "step is one event graph. hnl15_100_external (m_N = 15 GeV, cτ = 100 mm) is Wang et al.'s evaluation "
         "sample and was used by no model in training or selection. All numbers of this card: HNL samples only; "
         "Canto-L on TrackML is on its own card."),
    ]
    return [(h, t.replace(" %", "\u00a0%").replace(" et al.", "\u00a0et\u00a0al.")) for h, t in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output stem: OUT.png and OUT.pdf")
    ap.add_argument("--dump", help="also write every cell (shown text, raw value, source) to this CSV")
    a = ap.parse_args(argv)
    F.COLUMNS, F.GROUPS, F.TITLE, F.SUBTITLE = COLUMNS, GROUPS, TITLE, SUBTITLE
    F.BOLD_FIRST_HEADER = False                         # no column singled out
    F.GEOM = {**F.GEOM, "cw": 2.75}
    S = F.Sources()
    items, ctx = build(S)
    if a.dump:
        F.dump(items, a.dump)
    print("\n".join(F.render(items, notes(ctx), a.out, ctx)))


if __name__ == "__main__":
    main()
