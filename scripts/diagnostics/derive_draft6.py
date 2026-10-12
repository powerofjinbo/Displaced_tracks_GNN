"""Derived values quoted in paper draft 6 that combine several lines of the diagnostic outputs (sums, ranges, ratios);
each line names its inputs. Writes results/diagnostics/derived_draft6.md.

    python3 scripts/diagnostics/derive_draft6.py
"""
import os
import re

D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "diagnostics")
L = ["# Derived values for paper draft 6 (scripts/diagnostics/derive_draft6.py)", ""]
# same builder setting (MWS+M 0.2): ttbar-trained seeds vs HNL-trained, ttbar prompt and ttbar fake rate
cmp_ = [ln for ln in open(os.path.join(D, "ttbar_cmp_6717924.txt")) if "mwslayermerge 0.2" in ln]
tt = [ln for ln in cmp_ if ln.strip().startswith("Canto-L (ttbar)")]
prompt = [float(re.search(r"\(trackbuild_val_ext\)\s+([\d.]+)", ln).group(1)) for ln in tt]
fake = [float(ln.split()[-1]) for ln in tt]
L.append(f"- ttbar_cmp_6717924.txt, MWS+M 0.2, Canto-L (ttbar) s0-s2: ttbar prompt {min(prompt):.2f}-{max(prompt):.2f} %, "
         f"ttbar fake rate {min(fake):.2f}-{max(fake):.2f} %")
# ttbar training events: high-pT oblique particles by first-hit radius
rows = {}
for ln in open(os.path.join(D, "oblique_truth_6713296.txt")):
    m = re.match(r"\((\d+), (\d+)\]\s+(\d+)\s+(\d+)\s*$", ln)
    if m:
        rows[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
tot = sum(rows.values())
L.append(f"- oblique_truth_6713296.txt, ttbar train (2000 events), pT > 1.5 GeV and ratio > 0.7: {tot} particles, "
         f"{rows[(0, 50)]} with r_first <= 50 mm, {sum(v for (a, b), v in rows.items() if a >= 200)} beyond 200 mm")
# displaced ttbar targets: share and efficiency range over the |d0| bins (Canto-L and acorn, same builder)
txt = open(os.path.join(D, "ttbar_displaced_6717957.txt")).read()
n_all, n_sel = map(int, re.search(r"ttbar targets per val: (\d+)\s+with 1<=\|d0\|<100: (\d+)", txt).groups())
blk = txt.split("== d0_bin")[1].split("==")[0]
effs = []
for ln in blk.splitlines():
    m = re.match(r"\[[\d.]+, [\d.]+\)\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", ln)
    if m:
        effs += [float(x) for x in m.groups()]
L.append(f"- ttbar_displaced_6717957.txt: {n_sel} of {n_all} ttbar targets ({100 * n_sel / n_all:.0f} %) with 1 <= |d0| < 100 mm; "
         f"efficiency per |d0| bin {min(effs):.0f}-{max(effs):.0f} % (Canto-L ttbar, Canto-L HNL, acorn ttbar LM, acorn HNL LM)")
# classifier-stage loss rate by pT (Canto-L ttbar)
loss = {}
for ln in open(os.path.join(D, "ttbar_loss_6717959.txt")):
    m = re.match(r"\[([\d.]+), ([\d.]+)\)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s*$", ln)
    if m:
        loss[float(m.group(1))] = float(m.group(3))
L.append(f"- ttbar_loss_6717959.txt: classifier-stage loss {loss[1.0]:.0f} % at pT 1-2 GeV, {loss[10.0]:.0f} % above 10 GeV")
m = re.search(r"'classifier': (\d+), 'graph': \d+\} (\d+)", open(os.path.join(D, "ttbar_loss_6717959.txt")).read())
k, n = int(m.group(1)), int(m.group(2))
L.append(f"- ttbar_loss_6717959.txt: HNL daughters at 1-100 mm lost at the classifier stage: {k} of {n} ({100 * k / n:.1f} %)")
open(os.path.join(D, "derived_draft6.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
