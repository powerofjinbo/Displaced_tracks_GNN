#!/usr/bin/env python3
"""Export the Canto paper as a self-contained LaTeX source tree (Overleaf, arXiv).

The documents listed in DOCS are exported, each with its sections, tables and figures, and built with pdfLaTeX +
BibTeX. Since 2026-10-18 this is main.tex only (the first author dropped the Supplemental Material).

The working copy in paper/ carries bookkeeping comments: a trailing "% N:<ID>" on every line with a number (checked
against paper/NUMBERS.md) and header comments on the history of each file. The export drops every comment, copies only
the files that the documents read (sections, tables, the figures they include, references.bib) and adds
the .bbl of a fresh bibtex run of each, so that the tree compiles with pdfLaTeX + BibTeX as it is.

Comment rules (TeX semantics kept):
  * a line that holds only a comment is dropped (it contributes no token);
  * "text % comment" loses the whitespace and the comment (the newline then gives the one space the whitespace gave);
  * "text% comment" keeps a bare "%" (it suppresses the end-of-line space, e.g. after \\caption{\\label{...}%);
  * "\\%" is a literal percent sign, not a comment.

Usage: python3 scripts/paper_export.py OUTDIR   (OUTDIR must not exist; it is created, then zipped to OUTDIR.zip)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
DOCS = ("main.tex",)   # 2026-10-18: the paper only; the first author dropped the Supplemental Material
                        # (supplement.tex stays in the repository; add it back here to export it)
TEXLIVE = "/cvmfs/sft.cern.ch/lcg/external/texlive/2023/bin/x86_64-linux"


def comment_start(line: str) -> int:
    """Index of the first unescaped %, or -1."""
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2          # skip the escaped character (\%, \\, ...)
            continue
        if c == "%":
            return i
        i += 1
    return -1


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        k = comment_start(line)
        if k < 0:
            out.append(line.rstrip())
            continue
        head = line[:k]
        if head.strip() == "":
            continue                          # comment-only line
        if head[-1] in " \t":
            out.append(head.rstrip())         # "text % comment"
        else:
            out.append(head + "%")            # "text% comment": keep the space suppression
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)          # at most one blank line in a row
    return s.strip() + "\n"


def inputs(tex: str) -> list[str]:
    return re.findall(r"\\input\{([^}]+)\}", tex)


def graphics(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def main(outdir: str) -> None:
    out = Path(outdir).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    todo, seen, figs = list(DOCS), set(), []
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        src = PAPER / rel
        if not src.suffix:
            src = src.with_suffix(".tex")
            rel = rel + ".tex"
        tex = src.read_text()
        for r in inputs(tex):
            todo.append(r if r.endswith(".tex") else r + ".tex")
        figs += graphics(tex)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(strip_comments(tex))
    for f in sorted(set(figs)):
        src = PAPER / f
        if not src.exists() and (PAPER / (f + ".pdf")).exists():
            src, f = PAPER / (f + ".pdf"), f + ".pdf"
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(PAPER / "references.bib", out / "references.bib")
    left = [p for p in out.rglob("*.tex") if "% N:" in p.read_text()]
    if left:
        raise SystemExit(f"bookkeeping comments left in {left}")
    # build each document with pdfLaTeX + BibTeX, as Overleaf and arXiv do
    env = {"PATH": TEXLIVE + ":/usr/bin:/bin", "HOME": str(out)}
    for doc in DOCS:
        name = doc[:-4]
        for cmd in (["pdflatex", "-interaction=nonstopmode", name], ["bibtex", name],
                    ["pdflatex", "-interaction=nonstopmode", name], ["pdflatex", "-interaction=nonstopmode", name]):
            subprocess.run(cmd, cwd=out, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = (out / f"{name}.log").read_text(errors="replace")
        bad = [l for l in log.splitlines() if "undefined" in l.lower() or l.startswith("!")]
        if bad or not (out / f"{name}.pdf").exists():
            raise SystemExit(f"pdfLaTeX build problems in {doc}:\n" + "\n".join(bad[:20]))
    keep = {".tex", ".bib", ".bbl", ".pdf"}
    for p in list(out.iterdir()):
        if p.is_file() and (p.suffix not in keep or p.name.endswith("Notes.bib")):
            p.unlink()                                   # build by-products of this fresh export only (REVTeX writes mainNotes.bib)
    for doc in DOCS:
        name = doc[:-4]
        (out / f"{name}.pdf").rename(out.parent / f"{out.name}_{name}_build.pdf")
    (out / "README.txt").write_text(
        "Canto: a graph neural network tracker for displaced particles\n\n"
        "Main document: main.tex (REVTeX 4.2). Compiler: pdfLaTeX, bibliography: BibTeX (references.bib; main.bbl\n"
        "is included for arXiv). Figures are PDFs under figures/.\n")
    zpath = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))
    print(zpath)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
