"""Common figure style (brief §9): mplhep, Agg backend, PNG + PDF, simulation label on every figure."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

LABEL = "ACTS generic detector simulation, no pileup"
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                       "figures")
plt.style.use(hep.style.ROOT)


def new(figsize=(8, 6), **kw):
    fig, ax = plt.subplots(figsize=figsize, **kw)
    return fig, ax


def label(ax, extra: str | None = None, loc: str = "upper left"):
    """Simulation label (and an optional second line) inside the axes; keep the data clear of it with set_ylim."""
    text = LABEL + ("\n" + extra if extra else "")
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=13)


def axis_labels(ax, xlabel: str, ylabel: str, size: int = 18):
    ax.set_xlabel(xlabel, fontsize=size)
    ax.set_ylabel(ylabel, fontsize=size)


def save(fig, name: str, subdir: str = "") -> list[str]:
    d = os.path.join(FIG_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        out.append(p)
    plt.close(fig)
    return out
