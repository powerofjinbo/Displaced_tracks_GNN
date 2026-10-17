"""Evaluation conditions of addendum H on overlay events (docs/preregistration_pileup.md section 5). Truth filters only;
no model code changes.

  C-full  ("full")  every hit, noise included: hnl.trackml.events.with_condition(ev, "full"), unchanged.
  C-1GeV  ("1gev")  hits of particles with pT > 1 GeV, noise removed: hnl.trackml.events.with_condition(ev, "1gev"),
                    unchanged (the model card's condition).
  C-hs    ("hs")    hits of the host hard scatter and of the gun: particles with vertex field (particle_id >> 52) 1
                    (the hard-scatter collision; its simulation secondaries keep the field, as in the converter's
                    is_hard_scatter = vprim == 1) or 1001-1010 (the gun). Every other pileup particle and every noise
                    hit is removed. This filter is new and is committed before it runs.

C-hs removes whole particles, so no chain of true edges breaks (hnl.trackml.events.subset keeps an edge when both hits
stay). Unlike the frozen conditions, C-hs also cuts the particles table to the kept particles. In C-1GeV every target
keeps all its hits (targets have pT > 1 GeV), so the frozen evaluation (hnl.trackml.evaluate.event_counts) sees only
targets with hits. In C-hs a pileup target loses all its hits, and in the full table it would count as an unmatched
target. Closure events (vertex fields 1024-2024, hnl.overlay.closure) run in C-full and C-1GeV only (addendum H section
10), so `hs_mask` refuses them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hnl.trackml import events as E

HS_VERTEX = 1                         # vertex field of the hard-scatter collision (hnl.overlay.gun.hard_scatter_vertex)
GUN_VERTEX = (1001, 1010)             # = hnl.overlay.writer.GUN_VERTEX
CLOSURE_VERTEX = (1024, 2024)         # = hnl.overlay.closure.CLOSURE_VERTEX
CONDITIONS = {"full": "C-full", "1gev": "C-1GeV", "hs": "C-hs"}


def vertex_field(pid) -> np.ndarray:
    return np.asarray(pid, np.int64) >> 52


def is_gun(pid) -> np.ndarray:
    v = vertex_field(pid)
    return (v >= GUN_VERTEX[0]) & (v <= GUN_VERTEX[1])


def is_closure(pid) -> np.ndarray:
    v = vertex_field(pid)
    return (v >= CLOSURE_VERTEX[0]) & (v <= CLOSURE_VERTEX[1])


def hs_particle(pid) -> np.ndarray:
    """Particles kept in C-hs: vertex field 1 (host hard scatter) or 1001-1010 (gun). Noise (pid 0) is never kept."""
    p = np.asarray(pid, np.int64)
    return (p != 0) & ((vertex_field(p) == HS_VERTEX) | is_gun(p))


def hs_mask(ev: E.Event) -> np.ndarray:
    """Hits kept in C-hs (bool per hit of the full event)."""
    if ev.condition != "full" or ev.rows is not None:
        raise ValueError(f"event {ev.event}: C-hs is taken from the full event, not from condition {ev.condition}")
    if is_closure(ev.pid).any():
        raise ValueError(f"event {ev.event}: closure events are not run in C-hs (addendum H section 10)")
    return hs_particle(ev.pid)


def with_condition(ev: E.Event, condition: str) -> E.Event:
    """The event in one condition of addendum H: "full" and "1gev" through the frozen hnl.trackml.events.with_condition,
    "hs" through `hs_mask` with the particles table cut to the kept particles."""
    if condition in ("full", "1gev"):
        return E.with_condition(ev, condition)
    if condition != "hs":
        raise ValueError(f"condition {condition!r} not in {tuple(CONDITIONS)}")
    if ev.particles is None:
        raise ValueError("condition hs needs the particles table")
    keep = hs_mask(ev)
    sub = E.subset(ev, keep, "hs")
    p: pd.DataFrame = ev.particles
    return replace(sub, particles=p[hs_particle(p.particle_id.to_numpy())].reset_index(drop=True))


def counts(ev: E.Event) -> dict:
    """Hits and particles of the event by kind (descriptive: what each condition keeps)."""
    pid = np.asarray(ev.pid, np.int64)
    v = vertex_field(pid)
    return {"hits": int(len(pid)), "noise": int((pid == 0).sum()), "hard_scatter": int(((v == HS_VERTEX) & (pid != 0))
                                                                                       .sum()),
            "gun": int(is_gun(pid).sum()), "closure": int(is_closure(pid).sum()),
            "kept_hs": int(hs_particle(pid).sum()) if not is_closure(pid).any() else None}
