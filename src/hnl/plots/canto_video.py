"""Animated explainer video of the Canto pipeline (matplotlib animation + ffmpeg), narrated.

Every number shown on screen is taken from docs/canto_facts.md; ``check`` verifies that each number token of every
on-screen string occurs in that file and that no two visible texts overlap or leave the frame, and the narration
timing (subtitles, holds on static frames, invisible cuts, pauses, no dead air, .srt line lengths; see cmd_check).

Each scene is one function that builds a ``Scene`` (a full-frame 1920x1080 matplotlib figure plus time-dependent
updaters). ``render`` writes one MP4 clip per scene (FFMpegWriter, libx264, yuv420p, 30 fps) in parallel worker
processes and joins the clips with ffmpeg's concat demuxer.

Narration: canto_narration.json (next to this file, read once at import) gives each scene its lines: a window
[t0, t1] in scene time, the subtitle ("display", same markup as on-screen text) and the text spoken ("speech").
``Scene.narrate`` shows the subtitles and records the lines; ``lines`` writes the canto_tts.py input (ids
"<scene number:02d>_<scene>_<line:02d>", text = speech) to figures/video/narration/lines.json, and canto_tts.py
(Kokoro-82M, its own env, run through sbatch) writes one 24 kHz WAV per line and durations.json (each line's length,
silences included) to the same directory. Kokoro leaves ~0.3 s of silence before and ~0.5 s after the voice; a line
is placed trimmed to its voice plus TRIM_KEEP (0.1 s) of silence on each side, and "length" below is that trimmed
length.
Time warp (``Warp``): a line is spoken from 0.15 s after its window opens and needs the window open 0.35 s after
it ends. When the window is too short, the scene is held on a static frame (at t1 - 0.02, subtitle still fully on,
or at the latest earlier scene time of the window where nothing moves) for the difference, rounded up to whole frames.
When the window is longer, its slack after the speech is silent, and the static frames of that slack (nothing changes
on screen) are cut, so the picture never stands still without narration; frames that move stay. What moves when is
known exactly: every on-screen change goes through prog(), which ``scene_motion`` instruments. A scene's video length
is its scene time plus its holds minus its cuts; the render maps each video frame to a scene time. The full render
joins the clips, places every line at its video start time in one 24 kHz mono WAV (canto_explainer.wav, the mix as
synthesised), muxes it into canto_explainer.mp4 with the video stream copied: two-pass EBU R128 loudness
normalisation (ffmpeg loudnorm, -16 LUFS integrated, true peak at most -2 dBTP before the encoder, so at most -1.5
dBTP after it), AAC 192 kb/s at 48 kHz (AAC caps a 24 kHz mono stream at 144 kb/s), cut to the video's length, and
writes the subtitles to canto_explainer.srt (cues of at most two lines of at most 42 characters; a line too long for
that is split at a clause into two consecutive cues within its window).
Clips are cached per scene: next to each clip a .json records what it was rendered from (the scene time of every
frame, subtitles, holds, cuts, the scene function's source); a clip whose record differs from the current one is
stale and is not joined.

    python -m hnl.plots.canto_video lines [--submit]   # TTS input; prints (--submit: runs) the synthesis sbatch
    python -m hnl.plots.canto_video render [--scenes 2,11 | --scenes stale] [--preview] [--max-seconds 7.5] [--jobs 16]
    python -m hnl.plots.canto_video frames --scene 3 --times 2,8.5 [--outdir DIR] [--dpi 100]
    python -m hnl.plots.canto_video check
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import re
import subprocess
import time
import unicodedata
import wave
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont, fontManager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FFMPEG = "/work/submit/jinboz1/hnl/env/mamba/envs/media/bin/ffmpeg"
FFPROBE = "/work/submit/jinboz1/hnl/env/mamba/envs/media/bin/ffprobe"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "figures" / "video"
FACTS = REPO / "docs" / "canto_facts.md"

# narration: the lines of every scene, read once; the TTS output lives in figures/video/narration
NARR_JSON = Path(__file__).resolve().with_name("canto_narration.json")
NARRATION = json.loads(NARR_JSON.read_text(encoding="utf-8"))
LINES = NARRATION["scenes"]
NARR_DIR = OUT / "narration"
TTS_PY = "/work/submit/jinboz1/hnl/env/mamba/envs/tts/bin/python"
TTS_SCRIPT = Path(__file__).resolve().with_name("canto_tts.py")
SR = 24000                      # TTS and soundtrack sample rate (Hz)
SPEECH_LEAD, SPEECH_TAIL = 0.15, 0.35  # speech starts this long after its window opens / ends before it closes
TRIM_KEEP, TRIM_DB = 0.1, -50.0  # a line is placed as its voice (10 ms frames above TRIM_DB dBFS RMS) +- TRIM_KEEP s
HOLD_BACK = 0.02                # a hold freezes the frame at scene time t1 - HOLD_BACK (subtitle still fully on)
LOUDNORM = {"I": -16.0, "TP": -2.0, "LRA": 11.0}  # mux: EBU R128 target (LUFS, dBTP before the AAC encoder, LU)
SRT_WIDTH = 42                  # .srt: at most two lines of at most this many characters per cue
SUB_IN, SUB_OUT = 0.25, 0.2     # subtitle fade-in / fade-out (s); windows are >= SUB_OUT apart, so no cross-fade
SUB_Y = 108                     # subtitle centre line (px from the bottom)

# ----------------------------------------------------------------------------------------------------------------
# style
W, H, FPS = 1920, 1080, 30
BG, INK, INK2, MUTED, FAINT = "#FAF9F5", "#141413", "#3D3D3A", "#6B6A64", "#A29F96"
HAIR, LAYER, CARD, CARD2 = "#DDD9CF", "#D2CEC4", "#F2F0E9", "#E8E5DB"
ORANGE, ORANGE_D, ORANGE_M, ORANGE_T = "#D97757", "#B95F41", "#E8A68C", "#F6E2D9"
BLUE, BLUE_T = "#5B7DA1", "#DDE5EE"
BROWN, BROWN_M, BROWN_T = "#8C564B", "#B98D83", "#EEDFDB"
TRACK = ("#008300", "#4a3aa7", "#eda100", "#e87ba4")  # validated all-pairs set (dataviz validator, light mode)
PROMPT, HIT, EDGE0, LOW = "#8E8B82", "#3B3A36", "#B7B3A9", "#E3E0D7"
TITLE, H2, BODY, LABEL, SMALL, CAP = 40, 26, 23, 22, 21, 25  # points; 1 pt = 100/72 px at 1080p
PX_PT = 14.0                     # scene 6: px per point of edge-TPR change, same for dD and dT
SIM = "ACTS generic detector simulation, no pileup"
FONTS = ["Noto Sans", "DejaVu Sans"]
# Noto Sans (SIL Open Font License) is shipped next to the video: some compute nodes lack /usr/share/fonts/google-noto
FONT_DIR = OUT / "fonts"
for _f in sorted(FONT_DIR.glob("NotoSans-*.ttf")):
    fontManager.addfont(str(_f))


def check_fonts():
    """Fail instead of silently rendering with a fallback font."""
    for w in ("normal", "semibold", "bold"):
        path = findfont(FontProperties(family="Noto Sans", weight=w), fallback_to_default=False)
        if "NotoSans" not in Path(path).name:
            raise RuntimeError(f"Noto Sans ({w}) not found, got {path}")

matplotlib.rcParams.update({
    "animation.ffmpeg_path": FFMPEG,
    "font.family": FONTS,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
})


# ----------------------------------------------------------------------------------------------------------------
# timing helpers: every motion and fade goes through the ease-in-out cubic


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 4.0 * u ** 3 if u < 0.5 else 1.0 - (-2.0 * u + 2.0) ** 3 / 2.0


_MOTION = None  # scene_motion() collects here the (t0, d) of every transition the updaters evaluate


def prog(t: float, t0: float, d: float) -> float:
    """Eased progress of a transition that starts at t0 and lasts d seconds."""
    if _MOTION is not None:
        _MOTION.add((float(t0), float(d)))
    if d <= 0:
        return 1.0 if t >= t0 else 0.0
    return ease((t - t0) / d)


def fadeval(t, t0=None, d=0.6, t1=None, d1=0.5) -> float:
    a = 1.0 if t0 is None else prog(t, t0, d)
    if t1 is not None:
        a *= 1.0 - prog(t, t1, d1)
    return a


def lerp(a, b, u):
    return a + (b - a) * u


def mix(c1, c2, u):
    a, b = np.array(to_rgba(c1)), np.array(to_rgba(c2))
    return tuple(a + (b - a) * min(max(u, 0.0), 1.0))


# ----------------------------------------------------------------------------------------------------------------
# rich text: inline markup laid out from measured run widths
#   **bold**   [[o:orange]] [[b:blue]] [[n:brown]] [[m:muted]] [[k:ink]]   _{sub}  ^{sup}   {tt} = t t-bar   \n


COLMAP = {"o": ORANGE, "b": BLUE, "n": BROWN, "m": MUTED, "k": INK, "f": FAINT, "d": ORANGE_D}
BOLDER = {"light": "normal", "normal": "semibold", "medium": "semibold", "semibold": "bold", "bold": "bold"}


class Run:
    __slots__ = ("text", "color", "weight", "scale", "dy", "bar", "w", "x")

    def __init__(self, text, color, weight, scale=1.0, dy=0.0, bar=False):
        self.text, self.color, self.weight, self.scale, self.dy, self.bar = text, color, weight, scale, dy, bar
        self.w = self.x = 0.0


def parse_markup(s: str, color, weight):
    lines, buf, cols = [[]], [], [color]
    state = {"bold": False, "script": None}

    def cur_weight():
        return BOLDER[weight] if state["bold"] else weight

    def flush():
        if buf:
            sc, dy = {"sub": (0.7, -0.17), "sup": (0.7, 0.40)}.get(state["script"], (1.0, 0.0))
            lines[-1].append(Run("".join(buf), cols[-1], cur_weight(), sc, dy))
            buf.clear()

    i, n = 0, len(s)
    while i < n:
        if s.startswith("**", i):
            flush(); state["bold"] = not state["bold"]; i += 2
        elif s.startswith("[[", i):
            flush(); j = s.index(":", i); cols.append(COLMAP[s[i + 2:j]]); i = j + 1
        elif s.startswith("]]", i):
            flush(); cols.pop(); i += 2
        elif s.startswith("{tt}", i):
            flush(); lines[-1].append(Run("tt", cols[-1], cur_weight(), bar=True)); i += 4
        elif s.startswith("_{", i) or s.startswith("^{", i):
            flush(); state["script"] = "sub" if s[i] == "_" else "sup"; i += 2
        elif s[i] == "}" and state["script"]:
            flush(); state["script"] = None; i += 1
        elif s[i] == "\n":
            flush(); lines.append([]); i += 1
        else:
            buf.append(s[i]); i += 1
    flush()
    return lines


def plain(s: str) -> str:
    s = s.replace("{tt}", "tt̄")
    s = re.sub(r"\[\[[a-z]+:", "", s).replace("]]", "").replace("**", "")
    return re.sub(r"[_^]\{([^}]*)\}", r"\1", s)


class Rich:
    """One or more lines of text with inline markup; supports alpha, offset, recolouring and a background box."""

    def __init__(self, sc, x, y, s, size, color, weight, ha, va, lsp, z, bg=None):
        self.sc, self.s, self.size, self.color = sc, s, size, color
        self.plain = plain(s)
        self.alpha, self.dx, self.dy = 1.0, 0.0, 0.0
        em = size / 0.72  # data units (1 data unit = 1 px at 1080p = 0.72 pt)
        lines = parse_markup(s, color, weight)
        lh = em * lsp
        widths = []
        for line in lines:
            x0 = 0.0
            for r in line:
                r.w = sc.measure(r.text, size * r.scale, r.weight)
                r.x = x0
                x0 += r.w
            widths.append(x0)
        nl = len(lines)
        if va == "baseline":
            yb = y
        elif va == "center":
            yb = y + (nl - 1) * lh / 2 - 0.36 * em
        elif va == "top":
            yb = y - 0.76 * em
        else:  # bottom
            yb = y + (nl - 1) * lh + 0.25 * em
        self.items = []
        boxes = []
        for k, line in enumerate(lines):
            wl = widths[k]
            xl = x if ha == "left" else (x - wl / 2 if ha == "center" else x - wl)
            ybl = yb - k * lh
            boxes.append((xl, ybl - 0.25 * em, xl + wl, ybl + 0.76 * em))
            for r in line:
                fs = size * r.scale
                tx, ty = xl + r.x, ybl + r.dy * em
                art = sc.ax.text(tx, ty, r.text, fontsize=fs, color=r.color, fontweight=r.weight, ha="left",
                                 va="baseline", zorder=z)
                self.items.append([art, tx, ty, "t", r.color == color])
                if r.bar:  # the bar of t-bar, centred over the second t
                    wt = sc.measure("t", fs, r.weight)
                    emr = fs / 0.72
                    bx = np.array([tx + wt + 0.015 * emr, tx + 2 * wt - 0.045 * emr])
                    by = np.array([ty + 0.80 * emr] * 2)
                    ln = Line2D(bx, by, color=r.color, lw=0.07 * fs, solid_capstyle="butt", zorder=z)
                    sc.ax.add_line(ln)
                    self.items.append([ln, bx, by, "l", r.color == color])
        self.box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                    max(b[3] for b in boxes))
        self.width = max(widths) if widths else 0.0
        self.lines = nl
        if bg is not None:  # a box in the background colour keeps guide lines and layer circles out of the label
            bx0, by0, bx1, by1 = self.box
            px, py = 0.22 * em, 0.12 * em
            patch = FancyBboxPatch((bx0 - px, by0 - py), bx1 - bx0 + 2 * px, by1 - by0 + 2 * py,
                                   boxstyle=f"round,pad=0,rounding_size={0.25 * em}", facecolor=bg,
                                   edgecolor="none", zorder=z - 0.5)
            sc.ax.add_patch(patch)
            self.items.append([patch, bx0 - px, by0 - py, "p", False])

    def set_alpha(self, a):
        self.alpha = a
        vis = a > 0.003
        for it in self.items:
            it[0].set_visible(vis)
            if vis:
                it[0].set_alpha(min(1.0, a))

    def set_offset(self, dx, dy):
        if dx == self.dx and dy == self.dy:
            return
        self.dx, self.dy = dx, dy
        for art, bx, by, kind, _ in self.items:
            if kind == "t":
                art.set_position((bx + dx, by + dy))
            elif kind == "p":
                art.set_x(bx + dx)
                art.set_y(by + dy)
            else:
                art.set_data(bx + dx, by + dy)

    def recolor(self, c):
        """Change the colour of the runs that use the default colour (markup colours stay)."""
        for art, _, _, _, default in self.items:
            if default:
                art.set_color(c)

    def bbox(self):
        x0, y0, x1, y1 = self.box
        return x0 + self.dx, y0 + self.dy, x1 + self.dx, y1 + self.dy


# ----------------------------------------------------------------------------------------------------------------
# animated primitives


def show(obj, a):
    """Set alpha (and visibility) of an artist, a helper object or a list of them."""
    if isinstance(obj, (list, tuple)):
        for o in obj:
            show(o, a)
    elif isinstance(obj, Artist):
        vis = a > 0.003
        obj.set_visible(vis)
        if vis:
            obj.set_alpha(min(1.0, a))
    else:
        obj.set_alpha(a)


class GrowLine:
    """A polyline drawn up to a fraction of its arc length."""

    def __init__(self, sc, xs, ys, color=INK, lw=2.0, z=5, ls="-", alpha=1.0, cap="round"):
        self.x, self.y = np.asarray(xs, float), np.asarray(ys, float)
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])
        self.L = float(self.s[-1])
        self.line = Line2D([], [], color=color, lw=lw, zorder=z, linestyle=ls, solid_capstyle=cap, dash_capstyle=cap)
        sc.ax.add_line(self.line)
        self.p, self.a = 0.0, alpha
        self._apply()

    def set(self, p=None, alpha=None):
        if p is not None:
            self.p = p
        if alpha is not None:
            self.a = alpha
        self._apply()

    def set_alpha(self, a):
        self.set(alpha=a)

    def _apply(self):
        sh = self.p * self.L
        if sh <= 1e-6 or self.a <= 0.003:
            self.line.set_visible(False)
            return
        k = int(np.searchsorted(self.s, sh))
        if k >= len(self.s):
            xs, ys = self.x, self.y
        else:
            f = (sh - self.s[k - 1]) / max(self.s[k] - self.s[k - 1], 1e-9)
            xs = np.append(self.x[:k], self.x[k - 1] + f * (self.x[k] - self.x[k - 1]))
            ys = np.append(self.y[:k], self.y[k - 1] + f * (self.y[k] - self.y[k - 1]))
        self.line.set_data(xs, ys)
        self.line.set_alpha(min(1.0, self.a))
        self.line.set_visible(True)

    def point(self, p):
        sh = min(max(p, 0.0), 1.0) * self.L
        return np.interp(sh, self.s, self.x), np.interp(sh, self.s, self.y)


class Arrow:
    """A straight arrow that grows from p0 to p1."""

    def __init__(self, sc, p0, p1, color=MUTED, lw=2.4, head=14.0, z=8):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.head = head
        self.line = Line2D([], [], color=color, lw=lw, zorder=z, solid_capstyle="round")
        self.tri = Polygon(np.zeros((3, 2)), closed=True, facecolor=color, edgecolor="none", zorder=z)
        sc.ax.add_line(self.line)
        sc.ax.add_patch(self.tri)
        self.p, self.a = 0.0, 1.0
        self._apply()

    def set(self, p=None, alpha=None):
        if p is not None:
            self.p = p
        if alpha is not None:
            self.a = alpha
        self._apply()

    def set_alpha(self, a):
        self.set(alpha=a)

    def _apply(self):
        if self.p <= 0.002 or self.a <= 0.003:
            self.line.set_visible(False)
            self.tri.set_visible(False)
            return
        d = self.p1 - self.p0
        L = float(np.hypot(*d))
        u = d / L
        nrm = np.array([-u[1], u[0]])
        tip = self.p0 + d * self.p
        hl = min(self.head, L * self.p)
        base = tip - u * hl
        self.line.set_data([self.p0[0], base[0] + u[0] * 1.0], [self.p0[1], base[1] + u[1] * 1.0])
        self.tri.set_xy([tip, base + nrm * hl * 0.48, base - nrm * hl * 0.48])
        for art in (self.line, self.tri):
            art.set_visible(True)
            art.set_alpha(min(1.0, self.a))


class HBar:
    """Horizontal bar with a rounded data end and a square baseline (sign -1 grows to the left)."""

    def __init__(self, sc, x0, yc, h, color, r=6.0, z=10, sign=1):
        self.x0, self.yc, self.h, self.r, self.sign = x0, yc, h, r, sign
        self.patch = FancyBboxPatch((x0, yc - h / 2), 1.0, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=color, edgecolor="none", zorder=z)
        sc.ax.add_patch(self.patch)
        clip = Rectangle((x0, yc - h) if sign > 0 else (x0 - 5000, yc - h), 5000, 2 * h, transform=sc.ax.transData)
        self.patch.set_clip_path(clip)
        self.L, self.a = 0.0, 1.0
        self._apply()

    def set(self, L=None, alpha=None):
        if L is not None:
            self.L = L
        if alpha is not None:
            self.a = alpha
        self._apply()

    def set_alpha(self, a):
        self.set(alpha=a)

    def set_color(self, c):
        self.patch.set_facecolor(c)

    def _apply(self):
        if self.L < 0.4 or self.a <= 0.003:
            self.patch.set_visible(False)
            return
        r = min(self.r, (self.L + self.r) / 2, self.h / 2)
        self.patch.set_boxstyle("round", pad=0, rounding_size=r)
        if self.sign > 0:
            self.patch.set_bounds(self.x0 - self.r, self.yc - self.h / 2, self.L + self.r, self.h)
        else:
            self.patch.set_bounds(self.x0 - self.L, self.yc - self.h / 2, self.L + self.r, self.h)
        self.patch.set_alpha(min(1.0, self.a))
        self.patch.set_visible(True)


class Pill:
    """Rounded label box; ``style(u)`` blends from the idle look to the active look."""

    def __init__(self, sc, xc, yc, text, size=LABEL, h=62, w=None, pad=30, idle=(CARD, MUTED), active=(CARD2, INK),
                 edge=None, weight="normal", z=6):
        self.rich = sc.T(xc, yc, text, size=size, color=idle[1], weight=weight, ha="center", va="center", z=z + 1)
        self.w = w if w else self.rich.width + 2 * pad
        self.h, self.xc, self.yc = h, xc, yc
        self.box = sc.box(xc - self.w / 2, yc - h / 2, self.w, h, fc=idle[0], r=h / 2, z=z,
                          ec=edge or "none", lw=1.6 if edge else 0.0)
        self.idle, self.active, self.edge = idle, active, edge
        self.a = 1.0

    def style(self, u):
        self.box.set_facecolor(mix(self.idle[0], self.active[0], u))
        self.rich.recolor(mix(self.idle[1], self.active[1], u))
        if self.edge:
            self.box.set_edgecolor(mix(self.idle[0], self.edge, u))

    def set_alpha(self, a):
        self.a = a
        show(self.box, a)
        self.rich.set_alpha(a)

    @property
    def left(self):
        return self.xc - self.w / 2

    @property
    def right(self):
        return self.xc + self.w / 2


def pill_row(sc, texts, y, gap=90, size=LABEL, pad=30, **kw):
    """Pills of measured widths with equal gaps, centred on the frame."""
    ws = [sc.text_width(s, size) + 2 * pad for s in texts]
    x = W / 2 - (sum(ws) + gap * (len(ws) - 1)) / 2
    pills = []
    for s, w in zip(texts, ws):
        pills.append(Pill(sc, x + w / 2, y, s, size=size, w=w, **kw))
        x += w + gap
    return pills


class GraphView:
    """Hits (circles) and edges (lines) whose growth, colour, width and alpha are set per frame."""

    def __init__(self, sc, pts, edges, r=7.0, node_color=HIT, edge_color=EDGE0, lw=2.0, z_e=6, z_n=10):
        self.pts = np.asarray(pts, float)
        self.edges = list(edges)
        self.nodes = [sc.circle(x, y, r, fc=node_color, ec=BG, lw=1.5, z=z_n) for x, y in self.pts]
        self.lines = []
        for _ in self.edges:
            ln = Line2D([], [], color=edge_color, lw=lw, zorder=z_e, solid_capstyle="round")
            sc.ax.add_line(ln)
            self.lines.append(ln)

    def edge(self, i, grow=1.0, color=None, lw=None, alpha=1.0, ls=None):
        ln = self.lines[i]
        if grow <= 0.002 or alpha <= 0.003:
            ln.set_visible(False)
            return
        u, v = self.edges[i]
        p, q = self.pts[u], self.pts[v]
        e = p + (q - p) * grow
        ln.set_data([p[0], e[0]], [p[1], e[1]])
        if color is not None:
            ln.set_color(color)
        if lw is not None:
            ln.set_linewidth(lw)
        if ls is not None:
            ln.set_linestyle(ls)
        ln.set_alpha(min(1.0, alpha))
        ln.set_visible(True)

    def node(self, j, color=None, alpha=1.0):
        c = self.nodes[j]
        show(c, alpha)
        if color is not None:
            c.set_facecolor(color)


# ----------------------------------------------------------------------------------------------------------------
# scene base


class Scene:
    def __init__(self, name, duration, title=None, accent=ORANGE, ref=None, footer=None):
        self.name, self.duration = name, float(duration)
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_axis_off()
        self.ax = ax
        self.renderer = self.fig.canvas.get_renderer()
        self.updaters, self.riches, self._mcache = [], [], {}
        self.lines = []  # narration lines, filled by narrate()
        self.cover = Rectangle((0, 0), W, H, facecolor=BG, edgecolor="none", zorder=1000)
        ax.add_patch(self.cover)
        self.cover_alpha = 1.0
        if title:
            self.T(120, 966, title, size=TITLE, weight="semibold", t0=0.15, d=0.7, rise=10)
            self.fade(self.seg(124, 930, 204, 930, accent, lw=5), 0.35, 0.7)
        if ref:
            self.T(W - 80, 968, ref, size=SMALL, color=MUTED, ha="right", t0=0.5)
        if footer:
            self.T(W - 80, 36, footer, size=SMALL, color=MUTED, ha="right", t0=0.5)

    # --- infrastructure
    def measure(self, s, size, weight):
        key = (s, size, weight)
        w = self._mcache.get(key)
        if w is None:
            fp = FontProperties(family=FONTS, size=size, weight=weight)
            w = self.renderer.get_text_width_height_descent(s, fp, ismath=False)[0] * 100.0 / self.fig.dpi
            self._mcache[key] = w
        return w

    def text_width(self, s, size=BODY, weight="normal"):
        best = 0.0
        for line in parse_markup(s, INK, weight):
            best = max(best, sum(self.measure(r.text, size * r.scale, r.weight) for r in line))
        return best

    def on(self, f):
        self.updaters.append(f)
        return f

    def cover_at(self, t):
        """Alpha of the background cover: the scene fades in over 0.5 s and out over its last 0.6 s."""
        return min(1.0, max(0.0, (1.0 - prog(t, 0.0, 0.5)) + prog(t, self.duration - 0.6, 0.6)))

    def update(self, t):
        for f in self.updaters:
            f(t)
        self.cover_alpha = self.cover_at(t)
        self.cover.set_alpha(self.cover_alpha)
        self.cover.set_visible(self.cover_alpha > 0.002)

    # --- elements
    def T(self, x, y, s, size=BODY, color=INK, weight="normal", ha="left", va="baseline", lsp=1.3, z=30,
          t0=None, d=0.6, t1=None, d1=0.5, rise=0.0, alpha=1.0, bg=None):
        r = Rich(self, x, y, s, size, color, weight, ha, va, lsp, z, bg)
        self.riches.append(r)
        if t0 is None and t1 is None:
            r.set_alpha(alpha)
        else:
            def f(t, r=r):
                r.set_alpha(alpha * fadeval(t, t0, d, t1, d1))
                if rise and t0 is not None:
                    r.set_offset(0.0, -rise * (1.0 - prog(t, t0, d)))
            self.updaters.append(f)
            f(0.0)
        return r

    def narrate(self, lines):
        """Subtitles from the narration lines of this scene ({"t0", "t1", "display", "speech"} each): the display text
        is shown from t0 to t1 (scene time); the lines are recorded for the soundtrack and the time warp."""
        k = scene_number(self.name)
        for i, ln in enumerate(lines):
            t0, t1 = float(ln["t0"]), float(ln["t1"])
            sub = self.T(W / 2, SUB_Y, ln["display"], size=CAP, color=INK2, ha="center", va="center",
                         t0=t0, d=SUB_IN, t1=t1, d1=SUB_OUT)
            self.lines.append({"id": line_id(k, self.name, i), "t0": t0, "t1": t1, "display": ln["display"],
                               "speech": ln["speech"], "sub": sub})

    def sub_alpha(self, line, t):
        """Opacity of a narration line's subtitle on screen at scene time t (its own fade times the cover)."""
        return fadeval(t, line["t0"], SUB_IN, line["t1"], SUB_OUT) * (1.0 - self.cover_at(t))

    def fade(self, objs, t0=None, d=0.6, t1=None, d1=0.5, alpha=1.0):
        def f(t):
            show(objs, alpha * fadeval(t, t0, d, t1, d1))
        self.updaters.append(f)
        f(0.0)
        return objs

    def circle(self, x, y, r, fc=HIT, ec="none", lw=0.0, z=10, alpha=1.0, ls="-"):
        c = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha, linestyle=ls)
        self.ax.add_patch(c)
        return c

    def box(self, x, y, w, h, fc=CARD, ec="none", lw=0.0, r=16, z=2, alpha=1.0):
        b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=z, alpha=alpha)
        self.ax.add_patch(b)
        return b

    def seg(self, x0, y0, x1, y1, color=INK, lw=2.0, z=5, ls="-", alpha=1.0, cap="round"):
        ln = Line2D([x0, x1], [y0, y1], color=color, lw=lw, zorder=z, linestyle=ls, alpha=alpha,
                    solid_capstyle=cap, dash_capstyle=cap)
        self.ax.add_line(ln)
        return ln

    def poly(self, xs, ys, color=INK, lw=2.0, z=5, ls="-", alpha=1.0):
        ln = Line2D(xs, ys, color=color, lw=lw, zorder=z, linestyle=ls, alpha=alpha)
        self.ax.add_line(ln)
        return ln

    def legend_row(self, x, y, items, size=LABEL, gap=44, t0=None):
        """Horizontal legend: items = [(kind, colour, text)], kind in {"sq", "line", "dash", "dot"}."""
        objs = []
        for kind, col, text in items:
            if kind == "sq":
                objs.append(self.box(x, y - 11, 22, 22, fc=col, r=5, z=12))
                tx = x + 34
            elif kind == "dot":
                objs.append(self.circle(x + 9, y, 8, fc=col, z=12))
                tx = x + 30
            else:
                objs.append(self.seg(x, y, x + 40, y, col, lw=3, z=12, ls=(0, (4, 3.5)) if kind == "dash" else "-"))
                tx = x + 52
            r = self.T(tx, y, text, size=size, color=INK2, va="center")
            objs.append(r)
            x = tx + r.width + gap
        if t0 is not None:
            self.fade(objs, t0)
        return objs


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers


def helix(p0, phi0, kappa, smax, n=500):
    """Transverse helix (a circle arc) from p0 with heading phi0 and signed curvature kappa (1/px)."""
    s = np.linspace(0.0, smax, n)
    if abs(kappa) < 1e-12:
        return s, p0[0] + s * math.cos(phi0), p0[1] + s * math.sin(phi0)
    x = p0[0] + (np.sin(phi0 + kappa * s) - math.sin(phi0)) / kappa
    y = p0[1] - (np.cos(phi0 + kappa * s) - math.cos(phi0)) / kappa
    return s, x, y


def clip_r(x, y, c, rmax):
    r = np.hypot(x - c[0], y - c[1])
    out = np.nonzero(r > rmax)[0]
    k = out[0] if len(out) else len(x)
    return x[:k], y[:k]


def crossings(x, y, c, radii):
    """First crossing of each layer radius along the path: [(arc length, x, y)] for layers the path reaches."""
    r = np.hypot(x - c[0], y - c[1])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    out = []
    for R in radii:
        if r[0] >= R:
            continue
        idx = np.nonzero(r >= R)[0]
        if len(idx) == 0:
            continue
        k = idx[0]
        f = (R - r[k - 1]) / max(r[k] - r[k - 1], 1e-9)
        out.append((s[k - 1] + f * (s[k] - s[k - 1]), x[k - 1] + f * (x[k] - x[k - 1]), y[k - 1] + f * (y[k] - y[k - 1])))
    return out


def bezier(p0, p1, p2, n=60):
    u = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - u) ** 2 * np.asarray(p0, float) + 2 * u * (1 - u) * np.asarray(p1, float) + u ** 2 * np.asarray(p2, float)
    return pts[:, 0], pts[:, 1]


def through(p0, q, p2):
    """Control point of the quadratic Bezier from p0 to p2 that passes through q at its midpoint."""
    return 2 * np.asarray(q, float) - (np.asarray(p0, float) + np.asarray(p2, float)) / 2


def chain_band(sc, pts, color=ORANGE, lw=30, z=4):
    """A translucent rounded band through a few hits (a triplet highlight)."""
    xs, ys = zip(*pts)
    ln = Line2D(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
    sc.ax.add_line(ln)
    ln.set_visible(False)
    return ln


# ================================================================================================================
# scenes


def scene_title():
    sc = Scene("title", 6.2)
    xs = np.linspace(700, 1220, 7)
    ys = 715 + 44 * np.sin(np.pi * (xs - 700) / 520)
    hits = [sc.circle(x, y, 8, fc=INK, ec=BG, lw=1.6, z=12) for x, y in zip(xs, ys)]
    for k, c in enumerate(hits):
        sc.fade(c, 0.45 + 0.12 * k, 0.5)
    arcs = []
    for k in range(5):
        p0, p1, p2 = (xs[k], ys[k]), (xs[k + 1], ys[k + 1]), (xs[k + 2], ys[k + 2])
        side = 32 if k % 2 == 0 else -32
        bx, by = bezier(p0, through(p0, (p1[0], p1[1] + side), p2), p2)
        arcs.append(GrowLine(sc, bx, by, color=ORANGE, lw=3.2, z=8))

    @sc.on
    def _(t):
        for k, g in enumerate(arcs):
            g.set(prog(t, 1.25 + 0.32 * k, 0.9))

    sc.T(W / 2, 505, "Canto", size=132, weight="semibold", ha="center", t0=0.3, d=1.0, rise=16)
    line = sc.seg(W / 2, 468, W / 2, 468, ORANGE, lw=4)

    @sc.on
    def _(t):
        half = 330 * prog(t, 1.0, 1.1)
        line.set_data([W / 2 - half, W / 2 + half], [468, 468])
        line.set_visible(half > 0.5)

    sc.T(W / 2, 408, "[[o:**C**]]hained-triplet, origin-[[o:**A**]]gnostic [[o:**N**]]etworks for "
                     "[[o:**T**]]racking [[o:**O**]]ff-vertex", size=31, color=INK2, ha="center", va="center",
         t0=1.6, d=0.8, rise=10)
    sc.T(W / 2, 340, "Displaced-track reconstruction trained on {tt} only", size=26, color=MUTED, ha="center",
         va="center", t0=2.3, d=0.8, rise=10)
    sc.narrate(LINES["title"])
    return sc


def scene_problem():
    sc = Scene("problem", 27.0, title="The problem: tracks that start late")
    C = np.array([590.0, 548.0])
    R = [42, 68, 94, 120, 165, 210, 255, 300, 335, 370]
    inner = sc.circle(C[0], C[1], 184, fc=ORANGE_T, z=1)
    sc.fade(inner, 12.6, 1.0, alpha=0.75)
    for i, r in enumerate(R):
        sc.fade(sc.circle(C[0], C[1], r, fc="none", ec=LAYER, lw=1.9 if i < 4 else 1.6, z=2), 0.4 + 0.07 * i, 0.6)
    sc.fade(sc.circle(C[0], C[1], 5.5, fc=INK, z=14), 0.9, 0.5)

    # prompt tracks: start at the collision point, one hit per layer
    specs = [(72, 1 / 900), (108, 1 / 650), (144, -1 / 800), (178, 1 / 1000), (212, -1 / 700), (246, 1 / 850),
             (282, -1 / 620), (316, 1 / 760), (350, -1 / 950)]
    prompt = []
    for k, (phi, kap) in enumerate(specs):
        _, x, y = helix(C, math.radians(phi), kap, 560)
        x, y = clip_r(x, y, C, 380)
        g = GrowLine(sc, x, y, color=PROMPT, lw=2.2, z=5)
        hits = [(sx, sc.circle(hx, hy, 5.2, fc=HIT, ec=BG, lw=1.2, z=11)) for sx, hx, hy in crossings(x, y, C, R)]
        prompt.append((g, hits, 1.2 + 0.2 * k))

    @sc.on
    def _(t):
        dim = 1.0 - 0.62 * prog(t, 6.6, 1.0)
        for g, hits, t0 in prompt:
            g.set(prog(t, t0, 2.6), dim)
            sh = g.p * g.L
            for sx, c in hits:
                show(c, dim * ease((sh - sx) / 14.0))

    # the heavy neutral lepton and its displaced decay; phiN and the muon headings put the impact-parameter
    # segment (below) midway between the prompt tracks at 316 and 350 degrees
    phiN = math.radians(23)
    V = C + 190 * np.array([math.cos(phiN), math.sin(phiN)])
    nline = GrowLine(sc, [C[0], V[0]], [C[1], V[1]], color=MUTED, lw=2.4, z=6, ls=(0, (5, 4)))
    vtx = sc.circle(V[0], V[1], 8.5, fc=BG, ec=ORANGE, lw=2.6, z=13)
    sc.fade(vtx, 9.3, 0.5)
    muons = []
    for phi, kap in ((math.radians(49), -1 / 620), (math.radians(1), 1 / 560)):
        _, x, y = helix(V, phi, kap, 400)
        x, y = clip_r(x, y, C, 380)
        g = GrowLine(sc, x, y, color=ORANGE, lw=2.9, z=7)
        hits = [(sx, sc.circle(hx, hy, 5.8, fc=ORANGE_D, ec=BG, lw=1.2, z=12)) for sx, hx, hy in crossings(x, y, C, R)]
        muons.append((g, hits, phi, kap))

    @sc.on
    def _(t):
        nline.set(prog(t, 7.2, 2.2))
        for j, (g, hits, _, _) in enumerate(muons):
            g.set(prog(t, 10.0 + 0.25 * j, 2.6))
            sh = g.p * g.L
            for sx, c in hits:
                show(c, ease((sh - sx) / 14.0))

    # "N" sits in the ring between the layers at r = 120 and 165, 25 degrees above its flight line
    uN = np.array([math.cos(phiN + math.radians(25)), math.sin(phiN + math.radians(25))])
    sc.T(C[0] + 142 * uN[0], C[1] + 142 * uN[1], "N", size=LABEL, color=INK2, weight="semibold", ha="center",
         va="center", t0=8.2)
    sc.T(V[0] + 22, V[1] - 40, "displaced vertex", size=SMALL, color=INK2, t0=9.6, bg=BG)
    for (g, _, _, _), lab in zip(muons, ("μ⁺", "μ⁻")):
        ex, ey = g.x[-1], g.y[-1]
        dx, dy = ex - g.x[-2], ey - g.y[-2]
        n = math.hypot(dx, dy)
        sc.T(ex + 26 * dx / n, ey + 26 * dy / n, lab, size=LABEL, color=INK2, ha="center", va="center", t0=12.4)

    # no inner hits: label outside the detector with a leader to the inner region
    lead = sc.seg(300, 197, C[0] - 184 * 0.707 + 4, C[1] - 184 * 0.707 + 4, FAINT, lw=1.6, z=3)
    sc.fade(lead, 13.0)
    sc.T(110, 170, "no hits on the inner layers", size=SMALL, color=INK2, t0=13.0)

    # impact parameter of the first muon: extrapolate backwards to the point of closest approach
    g1, _, phi1, kap1 = muons[0]
    _, bx, by = helix(V, phi1 + math.pi, -kap1, 320)
    dist = np.hypot(bx - C[0], by - C[1])
    kmin = int(np.argmin(dist))
    P = np.array([bx[kmin], by[kmin]])
    back = GrowLine(sc, bx[:kmin + 1], by[:kmin + 1], color=ORANGE, lw=2.2, z=6, ls=(0, (1.2, 3.2)))
    d0 = sc.seg(C[0], C[1], P[0], P[1], INK, lw=2.4, z=8)
    u = (P - C) / np.hypot(*(P - C))
    nrm = np.array([-u[1], u[0]])
    ticks = [sc.seg(*(Q + 7 * nrm), *(Q - 7 * nrm), INK, lw=2.2, z=8) for Q in (C, P)]
    # beside the segment the label would sit on a prompt track; just beyond its end tick (r = 142, between the
    # layers at 120 and 165) it is clear of both neighbouring prompt tracks
    lab = C + 142 * u
    sc.T(lab[0], lab[1], "d_{0}", size=LABEL, color=INK, weight="semibold", ha="center", va="center", t0=17.9)

    @sc.on
    def _(t):
        back.set(prog(t, 16.4, 1.4))
        show([d0] + ticks, prog(t, 17.6, 0.6))

    # legend
    lx = 1090
    entries = [("dot", INK, "collision point (beam line)", 0.9), ("line", PROMPT, "prompt track: starts at the collision", 1.3),
               ("hit", HIT, "hit: one per layer crossed", 1.7),
               ("dash", MUTED, "neutral long-lived particle (not seen)", 7.2),
               ("line", ORANGE, "displaced track: starts at the decay", 10.0)]
    for i, (kind, col, text, t0) in enumerate(entries):
        y = 850 - 52 * i
        if kind == "dot":
            obj = sc.circle(lx + 20, y, 6.5, fc=col, z=12)
        elif kind == "hit":
            obj = sc.circle(lx + 20, y, 5.8, fc=col, z=12)
        else:
            obj = sc.seg(lx, y, lx + 40, y, col, lw=2.8, z=12, ls=(0, (4, 3.5)) if kind == "dash" else "-")
        sc.fade(obj, t0)
        sc.T(lx + 62, y, text, size=LABEL, color=INK2, va="center", t0=t0)

    # benchmark card
    card = sc.box(lx, 190, 750, 360, fc=CARD, r=18, z=2)
    sc.fade(card, 18.2, 0.8)
    rows = [("**Benchmark: heavy neutral lepton (HNL)**", BODY, INK),
            ("pp → W → μ N,   N → ν μ⁺ μ⁻", BODY, INK),
            ("Pythia 8 + ACTS generic detector (Fatras), 2 T", SMALL, INK2),
            ("no pileup; {tt} sample: 10k events", SMALL, INK2),
            ("HNL: 10 GeV, cτ = 100 mm; 15 GeV, cτ = 200 mm", SMALL, INK2),
            ("test only: 15 GeV, cτ = 100 mm", SMALL, INK2),
            ("Wang et al., arXiv:2203.08800", SMALL, MUTED)]
    for i, (s, size, col) in enumerate(rows):
        sc.T(lx + 32, 505 - 46 * i, s, size=size, color=col, t0=18.4 + 0.12 * i)

    sc.narrate(LINES["problem"])
    return sc


def scene_pipeline():
    sc = Scene("pipeline", 22.0, title="The ML tracking pipeline")
    LX = [330 + 170 * i for i in range(8)]
    TY = [[742, 738, 732, 724, 714, 702, 688, 672], [600, 606, 610, 612, 611, 608, 602, 594],
          [440, 448, 458, 470, 484, 500, 518, 538], [330, 322, 316, 312, 310, 310, 312, 316]]
    for x in LX:
        sc.fade(sc.seg(x, 262, x, 770, HAIR, lw=1.4, z=1), 0.5, 0.8)
    pts, owner, layer = [], [], []
    for k, ys in enumerate(TY):
        for i, y in enumerate(ys):
            pts.append((LX[i], y))
            owner.append(k)
            layer.append(i)
    edges, true = [], []
    for a in range(len(pts)):
        for b in range(len(pts)):
            if layer[b] == layer[a] + 1 and abs(pts[a][1] - pts[b][1]) < 140:
                edges.append((a, b))
                true.append(owner[a] == owner[b])
    rng = np.random.default_rng(7)
    score = [0.82 + 0.16 * rng.random() if tr else 0.04 + 0.32 * rng.random() for tr in true]
    g = GraphView(sc, pts, edges, r=7.5)

    pills = pill_row(sc, ["1   Graph construction", "2   Edge classification", "3   Track building"], 852)
    arrows = [Arrow(sc, (pills[i].right + 14, 852), (pills[i + 1].left - 14, 852), color=FAINT, lw=2.2, head=12)
              for i in range(2)]
    win = ((5.4, 10.4), (10.6, 15.4), (15.6, 30.0))

    # score legend
    legend = [sc.box(1590 + 10 * i, 642, 10.5, 16, fc=mix(LOW, INK, i / 23), r=0, z=4) for i in range(24)]
    legend += [sc.T(1590, 690, "edge score", size=SMALL, color=INK2, weight="semibold"),
               sc.T(1590, 608, "fake", size=SMALL, color=MUTED), sc.T(1830, 608, "true", size=SMALL, color=MUTED,
                                                                       ha="right")]
    sc.fade(legend, 11.0, 0.8)

    @sc.on
    def _(t):
        pa = prog(t, 0.4, 0.8)
        for k, p in enumerate(pills):
            p.set_alpha(pa)
            a0, a1 = win[k]
            p.style(prog(t, a0, 0.5) * (1 - prog(t, a1, 0.5)))
        for a in arrows:
            a.set(1.0, pa)
        for j in range(len(pts)):
            col = HIT
            if t > 17.4:
                col = mix(HIT, TRACK[owner[j]], prog(t, 17.4 + 0.6 * owner[j], 0.8))
            g.node(j, col, prog(t, 0.8 + 0.12 * layer[j], 0.6))
        for i, (a, b) in enumerate(edges):
            grow = prog(t, 5.6 + 0.55 * layer[a], 0.8)
            s = score[i]
            u = prog(t, 11.0 + 0.18 * layer[a], 1.4)
            col = mix(EDGE0, mix(LOW, INK, s), u)
            lw = lerp(2.0, 1.2 + 2.4 * s, u)
            alpha = 1.0
            if s < 0.5:
                alpha = 1.0 - prog(t, 16.0, 1.2)
            elif t > 17.4:
                col = mix(col, TRACK[owner[a]], prog(t, 17.4 + 0.6 * owner[a], 0.8))
            g.edge(i, grow, col, lw, alpha)

    sc.narrate(LINES["pipeline"])
    return sc


def scene_dezoort():
    sc = Scene("dezoort", 18.0, title="DeZoort interaction network", accent=BROWN,
               ref="DeZoort et al., Comput. Softw. Big Sci. 5 (2021) 26", footer=SIM)
    # left: a destination hit with three incoming edges
    D = np.array([440.0, 705.0])
    S = [np.array(p) for p in ((190.0, 822.0), (170.0, 700.0), (205.0, 585.0))]
    ins = [Arrow(sc, s, D - 22 * (D - s) / np.hypot(*(D - s)), color=INK2, lw=2.6, head=13, z=6) for s in S]
    nodes = [sc.circle(p[0], p[1], 16, fc=BG, ec=INK, lw=2.2, z=10) for p in S]
    dnode = sc.circle(D[0], D[1], 18, fc=BG, ec=INK, lw=2.4, z=10)
    labs = [sc.T(D[0] + 30, D[1] - 8, "x_{dst}", size=LABEL, color=INK2, va="center"),
            sc.T(140, 640, "x_{src}", size=LABEL, color=INK2, va="center"),
            sc.T(300, 726, "e", size=LABEL, color=INK2, va="center")]
    sc.fade(nodes + [dnode] + labs, 0.7, 0.8)
    pulses = [sc.circle(0, 0, 6.5, fc=BROWN, z=9) for _ in S]

    # right: block diagram
    cx = [740, 995, 1250, 1505, 1760]
    names = ["R_{1}", "Σ", "O", "R_{2}", "score"]
    sub = ["relational", "aggregate", "object", "relational", "edge"]
    t_box = [1.6, 2.8, 4.0, 5.2, 6.4]
    boxes = []
    for i, x in enumerate(cx):
        b = sc.box(x - 85, 675, 170, 100, fc=CARD, r=16, z=3)
        r1 = sc.T(x, 725, names[i], size=30 if i < 4 else 25, weight="semibold", ha="center", va="center")
        r2 = sc.T(x, 638, sub[i], size=SMALL, color=MUTED, ha="center", va="center")
        sc.fade([r1, r2], t_box[i], 0.5)
        boxes.append(b)
    arr = [Arrow(sc, (cx[i] + 85 + 10, 725), (cx[i + 1] - 85 - 10, 725), color=INK2, lw=2.4, head=13) for i in range(4)]
    sc.T(cx[0], 806, "[x_{dst}, x_{src}, e]", size=SMALL, color=INK2, ha="center", t0=1.6)
    sc.T(cx[2], 806, "[x, agg]", size=SMALL, color=INK2, ha="center", t0=4.0)
    sc.T((cx[0] + cx[1]) / 2, 742, "4", size=SMALL, color=MUTED, ha="center", t0=2.5)
    sc.T((cx[2] + cx[3]) / 2, 742, "3", size=SMALL, color=MUTED, ha="center", t0=4.9)
    sc.T(cx[2], 572, "each model: Linear–ReLU–Linear–ReLU–Linear, hidden 40", size=SMALL,
         color=INK2, ha="center", t0=7.0)

    @sc.on
    def _(t):
        for i, b in enumerate(boxes):
            show(b, prog(t, t_box[i], 0.5))
            on = prog(t, t_box[i], 0.4) * (1 - prog(t, t_box[i] + 1.2, 0.4)) if i < 4 else prog(t, t_box[i], 0.4)
            b.set_facecolor(mix(CARD, BROWN_T, on))
        for i, a in enumerate(arr):
            a.set(prog(t, t_box[i] + 0.65, 0.5))
        # left illustration follows the active block
        r1 = prog(t, 1.6, 0.5) * (1 - prog(t, 5.0, 0.6))
        for k, a in enumerate(ins):
            a.set(1.0, prog(t, 0.9, 0.8))
            col = mix(INK2, BROWN, r1 if k != 1 else max(r1, prog(t, 5.2, 0.5)))
            a.line.set_color(col)
            a.tri.set_facecolor(col)
            a.line.set_linewidth(2.6 + (1.6 if (k == 1 and t > 5.2) else 0.0) * prog(t, 5.2, 0.5))
        for k, p in enumerate(pulses):
            u = prog(t, 2.9 + 0.12 * k, 0.9)
            pos = S[k] + (D - S[k]) * lerp(0.45, 0.93, u)
            p.center = (pos[0], pos[1])
            show(p, prog(t, 2.8 + 0.12 * k, 0.3) * (1 - prog(t, 3.9 + 0.12 * k, 0.35)))
        dnode.set_facecolor(mix(BG, BROWN_T, prog(t, 4.0, 0.5)))

    # parameter count
    sc.T(150, 372, "6,448", size=60, weight="semibold", t0=8.4, rise=8)
    sc.T(152, 312, "parameters (IN, one round)", size=LABEL, color=MUTED, t0=8.6)

    # result: fake-rate gauge
    card = sc.box(690, 196, 1150, 300, fc=CARD, r=18, z=2)
    sc.fade(card, 10.0, 0.7)
    sc.T(720, 448, "On our graphs ({tt}-trained, connected components)", size=BODY, weight="semibold", t0=10.2)
    gx0, gx1, gy = 740, 1790, 352
    fx = lambda v: gx0 + (gx1 - gx0) * v / 15.0  # noqa: E731
    track = sc.seg(gx0, gy, gx1, gy, CARD2, lw=9, z=3)
    target = sc.box(gx0 - 4, gy - 9, fx(2.94) - gx0 + 4, 18, fc=MUTED, r=8, z=4)
    sc.fade([track, target], 10.4, 0.6)
    sc.T(gx0, 388, "track fake-rate target ≤ 2.94 %", size=SMALL, color=INK2, t0=10.6)
    mk = [sc.circle(gx1, gy, 12, fc=BROWN, ec=BG, lw=2, z=6), sc.circle(gx1, gy, 12, fc=BROWN_M, ec=BG, lw=2, z=6)]
    sc.T(fx(12.5), 300, "IN: lowest 12.5 %", size=SMALL, color=INK2, ha="center", t0=13.1)
    sc.T(fx(6.1), 300, "IN-L: lowest 6.1 %", size=SMALL, color=INK2, ha="center", t0=13.3)
    sc.T(720, 232, "primary efficiency **84.8 %** (IN), 88.0 % (IN-L: 127,126 parameters)", size=BODY, color=INK,
         t0=14.0)

    @sc.on
    def _(t):
        for m, v, t0 in ((mk[0], 12.5, 11.0), (mk[1], 6.1, 11.3)):
            u = prog(t, t0, 1.8)
            m.center = (lerp(gx1, fx(v), u), gy)
            show(m, prog(t, t0 - 0.3, 0.4))

    sc.narrate(LINES["dezoort"])
    return sc


def _acorn_data():
    """Hits of four tracks in hit space and in a learned embedding (small rings, one per track).

    Edges: consecutive hits of each track (true) and two close pairs of neighbouring tracks (fake)."""
    lx = [45 + 60 * i for i in range(6)]
    ty = [[250, 236, 220, 202, 182, 160], [170, 186, 200, 212, 222, 230], [120, 118, 114, 108, 100, 90],
          [40, 50, 58, 64, 68, 70]]
    centres = [(110, 205), (205, 214), (120, 85), (215, 90)]
    facing = [(3, 0.0), (2, 180.0), (5, 0.0), (4, 180.0)]  # (hit index, direction) that faces the neighbour
    hit, emb, owner = [], [], []
    for k, ys in enumerate(ty):
        j0, a0 = facing[k]
        for i, y in enumerate(ys):
            hit.append((lx[i], y))
            ang = math.radians(a0 + 60.0 * (i - j0))
            emb.append((centres[k][0] + 23 * math.cos(ang), centres[k][1] + 23 * math.sin(ang)))
            owner.append(k)
    edges = [(6 * k + i, 6 * k + i + 1) for k in range(4) for i in range(5)]
    edges += [(6 * 1 + 2, 6 * 0 + 3), (6 * 3 + 4, 6 * 2 + 5)]
    return lx, np.array(hit), np.array(emb), owner, edges


def scene_acorn():
    sc = Scene("acorn", 25.2, title="acorn / Exa.TrkX", accent=BLUE,
               ref="Ju et al., Eur. Phys. J. C 81 (2021) 876", footer=SIM)
    lx, hit, emb, owner, edges = _acorn_data()
    PX, PY, PW, PH = [120, 555, 990, 1425], 560, 390, 300
    true = [owner[a] == owner[b] for a, b in edges]
    t_panel = [0.7, 6.8, 11.2, 14.2]
    heads = [("1   Metric learning", "MLP: hit (r, φ, z) → 12-D"),
             ("2   Radius graph", "r* = 0.16, 28.7k pairs\nper {tt} event"),
             ("3   Interaction GNN", "8 iterations, hidden 128"),
             ("4   Connected components", "at a score cut")]
    sc.T(120, 892, "Our acorn reproduction (pinned 99d15341).  Wang et al. 2022 pipeline: MLP 3D → 8D, filter MLP, "
                   "attention GNN, DBSCAN.", size=SMALL, color=MUTED, t0=0.6)
    for k, x0 in enumerate(PX):
        sc.fade(sc.box(x0, PY, PW, PH, fc=CARD, r=18, z=1), t_panel[k], 0.6)
        sc.T(x0 + PW / 2, 522, heads[k][0], size=LABEL, weight="semibold", ha="center", t0=t_panel[k] + 0.2)
        sc.T(x0 + PW / 2, 484, heads[k][1], size=SMALL, color=MUTED, ha="center", va="top", lsp=1.25,
             t0=t_panel[k] + 0.3)

    def P(k, p):
        return PX[k] + p[0], PY + p[1]

    # panel 1: hits move from hit space into the embedding
    lay1 = [sc.seg(PX[0] + x, PY + 22, PX[0] + x, PY + PH - 50, HAIR, lw=1.3, z=2) for x in lx]
    n1 = [sc.circle(*P(0, h), 6.5, fc=HIT, ec=CARD, lw=1.2, z=8) for h in hit]
    sc.T(PX[0] + 16, PY + PH - 34, "hit space", size=SMALL, color=MUTED, t0=1.0, t1=2.6, d1=0.4)
    sc.T(PX[0] + 16, PY + PH - 34, "embedding space", size=SMALL, color=MUTED, t0=3.2, d=0.5)

    # panel 2: radius graph in the embedding
    n2 = [sc.circle(*P(1, e), 6.5, fc=TRACK[owner[j]], ec=CARD, lw=1.2, z=8) for j, e in enumerate(emb)]
    rings = [sc.circle(*P(1, emb[j]), 52, fc="none", ec=BLUE, lw=1.8, z=4) for j in (3, 22)]
    e2 = [sc.seg(*P(1, emb[a]), *P(1, emb[b]), INK2, lw=1.6, z=5) for a, b in edges]
    sc.fade(n2, 7.0, 0.5)

    # panel 3: Interaction GNN on the same edges, drawn in hit space
    g3 = GraphView(sc, [P(2, h) for h in hit], edges, r=6.5, z_n=8)
    for c in g3.nodes:
        c.set_edgecolor(CARD)
    for k in range(1, 9):
        sc.T(PX[2] + 16, PY + PH - 34, f"iteration {k} / 8", size=SMALL, color=MUTED, t0=11.7 + 0.33 * (k - 1),
             d=0.1, t1=(11.7 + 0.33 * k - 0.12) if k < 8 else None, d1=0.1)

    # panel 4: connected components above the cut
    g4 = GraphView(sc, [P(3, h) for h in hit], edges, r=6.5, z_n=8)
    for c in g4.nodes:
        c.set_edgecolor(CARD)

    @sc.on
    def _(t):
        u = prog(t, 2.2, 2.2)
        for j, c in enumerate(n1):
            p = hit[j] + (emb[j] - hit[j]) * u
            c.center = P(0, p)
            c.set_facecolor(mix(HIT, TRACK[owner[j]], u))
            show(c, prog(t, 1.0 + 0.05 * (j % 6), 0.5))
        show(lay1, prog(t, 0.9, 0.5) * (1 - prog(t, 2.2, 1.2)))
        show(rings, 0.75 * prog(t, 7.4, 0.6) * (1 - 0.6 * prog(t, 9.6, 0.6)))
        for i, ln in enumerate(e2):
            show(ln, 0.85 * prog(t, 8.0 + 0.04 * i, 0.5))
        # panel 3
        a3 = prog(t, 11.4, 0.6)
        it = prog(t, 11.7, 2.64)
        for j in range(len(hit)):
            g3.node(j, HIT, a3)
        for i in range(len(edges)):
            fin, lwf = (INK, 2.6) if true[i] else (LOW, 1.2)
            g3.edge(i, 1.0, mix(EDGE0, fin, it), lerp(1.8, lwf, it), a3)
        # panel 4
        a4 = prog(t, 14.4, 0.6)
        cc = prog(t, 15.0, 1.0)
        for j in range(len(hit)):
            g4.node(j, mix(HIT, TRACK[owner[j]], cc), a4)
        for i, (a, b) in enumerate(edges):
            if true[i]:
                g4.edge(i, 1.0, mix(INK, TRACK[owner[a]], cc), 2.6, a4)
            else:
                g4.edge(i, 1.0, LOW, 1.2, a4 * (1 - cc))

    # numbers: paper vs our reproduction
    tx0, c1, c2 = 120, 1070, 1570
    sc.T(c1, 372, "Paper (Wang et al. 2022)", size=LABEL, weight="semibold", ha="center", t0=18.8)
    sc.T(c2, 372, "Our reproduction (test)", size=LABEL, weight="semibold", ha="center", t0=18.8)
    sc.fade(sc.seg(tx0, 352, 1800, 352, HAIR, lw=1.4, z=2), 18.8)
    rows = [("all tracks (all targets)", "0.9561 ± 0.0006", "**0.9554 ± 0.0034**"),
            ("prompt W muons", "0.990 ± 0.002", "0.983 ± 0.002"),
            ("displaced HNL daughters", "0.916 ± 0.005", "–")]
    for i, (a, b, c) in enumerate(rows):
        y = 316 - 42 * i
        t0 = 19.2 + 0.35 * i
        sc.T(tx0, y, a, size=LABEL, color=INK2, va="center", t0=t0)
        sc.T(c1, y, b, size=LABEL, ha="center", va="center", t0=t0)
        sc.T(c2, y, c, size=LABEL, ha="center", va="center", t0=t0)
    sc.narrate(LINES["acorn"])
    return sc


def scene_failure():
    sc = Scene("failure", 27.0, title="Why it fails on displaced tracks", footer=SIM)
    panels = [(120, 540, 820, 350, 0.7), (980, 540, 820, 350, 6.9), (120, 165, 820, 340, 13.3),
              (980, 165, 820, 340, 17.4)]
    heads = ["1   The pointing shortcut", "2   Start-radius sensitivity", "3   The graph misses displaced pairs",
             "4   Connected components merges tracks"]
    for (x, y, w, h, t0), s in zip(panels, heads):
        sc.fade(sc.box(x, y, w, h, fc=CARD, r=18, z=1), t0, 0.6)
        sc.T(x + 30, y + h - 44, s, size=BODY, weight="semibold", t0=t0 + 0.15)

    # panel 1: edge TPR change between the prompt and the displaced |d0| bin; the bin labels head the two columns,
    # so the two-line value label of the acorn line has the space below it to itself
    xa, xb, yr, k = 300, 560, 772, PX_PT
    sc.T(xa, 800, "|d_{0}| 0.1–1 mm", size=SMALL, color=INK2, ha="center", t0=1.3)
    sc.T(xb, 800, "|d_{0}| 100–300 mm", size=SMALL, color=INK2, ha="center", t0=1.3)
    sc.fade(sc.seg(xa - 30, yr, xb + 30, yr, HAIR, lw=1.4, z=2), 1.3)
    sc.T(150, 562, "edge TPR change (points)", size=SMALL, color=MUTED, t0=1.3)
    lines = []
    for val, col, lab, t0 in ((-10.2, BLUE, "**−10.2** acorn, {tt}-trained", 2.1),
                              (-6.0, BLUE_T, "**−6.0** equal-budget IGNN", 2.5)):
        ln = sc.seg(xa, yr, xa, yr, col if col != BLUE_T else "#9DB3CB", lw=3.2, z=5)
        dot = sc.circle(xa, yr, 7.5, fc=col if col != BLUE_T else "#9DB3CB", ec=CARD, lw=1.5, z=6)
        lines.append((ln, dot, val, t0))
        sc.T(xb + 22, yr + val * k, lab, size=SMALL, color=INK, va="center", t0=t0 + 2.0)
        if val < -10:
            sc.T(xb + 22, yr + val * k - 34, "arm (i), seed 1", size=SMALL, color=MUTED, va="center", t0=t0 + 2.2)
    sc.fade(sc.circle(xa, yr, 7.5, fc=INK2, ec=CARD, lw=1.5, z=7), 1.6)

    @sc.on
    def _(t):
        for ln, dot, val, t0 in lines:
            u = prog(t, t0, 2.0)
            xe, ye = lerp(xa, xb, u), yr + val * k * u
            ln.set_data([xa, xe], [yr, ye])
            dot.center = (xe, ye)
            show([ln, dot], prog(t, t0 - 0.2, 0.3))

    # panel 2: truncation probe
    O = np.array([1040.0, 585.0])
    radii = [34, 58, 82, 106, 140, 174, 208, 242]
    phi = math.radians(38)
    for r in radii:
        w = Arc((O[0], O[1]), 2 * r, 2 * r, theta1=14, theta2=66, edgecolor=LAYER, lw=1.6, zorder=2)
        sc.ax.add_patch(w)
        sc.fade(w, 7.3)
    _, x, y = helix(O, phi, -1 / 900, 300)
    x, y = clip_r(x, y, O, 250)
    cr = crossings(x, y, O, radii)
    trk = GrowLine(sc, x, y, color=PROMPT, lw=2.4, z=4)
    hits = [sc.circle(hx, hy, 6.2, fc=HIT, ec=CARD, lw=1.2, z=8) for _, hx, hy in cr]
    ghosts = [sc.circle(hx, hy, 6.8, fc="none", ec=MUTED, lw=1.7, z=8) for _, hx, hy in cr[:4]]
    sc.fade(sc.circle(O[0], O[1], 5, fc=INK, z=9), 7.3)
    rm = [8.9 + 0.8 * i for i in range(4)]
    for i in range(5):
        sc.T(1285, 800, f"inner hits removed: {i}", size=SMALL, color=INK2, t0=8.4 if i == 0 else rm[i - 1] + 0.27,
             d=0.25, t1=rm[i] if i < 4 else None, d1=0.25)
    bar = HBar(sc, 1770, 700, 26, "#9DB3CB", sign=-1)
    sc.fade(sc.seg(1770, 670, 1770, 730, INK2, lw=1.6, z=6), 12.2)
    sc.T(1770, 620, "edge TPR change, dT = **−5.3** points", size=SMALL, color=INK, ha="right", t0=13.0)
    sc.T(1770, 584, "(equal-budget IGNN, depth 0 → 4)", size=SMALL, color=MUTED, ha="right", t0=13.0)
    sc.T(1285, 764, "prompt particles, 300 HNL val events", size=SMALL, color=MUTED, t0=8.4)

    @sc.on
    def _(t):
        a = prog(t, 7.3, 0.5)
        trk.set(prog(t, 7.6, 1.0), a)
        for i, c in enumerate(hits):
            gone = prog(t, rm[i], 0.5) if i < 4 else 0.0
            show(c, a * ease((trk.p * trk.L - cr[i][0]) / 12.0) * (1 - gone))
            if i < 4:  # the ghosts stay until the scene fades out (the cover fades them with everything else)
                show(ghosts[i], 0.9 * gone)
        bar.set(5.3 * PX_PT * prog(t, 12.2, 0.9), 1.0)

    # panel 3: graph ceilings
    sc.T(150, 418, "primary graph ceiling (perfect edge classifier)", size=SMALL, color=MUTED, t0=13.6)
    for x, v, lab, t0 in ((150, "0.984", "graph trained on {tt}\narm (i)", 14.0),
                          (520, "0.993", "graph trained on HNL\narm (ii)", 14.5)):
        sc.T(x, 318, v, size=50, weight="semibold", t0=t0, rise=8)
        sc.T(x + 2, 262, lab, size=SMALL, color=INK2, t0=t0 + 0.2)

    # panel 4: connected components merges two close tracks
    xs = np.array([1020, 1062, 1104, 1146, 1188, 1230])
    ya = np.array([212, 246, 280, 312, 344, 374])
    yb = np.array([262, 291, 320, 348, 376, 404])
    gp = [(x, y) for x, y in zip(xs, ya)] + [(x, y) for x, y in zip(xs, yb)]
    ge = [(i, i + 1) for i in range(5)] + [(6 + i, 7 + i) for i in range(5)] + [(2, 9)]
    g4 = GraphView(sc, gp, ge, r=6.5, z_n=8)
    for c in g4.nodes:
        c.set_edgecolor(CARD)
    sc.T(1275, 392, "one fake edge joins two tracks:", size=SMALL, color=INK2, t0=19.6)
    sc.T(1275, 355, "one candidate, two particles", size=SMALL, color=INK2, t0=20.6)
    sc.T(1275, 298, "{tt} prompt efficiency, acorn + CC:", size=SMALL, color=INK2, t0=21.6)
    sc.T(1275, 261, "**87 %** {tt}-trained, 85 % HNL-trained", size=SMALL, color=INK, t0=21.6)
    sc.T(1275, 214, "{tt}-trained + our builder: 95.5 %", size=SMALL, color=MUTED, t0=22.6)

    @sc.on
    def _(t):
        a = prog(t, 17.8, 0.6)
        m = prog(t, 20.4, 1.0)
        for j in range(12):
            base = TRACK[1] if j < 6 else TRACK[2]
            g4.node(j, mix(base, TRACK[1], m) if j >= 6 else base, a)
        for i, (u, v) in enumerate(ge):
            if i == 10:
                g4.edge(i, prog(t, 19.2, 0.6), mix(INK, TRACK[1], m), 2.6, a)
            else:
                base = TRACK[1] if u < 6 else TRACK[2]
                g4.edge(i, 1.0, mix(base, TRACK[1], m) if u >= 6 else base, 2.6, a)

    sc.narrate(LINES["failure"])
    return sc


def scene_canto():
    sc = Scene("canto", 18.2, title="Canto: a track as a chain of triplets")
    sc.T(120, 868, "[[o:**C**]]hained-triplet, origin-[[o:**A**]]gnostic [[o:**N**]]etworks for "
                   "[[o:**T**]]racking [[o:**O**]]ff-vertex", size=BODY, color=INK2, t0=0.5)
    # tercets of a canto: rhyme scheme ABA BCB CDC, the middle line of each tercet rhymes with the next tercet
    rhymes = ["ABA", "BCB", "CDC"]
    lens = [[380, 430, 360], [410, 350, 420], [370, 440, 390]]
    x0, ytop = 150, 760
    letter_pos = {}
    objs_t = []
    for k, (rh, ln) in enumerate(zip(rhymes, lens)):
        for i in range(3):
            y = ytop - k * 160 - i * 46
            t0 = 1.0 + 0.45 * k + 0.12 * i
            bar = sc.seg(x0, y, x0 + ln[i], y, CARD2, lw=12, z=3)
            circ = sc.circle(x0 + ln[i] + 34, y, 19, fc=CARD, ec=INK2, lw=1.6, z=5)
            lab = sc.T(x0 + ln[i] + 34, y, rh[i], size=SMALL, weight="semibold", ha="center", va="center", z=31)
            sc.fade([bar, circ, lab], t0, 0.5)
            letter_pos[(k, i)] = (x0 + ln[i] + 34, y)
            objs_t.append(circ)
    links = []
    for k in range(2):
        m = np.array(letter_pos[(k, 1)])
        for i in (0, 2):
            q = np.array(letter_pos[(k + 1, i)])
            # control point right of every letter between the two ends, so no arc runs under another letter
            xmax = max(p[0] for p in letter_pos.values() if q[1] - 20 <= p[1] <= m[1] + 20)
            ctrl = (xmax + 129 + 40 * i, (m[1] + q[1]) / 2)
            bx, by = bezier(m + (19, 0), ctrl, q + (19, 0))
            links.append(GrowLine(sc, bx, by, color=ORANGE, lw=2.6, z=4))
    sc.T(x0, 300, "a canto: interlocking tercets (ABA BCB CDC …)", size=SMALL, color=MUTED, t0=3.4)

    @sc.on
    def _(t):
        for j, g in enumerate(links):
            g.set(prog(t, 3.0 + 0.35 * j, 0.9))

    # a track as chained triplets
    xs = np.linspace(1040, 1760, 7)
    ys = 380 + 0.00062 * (xs - 1040) ** 2 + 0.12 * (xs - 1040)
    pts = list(zip(xs, ys))
    trk = GrowLine(sc, xs, ys, color=INK2, lw=2.4, z=5)
    hits = [sc.circle(x, y, 9, fc=INK, ec=BG, lw=1.8, z=10) for x, y in pts]
    names = "abcdefg"
    for i, (x, y) in enumerate(pts):
        sc.T(x + 4, y - 40, names[i], size=LABEL, color=INK2, ha="center", va="center", t0=6.2 + 0.08 * i)
    bands = [chain_band(sc, pts[i:i + 3], color=ORANGE, lw=34, z=3) for i in range(5)]
    for i in range(5):
        mx, my = pts[i + 1]
        sc.T(mx - 46, my + 64, f"({names[i]}, {names[i + 1]}, {names[i + 2]})", size=BODY, ha="right", va="center",
             t0=6.9 + 1.15 * i, d=0.4, t1=(6.9 + 1.15 * (i + 1) - 0.35) if i < 4 else None, d1=0.3)
    sc.T(1400, 300, "consecutive triplets share two hits", size=SMALL, color=MUTED, ha="center", t0=8.2)

    @sc.on
    def _(t):
        trk.set(prog(t, 5.8, 1.0), 1.0)
        for i, c in enumerate(hits):
            show(c, prog(t, 5.9 + 0.1 * i, 0.4))
        for i, b in enumerate(bands):
            t0 = 6.9 + 1.15 * i
            cur = prog(t, t0, 0.5)
            later = prog(t, t0 + 1.15, 0.5) if i < 4 else 0.0
            show(b, 0.34 * cur * (1 - 0.45 * later))

    # three learned stages
    stages = ["Learned graph", "Triplet edge classifier", "Track builder on learned scores"]
    pills = pill_row(sc, stages, 222, idle=(CARD, INK2), active=(ORANGE_T, INK))
    arr = [Arrow(sc, (pills[i].right + 12, 222), (pills[i + 1].left - 12, 222), color=FAINT, lw=2.2, head=12)
           for i in range(2)]

    @sc.on
    def _(t):
        for i, p in enumerate(pills):
            p.set_alpha(prog(t, 12.9 + 0.35 * i, 0.5))
            p.style(prog(t, 13.0 + 0.35 * i, 0.6))
        for i, a in enumerate(arr):
            a.set(prog(t, 13.2 + 0.35 * i, 0.4))

    sc.narrate(LINES["canto"])
    return sc


LADDER = [("IGNN (acorn architecture)", -6.0, -5.3, 0), ("+ trunc", 0.5, 0.0, 1), ("L3 (absolute)", -4.2, -8.4, 0),
          ("+ trunc", 0.7, -0.1, 1), ("L0 (exactly invariant)", -3.8, -6.7, 0), ("+ trunc", 3.1, -0.1, 1),
          ("L1 (symmetric + regions)", -4.3, -7.4, 0), ("+ trunc", 2.4, -0.1, 2)]


def fmt_pts(v):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f}"


def scene_classifier():
    sc = Scene("classifier", 33.4, title="The Canto edge classifier", footer=SIM)
    PX, PW, PY, PHh = [120, 690, 1260], 540, 290, 590
    t_p = [0.6, 7.8, 13.8]
    heads = ["Triplet messages", "Production-point invariance", "Inner-hit truncation"]
    part1 = []
    for k, x0 in enumerate(PX):
        b = sc.box(x0, PY, PW, PHh, fc=CARD, r=18, z=1)
        r = sc.T(x0 + 30, PY + PHh - 46, heads[k], size=BODY, weight="semibold")
        part1.append((b, r, t_p[k]))

    # (a) triplet messages
    a_, b_, c_ = np.array([210.0, 560.0]), np.array([380.0, 650.0]), np.array([570.0, 690.0])
    band = chain_band(sc, [a_, b_, c_], color=ORANGE, lw=36, z=2)
    e_ab = sc.seg(*a_, *b_, INK2, lw=3, z=4)
    e_bc = sc.seg(*b_, *c_, INK2, lw=3, z=4)
    hn = [sc.circle(p[0], p[1], 10, fc=INK, ec=CARD, lw=1.8, z=8) for p in (a_, b_, c_)]
    halo = sc.circle(b_[0], b_[1], 22, fc=ORANGE_T, ec=ORANGE, lw=2, z=6)
    lab_a = [sc.T(p[0], p[1] - 40, s, size=LABEL, color=INK2, ha="center", va="center") for p, s in
             ((a_, "a"), (b_, "b"), (c_, "c"))]
    pulse = sc.circle(0, 0, 8, fc=ORANGE, z=9)
    pa_txt = [sc.T(150, 452, "one message from two edges that", size=SMALL, color=INK2),
              sc.T(150, 415, "share a hit: (a → b) and (b → c)", size=SMALL, color=INK2),
              sc.T(150, 345, "8 iterations, hidden 128,", size=SMALL, color=MUTED),
              sc.T(150, 308, "347,777 parameters", size=SMALL, color=MUTED)]

    # (b) translation invariance: a shallow track is shifted along its direction (r grows, phi stays) and then
    # rotated by +20 degrees about the beam line (phi grows, r stays), so both steps lengthen the absolute-input
    # bars; with a steeper track or a -20 degree turn the phi term would undo the r term
    O = np.array([800.0, 575.0])
    base = O + np.array([[40.0, 4.0], [110.0, 16.0], [180.0, 24.0], [250.0, 28.0]])
    shift = np.array([110.0, 12.0])
    bn = [sc.circle(p[0], p[1], 8.5, fc=INK, ec=CARD, lw=1.6, z=8) for p in base]
    be = [sc.seg(0, 0, 1, 1, INK2, lw=2.6, z=5) for _ in range(3)]
    beam = [sc.circle(O[0], O[1], 9, fc="none", ec=INK2, lw=1.8, z=6), sc.circle(O[0], O[1], 3, fc=INK2, z=6)]
    beam_lab = sc.T(O[0] - 4, O[1] - 36, "beam line", size=SMALL, color=MUTED, ha="center", va="center")
    glyph_y = [474, 418]
    glyph_lab = [sc.T(720, glyph_y[0] + 14, "absolute (r, φ, z)", size=SMALL, color=INK2, va="center"),
                 sc.T(720, glyph_y[1] + 14, "triplet inputs", size=SMALL, color=INK2, va="center")]
    gbars = [[sc.box(1040 + 34 * i, glyph_y[j], 22, 10, fc=BLUE if j == 0 else ORANGE, r=4, z=6) for i in range(4)]
             for j in range(2)]
    tri_h = [30, 18, 38, 24]
    status = [sc.T(720, 384, "absolute inputs change;", size=SMALL, color=MUTED),
              sc.T(720, 347, "Canto's triplet inputs do not", size=SMALL, color=INK, weight="semibold"),
              sc.T(720, 310, "(apart from a 6-region one-hot)", size=SMALL, color=MUTED)]
    pb_txt = []

    # (c) truncation augmentation
    O2 = np.array([1300.0, 470.0])
    tpts = [O2 + (k + 1) * np.array([54.0, 33.0]) + np.array([0.0, -0.9 * (k + 1) ** 2]) for k in range(8)]
    tn = [sc.circle(p[0], p[1], 8, fc=INK, ec=CARD, lw=1.6, z=8) for p in tpts]
    te = [sc.seg(*tpts[k], *tpts[k + 1], INK2, lw=2.6, z=5) for k in range(7)]
    te0 = sc.seg(*O2, *tpts[0], FAINT, lw=1.8, z=4, ls=(0, (2, 3)))
    ghost = [sc.circle(p[0], p[1], 8, fc="none", ec=FAINT, lw=1.5, z=8) for p in tpts[:3]]
    beam2 = [sc.circle(O2[0], O2[1], 9, fc="none", ec=INK2, lw=1.8, z=6), sc.circle(O2[0], O2[1], 3, fc=INK2, z=6)]
    brace = sc.T(tpts[1][0] + 18, tpts[1][1] - 48, "m innermost hits removed", size=SMALL, color=INK2)
    pc_txt = [sc.T(1290, 382, "30 % of {tt} particles with ≥ 6 hits", size=SMALL, color=INK2),
              sc.T(1290, 345, "lose their m innermost hits,", size=SMALL, color=INK2),
              sc.T(1290, 308, "m ~ U{1, …, n − 5}; training only", size=SMALL, color=MUTED)]

    def out1(t):
        return 1.0 - prog(t, 19.0, 0.6)

    @sc.on
    def _(t):
        o = out1(t)
        for b, r, t0 in part1:
            show([b, r], prog(t, t0, 0.6) * o)
        # (a)
        a = prog(t, 1.0, 0.6) * o
        show(hn + lab_a, a)
        show(e_ab, prog(t, 1.6, 0.5) * o)
        show(e_bc, prog(t, 1.9, 0.5) * o)
        show(band, 0.30 * prog(t, 2.6, 0.6) * o)
        show(pa_txt[:2], prog(t, 2.6, 0.6) * o)
        show(pa_txt[2:], prog(t, 5.4, 0.6) * o)
        t0p = 3.2 if t < 4.75 else 4.8
        u = prog(t, t0p, 1.4)
        pos = a_ + (b_ - a_) * min(1.0, 2 * u) + (c_ - b_) * max(0.0, 2 * u - 1)
        pulse.center = (pos[0], pos[1])
        show(pulse, o * prog(t, t0p - 0.15, 0.25) * (1 - prog(t, t0p + 1.3, 0.25)))
        show(halo, 0.9 * prog(t, 6.0, 0.6) * o)
        # (b)
        b = prog(t, 8.2, 0.6) * o
        u = prog(t, 9.4, 2.0)
        rot = math.radians(20) * prog(t, 12.0, 1.6)
        pts = base + shift * u
        cr, sr = math.cos(rot), math.sin(rot)
        pts = np.array([O + np.array([cr * (p[0] - O[0]) - sr * (p[1] - O[1]), sr * (p[0] - O[0]) + cr * (p[1] - O[1])])
                        for p in pts])
        for i, c in enumerate(bn):
            c.center = (pts[i][0], pts[i][1])
        for i, ln in enumerate(be):
            ln.set_data([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]])
        show(bn + be + beam + [beam_lab], b)
        show(glyph_lab, prog(t, 8.8, 0.6) * o)
        for i, gb in enumerate(gbars[0]):  # absolute inputs: r grows with the shift, phi with the rotation
            rel = pts[i] - O
            rr, ph = math.hypot(*rel), math.degrees(math.atan2(rel[1], rel[0]))
            gb.set_bounds(1040 + 34 * i, glyph_y[0], 22, 6 + 0.11 * rr + 0.6 * ph)
        for i, gb in enumerate(gbars[1]):
            gb.set_bounds(1040 + 34 * i, glyph_y[1], 22, tri_h[i])
        show(gbars[0] + gbars[1], prog(t, 8.8, 0.6) * o)
        show(status, prog(t, 11.0, 0.6) * o)
        show(pb_txt, prog(t, 11.4, 0.6) * o)
        # (c)
        c = prog(t, 14.2, 0.6) * o
        gone = prog(t, 15.6, 1.2)
        show(beam2, c)
        for k, n in enumerate(tn):
            show(n, c * (1 - gone) if k < 3 else c)
        for k, e in enumerate(te):
            show(e, c * (1 - gone) if k < 3 else c)
        show(te0, 0.8 * c * (1 - gone))
        show(ghost, c * gone)
        show(brace, prog(t, 16.8, 0.6) * o)
        show(pc_txt, prog(t, 15.0, 0.6) * o)

    # part 2: the symmetry ladder
    sc.T(120, 868, "Symmetry ladder (screening budget: 15 epochs × first 4000 {tt} graphs, arm (i) seed 0; val)",
         size=SMALL, color=MUTED, t0=19.6)
    xl, xd, xt, kpx = 560, 900, 1580, 34.0
    rows_y = [790 - 64 * i for i in range(8)]
    sc.T(xd, 832, "dD (points)", size=LABEL, weight="semibold", ha="center", t0=19.6)
    sc.T(xt, 832, "dT (points)", size=LABEL, weight="semibold", ha="center", t0=19.6)
    sc.fade([sc.seg(xd, rows_y[-1] - 30, xd, rows_y[0] + 30, INK2, lw=1.4, z=3),
             sc.seg(xt, rows_y[-1] - 30, xt, rows_y[0] + 30, INK2, lw=1.4, z=3)], 19.6)
    hl = sc.box(130, rows_y[-1] - 28, 1720, 56, fc=ORANGE_T, r=12, z=1)
    sc.fade(hl, 28.0, 0.8)
    bars = []
    for i, (name, dd, dt, kind) in enumerate(LADDER):
        y = rows_y[i]
        t0 = 20.4 + 1.25 * (i // 2) + 0.3 * (i % 2)
        col = {0: "#BDB9AF", 1: ORANGE_M, 2: ORANGE}[kind]
        sc.T(xl, y, name, size=LABEL, color=INK2 if kind == 0 else INK, weight="semibold" if kind == 2 else "normal",
             ha="right", va="center", t0=t0 - 0.2)
        bd = HBar(sc, xd, y, 28, col, sign=1 if dd > 0 else -1)
        bt = HBar(sc, xt, y, 28, col, sign=1 if dt > 0 else -1)
        for v, x0, bb in ((dd, xd, bd), (dt, xt, bt)):
            xe = x0 + kpx * v
            if v > 0:
                sc.T(xe + 10, y, fmt_pts(v), size=SMALL, color=INK, ha="left", va="center", t0=t0 + 0.8)
            elif v < 0:
                sc.T(xe - 10, y, fmt_pts(v), size=SMALL, color=INK, ha="right", va="center", t0=t0 + 0.8)
            else:
                sc.T(x0 + 10, y, fmt_pts(v), size=SMALL, color=INK, ha="left", va="center", t0=t0 + 0.8)
            bars.append((bb, abs(v) * kpx, t0))
    sc.legend_row(560, 222, [("sq", "#BDB9AF", "no augmentation"), ("sq", ORANGE_M, "+ inner-hit truncation"),
                             ("sq", ORANGE, "Canto classifier (L1 + trunc)")], size=SMALL, t0=20.2)

    @sc.on
    def _(t):
        for bb, L, t0 in bars:
            bb.set(L * prog(t, t0, 0.8))

    sc.narrate(LINES["classifier"])
    return sc


def scene_branches():
    sc = Scene("branches", 40.7, title="Three branches: Canto-B, Canto-H, Canto-L", footer=SIM)
    LX = [200 + 140 * i for i in range(7)]
    tracks = {"P1": [800, 795, 786, 773, 756, 735, 710], "P2": [650, 652, 651, 647, 640, 630, 617],
              "D": [None, None, 400, 425, 460, 505, 560], "P4": [330, 316, 314, 324, 346, None, None]}
    idx, pts, owner = {}, [], []
    for name, ys in tracks.items():
        for i, y in enumerate(ys):
            if y is not None:
                idx[(name, i)] = len(pts)
                pts.append((LX[i], y))
                owner.append(name)
    chain = lambda nm, pairs: [(idx[(nm, a)], idx[(nm, b)]) for a, b in pairs]  # noqa: E731
    base = (chain("P1", [(i, i + 1) for i in range(6)]) + chain("P2", [(i, i + 1) for i in range(6)])
            + chain("D", [(2, 3), (4, 5), (5, 6)]) + chain("P4", [(0, 1), (1, 2), (3, 4)])
            + [(idx[("P2", 0)], idx[("P1", 1)]), (idx[("D", 5)], idx[("P2", 6)])])
    nbase = len(base)
    # wider-radius candidates: two true edges (displaced D, low-pT P4) and three fakes
    cand = [(idx[("D", 3)], idx[("D", 4)]), (idx[("P4", 2)], idx[("P4", 3)]), (idx[("P4", 3)], idx[("D", 4)]),
            (idx[("D", 3)], idx[("P4", 4)]), (idx[("D", 2)], idx[("P4", 3)])]
    halos = [chain_band(sc, [pts[a], pts[b]], ORANGE, 16, 3) for a, b in cand[:2]]
    solid = [sc.seg(*pts[a], *pts[b], ORANGE, lw=2.6, z=6) for a, b in cand[:2]]
    g = GraphView(sc, pts, base + cand, r=7.5)
    tH, tL, t1o = 6.6, 12.6, 18.2  # part 1: Canto-H act from tH, Canto-L act from tL, everything out at t1o
    for x in LX:
        sc.fade(sc.seg(x, 268, x, 848, HAIR, lw=1.3, z=1), 0.5, 0.8, t1=t1o, d1=0.6)
    gap = np.array(pts[idx[("D", 3)]]) / 2 + np.array(pts[idx[("D", 4)]]) / 2
    sc.T(560, 592, "missing edge", size=SMALL, color=INK2, t0=3.0, t1=6.5, bg=BG)
    miss = sc.seg(gap[0] - 2, gap[1] + 14, 640, 576, FAINT, lw=1.5, z=3)
    sc.fade(miss, 3.0, t1=6.5)
    trip_true = [chain_band(sc, [pts[idx[("D", a)]] for a in (2, 3, 4)], ORANGE, 26, 3),
                 chain_band(sc, [pts[idx[("D", a)]] for a in (3, 4, 5)], ORANGE, 26, 3)]
    trip_low = chain_band(sc, [pts[idx[("P4", a)]] for a in (1, 2, 3)], "#9C988D", 26, 3)
    # the low-pT label stays 3 s and is gone before the caption switches to Canto-L
    sc.T(250, 252, "low-pT track: turning angle outside the hand cut", size=SMALL, color=INK2, t0=8.8, t1=11.8,
         bg=BG)
    sc.T(250, 252, "the learned support keeps its edge", size=SMALL, color=INK2, t0=tL + 2.6, t1=tL + 5.2, bg=BG)

    # cards
    cx, cw = 1180, 660
    cards = [("B", "**Canto-B**   base graph", ["the {tt}-trained metric-learning graph", "at r* = 0.16"]),
             ("H", "**Canto-H**   helix union (hand cuts)", ["+ wider-radius candidates, r_{w} = 0.24,",
                                                             "kept by helix-triplet support:",
                                                             "|Δcot θ| < 5σ, pT ≥ 0.5 GeV arc,",
                                                             "mutual top-2"]),
             ("L", "**Canto-L**   learned triplet support", ["gradient-boosted trees on 16 invariant",
                                                             "triplet features (holdout AUC 0.998);",
                                                             "mutual top-5, p > 0.005"])]
    cy, ch = [710, 466, 222], [150, 220, 220]
    card_objs = []
    for k, (letter, head, body) in enumerate(cards):
        y0 = cy[k]
        b = sc.box(cx, y0, cw, ch[k], fc=CARD, r=18, z=1)
        badge = sc.circle(cx + 46, y0 + ch[k] - 40, 22, fc=ORANGE, z=4)
        bl = sc.T(cx + 46, y0 + ch[k] - 40, letter, size=LABEL, color="#FFFFFF", weight="bold", ha="center", va="center")
        hd = sc.T(cx + 84, y0 + ch[k] - 40, head, size=LABEL, va="center")
        body_r = [sc.T(cx + 84, y0 + ch[k] - 84 - 36 * i, s, size=SMALL, color=INK2, va="center")
                  for i, s in enumerate(body)]
        card_objs.append(([b, badge, bl, hd] + body_r, 0.6 + 0.3 * k))
    act = [(1.0, tH), (tH, tL), (tL, t1o)]
    dashed = (0, (3, 3))

    @sc.on
    def _(t):
        o = 1 - prog(t, t1o, 0.6)
        for k, (objs, t0) in enumerate(card_objs):
            on = prog(t, act[k][0], 0.5) * (1 - prog(t, act[k][1], 0.5))
            show(objs, prog(t, t0, 0.6) * o * (0.45 + 0.55 * on))
            objs[0].set_facecolor(mix(CARD, ORANGE_T, on))
        a = prog(t, 0.8, 0.8) * o
        for j in range(len(pts)):
            g.node(j, ORANGE_D if owner[j] == "D" else HIT, a)
        for i in range(nbase):
            fake = i >= nbase - 2
            col = ORANGE if owner[base[i][0]] == "D" and not fake else (EDGE0 if fake else INK2)
            g.edge(i, prog(t, 1.4 + 0.05 * i, 0.6), col, 1.6 if fake else 2.6, o)
        own_col = [ORANGE, INK2]
        if t < tL:  # Canto-H: hand-cut helix-triplet support keeps only the displaced edge
            hph = prog(t, 6.9, 0.6) * (1 - prog(t, tL - 0.4, 0.4))
            for j, i in enumerate(range(nbase, nbase + len(cand))):
                if j == 0:
                    col = mix(MUTED, ORANGE, prog(t, 8.9, 0.6))
                    g.edge(i, 1.0, col, 2.6, hph * (1 - prog(t, 9.1, 0.5)), dashed)
                    solid[0].set_color(col)
                    show(solid[0], hph * prog(t, 9.1, 0.5))
                else:
                    g.edge(i, 1.0, MUTED, 1.8, hph * (1 - prog(t, 9.6 + 0.15 * j, 0.6)), dashed)
            show(solid[1], 0.0)
            show(halos[0], 0.30 * prog(t, 9.4, 0.5) * (1 - prog(t, tL - 0.4, 0.4)))
            show(halos[1], 0.0)
        else:  # Canto-L: learned triplet probability keeps both true edges
            lph = prog(t, tL + 0.3, 0.6) * o
            for j, i in enumerate(range(nbase, nbase + len(cand))):
                if j < 2:
                    col = mix(MUTED, own_col[j], prog(t, tL + 1.4, 0.8))
                    g.edge(i, 1.0, col, 2.6, lph * (1 - prog(t, tL + 2.1, 0.5)), dashed)
                    solid[j].set_color(col)
                    show(solid[j], lph * prog(t, tL + 2.1, 0.5))
                else:
                    g.edge(i, 1.0, mix(MUTED, LOW, prog(t, tL + 1.4, 0.8)), 1.8,
                           lph * (1 - prog(t, tL + 2.4 + 0.15 * j, 0.6)), dashed)
            show(halos, 0.30 * prog(t, tL + 2.4, 0.5) * o)
        show(trip_true, 0.32 * prog(t, 8.0, 0.6) * (1 - prog(t, 11.8, 0.5)))
        show(trip_low, 0.30 * prog(t, 8.8, 0.5) * (1 - prog(t, 11.8, 0.5)))

    # part 2: numbers
    t2, t2o = 18.8, 29.8
    cols = [860, 1240, 1620]
    heads2 = [("B", "Canto-B", "base graph"), ("H", "Canto-H", "hand-cut union"), ("L", "Canto-L", "learned union")]
    for (letter, nm, sub), x in zip(heads2, cols):
        sc.fade(sc.circle(x - 96, 816, 20, fc=ORANGE, z=4), t2, t1=t2o)
        sc.T(x - 96, 816, letter, size=SMALL, color="#FFFFFF", weight="bold", ha="center", va="center", t0=t2, t1=t2o)
        sc.T(x - 64, 816, f"**{nm}**", size=LABEL, va="center", t0=t2, t1=t2o)
        sc.T(x - 64, 776, sub, size=SMALL, color=MUTED, va="center", t0=t2, t1=t2o)
    table = [("pairs per {tt} event", ["28.7k", "35.5k", "33.9k"]),
             ("{tt} chain-edge efficiency", ["0.9914", "0.9933", "0.9961"]),
             ("primary graph ceiling", ["0.9842", "0.9947", "0.9947"]),
             ("|d_{0}| ≥ 300 mm ceiling", ["13 of 19", "17 of 19", "17 of 19"])]
    ty = [690, 600, 510, 420]
    for i, (lab, vals) in enumerate(table):
        t0 = t2 + 0.4 + 0.35 * i
        sc.T(120, ty[i], lab, size=LABEL, color=INK2, va="center", t0=t0, t1=t2o)
        for x, v in zip(cols, vals):
            sc.T(x, ty[i], v, size=BODY, ha="center", va="center", t0=t0, t1=t2o)
        sc.fade(sc.seg(120, ty[i] - 45, 1800, ty[i] - 45, HAIR, lw=1.2, z=1), t0, t1=t2o)
    hl_ceiling = sc.box(110, ty[2] - 32, 1700, 64, fc=ORANGE_T, r=12, z=0)
    sc.fade(hl_ceiling, t2 + 1.8, 0.6, t1=t2 + 4.8)
    ar = Arrow(sc, (cols[0] + 70, ty[2] + 30), (cols[1] - 70, ty[2] + 30), color=ORANGE, lw=2.4, head=12)
    hl_l = [sc.box(cols[2] - 110, ty[k] - 30, 220, 60, fc=ORANGE_T, r=12, z=0) for k in (0, 1)]
    sc.fade(hl_l, t2 + 5.2, 0.6, t1=t2 + 10.6)
    sc.T(120, 336, "val, arm (i) seed 0, 2999 events. Hand cuts reject 29 % of the true triplets of a {tt} event "
                   "(mostly pT < 0.5 GeV).", size=SMALL, color=MUTED, t0=t2 + 1.8, t1=t2o)

    @sc.on
    def _(t):
        ar.set(prog(t, t2 + 2.2, 0.8), 1 - prog(t, t2 + 4.8, 0.5))

    # part 3: size control, ceiling vs graph size; the x range starts at 20k pairs so that the base point has room
    # for its label on its left, apart from the r = 0.165 label below-right of the next point
    t3 = 30.4
    X = lambda k: 380 + (k - 20.0) / 37.0 * 1180  # noqa: E731
    Y = lambda c: 280 + (c - 0.9830) / 0.0125 * 520  # noqa: E731
    axes = [sc.seg(380, 280, 1580, 280, INK2, lw=1.4, z=2), sc.seg(380, 280, 380, 820, INK2, lw=1.4, z=2)]
    sc.fade(axes, t3)
    sc.T(980, 236, "pairs per {tt} event", size=LABEL, color=INK2, ha="center", t0=t3)
    ytitle = sc.ax.text(342, 550, "primary graph ceiling", rotation=90, ha="center", va="center", fontsize=LABEL,
                        color=INK2, zorder=30)
    sc.fade(ytitle, t3)
    radius = [(28.7, 0.9842, "base, r* = 0.16\n28.7k, 0.9842"), (31.3, 0.9850, "r = 0.165:  31.3k, 0.9850"),
              (34.1, 0.9865, "r = 0.17:  34.1k, 0.9865"), (54.4, 0.9947, "r = 0.20:  54.4k, 0.9947")]
    rx = [X(k) for k, _, _ in radius]
    ry = [Y(c) for _, c, _ in radius]
    # the line reaches r = 0.17 with the first caption and r = 0.20 with the second
    rlines = [GrowLine(sc, rx[:3], ry[:3], color="#BDB9AF", lw=2.6, z=4),
              GrowLine(sc, rx[2:], ry[2:], color="#BDB9AF", lw=2.6, z=4)]
    rdots = [sc.circle(x, y, 8, fc="#9C988D", ec=BG, lw=2, z=6) for x, y in zip(rx, ry)]
    t_dot = [t3 + 0.4, t3 + 1.4, t3 + 2.4, t3 + 7.2]
    for i, (k, c, s) in enumerate(radius):
        if i == 0:
            sc.T(rx[0] - 18, ry[0], s, size=SMALL, color=INK2, ha="right", va="center", t0=t_dot[0] + 0.2)
        else:
            sc.T(rx[i] + 16, ry[i] - 12, s, size=SMALL, color=INK2, va="top", t0=t_dot[i] + 0.2)
    hx, lxx, yy = X(35.5), X(33.9), Y(0.9947)
    odots = [sc.circle(lxx, yy, 9, fc=ORANGE, ec=BG, lw=2, z=7), sc.circle(hx, yy, 9, fc=ORANGE, ec=BG, lw=2, z=7)]
    sc.fade(odots, t3 + 3.0, 0.5)
    sc.T(lxx - 12, yy + 20, "Canto-L  33.9k", size=SMALL, color=INK, ha="right", va="bottom", t0=t3 + 3.2)
    sc.T(hx + 12, yy + 20, "Canto-H  35.5k", size=SMALL, color=INK, ha="left", va="bottom", t0=t3 + 3.2)
    sc.T(lxx - 12, yy - 14, "0.9947", size=SMALL, color=MUTED, ha="right", va="top", t0=t3 + 3.2)
    span = Arrow(sc, (hx + 16, yy), (rx[3] - 16, yy), color=ORANGE, lw=2.2, head=12)
    sc.T(hx + 16, yy - 16, "**+53 %** pairs for the same ceiling", size=SMALL, color=INK, ha="left", va="top",
         t0=t3 + 8.2)

    @sc.on
    def _(t):
        rlines[0].set(prog(t, t3 + 0.4, 2.0), 1.0)
        rlines[1].set(prog(t, t3 + 6.2, 1.0), 1.0)
        for i, d in enumerate(rdots):
            show(d, prog(t, t_dot[i] - 0.1, 0.4))
        span.set(prog(t, t3 + 7.8, 0.9))

    sc.narrate(LINES["branches"])
    return sc


def scene_trackbuild():
    sc = Scene("trackbuild", 33.4, title="Track building on the learned edge scores", footer=SIM)
    LX = [200 + 140 * i for i in range(7)]
    A = [780, 745, 705, 660, 610, 556, 498]
    B = [520, 556, 592, 627, 660, 692, 722]
    Cc = [380, 390, 397, 401, 402, 400, 395]
    pts = [(x, y) for x, y in zip(LX, A)] + [(x, y) for x, y in zip(LX, B)] + [(x, y) for x, y in zip(LX, Cc)]
    own = [0] * 7 + [1] * 7 + [2] * 7
    edges, sc_ = [], []
    sA = [0.97, 0.95, 0.93, 0.96, 0.94, 0.92]
    sB = [0.96, 0.94, 0.91, 0.95, 0.90, 0.42]
    sC = [0.95, 0.93, 0.97, 0.22, 0.94, 0.96]
    for off, ss in ((0, sA), (7, sB), (14, sC)):
        for i in range(6):
            edges.append((off + i, off + i + 1))
            sc_.append(ss[i])
    edges.append((3, 11))   # fake A3 -> B4 above s0
    sc_.append(0.72)
    edges.append((10, 4))   # fake B3 -> A4 below s0
    sc_.append(0.30)
    g = GraphView(sc, pts, edges, r=8)
    # part 1 timeline: greedy merge from 2.8, two-pass from t2p, fragment merging from tfm, everything out at t1o
    t2p, tfm, t1o = 9.2, 13.6, 17.2
    for x in LX:
        sc.fade(sc.seg(x, 330, x, 830, HAIR, lw=1.3, z=1), 0.5, 0.8, t1=t1o, d1=0.6)
    order = sorted(range(len(edges)), key=lambda i: -sc_[i])
    s0 = 0.5
    t_step = {}
    for r, i in enumerate([i for i in order if sc_[i] >= s0]):
        t_step[i] = 2.8 + 0.27 * r
    t_fake = t_step[18]
    colors = [TRACK[1], TRACK[2], TRACK[0]]
    frag2 = TRACK[3]
    rep = GrowLine(sc, *bezier(pts[4], (LX[4] + 70, (pts[4][1] + pts[11][1]) / 2), pts[11]), color=INK2, lw=2.4,
                   z=7, ls=(0, (3, 3)))
    sc.T(LX[4] + 82, (pts[4][1] + pts[11][1]) / 2, "same layer: repel", size=SMALL, color=INK, va="center",
         t0=t_fake + 0.2, t1=9.0, bg=BG)
    merge_line = GrowLine(sc, *bezier(pts[17], (LX[3] + 70, 430), pts[18]), color=TRACK[0], lw=2.2, z=7,
                          ls=(0, (1.5, 3)))
    sc.T(LX[3] + 20, 455, "helix-consistent fragments", size=SMALL, color=INK2, ha="left", t0=tfm + 0.8,
         t1=tfm + 3.2, bg=BG)
    sc.T(LX[5] + 20, pts[13][1] + 26, "leftover hit", size=SMALL, color=INK2, t0=t2p, t1=t2p + 3.2, bg=BG)

    @sc.on
    def _(t):
        o = 1 - prog(t, t1o, 0.6)
        a = prog(t, 0.6, 0.8) * o
        for i, (u, v) in enumerate(edges):
            s = sc_[i]
            base_col = mix(LOW, INK2, s)
            lw = 1.2 + 2.0 * s
            if i in t_step:
                acc = prog(t, t_step[i], 0.3)
                if i == 18:  # the fake edge is rejected by the same-layer repulsion
                    g.edge(i, 1.0, mix(base_col, INK2, acc), lw, a * (1 - prog(t, t_fake + 1.4, 0.8)))
                    continue
                col = colors[own[u]]
                if own[u] == 2 and u >= 17:
                    col = mix(frag2, colors[2], prog(t, tfm + 1.8, 0.8))
                g.edge(i, 1.0, mix(base_col, col, acc), lw, a)
            elif i == 11:  # B5 -> B6, accepted in the second pass at a lower s0
                acc = prog(t, t2p + 1.2, 0.6)
                g.edge(i, 1.0, mix(base_col, colors[1], acc), lw, a)
            elif i == 15:  # the C3 -> C4 gap, bridged by fragment merging
                g.edge(i, 1.0, base_col, lw, a * (1 - prog(t, tfm + 0.6, 0.6)))
            else:
                g.edge(i, 1.0, base_col, lw, a * (1 - prog(t, 8.6, 0.8)))
        for j in range(len(pts)):
            first = [t_step[i] for i in t_step if j in edges[i] and i != 18]
            tj = min(first) if first else 99.0
            col = colors[own[j]]
            if own[j] == 2 and j >= 18:
                col = mix(frag2, colors[2], prog(t, tfm + 1.8, 0.8))
            if j == 13:
                tj = t2p + 1.2
            g.node(j, mix(HIT, col, prog(t, tj, 0.3)), a)
        rep.set(prog(t, t_fake, 0.5), (1 - prog(t, 9.0, 0.6)) * o)
        merge_line.set(prog(t, tfm + 0.8, 0.9), (1 - prog(t, tfm + 2.8, 0.8)) * o)

    sc.T(1180, 860, "Same-layer Mutex Watershed (MWS-layer)", size=LABEL, weight="semibold", t0=1.2, t1=t1o)
    bullets = [("Edges above s_{0} attract.", 1.6), ("Two hits on the same layer repel:", 1.9),
               ("   a particle crosses a layer once.", 1.9), ("Greedy merge in score order,", 2.6),
               ("   respecting the repulsions.", 2.6), ("Two-pass: a second pass at a lower s_{0}", t2p),
               ("   on the leftover hits.", t2p), ("Fragment merging: helix consistency", tfm),
               ("   or a learned GBDT on fragment pairs.", tfm)]
    for i, (s, t0) in enumerate(bullets):
        sc.T(1180, 800 - 50 * i, s, size=LABEL, color=INK if not s.startswith("   ") else INK2, t0=t0, t1=t1o)

    # part 2: numbers; the footnote (two lines) stays on screen for the whole part
    t2 = 18.0
    sc.T(120, 820, "acorn classifier ({tt}-trained, 3 seeds)", size=BODY, weight="semibold", t0=t2)
    sc.T(1800, 820, "primary efficiency (validation)", size=SMALL, color=MUTED, ha="right", t0=t2)
    sc.T(120, 520, "Canto-B classifier (full budget, seed 1)", size=BODY, weight="semibold", t0=t2 + 4.2)
    chains = [([("CC", "0.923"), ("MWS-layer + merging", "0.969")], 700, t2 + 0.5, BLUE_T),
              ([("CC", "0.9715"), ("MWS-layer", "0.9782"), ("+ merging", "0.9812"), ("two-pass + merging", "0.9857")],
               400, t2 + 4.7, ORANGE_T)]
    for steps, y, t0, hl in chains:
        x = 120
        prev = None
        for k, (nm, v) in enumerate(steps):
            wbox = 330
            last = k == len(steps) - 1
            bx = sc.box(x, y - 58, wbox, 116, fc=hl if last else CARD, r=18, z=2)
            tk = t0 + 0.8 * k
            sc.fade(bx, tk, 0.5)
            sc.T(x + wbox / 2, y + 26, nm, size=SMALL, color=MUTED, ha="center", va="center", t0=tk)
            sc.T(x + wbox / 2, y - 18, v, size=36, weight="semibold", ha="center", va="center", t0=tk)
            if prev is not None:
                a = Arrow(sc, (prev + 12, y), (x - 12, y), color=FAINT, lw=2.2, head=12)
                sc.on(lambda t, a=a, tk=tk: a.set(prog(t, tk - 0.3, 0.4)))
            prev = x + wbox
            x += wbox + 80
    sc.T(120, 262, "Walkthrough + merging: a walk along high-score edges, then helix fragment merging.", size=SMALL,
         color=MUTED, t0=t2 + 1.0)
    sc.T(120, 225, "Builders in the results: previews, walkthrough + merging; Canto-B full budget, MWS-layer + merging "
                   "(2 seeds).", size=SMALL, color=MUTED, t0=t2 + 1.2)
    sc.narrate(LINES["trackbuild"])
    return sc


# acorn (HNL): the HNL-trained acorn baseline (trained on the HNL signal as in Wang et al.); acorn ({tt}): the same
# network trained on {tt} only
RESULTS = [("DeZoort IN", "FR* unreachable", "1", 84.8, "84.8 %", "12.51 %", BROWN),
           ("DeZoort IN-L", "FR* unreachable", "1", 88.0, "88.0 %", "6.09 %", BROWN),
           ("acorn ({tt})", "seeds 0, 2 > FR*", "3", 92.3, "92.3 % ± 1.3", "2.97 %", BLUE),
           ("acorn (HNL)", "best cut; seed 1 > FR*", "3", 95.6, "95.6 % ± 1.3", "2.92 %", BLUE),
           ("acorn ({tt}) + Canto builder", "", "3", 96.9, "96.9 % ± 0.3", "1.75 %", BLUE),
           ("acorn (HNL) + Canto builder", "", "3", 98.6, "98.6 % ± 0.2", "1.47 %", BLUE),
           ("Canto-B (full budget)", "", "2", 98.3, "98.3 % ± 0.2", "1.50 %", ORANGE),
           ("Canto-B (preview)", "", "1", 98.8, "98.8 %", "1.77 %", ORANGE_M),
           ("Canto-H (preview)", "", "1", 99.5, "99.5 %", "1.45 %", ORANGE_M),
           ("Canto-L (preview)", "", "1", 99.5, "99.5 %", "1.13 %", ORANGE_M)]


def scene_results():
    sc = Scene("results", 53.2, title="Results: primary efficiency (validation)", footer=SIM)
    sc.T(120, 892, "2999 events; each model at its own cut and builder with fake rate ≤ FR* = 2.94 % where reachable",
         size=SMALL, color=MUTED, t0=0.6)
    sc.T(120, 856, "builders: Canto previews walkthrough + merging; Canto-B full budget and acorn + Canto builder "
                   "MWS-layer + merging", size=SMALL, color=MUTED, t0=0.7)
    sc.legend_row(120, 822, [("sq", BROWN, "DeZoort"), ("sq", BLUE, "acorn classifier"),
                             ("sq", ORANGE, "Canto classifier"), ("sq", ORANGE_M, "Canto preview (1 seed)")],
                  size=SMALL, t0=0.8)
    # row labels end at xl (the longest, the reference acorn's, starts at the 120 margin); seeds at xs; bars x0-x100
    xl, xs, x0, x100 = 790, 815, 850, 1480
    rows_y = [740 - 50 * i for i in range(10)]
    sc.T(xs, 774, "seeds", size=SMALL, color=MUTED, ha="center", t0=1.0)
    sc.T(1840, 774, "fake rate", size=SMALL, color=MUTED, ha="right", t0=1.0)
    sc.fade(sc.seg(x0, rows_y[-1] - 26, x0, rows_y[0] + 26, INK2, lw=1.4, z=3), 1.0)
    sc.fade(sc.seg(x100, rows_y[-1] - 26, x100, rows_y[0] + 26, HAIR, lw=1.4, z=1), 1.0)
    sc.T(x100, 774, "100 %", size=SMALL, color=MUTED, ha="center", t0=1.0)
    # one group of bars per caption: DeZoort, acorn tt, acorn HNL, acorn + Canto builder, Canto
    t_bar = [3.4, 4.4, 7.6, 12.4, 17.8, 18.8, 23.6, 24.6, 25.6, 26.6]
    bars, row_objs = [], []
    for i, (name, tag, seeds, v, vtxt, fr, col) in enumerate(RESULTS):
        y = rows_y[i]
        t0 = t_bar[i]
        lab = name + (f"  [[m:({tag})]]" if tag else "")
        r1 = sc.T(xl, y, lab, size=SMALL if tag else LABEL, ha="right", va="center")
        r2 = sc.T(xs, y, seeds, size=SMALL, color=MUTED, ha="center", va="center")
        b = HBar(sc, x0, y, 30, col)
        r3 = sc.T(x100 + 16, y, vtxt, size=LABEL, weight="semibold" if i >= 6 else "normal", va="center")
        r4 = sc.T(1840, y, fr, size=LABEL, color=INK2, ha="right", va="center")
        bars.append((b, (x100 - x0) * v / 100.0, t0))
        row_objs.append(([r1, r2], [r3, r4], b, t0))
    # focus 1: reference acorn vs Canto-L preview; focus 2: only the rows whose numbers are in the decomposition boxes
    focus = {1: [3, 9], 2: [2, 4, 6]}
    t_f1, t_sw, t_f2 = 30.0, 36.6, 46.0

    @sc.on
    def _(t):
        f1 = prog(t, t_f1, 0.8) * (1 - prog(t, t_sw, 0.8))
        f2 = prog(t, t_sw, 0.8) * (1 - prog(t, t_f2, 0.8))
        for i, (names, vals, b, t0) in enumerate(row_objs):
            # f2 rises while f1 falls (f1 + f2 = 1 during the switch): the sum keeps a row that is dimmed in both
            # focus states dimmed, where the max of the two would brighten it half-way at the switch
            dim = 1 - 0.65 * min(1.0, f1 * (i not in focus[1]) + f2 * (i not in focus[2]))
            show(names, prog(t, t0 - 0.4, 0.5) * dim)
            show(vals, prog(t, t0 + 0.8, 0.5) * dim)
            b.set(bars[i][1] * prog(t, t0, 1.0), dim)

    # component decomposition with the same builder (MWS-layer + merging), three-line boxes sized to their text
    chain = [("acorn CC", "92.3 %", "3 seeds"), ("+ Canto builder", "96.9 %", "MWS-layer + merging"),
             ("+ Canto classifier", "98.7 %", "preview, same builder; full budget 98.3 %"),
             ("+ helix union graph", "99.4 %", "same builder; Canto-H preview")]
    ws = [max(sc.text_width(a, SMALL), sc.text_width(b, LABEL, "semibold"), sc.text_width(c, SMALL)) + 44
          for a, b, c in chain]
    gapc = 40
    x = W / 2 - (sum(ws) + gapc * (len(ws) - 1)) / 2
    prev = None
    for k, ((nm, v, q), w) in enumerate(zip(chain, ws)):
        tk = t_sw + 0.4 + 0.9 * k
        bx = sc.box(x, 142, w, 108, fc=ORANGE_T if k == 3 else CARD, r=16, z=2)
        sc.fade(bx, tk, 0.5, t1=t_f2, d1=0.6)
        sc.T(x + w / 2, 230, nm, size=SMALL, color=INK2, ha="center", va="center", t0=tk, t1=t_f2, d1=0.6)
        sc.T(x + w / 2, 199, v, size=LABEL, weight="semibold", ha="center", va="center", t0=tk, t1=t_f2, d1=0.6)
        sc.T(x + w / 2, 168, q, size=SMALL, color=MUTED, ha="center", va="center", t0=tk, t1=t_f2, d1=0.6)
        if prev is not None:
            a = Arrow(sc, (prev + 8, 196), (x - 8, 196), color=FAINT, lw=2.2, head=12)
            sc.on(lambda t, a=a, tk=tk: a.set(prog(t, tk - 0.3, 0.4), 1 - prog(t, t_f2, 0.6)))
        prev = x + w
        x += w + gapc
    sc.narrate(LINES["results"])
    return sc


TABLE = [  # docs/canto_facts.md section 7, verbatim; (model, builder, seeds, primary, |d0| 100-300, |d0| >= 300,
           # W muons, tt prompt, fake rate, kind: 2 Canto, 1 acorn, 0 DeZoort)
    ("Canto-L (preview)", "walk + merge", "1", "99.5 %", "98.8 %", "17/19", "99.8 %", "96.0 %", "1.13 %", 2),
    ("Canto-H (preview)", "walk + merge", "1", "99.5 %", "98.8 %", "17/19", "99.8 %", "95.9 %", "1.45 %", 2),
    ("Canto-B (preview)", "walk + merge", "1", "98.8 %", "96.4 %", "15/19", "99.8 %", "96.7 %", "1.77 %", 2),
    ("Canto-B (full budget)", "MWS + merge", "2", "98.3 % ± 0.2", "95.2 %", "28/38", "99.8 %", "96.3 %", "1.50 %", 2),
    ("acorn (HNL) + Canto builder", "MWS + merge", "3", "98.6 % ± 0.2", "96.8 %", "52/57", "99.8 %", "94.6 %",
     "1.47 %", 1),
    ("acorn ({tt}) + Canto builder", "MWS + merge", "3", "96.9 % ± 0.3", "93.2 %", "42/57", "99.8 %", "95.5 %",
     "1.75 %", 1),
    ("acorn (HNL) *", "CC, best cut", "3", "95.6 % ± 1.3", "94.4 %", "46/57", "98.8 %", "84.6 %",
     "2.92 %", 1),
    ("acorn ({tt}) *", "CC", "3", "92.3 % ± 1.3", "89.2 %", "37/57", "98.7 %", "87.2 %", "2.97 %", 1),
    ("DeZoort IN-L *", "CC", "1", "88.0 %", "82.6 %", "10/19", "96.6 %", "78.2 %", "6.09 %", 0),
    ("DeZoort IN *", "CC", "1", "84.8 %", "80.2 %", "9/19", "91.2 %", "62.6 %", "12.51 %", 0)]


def scene_table():
    sc = Scene("table", 14.6, title="All validation numbers", footer=SIM)
    cols = [(625, "builder"), (745, "seeds"), (870, "primary"), (1060, "|d_{0}| 100–300"),
            (1255, "|d_{0}| ≥ 300"), (1430, "W muons"), (1600, "{tt} prompt"), (1770, "fake rate")]
    sc.T(120, 868, "2999 events; fake rate ≤ FR* = 2.94 % where reachable; mean ± seed spread", size=SMALL,
         color=MUTED, t0=0.3)
    sc.T(120, 812, "model", size=SMALL, color=INK2, weight="semibold", va="center", t0=0.4)
    for x, h in cols:
        sc.T(x, 812, h, size=SMALL, color=INK2, weight="semibold", ha="center", va="center", t0=0.4)
    sc.fade(sc.seg(120, 786, 1840, 786, INK2, lw=1.4, z=2), 0.4)
    ys = [750 - 46 * i for i in range(len(TABLE))]
    sc.fade(sc.box(108, ys[3] - 21, 1744, ys[0] - ys[3] + 42, fc=ORANGE_T, r=10, z=1), 0.9, 0.6)
    for i, (name, *vals, kind) in enumerate(TABLE):
        t0 = 0.8 + 0.22 * i
        col = INK if kind == 2 else INK2
        sc.T(120, ys[i], name, size=SMALL, color=col, va="center", t0=t0, weight="semibold" if kind == 2 else "normal")
        for (x, _), v in zip(cols, vals):
            sc.T(x, ys[i], v, size=SMALL, color=col, ha="center", va="center", t0=t0 + 0.1)
    sc.T(120, 270, "* fake-rate target not reached by acorn ({tt}) seeds 0 and 2 and acorn (HNL) seed 1, shown at the "
                   "lowest fake rate.", size=SMALL, color=MUTED, t0=3.4)
    sc.T(120, 234, "|d_{0}| in mm; ≥ 300 mm: 19 particles per seed, 17/19 has a 68 % interval of 77–96 %.",
         size=SMALL, color=MUTED, t0=3.6)
    sc.T(120, 198, "walk: walkthrough; MWS: same-layer Mutex Watershed.", size=SMALL, color=MUTED, t0=3.8)
    sc.narrate(LINES["table"])
    return sc


def scene_outlook():
    sc = Scene("outlook", 14.4, title="Next steps", footer=SIM)
    steps = [("Final selection", "among Canto-B, Canto-H and Canto-L by 3-seed means (rule R8), then frozen"),
             ("One-time test evaluation", "the test set is evaluated once"),
             ("Preregistered primary claim", "test inefficiency at most half of the reference acorn's "
                                             "(0.926 → ≥ 0.963), ≥ 3σ;\nprompt efficiency not "
                                             "lower by more than 0.2 points")]
    ys = [760, 590, 420]
    t_in = [0.9, 2.5, 6.6]  # step 3, the preregistered claim, comes with the narration line that reads it
    line = sc.seg(200, ys[0], 200, ys[0], HAIR, lw=3, z=1)
    for k, ((head, body), y) in enumerate(zip(steps, ys)):
        t0 = t_in[k]
        sc.fade([sc.circle(200, y, 22, fc=ORANGE if k == 2 else BG, ec=ORANGE, lw=3, z=4)], t0, 0.5)
        sc.T(200, y, str(k + 1), size=SMALL, color="#FFFFFF" if k == 2 else INK, weight="bold", ha="center",
             va="center", z=31, t0=t0)
        sc.T(250, y + 4, head, size=28, weight="semibold", va="center", t0=t0, rise=8)
        sc.T(250, y - 44, body, size=LABEL, color=INK2, va="top", lsp=1.35, t0=t0 + 0.2)

    @sc.on
    def _(t):  # the timeline reaches each step as it appears
        y = lerp(ys[0], ys[1], prog(t, 1.2, 1.3)) + (ys[2] - ys[1]) * prog(t, t_in[2] - 0.3, 0.6)
        line.set_data([200, 200], [ys[0], y])
        line.set_visible(y < ys[0] - 0.5)

    sc.narrate(LINES["outlook"])
    return sc


def scene_end():
    sc = Scene("end", 6.2)
    sc.T(W / 2, 540, "Canto", size=110, weight="semibold", ha="center", t0=0.3, d=1.0, rise=12)
    line = sc.seg(W / 2, 505, W / 2, 505, ORANGE, lw=4)

    @sc.on
    def _(t):
        half = 260 * prog(t, 0.8, 1.0)
        line.set_data([W / 2 - half, W / 2 + half], [505, 505])
        line.set_visible(half > 0.5)

    sc.T(W / 2, 440, "ACTS generic detector simulation, no pileup; validation results", size=26, color=MUTED,
         ha="center", va="center", t0=1.4, d=0.8)
    sc.narrate(LINES["end"])
    return sc


DURATIONS = [6.2, 27.0, 22.0, 18.0, 25.2, 27.0, 18.2, 33.4, 40.7, 33.4, 53.2, 14.6, 14.4, 6.2]  # scene time; must match
# (a scene ends at least 0.6 s after its last narration window closes, so its fade-out never dims a held subtitle)

SCENES = [("title", scene_title), ("problem", scene_problem), ("pipeline", scene_pipeline),
          ("dezoort", scene_dezoort), ("acorn", scene_acorn), ("failure", scene_failure), ("canto", scene_canto),
          ("classifier", scene_classifier), ("branches", scene_branches), ("trackbuild", scene_trackbuild),
          ("results", scene_results), ("table", scene_table), ("outlook", scene_outlook), ("end", scene_end)]


# ================================================================================================================
# narration: line ids, TTS input and output, the time warp, the soundtrack and the subtitles file


def scene_number(name):
    """1-based position of a scene in the video."""
    for k, (nm, _) in enumerate(SCENES):
        if nm == name:
            return k + 1
    raise KeyError(f"unknown scene {name!r}")


def line_id(k, name, i):
    """TTS id of line i (0-based) of scene number k (1-based), e.g. '05_acorn_03'."""
    return f"{k:02d}_{name}_{i + 1:02d}"


def scene_lines(k):
    """The narration lines of scene index k (0-based) as recorded by Scene.narrate (without the subtitle object)."""
    name = SCENES[k][0]
    return [{"id": line_id(k + 1, name, i), "t0": float(ln["t0"]), "t1": float(ln["t1"]), "display": ln["display"],
             "speech": ln["speech"]} for i, ln in enumerate(LINES.get(name, []))]


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tts_hash(text, voice, speed):
    """canto_tts.py's manifest key of a line: sha1 of [text, voice, speed]."""
    return hashlib.sha1(json.dumps([text, voice, float(speed)], ensure_ascii=False).encode("utf-8")).hexdigest()


def voiced_span(x):
    """(i0, i1, v0, v1) of a line's samples x: its voice [v0, v1), the 10 ms frames from the first to the last whose
    RMS exceeds TRIM_DB dBFS, and the part placed in the soundtrack [i0, i1), the voice with TRIM_KEEP s of the
    silence around it (Kokoro leaves ~0.3 s before and ~0.5 s after the voice)."""
    f = SR // 100
    n = len(x) // f
    rms = np.sqrt(np.mean(np.square(x[: n * f].reshape(n, f), dtype=np.float64), axis=1))
    on = np.flatnonzero(rms > 10.0 ** (TRIM_DB / 20.0))
    if not len(on):
        return 0, len(x), 0, len(x)
    v0, v1 = int(on[0]) * f, int(on[-1] + 1) * f
    keep = int(round(TRIM_KEEP * SR))
    return max(0, v0 - keep), min(len(x), v1 + keep), v0, v1


def load_speech():
    """({line id: (i0, i1, v0, v1)}, [problems]): the voiced_span of every narration line's WAV. A line without audio,
    whose audio was made from another speech text, voice or speed (canto_tts.py's manifest), or whose WAV length
    disagrees with durations.json, is listed instead."""
    durations, manifest = read_json(NARR_DIR / "durations.json"), read_json(NARR_DIR / "manifest.json")
    spans, bad = {}, []
    for k in range(len(SCENES)):
        for ln in scene_lines(k):
            lid, wav = ln["id"], NARR_DIR / f"{ln['id']}.wav"
            if not wav.is_file() or lid not in durations:
                bad.append(f"{lid}: no audio in {NARR_DIR}")
            elif manifest.get(lid) != tts_hash(ln["speech"].strip(), NARRATION["voice"], NARRATION["speed"]):
                bad.append(f"{lid}: audio out of date (speech text, voice or speed changed)")
            else:
                x = read_wav(wav)
                if abs(len(x) / SR - float(durations[lid])) > 0.002:
                    bad.append(f"{lid}.wav lasts {len(x) / SR:.3f} s but durations.json says {durations[lid]} s")
                else:
                    spans[lid] = voiced_span(x)
    return spans, bad


def placed_lengths(spans):
    """{line id: seconds of its audio placed in the soundtrack}: the lengths the time warp works with."""
    return {lid: (i1 - i0) / SR for lid, (i0, i1, _, _) in spans.items()}


def scene_motion(sc):
    """The transitions of a built scene: the sorted (t0, d) of every prog() that its updaters (captions, subtitles and
    the cover included) evaluate at some frame. Every on-screen change goes through prog, so the picture changes only
    inside these intervals [t0, t0 + d] (d <= 0: a switch at t0); ``check`` confirms it on the pixels of every hold
    and cut."""
    global _MOTION
    _MOTION = set()
    try:
        for j in range(int(round(sc.duration * FPS)) + 1):
            sc.update(j / FPS)
        return sorted(_MOTION)
    finally:
        _MOTION = None


def _motion_job(k):
    sc = SCENES[k][1]()
    try:
        return scene_motion(sc)
    finally:
        plt.close(sc.fig)


def motions(jobs=1):
    """scene_motion of every scene, computed in `jobs` processes."""
    if jobs > 1:
        with get_context("fork").Pool(min(jobs, len(SCENES))) as pool:
            return pool.map(_motion_job, range(len(SCENES)))
    return [_motion_job(k) for k in range(len(SCENES))]


def is_static(motion, a, b):
    """Nothing changes on screen over the scene times [a, b] (a == b: nothing is in mid-transition at a)."""
    for t0, d in motion:
        if (t0 < b and t0 + d > a) if d > 0 else (a < t0 <= b):
            return False
    return True


class Warp:
    """Time warp of one scene: the scene time that every video frame shows (frames), from the scene's narration lines,
    the lengths of their placed (trimmed) audio, dur, and its motion (scene_motion).

    A line is spoken from SPEECH_LEAD after its window [t0, t1] opens and needs the window open until SPEECH_TAIL after
    the speech ends: need = SPEECH_LEAD + length + SPEECH_TAIL.
    - need > t1 - t0: the scene is held for the difference, rounded up to whole frames, on a static frame: at
      t1 - HOLD_BACK (subtitle still fully on) if nothing is in mid-transition there, else at the latest scene time of
      the window (subtitle faded in) where nothing is (none: at t1 - HOLD_BACK, and ``check`` reports it).
    - need < t1 - t0: the slack after the speech, [t0 + need, t1], is silent. Its static frames are cut (frame j is
      static when nothing changes over [j - 1, j + 1] / FPS, so cutting it is invisible), the latest first and at most
      the slack, so the picture never stands still without narration; frames that move stay and fill the pause.
    holds: [(scene time, frames, line id)]; cuts: [(scene frame index, line id)]."""

    def __init__(self, duration, lines, dur, motion):
        self.duration = float(duration)
        n_scene = int(round(self.duration * FPS))
        self.holds, cut = [], {}
        for ln in lines:
            if ln["id"] not in dur:
                continue
            t0, t1 = ln["t0"], ln["t1"]
            need = SPEECH_LEAD + dur[ln["id"]] + SPEECH_TAIL
            if need > t1 - t0 + 1e-9:
                h = t1 - HOLD_BACK
                if not is_static(motion, h, h):
                    ends = [x for a, d in motion if d > 0 for x in (a, a + d)]
                    ok = [x for x in ends if t0 + SUB_IN <= x <= h and is_static(motion, x, x)]
                    h = max(ok) if ok else h
                self.holds.append((h, math.ceil((need - (t1 - t0)) * FPS - 1e-6), ln["id"]))
            else:
                slack = [j for j in range(math.ceil((t0 + need) * FPS - 1e-6), min(n_scene, math.ceil(t1 * FPS)))
                         if j / FPS < t1 and is_static(motion, (j - 1) / FPS, (j + 1) / FPS)]
                n_cut = min(len(slack), int(math.floor((t1 - t0 - need) * FPS + 1e-6)))
                for j in slack[len(slack) - n_cut:]:
                    cut[j] = ln["id"]
        self.holds.sort()
        self.cuts = sorted(cut.items())
        frames, h = [], 0
        for j in range(n_scene):
            s = j / FPS
            while h < len(self.holds) and self.holds[h][0] < s:
                frames += [self.holds[h][0]] * self.holds[h][1]
                h += 1
            if j not in cut:
                frames.append(s)
        for s, n, _ in self.holds[h:]:
            frames += [s] * n
        self.frames = frames

    @property
    def length(self):
        return len(self.frames) / FPS

    def video_time(self, s):
        """Time from the start of the scene's video at which scene time s is shown (s outside the holds and cuts)."""
        return (s - sum(1 for j, _ in self.cuts if j / FPS < s) / FPS
                + sum(n for h, n, _ in self.holds if h < s) / FPS)

    def hold_starts(self):
        """Video frame index of the first frame of every hold."""
        return [self.frames.index(h) for h, _, _ in self.holds]

    def cut_runs(self):
        """[(first, last) scene frame index] of every run of consecutive cut frames."""
        runs = []
        for j, _ in self.cuts:
            if runs and runs[-1][1] == j - 1:
                runs[-1][1] = j
            else:
                runs.append([j, j])
        return [tuple(r) for r in runs]


def warps(dur, motion):
    return [Warp(DURATIONS[k], scene_lines(k), dur, motion[k]) for k in range(len(SCENES))]


def read_wav(path):
    """A 16-bit mono WAV at SR as float32 in [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            raise SystemExit(f"{path}: expected 16-bit mono {SR} Hz audio")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, x):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.round(np.clip(x, -1.0, 32767 / 32768) * 32768.0).astype("<i2").tobytes())
    os.replace(tmp, path)


def scene_audio(k, warp, spans, n_frames):
    """Soundtrack of the first n_frames video frames of scene k: every line, trimmed to its placed part, from its
    video start time."""
    buf = np.zeros(n_frames * (SR // FPS), np.float32)
    for ln in scene_lines(k):
        i0, i1, _, _ = spans[ln["id"]]
        a = read_wav(NARR_DIR / f"{ln['id']}.wav")[i0:i1]
        s0 = int(round(warp.video_time(ln["t0"] + SPEECH_LEAD) * SR))
        seg = a[: max(0, len(buf) - s0)]
        buf[s0:s0 + len(seg)] += seg
    return buf


def line_times(wp, ln, spans):
    """Video times, from the start of the scene, of a line: (window open, window close, speech start, speech end,
    voice start, voice end); the speech is the placed (trimmed) audio, the voice its voiced part."""
    i0, i1, v0, v1 = spans[ln["id"]]
    s0 = wp.video_time(ln["t0"] + SPEECH_LEAD)
    return (wp.video_time(ln["t0"]), wp.video_time(ln["t1"]), s0, s0 + (i1 - i0) / SR, s0 + (v0 - i0) / SR,
            s0 + (v1 - i0) / SR)


SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def srt_text(s):
    """Subtitle markup as plain text for the .srt (numeric subscripts as Unicode subscripts)."""
    return plain(re.sub(r"_\{(\d+)\}", lambda m: m.group(1).translate(SUBSCRIPT), s))


def srt_time(x):
    h, ms = divmod(int(round(x * 1000)), 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vis_len(s):
    """Characters as displayed (a combining mark, such as the bar of t-bar, takes no place)."""
    return sum(1 for c in s if not unicodedata.combining(c))


SHORT_WORDS = {"a", "an", "the", "of", "to", "and", "or", "on", "in", "at", "by", "with", "for", "from", "its", "their"}
CLAUSE = r"[,:;?.!] "  # a clause mark followed by a space (not the point of a decimal number)


def srt_wrap(s, width=SRT_WIDTH):
    """s as at most two lines of at most `width` characters, broken at a space: balanced, preferably after a clause
    mark, not after a short function word, never before '%' or after an arrow or a relation sign. None if impossible."""
    if vis_len(s) <= width:
        return [s]
    words, best = s.split(" "), None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if vis_len(a) > width or vis_len(b) > width or words[i] == "%" or words[i - 1] in ("→", "≥", "≤", "=", "~"):
            continue
        cost = (abs(vis_len(a) - vis_len(b)) - (20 if a[-1] in ",:;?.!" else 0)
                + (20 if words[i - 1].lower() in SHORT_WORDS else 0))
        if best is None or cost < best[0]:
            best = (cost, [a, b])
    return best[1] if best else None


def srt_cues(text, a, b, voice, phonemes=""):
    """The .srt cues [(start, end, [lines])] of a subtitle shown from a to b: one cue wrapped by srt_wrap or, when the
    text does not fit two lines, two consecutive cues split after a clause mark, both wrapped, switching at the point
    of the voice (voice_start, voice_end) that the phonemes before the same clause mark take (the share of the
    characters if the phonemes have another number of clause marks)."""
    lines = srt_wrap(text)
    if lines:
        return [(a, b, lines)]
    marks = [m.end() for m in re.finditer(CLAUSE, text)]
    best = None
    for k, pos in enumerate(marks):
        p, q = text[:pos].rstrip(), text[pos:]
        wp, wq = srt_wrap(p), srt_wrap(q)
        if wp and wq and (best is None or abs(vis_len(p) - vis_len(q)) < best[0]):
            best = (abs(vis_len(p) - vis_len(q)), k, p, wp, wq)
    if best is None:
        raise ValueError(f"subtitle {text!r} does not split into cues of two lines of {SRT_WIDTH} characters")
    _, k, p, wp, wq = best
    pm = [m.end() for m in re.finditer(CLAUSE, phonemes)]
    frac = pm[k] / len(phonemes) if len(pm) == len(marks) else vis_len(p) / vis_len(text)
    t = voice[0] + frac * (voice[1] - voice[0])
    return [(a, t, wp), (t, b, wq)]


def mmss(x):
    x = round(x, 1)
    m = int(x // 60)
    return f"{m}:{x - 60 * m:04.1f}"


# ================================================================================================================
# rendering, frames and checks


def clip_record(k, warp, dpi, t_max):
    """What the clip of scene k is rendered from; a cached clip whose .json differs is stale. The record holds the
    scene time of every frame (as a hash; holds and cuts in full) and the subtitles. Only the scene function's own
    source is hashed: a change in shared code (Scene, Rich, ...) needs an explicit re-render of the scenes."""
    name, fn = SCENES[k]
    n = len(warp.frames) if t_max is None else min(len(warp.frames), int(round(t_max * FPS)))
    return {"scene": name, "fps": FPS, "dpi": dpi, "frames": n, "max_seconds": t_max, "duration": warp.duration,
            "frame_times": hashlib.sha1(json.dumps([round(x, 6) for x in warp.frames[:n]]).encode()).hexdigest(),
            "holds": [[round(h, 6), m] for h, m, _ in warp.holds], "cuts": [list(r) for r in warp.cut_runs()],
            "subtitles": [[ln["t0"], ln["t1"], ln["display"]] for ln in scene_lines(k)],
            "sub_fade": [SUB_IN, SUB_OUT],
            "source": hashlib.sha1(inspect.getsource(fn).encode("utf-8")).hexdigest()}


def render_clip(job):
    k, path, dpi, frames, record = job
    t_start = time.time()
    meta = Path(path).with_suffix(".json")
    meta.unlink(missing_ok=True)  # an interrupted render leaves no record, so its clip counts as stale
    check_fonts()
    sc = SCENES[k][1]()
    if abs(sc.duration - DURATIONS[k]) > 1e-9:
        raise RuntimeError(f"scene {k + 1} {sc.name}: {sc.duration} s but DURATIONS says {DURATIONS[k]} s")
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-threads", "2"])
    with writer.saving(sc.fig, str(path), dpi):
        for s in frames:
            sc.update(s)
            writer.grab_frame(facecolor=BG)
    plt.close(sc.fig)
    write_json(meta, record)
    return k, str(path), len(frames), time.time() - t_start


def concat(paths, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()


def ffmpeg_json(stderr):
    """The last JSON object that an ffmpeg filter printed (loudnorm's print_format=json report)."""
    i = stderr.rindex("{")
    return json.loads(stderr[i:stderr.index("}", i) + 1])


def mux(video, wav, out, seconds):
    """The video stream copied and the soundtrack normalised to LOUDNORM by two-pass ffmpeg loudnorm: pass 1 measures,
    pass 2 applies with the measured values (a plain gain when that keeps the true peak under the target; otherwise
    loudnorm rides the gain within the loudness range target and its true-peak limiter catches the peaks), then
    48 kHz, cut (or padded) to the video's `seconds` and AAC 192 kb/s (AAC allows at most 144 kb/s for 24 kHz mono).
    Returns both passes' loudnorm reports."""
    target = f"I={LOUDNORM['I']}:TP={LOUDNORM['TP']}:LRA={LOUDNORM['LRA']}"
    p1 = ffmpeg_json(subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(wav), "-af",
                                     f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                                    capture_output=True, text=True, check=True).stderr)
    n = int(round(seconds * 48000))
    # loudnorm's frames can carry a timestamp gap (one 33 ms jump in a 5-minute track, samples all present): asetpts
    # renumbers them from the sample count, so the AAC stream lasts exactly as long as the video
    af = (f"loudnorm={target}:measured_I={p1['input_i']}:measured_TP={p1['input_tp']}:"
          f"measured_LRA={p1['input_lra']}:measured_thresh={p1['input_thresh']}:offset={p1['target_offset']}:"
          f"linear=true:print_format=json,aresample=48000,asetpts=N/SR/TB,apad=whole_len={n},atrim=end_sample={n}")
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-nostats", "-i", str(video), "-i", str(wav),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", af, "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-ac", "1", "-movflags", "+faststart", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"mux failed:\n{r.stderr[-3000:]}")
    os.replace(tmp, out)
    return {"pass1": p1, "pass2": ffmpeg_json(r.stderr)}


def loudness(path):
    """EBU R128 measurement of the first audio stream of path by ffmpeg's ebur128 filter (true peak by 4x
    oversampling): {"I": LUFS, "LRA": LU, "TP": dBTP}."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    summ = r.stderr[r.stderr.rindex("Summary:"):]
    num = lambda pat: float(re.search(pat + r"\s+(-?[0-9.]+|-inf)", summ).group(1))  # noqa: E731
    return {"I": num(r"\bI:"), "LRA": num(r"\bLRA:"), "TP": num(r"True peak:\s+Peak:")}


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=index,codec_type,codec_name,width,height,sample_rate,channels,bit_rate,duration,nb_frames",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"]


def assemble(join, clips, wps, spans, records, out):
    """Join the clips of the scenes in `join`; soundtrack (out .wav), mux (out), subtitles (out .srt) and a timeline of
    every scene and line in video time with the loudness and the pauses (out _timeline.json)."""
    video = out.with_name(f".{out.stem}.video.mp4")
    concat(clips, video)
    phonemes = read_json(NARR_DIR / "phonemes.json")
    parts, cues, scenes, lines, t = [], [], [], [], 0.0
    pauses = {"within": [], "across": []}  # silence before a line's voice: within its scene / at a scene change
    for k in join:
        n, wp = records[k]["frames"], wps[k]
        parts.append(scene_audio(k, wp, spans, n))
        scenes.append({"scene": SCENES[k][0], "number": k + 1, "start": round(t, 6), "frames": n,
                       "length": round(n / FPS, 6), "scene_time": wp.duration,
                       "holds": round(sum(m for _, m, _ in wp.holds) / FPS, 6), "cuts": round(len(wp.cuts) / FPS, 6)})
        for i, ln in enumerate(scene_lines(k)):
            w0, w1, s0, s1, v0, v1 = line_times(wp, ln, spans)
            w1 = min(w1, n / FPS)
            if w0 >= n / FPS:
                continue
            cues += srt_cues(srt_text(ln["display"]), t + w0, t + w1, (t + v0, t + v1), phonemes.get(ln["id"], ""))
            i0, i1, _, _ = spans[ln["id"]]
            pause = round(t + v0 - lines[-1]["voice_end"], 3) if lines else None
            if pause is not None:
                pauses["across" if i == 0 else "within"].append(pause)
            lines.append({"id": ln["id"], "scene": SCENES[k][0], "speech_start": round(t + s0, 6),
                          "speech_end": round(t + s1, 6), "voice_start": round(t + v0, 6),
                          "voice_end": round(t + v1, 6), "wav_trim": [round(i0 / SR, 4), round(i1 / SR, 4)],
                          "pause_before": pause, "window_open": round(t + w0, 6), "window_close": round(t + w1, 6),
                          "display": srt_text(ln["display"]), "speech": ln["speech"]})
        t += n / FPS
    wav = out.with_suffix(".wav")
    write_wav(wav, np.concatenate(parts))
    norm = mux(video, wav, out, t)
    video.unlink()
    out.with_suffix(".srt").write_text(
        "".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n" + "\n".join(txt) + "\n\n"
                for i, (a, b, txt) in enumerate(cues, 1)), encoding="utf-8")
    loud = loudness(out)
    write_json(out.with_name(f"{out.stem}_timeline.json"),
               {"fps": FPS, "sample_rate": SR, "length": round(t, 6), "trim": {"keep": TRIM_KEEP, "db": TRIM_DB},
                "speech_lead": SPEECH_LEAD, "speech_tail": SPEECH_TAIL,
                "loudness": {"target": LOUDNORM, "loudnorm": norm, "ebur128_mp4": loud}, "scenes": scenes,
                "lines": lines})
    for sc in scenes:
        print(f"scene {sc['number']:2d} {sc['scene']:<11s} starts at {mmss(sc['start'])}  ({sc['length']:.2f} s = "
              f"{sc['scene_time']:.1f} s + {sc['holds']:.2f} s of holds - {sc['cuts']:.2f} s cut)")
    for st in probe(out):
        print("stream", {key: st.get(key) for key in ("index", "codec_type", "codec_name", "width", "height",
                                                      "sample_rate", "channels", "bit_rate", "duration", "nb_frames")})
    for kind, ps in pauses.items():
        if ps:
            print(f"pauses before a line's voice, {kind} scenes: median {np.median(ps):.2f} s, max {max(ps):.2f} s")
    print(f"loudness: loudnorm {norm['pass2'].get('normalization_type')}, input {norm['pass1']['input_i']} LUFS / "
          f"{norm['pass1']['input_tp']} dBTP; MP4 (ebur128): I {loud['I']} LUFS, LRA {loud['LRA']} LU, "
          f"true peak {loud['TP']} dBTP")
    print(f"wrote {out} ({mmss(t)}, {t:.3f} s), {wav.name}, {out.with_suffix('.srt').name} ({len(cues)} cues), "
          f"{out.stem}_timeline.json", flush=True)


def scene_list(spec):
    if not spec:
        return list(range(len(SCENES)))
    return [int(s) - 1 for s in spec.split(",")]


def cmd_render(args):
    """Render the selected scenes in parallel (--scenes stale: those whose cached clip is missing or stale); a full
    render then joins all the clips with the narration, a preview only its own."""
    spans, bad = load_speech()
    if bad:
        raise SystemExit("narration audio missing or out of date; run `lines --submit` first:\n  " + "\n  ".join(bad))
    wps = warps(placed_lengths(spans), motions(args.jobs))
    dpi = 50 if args.preview else 100
    clip_dir = OUT / "preview" / "clips" if args.preview else OUT / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = lambda k: clip_dir / f"scene_{k + 1:02d}_{SCENES[k][0]}.mp4"  # noqa: E731
    records = [clip_record(k, wps[k], dpi, args.max_seconds) for k in range(len(SCENES))]

    def fresh(k):
        return clip(k).is_file() and read_json(clip(k).with_suffix(".json")) == records[k]

    idxs = [k for k in range(len(SCENES)) if not fresh(k)] if args.scenes == "stale" else scene_list(args.scenes)
    jobs = sorted([(k, clip(k), dpi, wps[k].frames[:records[k]["frames"]], records[k]) for k in idxs],
                  key=lambda j: -len(j[3]))
    t0 = time.time()
    if jobs:
        with get_context("fork").Pool(min(args.jobs, len(jobs))) as pool:
            for k, path, n, dt in pool.imap_unordered(render_clip, jobs):
                print(f"scene {k + 1:2d} {SCENES[k][0]:<11s} {n:5d} frames  {dt:6.1f} s  -> {path}", flush=True)
    else:
        print("every clip is up to date", flush=True)
    if args.no_concat:
        return
    join = idxs if args.preview else list(range(len(SCENES)))
    if not join:
        return
    stale = [f"{k + 1} {SCENES[k][0]}" for k in join if not fresh(k)]
    if stale:
        raise SystemExit(f"missing or stale clips, not joined (render them, e.g. --scenes stale): {', '.join(stale)}")
    out = Path(args.out) if args.out else (OUT / "preview" / "canto_preview.mp4" if args.preview
                                           else OUT / "canto_explainer.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    assemble(join, [clip(k) for k in join], wps, spans, records, out)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def cmd_lines(args):
    """Write the canto_tts.py input (text = speech) and print, or with --submit run, the sbatch command."""
    out = Path(args.out) if args.out else NARR_DIR / "lines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    items = [{"id": ln["id"], "text": ln["speech"]} for k in range(len(SCENES)) for ln in scene_lines(k)]
    write_json(out, items)
    print(f"{len(items)} lines -> {out}")
    wrap = (f". {REPO}/env/env.sh && {TTS_PY} {TTS_SCRIPT} --in {out} --out-dir {NARR_DIR} "
            f"--voice {NARRATION['voice']} --speed {NARRATION['speed']} --threads 8")
    logs = os.environ.get("HNL_LOGS", "/work/submit/jinboz1/hnl/logs")
    cmd = ["sbatch", "--parsable", "-p", "submit", "-c", "8", "--mem=16G", "-t", "01:00:00",
           "--exclude=submit00,submit01", "-J", "canto_tts", "-o", f"{logs}/%x_%j.out", "--wrap", wrap]
    if args.submit:
        job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        print(f"submitted canto_tts job {job} (log {logs}/canto_tts_{job}.out)")
    else:
        print(" ".join(cmd[:-1]) + f" \"{wrap}\"")


def cmd_frames(args):
    check_fonts()
    outdir = Path(args.outdir) if args.outdir else OUT / "check"
    outdir.mkdir(parents=True, exist_ok=True)
    idx = args.scene - 1
    sc = SCENES[idx][1]()
    for t in (float(x) for x in args.times.split(",")):
        sc.update(t)
        p = outdir / f"s{idx + 1:02d}_{SCENES[idx][0]}_t{t:05.2f}.png"
        sc.fig.savefig(p, dpi=args.dpi, facecolor=BG)
        print(p)
    plt.close(sc.fig)


def overlap(a, b, tol=2.0):
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def n_words(s):
    return len([w for w in plain(s).split() if any(c.isalnum() for c in w)])


PAUSE_MAX = 2.2   # check: no silence between two lines' voices longer than this (s)
DEAD_AIR = 1.5    # check: no stretch of video longer than this (s) with a still picture and no voice


def pixels(sc, s):
    """The frame of a built scene at scene time s, as an RGB array at the figure's dpi."""
    sc.update(s)
    sc.fig.canvas.draw()
    return np.asarray(sc.fig.canvas.buffer_rgba())[..., :3].copy()


def n_diff(x, y):
    """Number of pixels that differ between two frames."""
    return int(np.count_nonzero(np.any(x != y, axis=-1)))


def cmd_check(args):
    """Numbers against the facts sheet, text sizes, frame and overlap of every visible text (every 0.1 s of video and
    every hold), and the narration: windows, the scene fade, and per line in video time: the subtitle is on screen
    (opacity > 0.04) at least max(speech + 0.4 s, words / 3 s), fully legible (opacity >= 0.5, cover included) and
    alone while its line is spoken, and no two lines are spoken at once. The time warp: every hold freezes a static
    frame (nothing in mid-transition, and on the pixels the held frame is not between two different ones half a frame
    before and after), every cut is invisible (the frames before and after a run of cut frames are identical), no
    still picture waits more than DEAD_AIR s without a voice, and no silence between two lines exceeds PAUSE_MAX s.
    The .srt: every cue fits two lines of SRT_WIDTH characters."""
    facts = FACTS.read_text()
    problems = 0
    spans, bad = load_speech()
    for msg in bad:
        print(f"[audio] {msg}")
    problems += len(bad)
    dur = placed_lengths(spans)
    phonemes = read_json(NARR_DIR / "phonemes.json")
    total_scene, total_video, starts, voices = 0.0, 0.0, [], []
    for k, (name, fn) in enumerate(SCENES):
        sc = fn()
        if abs(sc.duration - DURATIONS[k]) > 1e-9:
            print(f"[duration] scene {k + 1} {name}: {sc.duration} s but DURATIONS says {DURATIONS[k]} s")
            problems += 1
        if abs(sc.duration * FPS - round(sc.duration * FPS)) > 1e-6:
            print(f"[duration] scene {k + 1} {name}: {sc.duration} s is not a whole number of frames")
            problems += 1
        # narration windows
        if [{key: ln[key] for key in ("id", "t0", "t1", "display", "speech")} for ln in sc.lines] != scene_lines(k):
            print(f"[narration] scene {k + 1} {name}: does not narrate its lines of {NARR_JSON.name}")
            problems += 1
        for ln in sc.lines:
            if not 0.0 <= ln["t0"] < ln["t1"]:
                print(f"[narration] {ln['id']}: window {ln['t0']}-{ln['t1']}")
                problems += 1
        for a, b in zip(sc.lines, sc.lines[1:]):
            if b["t0"] < a["t1"] + SUB_OUT - 1e-9:
                print(f"[narration] {b['id']} opens at {b['t0']} before {a['id']} has faded out ({a['t1']} + {SUB_OUT})")
                problems += 1
        if sc.lines and sc.lines[-1]["t1"] > sc.duration - 0.6 + 1e-9:
            print(f"[narration] scene {k + 1} {name}: the last window closes at {sc.lines[-1]['t1']} s, inside the "
                  f"scene fade-out ({sc.duration - 0.6:.1f}-{sc.duration:.1f} s)")
            problems += 1
        # numbers and sizes of every text, subtitles included
        for r in sc.riches:
            for tok in re.findall(r"\d+(?:[.,]\d+)*", r.plain):
                if tok not in facts:
                    print(f"[number] scene {k + 1} {name}: '{tok}' not in the facts sheet: {r.plain!r}")
                    problems += 1
            if r.size < SMALL - 1e-9:
                print(f"[size] scene {k + 1}: {r.size} pt < {SMALL} pt: {r.plain!r}")
                problems += 1
        # time warp; per-frame subtitle opacity
        motion = scene_motion(sc)
        wp = Warp(sc.duration, sc.lines, dur, motion)
        frames, n = wp.frames, len(wp.frames)
        starts.append((k + 1, name, total_video, wp.length, sc.duration, len(wp.cuts) / FPS))
        alpha = np.array([[sc.sub_alpha(ln, s) for s in frames] for ln in sc.lines]).reshape(len(sc.lines), n)
        spoken = []
        for j, ln in enumerate(sc.lines):
            if ln["id"] not in dur:
                continue
            d = dur[ln["id"]]
            on = np.count_nonzero(alpha[j] > 0.04) / FPS
            words = n_words(ln["display"])
            if on < d + 0.4 - 1e-6 or on < words / 3.0 - 1e-6:
                print(f"[subtitle] {ln['id']}: on screen {on:.2f} s; needs speech {d:.2f} s + 0.4 s and {words} words "
                      f"/ 3 = {words / 3:.2f} s")
                problems += 1
            w0, w1, a, b, v0, v1 = line_times(wp, ln, spans)
            spoken.append((a, b, ln["id"]))
            voices.append((total_video + v0, total_video + v1, ln["id"], j == 0))
            if b > wp.length + 1e-9:
                print(f"[speech] {ln['id']}: ends at {b:.2f} s, after the end of the scene ({wp.length:.2f} s)")
                problems += 1
            sl = slice(math.ceil(a * FPS - 1e-9), min(n, math.ceil(b * FPS - 1e-9)))  # frames shown while spoken
            if alpha[j, sl].size and alpha[j, sl].min() < 0.5:
                print(f"[subtitle] {ln['id']}: opacity {alpha[j, sl].min():.2f} < 0.5 while the line is spoken")
                problems += 1
            others = np.delete(alpha, j, axis=0)[:, sl]
            if others.size and others.max() > 0.04:
                print(f"[subtitle] {ln['id']}: another subtitle is visible while the line is spoken")
                problems += 1
            try:
                cues = srt_cues(srt_text(ln["display"]), w0, w1, (v0, v1), phonemes.get(ln["id"], ""))
            except ValueError as e:
                print(f"[srt] {ln['id']}: {e}")
                problems += 1
            else:
                for c0, c1, txt in cues:
                    if len(txt) > 2 or max(vis_len(x) for x in txt) > SRT_WIDTH or c1 - c0 < 1.0:
                        print(f"[srt] {ln['id']}: cue {c1 - c0:.2f} s {txt!r}")
                        problems += 1
        for (a0, b0, i0), (a1, b1, i1) in zip(spoken, spoken[1:]):
            if a1 < b0 - 1e-9:
                print(f"[speech] {i1} starts at {a1:.2f} s before {i0} ends at {b0:.2f} s")
                problems += 1
        # holds freeze a static frame; cuts are invisible; no still picture waits without a voice
        for h, m, lid in wp.holds:
            idx = [i for i, x in enumerate(frames) if x == h]
            before, at, after = (pixels(sc, x) for x in (h - 0.5 / FPS, h, h + 0.5 / FPS))
            nb, na = n_diff(before, at), n_diff(after, at)
            if not is_static(motion, h, h) or (nb and na):
                print(f"[hold] {lid}: {m} frames at scene time {h:.3f} (video {total_video + idx[0] / FPS:.2f} s) "
                      f"freeze a moving picture ({nb} / {na} pixels change half a frame before / after)")
                problems += 1
        for j0, j1 in wp.cut_runs():
            nd = n_diff(pixels(sc, (j0 - 1) / FPS), pixels(sc, (j1 + 1) / FPS))
            if nd or not is_static(motion, (j0 - 1) / FPS, (j1 + 1) / FPS):
                print(f"[cut] scene {k + 1} {name}: cutting scene frames {j0}-{j1} ({(j1 - j0 + 1) / FPS:.2f} s) "
                      f"skips a change ({nd} pixels)")
                problems += 1
        quiet = np.ones(n, bool)
        for v0, v1, _, _ in voices:
            v0, v1 = v0 - total_video, v1 - total_video
            quiet[max(0, math.floor(v0 * FPS)):max(0, min(n, math.ceil(v1 * FPS)))] = False
        still = np.array([i > 0 and is_static(motion, frames[i - 1], frames[i]) for i in range(n)])
        run, longest, at = 0, 0, 0
        for i in range(n):
            run = run + 1 if (quiet[i] and still[i]) else 0
            if run > longest:
                longest, at = run, i - run + 1
        if longest / FPS > DEAD_AIR:
            print(f"[dead air] scene {k + 1} {name}: {longest / FPS:.2f} s of still picture without a voice from "
                  f"video {total_video + at / FPS:.2f} s")
            problems += 1
        total_scene += sc.duration
        total_video += wp.length
        # frame and overlap of every visible text, every 0.1 s of video and on every hold
        holds = set(wp.hold_starts())
        seen = {}
        for i in sorted(set(range(0, n, 3)) | holds):
            s = frames[i]
            sc.update(s)
            vis = [r for r in sc.riches if r.alpha * (1.0 - sc.cover_alpha) > 0.04]
            for r in vis:
                x0, y0, x1, y1 = r.bbox()
                if x0 < 30 or x1 > W - 30 or y0 < 8 or y1 > H - 20:
                    seen.setdefault(("frame", r.plain, ""), []).append((s, i in holds))
            for a, b in itertools.combinations(vis, 2):
                if overlap(a.bbox(), b.bbox()):
                    seen.setdefault(("overlap", a.plain, b.plain), []).append((s, i in holds))
        for (kind, a, b), ts in seen.items():
            hold = " (also on a hold)" if any(h for _, h in ts) else ""
            print(f"[{kind}] scene {k + 1} {name} t={ts[0][0]:.2f}-{ts[-1][0]:.2f}{hold}: {a!r} "
                  f"{('/ ' + repr(b)) if b else ''}")
            problems += 1
        plt.close(sc.fig)
    # pauses between the voices of consecutive lines, over the whole video
    within, across = [], []
    for (_, b0, i0, _), (a1, _, i1, first) in zip(voices, voices[1:]):
        (across if first else within).append((a1 - b0, i1))
        if a1 - b0 > PAUSE_MAX:
            print(f"[pause] {a1 - b0:.2f} s of silence before {i1} (after {i0})")
            problems += 1
    for k, name, s, length, scene_time, cut in starts:
        print(f"scene {k:2d} {name:<11s} starts at {mmss(s)}  ({length:.2f} s = {scene_time:.1f} s + "
              f"{length - scene_time + cut:.2f} s of holds - {cut:.2f} s cut)")
    for kind, ps in (("within scenes", within), ("at scene changes", across)):
        if ps:
            top = max(ps)
            print(f"pauses between voices {kind}: median {np.median([p for p, _ in ps]):.2f} s, max {top[0]:.2f} s "
                  f"(before {top[1]})")
    print(f"total {mmss(total_video)} ({total_video:.2f} s = {total_scene:.1f} s of scenes + "
          f"{total_video - total_scene:.2f} s of holds and cuts); {problems} problem(s)")
    if problems:
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--scenes", default="", help="1-based scene numbers, e.g. 2,11 (default: all), or 'stale'")
    r.add_argument("--preview", action="store_true", help="960x540 (dpi 50)")
    r.add_argument("--max-seconds", type=float, default=None, help="cut every clip after this much video")
    r.add_argument("--jobs", type=int, default=16)
    r.add_argument("--out", default=None)
    r.add_argument("--no-concat", action="store_true")
    f = sub.add_parser("frames")
    f.add_argument("--scene", type=int, required=True)
    f.add_argument("--times", required=True, help="scene times (holds excluded)")
    f.add_argument("--outdir", default=None)
    f.add_argument("--dpi", type=int, default=100)
    sub.add_parser("check")
    ln = sub.add_parser("lines")
    ln.add_argument("--out", default=None, help=f"default {NARR_DIR / 'lines.json'}")
    ln.add_argument("--submit", action="store_true", help="submit the TTS job (sbatch)")
    args = ap.parse_args(argv)
    {"render": cmd_render, "frames": cmd_frames, "check": cmd_check, "lines": cmd_lines}[args.cmd](args)


if __name__ == "__main__":
    main()
