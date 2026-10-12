"""The Canto README banner: Tercet the gecko hangs the letters C, A, N, T, O on a wire in the dark.

Drawn on a 192 x 64 pixel canvas and enlarged 5x (nearest neighbour) to 960 x 320. The story is a list of frame
states written by a small director (enter dragging a sack of glowing letters, climb the ladder, hang five letters,
a particle track lights the word, turn to the viewer, hop, wave, sit, fade); render() draws one state. The GIF starts
on the finished sign (a readable first frame where animation is paused) and loops through the fade into the story.

    python canto_banner.py      # canto_banner.gif, canto_banner_final.png, banner_info.json into .. (branding/)
"""

from __future__ import annotations

import copy
import json
import math
import os

from PIL import Image

import tercet as T

NW, NH, SCALE = 192, 64, 5
FRAME_MS = 70                      # 14.3 frames per second
BG = (13, 17, 23)                  # GitHub dark
FLOOR_Y = 58
WIRE_Y = 23
WIRE_X = (44, 168)
LADDER_X = (34, 40)                # rails
LADDER_TOP = 21
LETTERS = "CANTO"
CX = [64, 80, 96, 112, 128]        # hook positions
HANG_TOP = 28                      # glyph top when hung (a 3-px thread below the hook)
CARRY_TOP = HANG_TOP + 3           # glyph top while it hangs from the mouth
GECKO_FLOOR_Y = FLOOR_Y - 13       # pose canvas top when the feet stand on the floor
GECKO_WIRE_Y = WIRE_Y - 13         # ... on the wire

COL = {
    "dust1": (34, 41, 56), "dust2": (52, 62, 82),
    "pool1": (24, 28, 36), "pool2": (33, 36, 44), "floor": (40, 44, 54), "floor_dim": (22, 26, 34),
    "wire": (66, 74, 92), "thread": (30, 36, 48), "hook": (120, 128, 148),
    "rail_lit": (150, 116, 86), "rail_mid": (108, 84, 64), "rail_dim": (70, 56, 46),
    "rope": (138, 112, 84),
    "spark": (255, 252, 236), "spark2": (255, 220, 150),
    "lpool1": (24, 25, 30), "lpool2": (30, 29, 32),
    "gpool1": (28, 28, 32), "gpool2": (40, 36, 36), "gpool3": (58, 48, 40),
    "trail": (120, 214, 190), "trail_hi": (220, 255, 245), "edge": (52, 104, 98),
}
# letter colours by light level: 0 unlit, 1-4 lighting, 5 the lit word, 6 a flare
LETTER_CORE = [(112, 102, 82), (150, 134, 100), (200, 178, 128), (240, 218, 166), (255, 240, 200), (255, 248, 222),
               (255, 255, 246)]
LETTER_EDGE = [(80, 72, 58), (110, 96, 70), (160, 132, 82), (214, 172, 96), (238, 192, 104), (246, 204, 118),
               (255, 228, 150)]
HALO1 = [None, (22, 24, 26), (32, 31, 28), (44, 39, 30), (54, 46, 32), (78, 58, 34), (98, 74, 40)]
HALO2 = [None, None, None, None, None, (34, 29, 27), (46, 38, 30)]
SPUTTER = [1, 0, 2, 1, 3, 4]        # a letter lights like a bulb coming on

GLYPHS = {
    "C": ["..#####..", ".#######.", "###...###", "##.......", "##.......", "##.......", "##.......",
          "##.......", "###...###", ".#######.", "..#####.."],
    "A": ["...###...", "..#####..", ".###.###.", ".##...##.", "##.....##", "##.....##", "#########",
          "#########", "##.....##", "##.....##", "##.....##"],
    "N": ["##.....##", "###....##", "####...##", "##.##..##", "##.##..##", "##..##.##", "##..##.##",
          "##...####", "##...####", "##....###", "##.....##"],
    "T": ["########", "########", "...##...", "...##...", "...##...", "...##...", "...##...", "...##...",
          "...##...", "...##...", "...##..."],
    "O": ["..#####..", ".#######.", "###...###", "##.....##", "##.....##", "##.....##", "##.....##",
          "##.....##", "###...###", ".#######.", "..#####.."],
}
DUST = [(10, 8), (27, 33), (58, 5), (90, 10), (141, 7), (176, 13), (185, 38), (8, 47), (160, 47), (118, 50)]


# ---------------------------------------------------------------------------------------------------- drawing

def setp(px, x, y, c):
    if 0 <= x < NW and 0 <= y < NH:
        px[x, y] = c


def line(px, x0, y0, x1, y1, c):
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for k in range(n + 1):
        setp(px, round(x0 + (x1 - x0) * k / n), round(y0 + (y1 - y0) * k / n), c)


def glyph(ch: str):
    g = GLYPHS[ch]
    return {(x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == "#"}, len(g[0])


def glyph_x0(ch, cx, off=0):
    return cx - glyph(ch)[1] // 2 + off


def attach_cols(ch: str) -> list[int]:
    """Where a thread meets the glyph's top row: the centre, or both top strokes where the centre is open (N)."""
    cells, w = glyph(ch)
    c = w // 2
    return [c] if (c, 0) in cells else [1, w - 2]


def draw_halo(px, ch, cx, top, level, off=0):
    cells, _ = glyph(ch)
    x0 = glyph_x0(ch, cx, off)
    ring1 = {(x + dx, y + dy) for (x, y) in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - cells
    if HALO2[level]:
        ring2 = {(x + dx, y + dy) for (x, y) in ring1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - ring1 - cells
        for (x, y) in ring2:
            setp(px, x0 + x, top + y, HALO2[level])
    if HALO1[level]:
        for (x, y) in ring1:
            setp(px, x0 + x, top + y, HALO1[level])


def draw_glyph(px, ch, cx, top, level, off=0):
    cells, _ = glyph(ch)
    x0 = glyph_x0(ch, cx, off)
    for (x, y) in cells:
        edge = (x + 1, y) not in cells or (x, y + 1) not in cells
        setp(px, x0 + x, top + y, LETTER_EDGE[level] if edge else LETTER_CORE[level])


def draw_threads(px, ch, x_from, y_from, top, off, cx):
    x0 = glyph_x0(ch, cx, off)
    for a in attach_cols(ch):
        line(px, x_from, y_from, x0 + a, top - 1, COL["hook"])


def swing(k, amp=2.5, period=10.0, decay=12.0):
    return round(amp * math.exp(-k / decay) * math.sin(2 * math.pi * k / period)) if k >= 0 else 0


def ellipse_pool(px, cx, cy, rx, ry, layers):
    """layers: [(r_max, colour, dithered)], innermost first."""
    for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
        for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
            r = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            for r_max, c, dith in layers:
                if r < r_max and (not dith or (x + y) % 2 == 0):
                    setp(px, x, y, c)
                    break


def static_scene() -> Image.Image:
    im = Image.new("RGBA", (NW, NH), BG + (255,))
    px = im.load()
    for (x, y) in DUST:
        px[x, y] = COL["dust1"] + (255,)
    cx0 = (LADDER_X[0] + LADDER_X[1]) / 2
    ellipse_pool(px, cx0, FLOOR_Y + 0.5, 26, 4.2, [(0.45, COL["pool2"] + (255,), False),
                                                     (1.0, COL["pool1"] + (255,), True)])
    for x in range(8, NW - 8):                          # the floor line, lit near the ladder
        d = abs(x - cx0)
        if d < 24:
            px[x, FLOOR_Y] = COL["floor"] + (255,)
        elif d < 70 or (x % 2 == 0 and d < 110):
            px[x, FLOOR_Y] = COL["floor_dim"] + (255,)
    for x in (WIRE_X[0] + 2, WIRE_X[1] - 2):            # threads from the ceiling
        for y in range(0, WIRE_Y):
            px[x, y] = COL["thread"] + (255,)
    for x in range(WIRE_X[0], WIRE_X[1] + 1):
        px[x, WIRE_Y] = COL["wire"] + (255,)
    for cx in CX:
        px[cx, WIRE_Y + 1] = COL["hook"] + (255,)
    for y in range(LADDER_TOP, FLOOR_Y):                # the ladder, lit from the floor
        c = COL["rail_lit"] if y > FLOOR_Y - 12 else COL["rail_mid"] if y > FLOOR_Y - 26 else COL["rail_dim"]
        for x in LADDER_X:
            px[x, y] = c + (255,)
        if (y - LADDER_TOP) % 4 == 2:
            for x in range(LADDER_X[0] + 1, LADDER_X[1]):
                px[x, y] = c + (255,)
    return im


STATIC = None

# the particle track: a smile-shaped arc under the word, its hits right under the letters
ARC = [(56, 46), (96, 49), (136, 46)]


def _circle(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


CXC, CYC, RC = _circle(*ARC)
A0 = math.atan2(ARC[0][1] - CYC, ARC[0][0] - CXC)
A1 = math.atan2(ARC[2][1] - CYC, ARC[2][0] - CXC)
TRACK_N = 20                        # frames from the first to the last point
HITS = [0.1, 0.3, 0.5, 0.7, 0.9]    # one under each letter


def arc_point(u):
    a = A0 + (A1 - A0) * u
    return round(CXC + RC * math.cos(a)), round(CYC + RC * math.sin(a))


def track_levels(k):
    """Letter levels while the track runs: each letter flares as the particle passes under it, then the whole word
    takes one breath."""
    if TRACK_N + 2 <= k < TRACK_N + 4:
        return [6] * 5
    out = []
    for h in HITS:
        d = k - h * TRACK_N
        out.append(6 if 0 <= d < 3 else (5 if d >= 3 else 4))
    return out


def draw_track(px, k):
    u = k / TRACK_N
    fade_k = k - TRACK_N
    shown = [h for h in HITS if h <= min(u, 1.0)]
    if fade_k < 14:
        edge_col = COL["edge"] if fade_k < 7 else COL["thread"]
        for a, b in zip(shown, shown[1:]):
            (x0, y0), (x1, y1) = arc_point(a), arc_point(b)
            line(px, x0, y0, x1, y1, edge_col)
        for h in shown:
            x, y = arc_point(h)
            setp(px, x, y, COL["trail"] if fade_k < 9 else COL["edge"])
    if u <= 1.0:
        (xa, ya), (xb, yb) = arc_point(max(u - 0.09, 0)), arc_point(max(u - 0.03, 0))
        line(px, xa, ya, xb, yb, COL["edge"])
        x, y = arc_point(u)
        line(px, xb, yb, x, y, COL["trail"])
        draw_spark(px, x, y, 0, COL["trail_hi"], COL["trail"])


def draw_spark(px, x, y, k, core=None, arm=None):
    core, arm = core or COL["spark"], arm or COL["spark2"]
    if k == 0:
        setp(px, x, y, core)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            setp(px, x + dx, y + dy, arm)
    elif k == 1:
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            setp(px, x + dx, y + dy, arm)


FADE_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def fade(im, a):
    a = min(FADE_STEPS, key=lambda s: abs(s - a))
    px = im.load()
    for y in range(NH):
        for x in range(NW):
            r, g, b, _ = px[x, y]
            px[x, y] = (round(BG[0] + (r - BG[0]) * a), round(BG[1] + (g - BG[1]) * a),
                        round(BG[2] + (b - BG[2]) * a), 255)
    return im


def blit(im, sprite, x, y):
    px, sp = im.load(), sprite.load()
    for j in range(sprite.height):
        for i in range(sprite.width):
            a = sp[i, j]
            if a[3] and 0 <= x + i < NW and 0 <= y + j < NH:
                px[x + i, y + j] = a


def render(st: dict, f: int) -> Image.Image:
    global STATIC
    if STATIC is None:
        STATIC = static_scene()
    im = STATIC.copy()
    px = im.load()
    g = st["g"]
    sag = st.get("sag", 0)
    if sag and g["visible"] and not g.get("rot") and not g.get("front"):     # the wire gives under the gecko
        a, b = g["x"] + 4, g["x"] + 13
        for x in range(max(a, WIRE_X[0]), min(b, WIRE_X[1]) + 1):
            px[x, WIRE_Y] = BG + (255,)
            px[x, WIRE_Y + sag] = COL["wire"] + (255,)
        if sag > 1:
            for x in (a - 1, b + 1):
                if WIRE_X[0] <= x <= WIRE_X[1]:
                    px[x, WIRE_Y + 1] = COL["wire"] + (255,)
    # warm floor pools: one under each lit letter, then the word's own pool in the finale
    levels = dict(st["level"])
    if st.get("track") is not None:
        levels = dict(enumerate(track_levels(st["track"])))
    for i, lev in levels.items():
        if lev >= 3:
            ellipse_pool(px, CX[i], FLOOR_Y + 0.5, 10, 2, [(0.5, COL["lpool2"] + (255,), False),
                                                          (1.0, COL["lpool1"] + (255,), True)])
    if st.get("pool") is not None:
        rx = 52 * min(1.0, (st["pool"] + 1) / TRACK_N)
        ellipse_pool(px, 96, FLOOR_Y + 0.5, max(rx, 4), 3.2, [(0.35, COL["gpool3"] + (255,), False),
                                                             (0.7, COL["gpool2"] + (255,), False),
                                                             (1.0, COL["gpool1"] + (255,), True)])
    tw = (f // 9) % 23                                  # dust: one mote twinkles at a time
    if tw < len(DUST) and (f % 9) < 3:
        px[DUST[tw]] = COL["dust2"] + (255,)
    # hung letters: halos, then threads (and hooks), then glyphs, so nothing covers a thread
    hung = []
    for i, ch in enumerate(LETTERS):
        if i in st["hung"]:
            k = f - st["hung"][i]
            hung.append((i, ch, swing(k, amp=3.5 if ch == "T" else 2.5), levels.get(i, 0)))
    for i, ch, off, lev in hung:
        draw_halo(px, ch, CX[i], HANG_TOP, lev, off)
    for i, ch, off, lev in hung:
        draw_threads(px, ch, CX[i], WIRE_Y + 1, HANG_TOP, off, CX[i])
        px[CX[i], WIRE_Y + 1] = COL["hook"] + (255,)
    for i, ch, off, lev in hung:
        draw_glyph(px, ch, CX[i], HANG_TOP, lev, off)
    if st.get("track") is not None:
        draw_track(px, st["track"])
    c = st.get("carry")
    if c is not None and not c.get("string"):           # tossed: it flies behind the gecko
        draw_glyph(px, LETTERS[c["i"]], c["x"], c["y"], 0)
    s = st["sack"]
    if s["mode"] == "floor":
        sx, sy = s["x"], s["y"]
        blit(im, T.sack_floor(), sx, sy)
        if st["rope"]:
            kx, ky = g["x"] + T.TAIL_KINK[0], g["y"] + T.TAIL_KINK[1] + g["pose"].get("body_dy", 0)
            mid = ((sx + 7 + kx) // 2, max(sy + 3, ky) + 1)
            line(px, sx + 7, sy + 3, *mid, COL["rope"])
            line(px, *mid, kx, ky, COL["rope"])
    elif s["mode"] == "air":
        blit(im, T.sack_floor(), s["x"], s["y"])
    elif s["mode"] == "empty":
        blit(im, T.sack_back(0), s["x"], s["y"])
    if g["visible"]:
        if g.get("front"):
            blit(im, T.front(**g["fpose"]), g["x"], g["y"])
        else:
            spr = T.pose(**g["pose"])
            if g.get("rot"):
                spr = spr.rotate(g["rot"], expand=True)
            blit(im, spr, g["x"], g["y"] + (sag if not g.get("rot") else 0))
    if c is not None and c.get("string"):               # in the mouth: string, then the letter
        draw_threads(px, LETTERS[c["i"]], c["string"][0], c["string"][1], c["y"], c.get("off", 0), c["x"])
        draw_glyph(px, LETTERS[c["i"]], c["x"], c["y"], 0, c.get("off", 0))
    for (x, y, k) in st.get("sparks", []):
        draw_spark(px, x, y, k)
    if st.get("fade", 1.0) < 1.0:
        im = fade(im, st["fade"])
    return im


# ---------------------------------------------------------------------------------------------------- director

class Director:
    """Writes the story as a list of frame states."""

    def __init__(self):
        self.frames: list[dict] = []
        self.st = {"g": {"x": -4, "y": GECKO_FLOOR_Y, "visible": True, "rot": 0, "front": False,
                         "pose": dict(eyes="open", legs="stand", tail="up"), "fpose": {}},
                   "sack": {"mode": "floor", "x": -16, "y": FLOOR_Y - 9}, "rope": True,
                   "hung": {}, "level": {}, "light": {}, "carry": None, "sparks": [], "track": None,
                   "pool": None, "sag": 0, "fade": 1.0}
        self.hero_index = None

    @property
    def f(self):
        return len(self.frames)

    def emit(self, n=1):
        for _ in range(n):
            for i, f0 in self.st["light"].items():
                if self.f >= f0:
                    self.st["level"][i] = SPUTTER[min(self.f - f0, len(SPUTTER) - 1)]
            self.frames.append(copy.deepcopy(self.st))
            self.st["sparks"] = [(x, y, s + 1) for (x, y, s) in self.st["sparks"] if s < 1]
            if self.st["track"] is not None:
                self.st["track"] += 1
            if self.st["pool"] is not None:
                self.st["pool"] += 1

    def P(self, **kw):
        """Set the whole side pose (unspecified parts back to the default)."""
        base = dict(eyes="open", legs="stand", tail="up")
        base.update(kw)
        self.st["g"]["pose"] = base

    def hold(self, n, sag=0, **kw):
        self.P(**kw)
        self.st["sag"] = sag
        self.emit(n)
        self.st["sag"] = 0

    def scurry(self, x_target, **kw):
        """Gecko steps: darts of 3 px per frame, a one-frame freeze (head up) after every three."""
        g = self.st["g"]
        sgn = 1 if x_target > g["x"] else -1
        k = 0
        while g["x"] != x_target:
            if k % 4 == 3:
                self.P(head_dy=-1, **kw)
            else:
                g["x"] += sgn * min(3, abs(x_target - g["x"]))
                leg = ("walk1", "stand", "walk2")[k % 4]
                self.P(legs=leg, body_dy=-1 if leg == "stand" else 0, **kw)
            self.emit()
            k += 1

    # -- the story
    def fade_in(self, n=3):
        for k in range(n):
            self.st["fade"] = (k + 1) / n
            self.P(eyes="squint", legs="wide", head_dy=1)
            self.emit()
        self.st["fade"] = 1.0

    def enter_dragging(self):
        g, s = self.st["g"], self.st["sack"]
        stop = LADDER_X[0] - T.W + 2
        k = 0
        while g["x"] < stop:
            tug = k % 6 == 5                            # every sixth frame a tug that does not move
            if not tug:
                g["x"] += 1
                if g["x"] - s["x"] > 13:
                    s["x"] += 1
            leg = "wide" if tug else ("walk1", "stand", "walk2", "stand")[(k // 2) % 4]
            self.P(eyes="squint" if (k // 6) % 3 != 2 else "open", legs=leg, head_dy=1)
            self.emit()
            k += 1
            if k == 12:                                 # a breather: a glance back at the sack, a puff
                self.hold(1, eyes="blink", tail="low", body_dy=1)
                self.hold(2, eyes="down", tail="low", body_dy=1, head_dy=1)
                self.hold(1, eyes="squint", tail="up")
        while g["x"] - s["x"] > 12:
            s["x"] += 1
            self.hold(1)

    def look_up(self):
        self.hold(2, eyes="up", head_dy=-1, tail="wag")
        self.hold(2, eyes="up", head_dy=-2, body_dy=1, tail="up")       # leans back: that is high
        self.hold(1, eyes="blink", head_dy=-1)
        self.hold(1, eyes="squint", head_dy=-1, tail="wag")              # a gulp

    def hoist(self):
        g, s = self.st["g"], self.st["sack"]
        self.hold(2, eyes="squint", legs="wide", body_dy=1)
        self.st["rope"] = False
        x0, y0 = s["x"], s["y"]
        x1, y1 = g["x"] + 2, g["y"] + 1
        s["mode"] = "air"
        for k in range(4):
            u = (k + 1) / 4
            s["x"] = round(x0 + (x1 - x0) * u)
            s["y"] = round(y0 + (y1 - y0) * u - 9 * math.sin(math.pi * u))
            self.P(eyes="up", tail="wag", body_dy=-1 if k < 2 else 0)
            self.emit()
        s["mode"] = "back"
        self.hold(2, eyes="squint", legs="wide", body_dy=1, sack=5)
        self.hold(1, eyes="happy", sack=5, tail="wag")

    def climb(self):
        g = self.st["g"]
        self.scurry(LADDER_X[0] - 10, sack=5)
        g["rot"] = 90
        g["x"], g["y"] = LADDER_X[0] - 5, FLOOR_Y - T.CW
        self.hold(1, eyes="up", sack=5)
        k = 0
        while g["y"] > LADDER_TOP - 9:
            g["y"] -= 2
            leg = ("walk1", "stand", "walk2", "stand")[k % 4]
            self.P(legs=leg, sack=5, tail="up" if k % 6 < 3 else "wag", eyes="up" if k % 10 != 7 else "blink")
            self.emit()
            k += 1
        self.hold(1, eyes="up", legs="tuck", sack=5)                     # over the top ...
        g["rot"] = 0
        g["x"], g["y"] = 33, GECKO_WIRE_Y - 3
        self.hold(1, eyes="happy", legs="tiptoe", sack=5, tail="wag")
        g["x"], g["y"] = 38, GECKO_WIRE_Y - 2
        self.hold(1, eyes="happy", legs="tuck", sack=5, tail="wag")
        g["x"], g["y"] = WIRE_X[0] - 2, GECKO_WIRE_Y                    # ... and onto the wire, which gives
        self.hold(1, sag=1, eyes="squint", legs="wide", sack=5, tail="out", body_dy=1)
        self.hold(1, eyes="open", sack=5, tail="out")
        self.hold(1, eyes="blink", sack=5)

    def fetch(self, i, n_left):
        """Toss the next letter out of the sack (it flies behind the head) and catch it by its string."""
        g = self.st["g"]
        self.hold(1, eyes="up", head_dy=-1, sack=n_left + 1, tail="wag", body_dy=1)
        sx, sy = g["x"] + 6, g["y"] + 2
        tx, ty = g["x"] + T.MOUTH[0], CARRY_TOP
        for k in range(5):
            u = (k + 1) / 6
            self.st["carry"] = {"i": i, "x": round(sx + (tx - sx) * u),
                                "y": round(sy + (ty - sy) * u - 6 * math.sin(math.pi * u))}
            self.P(eyes="up", sack=n_left, tail="wag", body_dy=-1 if k < 2 else 0)
            self.emit()
        self.carry_mouth(i, eyes="squint", head_dy=1, legs="wide", sack=n_left, tail="out")   # oof, heavy
        self.st["sag"] = 1
        self.emit()
        self.st["sag"] = 0

    def carry_mouth(self, i, sway=0, top=CARRY_TOP, x=None, **kw):
        g = self.st["g"]
        self.P(**kw)
        p = g["pose"]
        mx, my = T.mouth(p.get("head_dx", 0), p.get("head_dy", 0), p.get("body_dy", 0))
        self.st["carry"] = {"i": i, "x": (g["x"] + mx) if x is None else x, "y": top, "off": sway,
                            "string": (g["x"] + mx, g["y"] + my + 1)}

    def clip(self, i, n_left, stretch=0):
        """Lift the letter onto its hook, then let go."""
        legs = "tiptoe" if stretch else "wide"
        self.carry_mouth(i, eyes="squint", head_dy=-1, head_dx=stretch, legs=legs, body_dy=-1, sack=n_left,
                         tail="out", top=CARRY_TOP - 1, x=CX[i])
        self.emit()
        self.carry_mouth(i, eyes="squint", head_dy=-2, head_dx=stretch, legs=legs, body_dy=-1, sack=n_left,
                         tail="out", top=CARRY_TOP - 2, x=CX[i])
        self.emit()
        self.carry_mouth(i, eyes="down", head_dy=0, head_dx=stretch, sack=n_left, top=HANG_TOP, x=CX[i])
        self.st["sparks"].append((CX[i], WIRE_Y + 1, 0))
        self.emit()
        self.st["carry"] = None
        self.st["hung"][i] = self.f
        self.st["light"][i] = self.f + (4 if i == 0 else 8)
        self.hold(1, eyes="happy", head_dx=stretch, sack=n_left, tail="wag")

    def hang(self, i):
        g = self.st["g"]
        n_left = 4 - i
        target = CX[i] - T.MOUTH[0] - (4 if i == 1 else 0)        # A: stops short, has to stretch
        self.scurry(target, sack=n_left + 1)
        self.fetch(i, n_left)
        if i == 0:
            self.carry_mouth(i, sack=n_left)
            self.emit()
            self.clip(i, n_left)
            self.hold(2, eyes="down", head_dy=1, sack=n_left)            # watches its first letter come on
            self.hold(2, eyes="happy", sack=n_left, tail="wag")
        elif i == 1:
            self.carry_mouth(i, eyes="up", sack=n_left)
            self.emit()
            for dx in (1, 2, 3, 4):                                       # reaches, on tiptoe
                self.carry_mouth(i, eyes="squint", head_dx=dx, legs="tiptoe", body_dy=-1, sack=n_left, tail="out")
                self.emit()
            for dx, sway in ((4, 1), (3, -1)):                            # trembling
                self.carry_mouth(i, sway=sway, eyes="squint", head_dx=dx, legs="tiptoe", body_dy=-1, sack=n_left,
                                 tail="out")
                self.emit()
            self.clip(i, n_left, stretch=4)
            self.hold(2, eyes="blink", sack=n_left, body_dy=1)              # phew
        elif i == 3:
            # the heavy T nearly pulls the gecko off the wire
            for dx, sway, legs, hdy, tail, eyes, sag, bdy in (
                    (1, 2, "wide", 1, "out", "squint", 1, 0), (1, 3, "teeter", 2, "high", "wide", 2, 0),
                    (0, 3, "teeter", 2, "high", "wide", 2, 0), (0, -2, "teeter", 1, "wag", "wide", 1, 0),
                    (0, 2, "teeter", 1, "high", "squint", 1, 0), (-1, -1, "wide", 0, "out", "squint", 1, 1),
                    (-1, 0, "stand", 0, "up", "blink", 0, 0)):
                g["x"] += dx
                self.carry_mouth(i, sway=sway, eyes=eyes, legs=legs, head_dy=hdy, tail=tail, body_dy=bdy, sack=n_left)
                self.st["sag"] = sag
                self.emit()
            self.st["sag"] = 0
            self.clip(i, n_left)
            self.hold(1, eyes="happy", sack=n_left, tail="wag")
        else:
            self.carry_mouth(i, sack=n_left)
            self.emit()
            self.clip(i, n_left)
            if i == 2:
                self.hold(1, eyes="happy", sack=n_left, tail="up")

    def finale(self):
        g, sk = self.st["g"], self.st["sack"]
        # fling the empty sack away over its head
        self.hold(1, eyes="squint", legs="wide", body_dy=1, sack=0, tail="out")
        sk.update(mode="empty", x=g["x"] + 3, y=g["y"] + 2)
        for k, (dx, dy) in enumerate(((-2, -5), (-3, -6), (-4, -7))):
            sk["x"] += dx
            sk["y"] += dy
            self.hold(1, eyes="up" if k < 2 else "happy", tail="wag", body_dy=-1 if k == 0 else 0)
        sk["mode"] = "gone"
        # the particle track runs under the word and lights it letter by letter
        self.st["track"] = 0
        self.st["pool"] = 0
        self.st["light"] = {}
        self.st["sparks"].append((*arc_point(0), 0))
        for k in range(TRACK_N + 5):
            self.P(eyes="down", head_dy=1 if k < TRACK_N else 0, tail="up" if k < TRACK_N else "wag",
                   tail_glow=(k // 2) % 2 == 0 and k < TRACK_N)
            self.emit()
        self.st["track"] = None
        for i in range(5):
            self.st["level"][i] = 5
        # turn to the viewer, hop, wave, sit
        self.hold(1, eyes="blink", legs="tuck")
        g.update(front=True, x=g["x"] - 1, y=WIRE_Y - 14)
        self.front(1, eyes="happy")
        for dy in (-2, -3, -2, 0):
            g["y"] = WIRE_Y - 14 + dy
            self.front(1, eyes="happy")
        for arm in (1, 1, 2, 2, 1, 1, 2, 2):
            self.front(1, eyes="happy", arm=arm)
        self.front(1, eyes="open")
        self.hero_index = self.f
        for k in range(28):                             # the hero hold: a blink, the hit triplet glowing in turn
            glow = (0, 1, 2, None)[(k // 3) % 4]
            self.front(1, eyes="blink" if k in (13, 14) else "open", glow=glow)

    def front(self, n, **kw):
        self.st["g"]["fpose"] = kw
        self.emit(n)

    def fade_out(self, n=5):
        for k in range(n):
            self.st["fade"] = 1 - (k + 1) / n
            self.emit()

    def story(self):
        self.fade_in()
        self.enter_dragging()
        self.look_up()
        self.hoist()
        self.climb()
        for i in range(5):
            self.hang(i)
        self.finale()
        self.fade_out()
        return self.frames


# ---------------------------------------------------------------------------------------------------- output

def build(out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
    os.makedirs(out_dir, exist_ok=True)
    d = Director()
    states = d.story()
    frames = [render(st, f) for f, st in enumerate(states)]
    k0 = d.hero_index                                   # the GIF starts on the finished sign
    order = list(range(k0, len(frames))) + list(range(k0))
    colours = sorted({c for im in frames for c in im.convert("RGB").get_flattened_data()})
    if len(colours) > 256:
        raise SystemExit(f"{len(colours)} colours")
    flat = [v for c in colours for v in c] + [0] * (3 * (256 - len(colours)))
    index = {c: i for i, c in enumerate(colours)}
    out, dur, last = [], [], None
    for f in order:
        b = frames[f].tobytes()
        if b == last:
            dur[-1] += FRAME_MS
            continue
        rgb = frames[f].convert("RGB").resize((NW * SCALE, NH * SCALE), Image.NEAREST)
        p = Image.new("P", rgb.size)
        p.putpalette(flat)
        p.putdata([index[c] for c in rgb.get_flattened_data()])
        out.append(p)
        dur.append(FRAME_MS)
        last = b
    gif = os.path.join(out_dir, "canto_banner.gif")
    out[0].save(gif, save_all=True, append_images=out[1:], duration=dur, loop=0, optimize=False, disposal=1,
                palette=bytes(flat))
    frames[k0].convert("RGB").resize((NW * SCALE, NH * SCALE), Image.NEAREST).save(
        os.path.join(out_dir, "canto_banner_final.png"))
    info = {"frames": len(frames), "gif_frames": len(out), "seconds": round(len(frames) * FRAME_MS / 1000, 2),
            "colours": len(colours), "bytes": os.path.getsize(gif), "hero_frame": k0}
    json.dump(info, open(os.path.join(out_dir, "banner_info.json"), "w"), indent=1)
    return info, states, frames


if __name__ == "__main__":
    print(build()[0])
