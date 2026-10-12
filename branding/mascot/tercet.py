"""Tercet, the Canto mascot: a small orange gecko whose angular tail carries three teal "hits" (a hit triplet).

Side view: pixel parts on an 18 x 12 grid, facing right, composed in layer order (tail, body, legs, head, eyes, sack
on the back) with small offsets on a 22 x 14 canvas (one spare row above and below, four spare columns on the right
for a stretched neck). Front view: one 22 x 14 map (sitting, facing the viewer) with eye, arm and tail-glow
variants. Every frame of every asset is built from these parts, so the character stays the same in all of them.
"""

from __future__ import annotations

from PIL import Image, ImageOps

PAL = {
    "k": (96, 42, 28),      # deepest: far legs
    "d": (146, 64, 42),     # deep: crest, spine, feet
    "s": (186, 92, 64),     # shade
    "m": (217, 119, 87),    # body, Claude orange #D97757
    "l": (244, 196, 170),   # light: jaw, belly, toe pads
    "e": (26, 14, 10),      # eye
    "w": (255, 250, 240),   # eye glint
    "p": (196, 78, 74),     # cheek
    "h": (120, 214, 190),   # teal hits on the tail
    "H": (210, 255, 240),   # a hit glowing; the sweat drop
    # the sack and the letters peeking out of it
    "t": (74, 58, 44), "b": (124, 100, 74), "B": (158, 130, 98), "q": (92, 74, 56),
    "y": (240, 218, 166), "Y": (214, 172, 96), "G": (44, 39, 30), "g": (255, 226, 160),
}
W, H = 18, 12
CW, CH = 22, 14            # pose canvas
MOUTH = (16, 9)            # where a carried letter's string starts (canvas coordinates, facing right)
TAIL_KINK = (1, 8)         # where the drag rope is tied (canvas coordinates)


def _m(s: str, w: int = W, h: int = H) -> list[str]:
    rows = s.strip("\n").split("\n")
    assert len(rows) == h, (len(rows), s)
    for r in rows:
        assert len(r) == w, (len(r), r)
    return rows


def _pad(rows_from_top: list[str], top: int = 0) -> list[str]:
    out = ["." * W] * top + [r.ljust(W, ".") for r in rows_from_top]
    return out + ["." * W] * (H - len(out))


TAIL = {
    "up": _pad([".h", ".s", ".ss", ".hs", "..ms", "...hm"], 3),
    "wag": _pad([".h", "..s", "..s", "..ss", "..hs", "...ms", "...hm"], 2),
    "low": _pad(["h", ".s", ".hss", "...hm"], 5),
    "out": _pad(["h", ".shss", ".....hm"], 7),
    "curl": _pad(["..h", ".s", ".s", ".hsss"], 5),
    "high": _pad([".h", ".s", ".s", ".h", ".s", ".s", ".hs", "..ss", "...m"], 0),
}

BODY = _m("""
..................
..................
..................
..................
..................
..................
......d.d.........
.....mmmmms.......
....mmmmmmms......
.....mllllls......
..................
..................
""")

HEAD = _m("""
..................
..........dd......
.........dmmmm....
.........mmmmmmm..
.........mmmmmmmm.
.........mmmmmmmmm
.........smmmmmmmm
..........smllllll
...........sssss..
..................
..................
..................
""")

EYES = {   # (x, y, colour) over the head; they move with it
    "open": [(13, 3, "w"), (14, 3, "e"), (13, 4, "e"), (14, 4, "e"), (14, 5, "p")],
    "blink": [(13, 4, "e"), (14, 4, "e"), (14, 5, "p")],
    "happy": [(13, 4, "e"), (14, 3, "e"), (15, 4, "e"), (14, 5, "p")],
    "squint": [(13, 3, "e"), (14, 4, "e"), (13, 5, "e"), (15, 5, "p")],
    "up": [(13, 3, "e"), (14, 3, "e"), (13, 4, "w"), (14, 5, "p")],
    "down": [(13, 4, "e"), (14, 4, "e"), (13, 5, "e"), (14, 5, "e"), (15, 6, "p")],
    "wide": [(13, 3, "w"), (14, 3, "e"), (13, 4, "e"), (14, 4, "e"), (14, 5, "p"), (11, 2, "H")],
}

_LEG_ROWS = {   # rows 10 and 11 (row 9 only for the poses that use it)
    "stand": ["", ".....sd...sd", "....lsdl.lsdl"],
    "walk1": ["", ".....sd....sd", "...lsdl....lsdl"],
    "walk2": ["", "......sd..sd", ".....lsdllsdl"],
    "wide": ["", "....sd......sd", "..lsdl......lsdl"],
    "sit": ["", "....sss...sd", "...dddl..lsdl"],
    "wave": ["...........sd", ".....sd......dl", "....lsdl"],
    "tuck": ["", ".....dd...dd", ""],
    "tiptoe": [".....sd...sd", ".....sd...sd", "....lsdl.lsdl"],
    "teeter": ["............dl", ".....sd....sd", "....lsdl"],
}
LEGS = {k: _pad(v, 9) for k, v in _LEG_ROWS.items()}

SACK = {   # on the back (top-left at canvas (3, 2 + body_dy)), by the letters left inside
    5: ["..yY..", ".ytty.", "bbbgbb", "bBbbbq", "bbbbbq", ".qqqq."],
    4: ["...Y..", "..tty.", ".bbgbb", "bBbbbq", "bbbbbq", ".qqqq."],
    3: ["......", "..yY..", ".bttgb", "bBbbbq", "bbbbbq", ".qqqq."],
    2: ["......", "...y..", "..tt..", ".bgbbq", "bBbbbq", ".qqqq."],
    1: ["......", "......", "..tt..", ".bbgq.", "bBbbbq", ".qqqq."],
    0: ["......", "......", "......", "..tt..", ".bBbq.", "qqqqqq"],
}
SACK_FLOOR = ["..y..yy.", ".yYy.Yy.", "..tttt..", ".bbbgbb.", "bBbbbbbq", "bBbbbbbq", "bbbbbbbq", ".qqqqqq."]

FRONT = [
    "......................",
    "......................",
    ".....dmmmmmmd.........",
    "...mmmmmmmmmmmm.......",
    "..mweemmmmmmweem......",
    "..meeemmmmmmeeem......",
    "..mmmmmmmmmmmmmm......",
    "..mpmmemmmmemmpm......",
    "...mmmmeeeemmmm.......",
    "....smllllllms........",
    "..llsmllllllmsll......",
    "....smllllllms.....h..",
    "....dsmsmmsmsdmm....s.",
    "...ldl.....ldlsmhssh..",
]
FRONT_HAPPY = {(3, 4): "m", (4, 4): "e", (5, 4): "m", (2, 5): "m", (3, 5): "e", (4, 5): "m", (5, 5): "e",
               (12, 4): "m", (13, 4): "e", (14, 4): "m", (12, 5): "e", (13, 5): "m", (14, 5): "e"}
FRONT_BLINK = {(3, 4): "m", (4, 4): "m", (12, 4): "m", (13, 4): "m", (14, 4): "m", (3, 5): "e", (4, 5): "e",
               (5, 5): "e", (12, 5): "e", (13, 5): "e", (14, 5): "e"}
FRONT_ARM = {
    1: ({(14, 10): ".", (15, 10): "."}, {(14, 9): "s", (15, 8): "m", (16, 7): "m", (17, 6): "l", (17, 5): "l",
                                         (16, 5): "l"}),
    2: ({(14, 10): ".", (15, 10): "."}, {(14, 9): "s", (15, 9): "m", (16, 8): "m", (17, 8): "l", (18, 7): "l",
                                         (18, 8): "l"}),
}
FRONT_HITS = [(16, 13), (19, 13), (19, 11)]


def _put(im, rows, dx, dy, recolor=None):
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != ".":
                xx, yy = x + dx, y + dy
                if 0 <= xx < im.width and 0 <= yy < im.height:
                    cc = recolor.get(c, c) if recolor else c
                    im.putpixel((xx, yy), PAL[cc] + (255,))


def pose(eyes="open", legs="stand", tail="up", head_dy=0, head_dx=0, body_dy=0, tail_glow=False, sack=None,
         flip=False) -> Image.Image:
    """The gecko on a transparent CW x CH canvas, facing right (left with flip). head_dx > 0 stretches the neck;
    body_dy < 0 lifts the body and the legs stretch to stay attached; sack: letters left in the sack on its back."""
    im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    _put(im, TAIL[tail], 0, body_dy + 1, {"h": "H"} if tail_glow else None)
    _put(im, BODY, 0, body_dy + 1)
    _put(im, LEGS[legs], 0, 1)
    top = next(r for r in LEGS[legs][9:] if r.strip("."))
    for k in range(1, -body_dy + 1):                # lifted body: the legs reach up to it
        _put(im, [top], 0, 9 + 1 - k + (0 if LEGS[legs][9].strip(".") else 1))
    for k in range(head_dx):                        # the stretched neck
        for y, c in ((7, "m"), (8, "m"), (9, "s")):
            im.putpixel((11 + k, y + body_dy + head_dy), PAL[c] + (255,))
    _put(im, HEAD, head_dx, body_dy + head_dy + 1)
    for (x, y, c) in EYES[eyes]:
        im.putpixel((x + head_dx, y + body_dy + head_dy + 1), PAL[c] + (255,))
    if sack is not None:
        _put(im, SACK[sack], 3, body_dy + 2)
    return ImageOps.mirror(im) if flip else im


def front(eyes="open", arm=0, glow=None) -> Image.Image:
    """The gecko sitting, facing the viewer (22 x 14). eyes: open/happy/blink; arm 1 or 2: a waving paw; glow: the
    index of the tail hit that lights up."""
    grid = [list(r) for r in FRONT]
    if eyes == "happy":
        for (x, y), c in FRONT_HAPPY.items():
            grid[y][x] = c
    elif eyes == "blink":
        for (x, y), c in FRONT_BLINK.items():
            grid[y][x] = c
    if arm:
        clear, paw = FRONT_ARM[arm]
        for (x, y), c in list(clear.items()) + list(paw.items()):
            grid[y][x] = c
    if glow is not None:
        x, y = FRONT_HITS[glow]
        grid[y][x] = "H"
    im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    _put(im, ["".join(r) for r in grid], 0, 0)
    return im


def mouth(head_dx=0, head_dy=0, body_dy=0, flip=False) -> tuple[int, int]:
    x, y = MOUTH[0] + head_dx, MOUTH[1] + body_dy + head_dy
    return (CW - 1 - x if flip else x), y


def sack_floor() -> Image.Image:
    """The full sack on the floor, letters peeking out with a faint glow (10 x 10)."""
    im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    lit = {(x + 1, y + 1) for y, r in enumerate(SACK_FLOOR) for x, c in enumerate(r) if c in "yY"}
    for (x, y) in lit:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x + dx, y + dy) not in lit and 0 <= x + dx < 10 and 0 <= y + dy < 10:
                    im.putpixel((x + dx, y + dy), PAL["G"] + (255,))
    _put(im, SACK_FLOOR, 1, 1)
    return im


def sack_back(n: int) -> Image.Image:
    im = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    _put(im, SACK[n], 0, 0)
    return im
