"""The static mascot assets: concept sheet, transparent mascot, sprite sheet (+ JSON). Run from this directory:

    python assets.py            # writes into ..  (branding/)
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

import tercet as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BG = (13, 17, 23)
INK, MUTED, ACCENT = (230, 226, 216), (140, 146, 158), (217, 119, 87)

# concept B: "Hop", an upright jerboa-like hopper with a kinked tail; concept C: "Helix", an arched,
# chameleon-like walker whose tail coils into a helix
HOP = """
...........dd...
..........dmd.d.
..........dmddmd
.........mmmmmmd
........mmmmmmmd
........mmmmemmm
.......mmmmmmmpmm
........smmlllll
........mmmsss..
.......mmmmm....
......mmmmmms...
.h...mmmmmmls...
..s..mmmmmll....
...s.smmmmls....
....hssmssl.....
.......sd.......
......sdddddl...
"""
HELIX = """
..............dd....
.............dmmd...
.......mmmm.dmmmmd..
.....mmmmmmmmmmmmmm.
....mmmmmmmmmmmmemm.
...mmmmmmmmmmmmmmmpm
...smmmmmmmmmmmsllll
..s.sllllllllls.....
.s...sd.....sd......
s...hsdl...sdl......
s.s.................
.sh.................
"""


def font(size, bold=False):
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return ImageFont.truetype(os.path.join(d, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def from_rows(s: str) -> Image.Image:
    rows = s.strip("\n").split("\n")
    w = max(len(r) for r in rows)
    im = Image.new("RGBA", (w, len(rows)), (0, 0, 0, 0))
    T._put(im, rows, 0, 0)
    return im


def crop(im: Image.Image) -> Image.Image:
    return im.crop(im.getbbox())


def silhouette(im: Image.Image, colour=(232, 228, 220)) -> Image.Image:
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    px, po = im.load(), out.load()
    for y in range(im.height):
        for x in range(im.width):
            if px[x, y][3]:
                po[x, y] = colour + (255,)
    return out


def up(im, z):
    return im.resize((im.width * z, im.height * z), Image.NEAREST)


def concepts():
    gecko = crop(T.pose())
    cards = [
        ("A  Tercet", "a gecko with an angular tail", gecko,
         ["Low body, big head, a ridge and spines.", "Tail: three straight segments,", "three teal hits (a triplet).",
          "Climbs anything; reads at 18 px.", "CHOSEN"]),
        ("B  Hop", "an upright jerboa-like hopper", crop(from_rows(HOP)),
         ["Upright, huge hind foot, ears.", "Kinked tail with two hits.", "Expressive hops; taller than", "wide, so it fights the banner.",
          ""]),
        ("C  Helix", "an arched, coil-tailed walker", crop(from_rows(HELIX)),
         ["Arched back, casque, slow.", "Tail coils into a helix.", "Strong idea, but the coil", "blurs below 20 px.", ""]),
    ]
    cw, ch = 380, 470
    sheet = Image.new("RGBA", (cw * 3 + 40, ch + 150), BG + (255,))
    d = ImageDraw.Draw(sheet)
    d.text((20, 18), "Canto mascot concepts", font=font(26, True), fill=INK)
    d.text((20, 54), "three silhouettes, drawn on the pixel grid; shown at 8x, as a silhouette, and at the banner's 5x",
            font=font(14), fill=MUTED)
    for k, (name, sub, im, notes) in enumerate(cards):
        x0, y0 = 20 + k * cw, 90
        if notes[-1] == "CHOSEN":
            d.rounded_rectangle((x0 - 6, y0 - 6, x0 + cw - 26, y0 + ch - 6), radius=10, outline=ACCENT, width=3)
        d.text((x0 + 8, y0 + 4), name, font=font(20, True), fill=INK)
        d.text((x0 + 8, y0 + 32), sub, font=font(13), fill=MUTED)
        big = up(im, 8)
        sheet.alpha_composite(big, (x0 + (cw - 40 - big.width) // 2, y0 + 64 + (170 - big.height) // 2))
        sil = up(silhouette(im), 4)
        sheet.alpha_composite(sil, (x0 + 30, y0 + 250 + (80 - sil.height) // 2))
        small = up(im, 5)
        sheet.alpha_composite(small, (x0 + 200, y0 + 250 + (80 - small.height) // 2))
        for j, line in enumerate(notes):
            col = ACCENT if line == "CHOSEN" else MUTED
            d.text((x0 + 8, y0 + 344 + 20 * j), line, font=font(14, line == "CHOSEN"), fill=col)
    # the chosen palette
    for j, key in enumerate("kdsmleph"):
        x = 20 + j * 34
        d.rectangle((x, ch + 106, x + 26, ch + 132), fill=T.PAL[key])
    d.text((20 + 8 * 34 + 10, ch + 110), "Tercet palette: Claude orange #D97757, with teal hits", font=font(14), fill=MUTED)
    sheet.convert("RGB").save(os.path.join(OUT, "canto_mascot_concepts.png"))


def mascot(z=24):
    for name, im in (("canto_mascot.png", T.pose(eyes="open", tail="up")), ("canto_mascot_front.png", T.front())):
        im = crop(im)
        pad = 2
        canvas = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
        canvas.alpha_composite(im, (pad, pad))
        up(canvas, z).save(os.path.join(OUT, name))


SPRITES = [   # name, pose kwargs, rotation ("front": the front view)
    ("idle", dict(), 0), ("blink", dict(eyes="blink"), 0), ("walk_1", dict(legs="walk1"), 0),
    ("walk_2", dict(legs="stand", body_dy=-1), 0), ("walk_3", dict(legs="walk2"), 0),
    ("drag", dict(eyes="squint", legs="wide", head_dy=1), 0), ("look_up", dict(eyes="up", head_dy=-2, body_dy=1), 0),
    ("with_sack", dict(sack=5), 0), ("oof", dict(eyes="squint", legs="wide", head_dy=1, tail="out", sack=3), 0),
    ("lift", dict(eyes="squint", head_dy=-2, body_dy=-1, legs="wide", tail="out"), 0),
    ("stretch", dict(eyes="squint", head_dx=4, legs="tiptoe", body_dy=-1, tail="out"), 0),
    ("teeter", dict(eyes="wide", legs="teeter", tail="high", head_dy=2), 0), ("happy", dict(eyes="happy", tail="wag"), 0),
    ("inspect", dict(eyes="down", head_dy=1), 0), ("glow", dict(eyes="down", tail_glow=True), 0),
    ("sit", dict(legs="sit", tail="curl", body_dy=1, head_dy=-1), 0),
    ("climb_1", dict(legs="walk1", sack=5, eyes="up"), 90), ("climb_2", dict(legs="walk2", sack=5, eyes="up"), 90),
    ("front", dict(), "front"), ("front_happy", dict(eyes="happy"), "front"),
    ("front_wave_1", dict(eyes="happy", arm=1), "front"), ("front_wave_2", dict(eyes="happy", arm=2), "front"),
    ("front_blink", dict(eyes="blink"), "front"), ("front_glow", dict(glow=1), "front"),
]


def spritesheet(z=6, cell=24, cols=6):
    rows = (len(SPRITES) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * cell), (0, 0, 0, 0))
    meta = {"cell": cell, "scale": z, "units": "cell rects in native pixels; multiply by scale for the PNG",
            "facing": "right; climb_* face up; front_* face the viewer", "frames": {}}
    for n, (name, kw, rot) in enumerate(SPRITES):
        im = T.front(**kw) if rot == "front" else T.pose(**kw)
        if rot and rot != "front":
            im = im.rotate(rot, expand=True)
        im = crop(im)
        cx, cy = (n % cols) * cell, (n // cols) * cell
        x, y = cx + (cell - im.width) // 2, cy + cell - 2 - im.height
        sheet.alpha_composite(im, (x, y))
        meta["frames"][name] = {"cell": [cx, cy, cell, cell], "rect_px": [cx * z, cy * z, cell * z, cell * z],
                                "pose": kw, "view": "front" if rot == "front" else "side", "rotate": 0 if rot == "front" else rot}
    up(sheet, z).save(os.path.join(OUT, "canto_mascot_spritesheet.png"))
    json.dump(meta, open(os.path.join(OUT, "canto_mascot_spritesheet.json"), "w"), indent=1)


if __name__ == "__main__":
    concepts()
    mascot()
    spritesheet()
    print("written to", os.path.abspath(OUT))
