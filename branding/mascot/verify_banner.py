"""Checks of the delivered canto_banner.gif against the brief: size, loop, timing, crisp pixels, the word CANTO in the
hero frame (frame 0), the order in which the letters are hung, and identity with a fresh render.

    python verify_banner.py
"""

import os

from PIL import Image, ImageSequence

import canto_banner as B

GIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canto_banner.gif")


def main():
    im = Image.open(GIF)
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        durs.append(fr.info["duration"])
        frames.append(fr.convert("RGB"))
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    check("size 960x320", im.size == (960, 320), str(im.size))
    check("loops forever", im.info.get("loop") == 0, str(im.info.get("loop")))
    total = sum(durs) / 1000
    check("12-18 s", 12 <= total <= 18.0001, f"{total:.2f} s, {len(frames)} stored frames, {B.FRAME_MS} ms ticks")
    check("file < 8 MB", os.path.getsize(GIF) < 8e6, f"{os.path.getsize(GIF)} bytes")
    crisp = all(f.resize((192, 64), Image.NEAREST).resize((960, 320), Image.NEAREST).tobytes() == f.tobytes()
                for f in frames)
    check("every 5x5 block uniform", crisp)
    # fresh render, rotated to start on the hero frame, ticks expanded
    d = B.Director()
    states = d.story()
    order = list(range(d.hero_index, len(states))) + list(range(d.hero_index))
    ticks = [f for f, ms in zip(frames, durs) for _ in range(ms // B.FRAME_MS)]
    same = len(ticks) == len(order) and all(
        t.resize((192, 64), Image.NEAREST).tobytes() == B.render(states[k], k).convert("RGB").tobytes()
        for t, k in zip(ticks, order))
    check("GIF == fresh render of the story", same, f"{len(ticks)} ticks vs {len(order)} states")
    # the word in frame 0 (the hero frame): letter pixels exactly the five glyphs at their hooks
    f0 = frames[0].resize((192, 64), Image.NEAREST).load()
    lit = {B.LETTER_CORE[5], B.LETTER_EDGE[5]}
    expect = set()
    for i, ch in enumerate(B.LETTERS):
        cells, _ = B.glyph(ch)
        x0 = B.glyph_x0(ch, B.CX[i])
        expect |= {(x0 + x, B.HANG_TOP + y) for (x, y) in cells}
    got = {(x, y) for y in range(64) for x in range(192) if f0[x, y] in lit}
    check("frame 0 reads exactly CANTO", got == expect, f"{len(got)} letter pixels, {len(expect)} expected")
    hung_at = sorted((st_hung, i) for i, st_hung in states[-1]["hung"].items())
    check("hung in the order C, A, N, T, O", [B.LETTERS[i] for _, i in hung_at] == list("CANTO"),
          " ".join(f"{B.LETTERS[i]}@{f}" for f, i in hung_at))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
