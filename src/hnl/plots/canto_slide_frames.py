"""Still frames of the Canto explainer video for the slides (figures/slides/video_frames).

The frames are drawn by the video's own scene code, from a SNAPSHOT of hnl.plots.canto_video (that module keeps
changing; the snapshot fixes the drawing the slides were built on). A snapshot directory (on /work, so that batch jobs
can read it) holds canto_video_snapshot.py, a copy of canto_video.py with its REPO line hard-coded (so that it finds
the facts sheet and the Noto Sans files of figures/video/fonts from anywhere), canto_narration.json, the narration
file it reads (next to it), and SOURCE.json (the md5 and modification time of both sources). --make-snapshot DIR
writes one from the current files.

Each frame shows one scene at one time WITHOUT the subtitles and WITHOUT the scene title, reference and footer: a
slide sets its own title, reference and footer. It is drawn at 2x (200 dpi) and cropped to the drawn area of its
GROUP, the slides that build on one picture (e.g. the three pipeline stages): the union over the group's frames of
the pixels that differ from the background, plus a 12 px margin. The frames of a group share one crop box, so every
element keeps its place from slide to slide. crops.json in OUTDIR records each file's group and box (x0, y0, x1, y1
in 1920 x 1080 frame px, origin at the top left); the deck builder scales each group to fit the slide's content area
under the title. Times are chosen where an animation step is complete (no element half faded in or out);
--check-static verifies it: a frame must equal, pixel for pixel, the frames 0.1 s before and after it.

    python -m hnl.plots.canto_slide_frames                          # every frame the slides use (FRAMES)
    python -m hnl.plots.canto_slide_frames --only problem,results   # those groups' frames (crops of these groups)
    python -m hnl.plots.canto_slide_frames --check-static [--only ...]
    python -m hnl.plots.canto_slide_frames --windows frames         # static windows (from the snapshot's scene_motion)
    python -m hnl.plots.canto_slide_frames --scene problem --times 6.2,26.0 [--no-text]   # full frames, 1x
    python -m hnl.plots.canto_slide_frames --list                   # the scene names and durations
    python -m hnl.plots.canto_slide_frames --make-snapshot DIR      # copy canto_video.py + its narration to DIR

Output: OUTDIR/<scene>_<t>.png with t to one decimal (e.g. problem_26.0.png); the cover's triplet chain (no text,
fixed crop) is title_5.0_chain.png.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = Path("/work/submit/jinboz1/hnl/tmp/canto_slides_20261004/video_snapshot")
OUTDIR = REPO / "figures" / "slides" / "video_frames"
DPI = 200                      # 2x: the frames are 3840 x 2160 before cropping
MARGIN = 12                    # crop margin around the drawn area (frame px)
THRESH = 24                    # a pixel is drawn when its |RGB - background| sum exceeds this

# (scene, time in s, group, options): the frames of the deck, in deck order. A group's frames share one crop box.
# Options: "art" = no text at all; "crop" = a fixed (x0, y0, x1, y1) box in frame px instead of the group's;
# "tag" names the file suffix.
FRAMES = [
    ("title", 5.0, "cover", {"art": True, "crop": (670, 270, 1250, 395), "tag": "chain"}),  # the triplet chain
    ("problem", 6.2, "problem", {}),          # prompt tracks drawn, before they dim
    ("problem", 26.0, "problem", {}),         # displaced decay, no inner hits, d0, benchmark card
    ("pipeline", 10.0, "pipeline", {}),       # 1: graph construction (pill 1 active)
    ("pipeline", 15.0, "pipeline", {}),       # 2: edge scores (pill 2 active)
    ("pipeline", 21.0, "pipeline", {}),       # 3: tracks (pill 3 active)
    ("dezoort", 9.6, "dezoort", {}),          # the model and its parameter count
    ("dezoort", 17.0, "dezoort", {}),         # + the fake-rate gauge
    ("acorn", 18.4, "acorn", {}),             # the four stages
    ("acorn", 24.0, "acorn", {}),             # + paper vs reproduction
    ("failure", 6.6, "failure", {}),          # panel 1 (panel 2 fades in while panel 3 starts: no clean 1-2 frame)
    ("failure", 25.5, "failure", {}),         # all four panels
    ("canto", 17.0, "canto", {}),             # tercets, chained triplets, three stages
    ("classifier", 18.6, "classifier", {}),   # triplet messages, invariance, truncation
    ("classifier", 32.0, "ladder", {}),       # the symmetry ladder with the Canto row marked
    ("branches", 6.2, "branches", {}),        # Canto-B: a displaced edge is missing
    ("branches", 11.2, "branches", {}),       # Canto-H: the hand cuts keep it, but not the low-pT edge
    ("branches", 17.6, "branches", {}),       # Canto-L: the learned support keeps both
    ("branches", 39.6, "size", {}),           # size control: ceiling vs pairs, +53 % arrow
    ("trackbuild", 8.2, "trackbuild", {}),    # MWS-layer: the same-layer repulsion rejects the fake edge
    ("trackbuild", 16.3, "trackbuild", {}),   # + two-pass and fragment merging (static from 16.2 to 16.4)
    ("trackbuild", 32.0, "builder-numbers", {}),  # numbers: acorn and Canto-B with each builder
    ("results", 29.4, "results", {}),         # every bar
    ("results", 36.0, "results", {}),         # reference acorn vs the Canto-L preview
    ("results", 45.4, "results", {}),         # component decomposition
]
# The table, outlook and end scenes are text only: their slides set the same layout as live text instead.


def file_name(scene: str, t: float, opt: dict) -> str:
    return f"{scene}_{t:.1f}" + (f"_{opt['tag']}" if opt.get("tag") else "") + ".png"


def load_snapshot(snapshot_dir: Path = SNAPSHOT_DIR):
    """Import the snapshot as a module and make its scenes draw no title, reference, footer or subtitle."""
    path = Path(snapshot_dir) / "canto_video_snapshot.py"
    spec = importlib.util.spec_from_file_location("canto_video_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    class FrameScene(mod.Scene):
        def __init__(self, name, duration, title=None, accent=mod.ORANGE, ref=None, footer=None):
            super().__init__(name, duration, title=None, accent=accent, ref=None, footer=None)

        def narrate(self, lines):  # the subtitles are the narration: they go into the speaker notes
            return None

    mod.Scene = FrameScene                # the scene functions look the class up in the module at call time
    return mod


def draw(mod, sc, t: float, art: bool = False, dpi: int = DPI) -> np.ndarray:
    """The scene at time t as an RGB array (full frame, at dpi)."""
    from PIL import Image
    if not 0.5 <= t <= sc.duration - 0.6:
        raise SystemExit(f"{sc.name} t={t}: inside the scene's fade in or out (duration {sc.duration} s)")
    sc.update(t)
    if art:
        for r in sc.riches:
            r.set_alpha(0.0)
    buf = io.BytesIO()
    sc.fig.savefig(buf, dpi=dpi, facecolor=mod.BG, format="png")
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def drawn_box(img: np.ndarray, bg: str, scale: float) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) in frame px of the pixels that differ from the background colour bg."""
    rgb = np.array([int(bg[i:i + 2], 16) for i in (1, 3, 5)])
    ys, xs = np.nonzero(np.abs(img.astype(int) - rgb).sum(axis=2) > THRESH)
    if len(xs) == 0:
        return None
    return (int(np.floor(xs.min() / scale)), int(np.floor(ys.min() / scale)), int(np.ceil((xs.max() + 1) / scale)),
            int(np.ceil((ys.max() + 1) / scale)))


def render_frames(mod, frames, outdir: Path, dpi: int = DPI, merge: bool = False) -> dict:
    """Draw `frames` (entries of FRAMES), crop every group to its union box and write the PNGs and crops.json (merged
    into the existing crops.json with merge=True, i.e. when only some groups are drawn)."""
    from PIL import Image
    outdir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 100.0
    imgs, boxes = {}, {}
    for scene in dict.fromkeys(f[0] for f in frames):           # one Scene object per scene, in order
        sc = dict(mod.SCENES)[scene]()
        for sc_name, t, group, opt in frames:
            if sc_name != scene:
                continue
            img = draw(mod, sc, t, opt.get("art", False), dpi)
            name = file_name(scene, t, opt)
            imgs[name] = (img, group, opt)
            if "crop" not in opt:
                b = drawn_box(img, mod.BG, scale)
                if b is None:
                    raise SystemExit(f"{name}: nothing drawn")
                g = boxes.get(group)
                boxes[group] = b if g is None else (min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]),
                                                    max(g[3], b[3]))
        import matplotlib.pyplot as plt
        plt.close(sc.fig)
    crops_path = outdir / "crops.json"
    crops = json.loads(crops_path.read_text()) if merge and crops_path.exists() else {}
    for name, (img, group, opt) in imgs.items():
        if "crop" in opt:
            box = tuple(opt["crop"])
        else:
            b = boxes[group]
            box = (max(0, b[0] - MARGIN), max(0, b[1] - MARGIN), min(mod.W, b[2] + MARGIN), min(mod.H, b[3] + MARGIN))
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        Image.fromarray(img[y0:y1, x0:x1]).save(outdir / name, optimize=True)
        crops[name] = {"group": group, "box": list(box), "scale": scale}
        print(outdir / name, box, flush=True)
    crops_path.write_text(json.dumps(dict(sorted(crops.items())), indent=1) + "\n")
    return crops


def check_static(mod, frames, dt: float = 0.1, dpi: int = 100) -> int:
    """Every frame must equal the frames dt before and after it, pixel for pixel: no element is moving or fading
    (the drawing is deterministic, so a static scene gives identical pixels)."""
    bad = 0
    for scene in dict.fromkeys(f[0] for f in frames):
        sc = dict(mod.SCENES)[scene]()
        for sc_name, t, group, opt in frames:
            if sc_name != scene:
                continue
            ref = draw(mod, sc, t, opt.get("art", False), dpi)
            for u in (t - dt, t + dt):
                d = np.abs(draw(mod, sc, u, opt.get("art", False), dpi).astype(int) - ref.astype(int)).sum(axis=2)
                n = int((d > 0).sum())
                print(f"{scene} t={t:.1f} vs {u:.2f}: {n} px differ" + ("" if n == 0 else "  <-- NOT STATIC"),
                      flush=True)
                bad += n > 0
        import matplotlib.pyplot as plt
        plt.close(sc.fig)
    print(f"{bad} comparison(s) not static")
    return bad


def static_windows(mod, scene: str, min_len: float = 0.15) -> list[tuple[float, float]]:
    """The static windows of a scene as drawn for the slides (no title, reference, footer or subtitles): the gaps
    between the transitions that the snapshot's scene_motion() reports (every on-screen change goes through prog)."""
    sc = dict(mod.SCENES)[scene]()
    iv = sorted((t0, t0 + max(d, 0.0)) for t0, d in mod.scene_motion(sc))
    import matplotlib.pyplot as plt
    plt.close(sc.fig)
    out, t = [], 0.0
    for a, b in iv:
        if a - t >= min_len:
            out.append((round(t, 3), round(a, 3)))
        t = max(t, b)
    if sc.duration - t >= min_len:
        out.append((round(t, 3), round(sc.duration, 3)))
    return out


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_snapshot(dest: Path) -> None:
    """Copy the current canto_video.py (REPO hard-coded) and canto_narration.json to dest, with SOURCE.json."""
    src = REPO / "src" / "hnl" / "plots"
    video, narr = src / "canto_video.py", src / "canto_narration.json"
    text = video.read_text(encoding="utf-8")
    old = "REPO = Path(__file__).resolve().parents[3]"
    if text.count(old) != 1:
        raise SystemExit(f"{video}: expected one line {old!r}")
    text = text.replace(old, f'REPO = Path("{REPO}")  # snapshot: hard-coded (was Path(__file__).resolve().parents[3])')
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "canto_video_snapshot.py").write_text(text, encoding="utf-8")
    shutil.copyfile(narr, dest / "canto_narration.json")
    info = {"copied_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": {str(p): {"md5": md5(p), "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                           time.gmtime(p.stat().st_mtime))}
                        for p in (video, narr)}}
    (dest / "SOURCE.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=str(SNAPSHOT_DIR), help="the snapshot directory to draw with")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--only", default="", help="comma-separated groups: render only their frames")
    ap.add_argument("--check-static", action="store_true", help="check that the frames are static (+-0.1 s)")
    ap.add_argument("--scene", default=None, help="one scene, with --times: full frames at 1x (instead of FRAMES)")
    ap.add_argument("--times", default="", help="comma-separated times in s")
    ap.add_argument("--no-text", action="store_true", help="leave out every text element too (with --scene)")
    ap.add_argument("--list", action="store_true", help="print the scenes and their durations")
    ap.add_argument("--windows", default=None, metavar="SCENES",
                    help="comma-separated scenes (or 'frames'): their static windows and the FRAMES times in them")
    ap.add_argument("--make-snapshot", default=None, metavar="DIR", help="copy the current video code to DIR")
    a = ap.parse_args(argv)
    if a.make_snapshot:
        make_snapshot(Path(a.make_snapshot))
        return
    mod = load_snapshot(Path(a.snapshot))
    mod.check_fonts()
    if a.list:
        for (name, _), d in zip(mod.SCENES, mod.DURATIONS):
            print(f"{name:<11s} {d:5.1f} s")
        return
    if a.windows:
        names = list(dict.fromkeys(f[0] for f in FRAMES)) if a.windows == "frames" else a.windows.split(",")
        for name in names:
            w = static_windows(mod, name)
            print(f"{name}: " + "  ".join(f"[{x:g}, {y:g}]" for x, y in w))
            for sc_name, t, group, _ in FRAMES:
                if sc_name == name:
                    inside = [x for x in w if x[0] <= t <= x[1]]
                    print(f"    frame {t:g} ({group}): " + (f"in [{inside[0][0]:g}, {inside[0][1]:g}]" if inside
                                                             else "NOT in a static window"))
        return
    outdir = Path(a.outdir)
    if a.scene:
        from PIL import Image
        outdir.mkdir(parents=True, exist_ok=True)
        sc = dict(mod.SCENES)[a.scene]()
        for t in (float(v) for v in a.times.split(",")):
            p = outdir / f"{a.scene}_{t:.1f}{'_art' if a.no_text else ''}_full.png"
            Image.fromarray(draw(mod, sc, t, a.no_text, 100)).save(p)
            print(p)
        return
    only = {s for s in a.only.split(",") if s}
    frames = [f for f in FRAMES if not only or f[2] in only]
    if a.check_static:
        sys.exit(1 if check_static(mod, frames) else 0)
    render_frames(mod, frames, outdir, merge=bool(only))


if __name__ == "__main__":
    main()
