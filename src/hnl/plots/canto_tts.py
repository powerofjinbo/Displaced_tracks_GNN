"""Offline English narration for the Canto explainer video: Kokoro-82M text-to-speech on CPU.

Runs in its own micromamba env (NOT the training venv) and imports nothing from the hnl package:

    env    /work/submit/jinboz1/hnl/env/mamba/envs/tts
           python 3.11, torch 2.14.1+cpu, kokoro 0.9.4, misaki[en] 0.9.4 (G2P; espeak-ng 1.52 for out-of-vocabulary
           words comes from the espeakng-loader wheel), spaCy 3.8 + en_core_web_sm 3.8.0, soundfile 0.14
    model  /work/submit/jinboz1/hnl/cache/tts/Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0, HF revision
           f3ff3571791e39611d31c381e3a41a3af07b4987: config.json, kokoro-v1_0.pth, voices/{a,b}{f,m}_*.pt);
           other location: --model-dir or $CANTO_TTS_MODEL_DIR. Nothing is downloaded at run time (HF_HUB_OFFLINE=1).

    source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh
    /work/submit/jinboz1/hnl/env/mamba/envs/tts/bin/python \\
        /work/submit/jinboz1/hnl/hnl-displaced/src/hnl/plots/canto_tts.py \\
        --in LINES.json --out-dir DIR [--voice af_heart] [--speed 1.0] [--threads 8]

LINES.json is a list of {"id": str, "text": str} (id: letters, digits, '.', '_', '-'). Writes DIR/<id>.wav (24 kHz
mono 16-bit PCM), DIR/durations.json {id: seconds}, DIR/manifest.json {id: sha1(text, voice, speed)} and
DIR/phonemes.json {id: phonemes actually spoken}. A line whose wav exists and whose manifest hash matches is skipped,
so editing one line re-synthesises only that line. Entries of ids absent from LINES.json are kept (several scripts may
share one DIR). Each line is synthesised with a fixed torch seed (--seed): a re-run with the same --threads reproduces
the wav bit for bit (checked across the login node and a compute node).
Speed with 8 threads: 3.8x real time on a compute node (Xeon Silver 4310), 3.3x on the submit01 login node, plus
15-50 s of start-up; ~2.3 GB RAM. A 5-minute script takes ~2 min: run it through sbatch (-p submit -c 8 --mem=8G).

Voices: af_heart (American, female; the only grade-A voice), af_bella, am_michael, am_fenrir, am_puck (American);
bf_emma, bm_george, bm_fable (British); the first letter picks the G2P accent. "af_heart,af_bella" averages voices.

Writing text for the ear (checked on misaki's phonemes and with a Whisper transcription of the audio):
- Decimals and percentages may stay as digits: "99.5 %" and "ninety-nine point five percent" are read alike (only
  the stress on "point" differs). A sign breaks a decimal: "+2.4" is read "plus two four" and "−10.2" (Unicode
  minus) "minus ten two"; write "plus two point four" (ASCII "-10.2" works). "+-" is read "plus": write "plus or minus".
- Four-digit integers are read as years ("2999" -> "twenty-nine ninety-nine"): write "2,999". "10k" is "ten kay".
- Spell out units and symbols: "mm" becomes a bare "m" sound (write "millimeters"), "GeV" is read "G V" (write
  "G-e-V"), "μm" "myoom"; "17/19" is read "seventeen nineteen" (write "17 of 19"), "FR*" "F R asterisk" (write
  "F R star"), arrows are dropped, ">=" is "greater or equal to" (write "at least"). Write "for example", "that is",
  "versus" instead of e.g., i.e., vs.
- Works as written: "t t-bar" or "ttbar" (tee tee-bar), "d0" or "d zero" (dee zero; avoid "|d0|"), "pT", "GNN",
  "HNL", "acorn" (ay-corn), "Canto", "3 sigma", "2026". Write "Canto B", not "Canto-B" (read as one word).
- Force a pronunciation with misaki's link syntax: "[Exa](/ˈɛksə/) track X" (Exa.TrkX is garbled otherwise).
- Keep a line to one to three sentences (under ~400 phonemes; see phonemes.json): Kokoro rushes on long inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

DEFAULT_MODEL_DIR = Path(os.environ.get("CANTO_TTS_MODEL_DIR", "/work/submit/jinboz1/hnl/cache/tts/Kokoro-82M"))
REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RUSH_PHONEMES = 400  # Kokoro's VOICES.md: quality drops ("rushing") above ~400 tokens per chunk


def line_hash(text: str, voice: str, speed: float) -> str:
    return hashlib.sha1(json.dumps([text, voice, float(speed)], ensure_ascii=False).encode("utf-8")).hexdigest()


def read_lines(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit(f"{path}: expected a JSON list of {{'id': str, 'text': str}}")
    lines, seen = [], set()
    for i, item in enumerate(data):
        if not (isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("text"), str)):
            sys.exit(f"{path}: entry {i} is not {{'id': str, 'text': str}}")
        lid, text = item["id"], item["text"].strip()
        if not ID_RE.fullmatch(lid):
            sys.exit(f"{path}: id {lid!r} is not a plain file name (letters, digits, '.', '_', '-')")
        if lid in seen:
            sys.exit(f"{path}: duplicate id {lid!r}")
        if not text:
            sys.exit(f"{path}: id {lid!r} has an empty text")
        seen.add(lid)
        lines.append((lid, text))
    return lines


def voice_files(model_dir: Path, voice: str) -> tuple[str, str]:
    """(lang_code, comma-joined .pt paths) for 'af_heart' or a mix such as 'af_heart,af_bella'."""
    names = voice.split(",")
    langs = {n[:1] for n in names}
    if len(langs) != 1 or not langs <= {"a", "b"}:
        sys.exit(f"--voice {voice}: English voices only (a* American, b* British), one accent per mix")
    paths = [model_dir / "voices" / f"{n}.pt" for n in names]
    for p in paths:
        if not p.is_file():
            have = sorted(q.stem for q in (model_dir / "voices").glob("*.pt"))
            sys.exit(f"voice file {p} not found; available: {' '.join(have)}")
    return langs.pop(), ",".join(str(p) for p in paths)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Engine:
    """Kokoro model + G2P pipeline, loaded only when some line needs synthesis (imports + load: 15-50 s on NFS)."""

    def __init__(self, model_dir: Path, lang: str, threads: int):
        os.environ["HF_HUB_OFFLINE"] = "1"       # all files are local; never touch the network
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        import torch
        from loguru import logger
        from kokoro import KModel, KPipeline

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        torch.set_num_threads(threads)
        self.torch = torch
        model = KModel(repo_id=REPO_ID, config=str(model_dir / "config.json"),
                       model=str(model_dir / "kokoro-v1_0.pth")).eval()
        # phonemizer copies libespeak-ng.so into tempfile.mkdtemp() and rmtree()s it at exit. With $TMPDIR on NFS
        # (/work) the still-mapped copy leaves a .nfs* file, rmtree fails and an empty tmp* dir leaks per run, so the
        # copy (0.7 MB, removed at exit) goes to node-local /tmp.
        if os.access("/tmp", os.W_OK):
            tempfile.tempdir = "/tmp"
        self.pipe = KPipeline(lang_code=lang, repo_id=REPO_ID, model=model)

    def synth(self, text: str, voice: str, speed: float, seed: int):
        torch = self.torch
        torch.manual_seed(seed)  # the iSTFTNet source adds random noise and phase
        audio, phonemes = [], []
        with torch.inference_mode():
            for r in self.pipe(text, voice=voice, speed=speed):
                if r.audio is None:
                    continue
                if len(r.phonemes) > RUSH_PHONEMES:
                    print(f"  warning: chunk of {len(r.phonemes)} phonemes may sound rushed; split the line",
                          file=sys.stderr)
                audio.append(r.audio.float().cpu())
                phonemes.append(r.phonemes)
        if not audio:
            sys.exit(f"no audio produced for text {text!r}")
        return torch.cat(audio).clamp(-1.0, 1.0).numpy(), " | ".join(phonemes)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="LINES.json: [{'id': str, 'text': str}, ...]")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice, e.g. af_heart, am_michael, bf_emma")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking-rate factor (1.0 = model default)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--threads", type=int, default=min(8, len(os.sched_getaffinity(0))))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not 0.5 <= args.speed <= 2.0:
        ap.error("--speed must be within [0.5, 2]")

    import soundfile as sf

    lines = read_lines(args.inp)
    lang, voice_pt = voice_files(args.model_dir, args.voice)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_json(out / "manifest.json")
    phonemes = load_json(out / "phonemes.json")

    engine, n_new, audio_s, compute_s = None, 0, 0.0, 0.0
    for lid, text in lines:
        key = line_hash(text, args.voice, args.speed)
        wav = out / f"{lid}.wav"
        if wav.is_file() and manifest.get(lid) == key:
            print(f"skip  {lid}")
            continue
        if engine is None:
            t0 = time.perf_counter()
            engine = Engine(args.model_dir, lang, args.threads)
            print(f"loaded Kokoro-82M ({args.voice}, lang '{lang}', {args.threads} threads) in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        t0 = time.perf_counter()
        audio, ps = engine.synth(text, voice_pt, args.speed, args.seed)
        dt = time.perf_counter() - t0
        tmp = wav.with_name(f".{wav.name}.tmp")
        sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        os.replace(tmp, wav)
        manifest[lid], phonemes[lid] = key, ps
        write_json(out / "manifest.json", manifest)
        write_json(out / "phonemes.json", phonemes)
        n_new, audio_s, compute_s = n_new + 1, audio_s + len(audio) / SAMPLE_RATE, compute_s + dt
        print(f"synth {lid}: {len(audio) / SAMPLE_RATE:.2f} s audio in {dt:.2f} s", flush=True)

    durations = {}
    for lid in [lid for lid, _ in lines] + sorted(manifest):
        wav = out / f"{lid}.wav"
        if lid not in durations and wav.is_file():
            info = sf.info(str(wav))
            durations[lid] = round(info.frames / info.samplerate, 3)
    write_json(out / "durations.json", durations)
    total = sum(durations[lid] for lid, _ in lines)
    print(f"{len(lines)} lines ({n_new} synthesised, {len(lines) - n_new} skipped), {total:.1f} s of narration "
          f"-> {out}")
    if n_new:
        print(f"synthesis speed: {audio_s:.1f} s audio / {compute_s:.1f} s compute = {audio_s / compute_s:.2f}x "
              "real time")


if __name__ == "__main__":
    main()
