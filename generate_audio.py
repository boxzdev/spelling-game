#!/usr/bin/env python3
"""
Generate an audio file for every word in words.json.

Run this on your computer (not on GitHub). It creates:
    audio/kokoro/<word>.wav
    audio/piper/<word>.wav
Then commit the audio/ folder next to index.html.

--- Kokoro (most natural) ---
  pip install kokoro soundfile numpy
  (also install espeak-ng: https://github.com/espeak-ng/espeak-ng)
  python generate_audio.py --engine kokoro

--- Piper (fast, light) ---
  pip install piper-tts
  python -m piper.download_voices en_US-lessac-medium
  python generate_audio.py --engine piper --piper-model en_US-lessac-medium
  (older Piper: pass the path to the .onnx file instead of the voice name)

--- Both ---
  python generate_audio.py --engine both --piper-model en_US-lessac-medium

Tip: test with a few words first:  --limit 5
Files that already exist are skipped, so it's safe to re-run after adding words.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def load_words(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words = []
    for level in ("easy", "normal", "hard"):
        words.extend(w.strip().lower() for w in data.get(level, []) if w.strip())
    return list(dict.fromkeys(words))  # remove duplicates, keep order


def make_kokoro(words, out_dir, voice, speed):
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError:
        sys.exit("Kokoro isn't installed. Run: pip install kokoro soundfile numpy "
                 "(and install espeak-ng)")

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")  # "a" = American English
    total = len(words)

    for i, word in enumerate(words, 1):
        target = out_dir / f"{word}.wav"
        if target.exists():
            continue
        chunks = [np.asarray(audio) for _, _, audio in pipeline(word, voice=voice, speed=speed)]
        if not chunks:
            print(f"[kokoro] {i}/{total} {word}: no audio produced, skipped")
            continue
        sf.write(str(target), np.concatenate(chunks), 24000)
        print(f"[kokoro] {i}/{total} {word}")


def make_piper(words, out_dir, model):
    if not model:
        sys.exit("Piper needs a voice. Example: --piper-model en_US-lessac-medium")

    exe = shutil.which("piper")
    base = [exe] if exe else [sys.executable, "-m", "piper"]
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(words)
    flags = ["--output_file", "--output-file", "-f"]  # differs between Piper versions
    working_flag = None

    for i, word in enumerate(words, 1):
        target = out_dir / f"{word}.wav"
        if target.exists():
            continue

        last_error = ""
        for flag in ([working_flag] if working_flag else flags):
            result = subprocess.run(
                base + ["--model", model, flag, str(target)],
                input=word, text=True, capture_output=True,
            )
            if result.returncode == 0 and target.exists():
                working_flag = flag
                break
            last_error = result.stderr.strip()
        else:
            sys.exit(f"Piper failed on '{word}':\n{last_error}")

        print(f"[piper] {i}/{total} {word}")


def main():
    p = argparse.ArgumentParser(description="Generate word audio from words.json")
    p.add_argument("--engine", choices=["kokoro", "piper", "both"], required=True)
    p.add_argument("--words", default="words.json", help="path to words.json")
    p.add_argument("--out", default="audio", help="output folder")
    p.add_argument("--kokoro-voice", default="af_heart", help="Kokoro voice name")
    p.add_argument("--speed", type=float, default=0.9, help="Kokoro speaking speed")
    p.add_argument("--piper-model", default=None, help="Piper voice name or .onnx path")
    p.add_argument("--limit", type=int, default=None, help="only do the first N words (for testing)")
    args = p.parse_args()

    words = load_words(args.words)
    if args.limit:
        words = words[: args.limit]
    print(f"{len(words)} words")

    out = Path(args.out)
    if args.engine in ("kokoro", "both"):
        make_kokoro(words, out / "kokoro", args.kokoro_voice, args.speed)
    if args.engine in ("piper", "both"):
        make_piper(words, out / "piper", args.piper_model)

    print("Done. Commit the audio/ folder next to index.html.")


if __name__ == "__main__":
    main()
