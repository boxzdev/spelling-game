#!/usr/bin/env python3
"""
Generate an audio file for every word in words.json.

Run this on your computer (not on GitHub). It creates:
    audio/kokoro/<word>.wav
    audio/piper/<word>.wav
    audio/edge/<voice>/<word>.mp3
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

--- Edge neural voices (free, very natural, needs internet while generating) ---
  pip install edge-tts
  python generate_audio.py --engine edge
  (makes 4 voices by default; choose your own with --edge-voices en-US-AriaNeural,en-US-GuyNeural)
  The game's "Edge neural" option plays these files, so generate them first.

--- Kokoro + Piper ---
  python generate_audio.py --engine both --piper-model en_US-lessac-medium

--- Everything ---
  python generate_audio.py --engine all --piper-model en_US-lessac-medium

Note: the "Kitten" voice in the game runs live in the browser, so it needs no files.

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


DEFAULT_EDGE_VOICES = "en-US-AriaNeural,en-US-GuyNeural,en-GB-SoniaNeural,en-GB-RyanNeural"


def make_edge(words, out_dir, voices, rate):
    try:
        import edge_tts
    except ImportError:
        sys.exit("edge-tts isn't installed. Run: pip install edge-tts")
    import asyncio

    async def run():
        total = len(words)
        for voice in voices:
            voice_dir = out_dir / voice
            voice_dir.mkdir(parents=True, exist_ok=True)
            for i, word in enumerate(words, 1):
                target = voice_dir / f"{word}.mp3"
                if target.exists() and target.stat().st_size > 0:
                    continue
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(word, voice, rate=rate).save(str(target))
                        if target.stat().st_size == 0:
                            raise RuntimeError("empty audio file")
                        break
                    except Exception as err:
                        target.unlink(missing_ok=True)
                        if attempt == 2:
                            sys.exit(f"Edge failed on '{word}' ({voice}): {err}")
                        await asyncio.sleep(2)
                print(f"[edge:{voice}] {i}/{total} {word}")

    asyncio.run(run())


def main():
    p = argparse.ArgumentParser(description="Generate word audio from words.json")
    p.add_argument("--engine", choices=["kokoro", "piper", "edge", "both", "all"], required=True)
    p.add_argument("--words", default="words.json", help="path to words.json")
    p.add_argument("--out", default="audio", help="output folder")
    p.add_argument("--kokoro-voice", default="af_heart", help="Kokoro voice name")
    p.add_argument("--speed", type=float, default=0.9, help="Kokoro speaking speed")
    p.add_argument("--piper-model", default=None, help="Piper voice name or .onnx path")
    p.add_argument("--edge-voices", default=DEFAULT_EDGE_VOICES, help="comma-separated Edge voice names")
    p.add_argument("--edge-rate", default="-10%", help="Edge speaking rate, e.g. -10%% or +0%%")
    p.add_argument("--limit", type=int, default=None, help="only do the first N words (for testing)")
    args = p.parse_args()

    words = load_words(args.words)
    if args.limit:
        words = words[: args.limit]
    print(f"{len(words)} words")

    out = Path(args.out)
    if args.engine in ("kokoro", "both", "all"):
        make_kokoro(words, out / "kokoro", args.kokoro_voice, args.speed)
    if args.engine in ("piper", "both", "all"):
        make_piper(words, out / "piper", args.piper_model)
    if args.engine in ("edge", "all"):
        voices = [v.strip() for v in args.edge_voices.split(",") if v.strip()]
        make_edge(words, out / "edge", voices, args.edge_rate)

    print("Done. Commit the audio/ folder next to index.html.")


if __name__ == "__main__":
    main()
