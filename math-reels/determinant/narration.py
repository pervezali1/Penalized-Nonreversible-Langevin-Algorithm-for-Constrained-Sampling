"""Narration for the determinant reel: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), voice am_michael
(American English, male) at 1.05x speed. One line per segment of the reel. Running this file synthesises the
lines into <workdir>/voice_lines/ and writes narration_timing.json next to it (committed), which the scene
reads to size each segment so its line fits; the scene therefore never needs Kokoro to be installed.

The model files are not in the repo (325 MB). Download them once:
    curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    curl -L -o voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
and point KOKORO_DIR at the folder holding them.        python narration.py <workdir>"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIMING = os.path.join(HERE, "narration_timing.json")
VOICE, SPEED, LANG = "am_michael", 1.05, "en-us"

# one line per segment: 0 hook, 1 intro, 2 stretch, 3 shear, 4 general, 5 flatten, 6 flat, 7 mirror, 8 close.
# numbers are spelled out so the voice reads them the way the card shows them
SCRIPT = [
    "The determinant is a volume.",
    "A three by three matrix moves the three edges of a cube. Its columns are where those edges land.",
    "Stretch it by two, one, and one and a half. The volume grows by exactly that: three.",
    "Now slant it. Same base, same height, so the volume stays three.",
    "Any matrix at all: its volume is the determinant. Here, two point two three. A million random points measure the same.",
    "Tip the third edge into the plane of the other two, and the volume drains away.",
    "Determinant zero. Crushed flat, with no way back. No inverse.",
    "Keep going, and it comes out the other side: the mirror image. Same volume, but the determinant is minus two point two three.",
    "So the determinant is how much a matrix scales volume. Zero means flattened. Negative means flipped.",
]


def load_timing():
    """Durations (s) of the synthesised lines, or None if the narration has not been made."""
    if not os.path.exists(TIMING):
        return None
    with open(TIMING) as fh:
        d = json.load(fh)
    if d.get("script") != SCRIPT or d.get("voice") != VOICE or d.get("speed") != SPEED:
        return None
    return d["durations"]


def synth(workdir):
    """Synthesise every line (24 kHz mono float32 wavs in workdir/voice_lines) and write narration_timing.json."""
    import soundfile as sf
    from kokoro_onnx import Kokoro
    kdir = os.environ.get("KOKORO_DIR", os.path.expanduser("~/.cache/kokoro"))
    k = Kokoro(os.path.join(kdir, "kokoro-v1.0.onnx"), os.path.join(kdir, "voices-v1.0.bin"))
    out = os.path.join(workdir, "voice_lines")
    os.makedirs(out, exist_ok=True)
    durs = []
    for i, line in enumerate(SCRIPT):
        s, sr = k.create(line, voice=VOICE, speed=SPEED, lang=LANG)
        s = np.asarray(s, np.float32)
        # trim the model's leading and trailing near-silence so the timing is the speech itself
        idx = np.flatnonzero(np.abs(s) > 0.01)
        s = s[max(0, idx[0] - int(0.02 * sr)):idx[-1] + int(0.08 * sr)]
        sf.write(os.path.join(out, f"line{i}.wav"), s, sr)
        durs.append(round(len(s) / sr, 3))
        print(f"line {i}: {durs[-1]:.2f} s  {line}")
    with open(TIMING, "w") as fh:
        json.dump(dict(engine="kokoro-onnx v1.0", voice=VOICE, speed=SPEED, lang=LANG, script=SCRIPT, durations=durs, sample_rate=sr), fh, indent=1)
    return durs


def lines(workdir):
    """The synthesised lines as (array, sample rate) pairs, read from workdir/voice_lines."""
    import soundfile as sf
    res = []
    for i in range(len(SCRIPT)):
        s, sr = sf.read(os.path.join(workdir, "voice_lines", f"line{i}.wav"), dtype="float32")
        res.append((s, sr))
    return res


if __name__ == "__main__":
    synth(os.path.abspath(sys.argv[1]))
