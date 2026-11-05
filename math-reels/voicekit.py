"""Narration for the reels: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), by default the am_michael voice
(American English, male) at 1.05x. A reel defines SCRIPT (one line per segment); synth() voices the lines into
<workdir>/voice_lines and writes <reel>/narration_timing.json (committed), which the scene reads to size its
segments, so a scene never needs Kokoro installed. mix() lays the lines over the music, ducking it under each.

The model files are not in the repo (325 MB). Download them once and point KOKORO_DIR at their folder:
    curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    curl -L -o voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"""

import json
import math
import os

import numpy as np

VOICE, SPEED, LANG = "am_michael", 1.05, "en-us"
LEAD, TAIL = 0.25, 0.6            # a segment starts its line LEAD s in and leaves TAIL s after it ends

# Pronunciation fixes. Kokoro reads phonemes made by espeak-ng, which gets a few of our words wrong. Each fix is
# (name, the words in the text that trigger it (regex), espeak's phonemes, the right phonemes). A line that
# triggers no fix is voiced from its text exactly as before; a line that does is voiced from corrected phonemes.
PRONUNCIATION = [
    ("eigen", r"(?i)eigen", "aɪdʒən", "aɪɡən"),              # espeak: "EYE-jen" (soft g); it's "EYE-gun"
    ("ringed", r"(?i)\bringed\b", "ɹˈɪŋɡd", "ɹˈɪŋd"),        # espeak adds a hard g: "ring-gd"
    ("A-times", r"\bA times\b", "ɐ tˈaɪmz", "ˈeɪ tˈaɪmz"),   # the matrix A (the letter "ay"), not the article "uh"
    ("ReLU", r"\bReLU\b", "ɹˌiː ˌɛljˈuː", "ɹˈeɪluː"),       # espeak spells it out, "R-E-L-U"; it's said "RAY-loo"
]


def fixes_for(line):
    import re
    return [f for f in PRONUNCIATION if re.search(f[1], line)]


def pron_ids(script):
    """The fixes a script triggers, per line ([] for none): stored with the timing, so a new fix re-voices the line."""
    return [[f[0] for f in fixes_for(line)] for line in script]


def phonemes(line, tokenizer):
    """The phonemes Kokoro should read for line, with the pronunciation fixes applied (None: use the text as is)."""
    fx = fixes_for(line)
    if not fx:
        return None
    ps = tokenizer.phonemize(line, LANG)
    for name, _, wrong, right in fx:
        if wrong not in ps:
            raise ValueError(f"pronunciation fix {name!r}: {wrong!r} not in espeak's phonemes for {line!r}: {ps}")
        ps = ps.replace(wrong, right)
    return ps


def load_timing(path, script):
    """Durations (s) of the voiced lines, or None if they are missing or were made from a different script."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    if d.get("script") != script or d.get("voice") != VOICE or d.get("speed") != SPEED:
        return None
    if d.get("pronunciation", [[] for _ in script]) != pron_ids(script):
        return None
    return d["durations"]


def segments(seg_min, durs, first_lead=0.15):
    """Segment boundaries: each segment lasts at least seg_min[k], or long enough for its line plus lead and tail.
    Returns (boundaries [0, ..., end], line start times)."""
    seg = [0.0]
    for k, m in enumerate(seg_min):
        need = ((first_lead if k == 0 else LEAD) + durs[k] + TAIL) if durs else 0.0
        seg.append(seg[-1] + max(m, round(need, 2)))
    starts = [first_lead] + [seg[k] + LEAD for k in range(1, len(seg_min))]
    return seg, starts


def synth(script, workdir, timing_path):
    import soundfile as sf
    from kokoro_onnx import Kokoro
    kdir = os.environ.get("KOKORO_DIR", os.path.expanduser("~/.cache/kokoro"))
    k = Kokoro(os.path.join(kdir, "kokoro-v1.0.onnx"), os.path.join(kdir, "voices-v1.0.bin"))
    out = os.path.join(workdir, "voice_lines")
    os.makedirs(out, exist_ok=True)
    durs = []
    sr = 24000
    for i, line in enumerate(script):
        ps = phonemes(line, k.tokenizer)
        if ps is None:
            s, sr = k.create(line, voice=VOICE, speed=SPEED, lang=LANG)
        else:
            print(f"line {i}: fixed pronunciation -> {ps}")
            s, sr = k.create(ps, voice=VOICE, speed=SPEED, lang=LANG, is_phonemes=True)
        s = np.asarray(s, np.float32)
        idx = np.flatnonzero(np.abs(s) > 0.01)
        s = s[max(0, idx[0] - int(0.02 * sr)):idx[-1] + int(0.08 * sr)]
        sf.write(os.path.join(out, f"line{i}.wav"), s, sr)
        durs.append(round(len(s) / sr, 3))
        print(f"line {i}: {durs[-1]:.2f} s  {line}")
    with open(timing_path, "w") as fh:
        d = dict(engine="kokoro-onnx v1.0", voice=VOICE, speed=SPEED, lang=LANG, script=script, durations=durs, sample_rate=sr)
        if any(pron_ids(script)):
            d["pronunciation"] = pron_ids(script)
        json.dump(d, fh, indent=1)
    return durs


def mix(music, sr, workdir, starts, ns, duck_db=-10.0, music_lufs=-17.0, voice_lufs=-16.5):
    """music: (ns, 2) float array. Returns music ducked under the voice lines (placed at starts) plus the voice."""
    import pyloudnorm
    import soundfile as sf
    from scipy.signal import butter, resample_poly, sosfilt
    meter = pyloudnorm.Meter(sr)
    voice = np.zeros(ns)
    spans = []
    for i, t0 in enumerate(starts):
        v, sr_v = sf.read(os.path.join(workdir, "voice_lines", f"line{i}.wav"), dtype="float32")
        v = resample_poly(np.asarray(v, np.float64), sr, sr_v)
        v = sosfilt(butter(2, 80, "high", fs=sr, output="sos"), v)
        i0 = int(round(t0 * sr))
        voice[i0:i0 + len(v)] += v[:ns - i0]
        spans.append((t0, t0 + len(v) / sr))
    tt = np.arange(ns) / sr
    music = music * 10 ** ((music_lufs - meter.integrated_loudness(music)) / 20)
    voice *= 10 ** ((voice_lufs - meter.integrated_loudness(voice[np.abs(voice) > 1e-4])) / 20)
    g = np.ones(ns)
    for a, b in spans:
        k = np.clip(np.minimum((tt - (a - 0.25)) / 0.2, ((b + 0.35) - tt) / 0.3), 0, 1)
        g = np.minimum(g, 1 - (1 - 10 ** (duck_db / 20)) * k)
    return music * g[:, None] + voice[:, None]


def finish(out, sr, target=-16.0, ceiling=-1.0):
    """Loudness to target LUFS, a soft limiter for the true peak, and back up as far as the ceiling allows."""
    import pyloudnorm
    from scipy.signal import resample_poly
    meter = pyloudnorm.Meter(sr)

    def db(x):
        return 10 ** (x / 20)

    out = out * db(target - meter.integrated_loudness(out))
    for _ in range(3):
        if np.max(np.abs(resample_poly(out, 4, 1, axis=0))) <= db(ceiling):
            break
        out = np.tanh(out / db(ceiling) * 0.9) * db(ceiling) / 0.9 * 0.98
    loud = meter.integrated_loudness(out)
    tp = 20 * math.log10(np.max(np.abs(resample_poly(out, 4, 1, axis=0))))
    out = out * db(min(target - loud, ceiling - tp))
    loud = meter.integrated_loudness(out)
    tp = 20 * math.log10(np.max(np.abs(resample_poly(out, 4, 1, axis=0))))
    return out, loud, tp
