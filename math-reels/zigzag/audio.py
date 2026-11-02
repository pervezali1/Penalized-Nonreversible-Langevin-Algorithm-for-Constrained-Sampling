"""Soundtrack for the zig-zag reel: Kokoro narration (voicekit) over the gentle bed of reels 38-44: a soft chord; a
glide as each bowl is stretched; a soft tick for every step of every run (thinned where steps come faster than 25 a
second), pitched up as the ball nears the bottom and panned with its side of the valley; a chord when each run
arrives; bells at the headings. Circular over the reel."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import voicekit  # noqa: E402
from scene import FPS, MORPH, N_FRAMES, PATH, RUN_T, SEG, STEPS, T_END, VOICE_AT, Scene, progress  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR


def db(x):
    return 10 ** (x / 20)


def integer_cycles(inc):
    C = inc.sum() / (2 * math.pi)
    return inc + 2 * math.pi * ((round(C) - C) / DUR) / SR


def place(buf, clip, t0):
    i0 = int(round(t0 * SR)) % NS
    idx = (i0 + np.arange(len(clip))) % NS
    if buf.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    np.add.at(buf, idx, clip)


def glide(t0, t1, f0, f1, level=-31):
    tt = np.arange(int((t1 - t0 + 0.6) * SR)) / SR
    fr = f0 * (f1 / f0) ** np.clip(tt / (t1 - t0), 0, 1)
    return np.sin(np.cumsum(2 * math.pi * fr / SR)) * np.minimum(1, tt / 0.3) * np.minimum(1, (tt[-1] - tt) / 0.5) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.minimum(1, np.maximum(0, (TT - 1.0) / 2.5)) * (1 - np.clip((TT - T_END) / 0.55, 0, 1))
    for base, pan, lv in ((110.0, -0.4, -34), (164.81, 0.1, -36), (220.0, -0.1, -38), (277.18, 0.4, -40)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    for t0, t1, k0, k1 in MORPH:
        place(mix, glide(t0, t1, 220.0, 220.0 * (1.5 if k1 > k0 else 0.75), level=-34), t0)
    for key in RUN_T:
        t0, t1 = RUN_T[key]
        ts = np.linspace(t0, t1, 4000)
        pr = np.array([progress(key, t) for t in ts])
        last = -1.0
        path = PATH[key]
        d0 = np.linalg.norm(path[0])
        for k in range(1, STEPS[key] + 1):
            tk = float(ts[np.argmin(np.abs(pr - k))])
            if tk - last < 0.04:
                continue
            last = tk
            closeness = 1 - min(1.0, np.linalg.norm(path[k]) / d0)
            pan = float(np.clip(path[k][1] / 2.0, -0.8, 0.8))
            place(mix, A.pluck(392.0 * 2 ** (1.2 * closeness), 0.35, pan) * 0.35, tk)
        for i, f in enumerate((261.63, 329.63, 392.0)):
            place(mix, A.pluck(f, 0.5, -0.4 + 0.4 * i) * 0.5, t1 + 0.05 + 0.08 * i)
    bell = A.bell_e6() * 0.4
    for c in sc.cues[1:-1]:
        place(mix, bell, c[0])
    out = A.circ_reverb(mix, rt60=3.0, wet=0.3)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    out = voicekit.mix(out, SR, os.path.dirname(os.path.abspath(out_wav)), VOICE_AT, NS)
    out, loud, tp = voicekit.finish(out, SR)
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
