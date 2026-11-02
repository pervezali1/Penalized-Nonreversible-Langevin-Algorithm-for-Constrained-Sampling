"""Soundtrack for the least-squares reel: Kokoro narration (voicekit) over the gentle bed of reels 38-43: a soft
chord; a pluck for each arrow (y, then (1,1,1) and x); a soft rising glide as the plane appears; a tick for each
line tried; a falling glide into the shadow and a chord when it lands; bells at the headings. Circular over the reel."""

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
from scene import FPS, N_FRAMES, SEG, SETTLE, T_END, TRIES, TRY, VOICE_AT, Scene, sse  # noqa: E402

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
    for t0, f in ((SEG[2] + 0.9, 329.63), (SEG[3] + 1.0, 261.63), (SEG[3] + 2.6, 392.0)):
        place(mix, A.pluck(f, 0.7, 0.0) * 0.7, t0)
    place(mix, glide(SEG[3] + 4.2, SEG[3] + 5.4, 196.0, 293.66, level=-33), SEG[3] + 4.2)
    n = len(TRIES)
    for i, (a, b) in enumerate(TRIES):
        ti = TRY[0] + (TRY[1] - TRY[0]) * (i / (n - 0.4))
        place(mix, A.pluck(440.0 * 2 ** (-0.25 * sse(a, b)), 0.5, -0.3 + 0.2 * i) * 0.5, ti)
    place(mix, glide(*SETTLE, 392.0, 261.63), SETTLE[0])
    for i, f in enumerate((261.63, 329.63, 392.0)):
        place(mix, A.pluck(f, 0.6, -0.4 + 0.4 * i) * 0.6, SETTLE[1] + 0.1 * i)
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
