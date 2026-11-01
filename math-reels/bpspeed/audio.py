"""Soundtrack for the backpropagation-speed reel: Kokoro narration (voicekit) over the gentle bed of reels 23-40: a soft
chord; during the nudging a tick for each pair of runs (a few slow ones, then a rising blur); the two backpropagation
passes as a rising and a falling glide; a chime when the slopes match; a long low tone under the hours-long bar and
a short bright one under the backpropagation bar; bells at the headings. Circular over the reel."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import voicekit  # noqa: E402
from scene import BP_BWD, BP_FWD, FPS, LOOP_FADE, N_FRAMES, NP, NUDGE, SEG, VOICE_AT, Scene, nudge_index  # noqa: E402

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


def tick(rng, level):
    n = int(0.04 * SR)
    t = np.arange(n) / SR
    v = sosfilt(butter(2, [1500, 6000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.005)
    return v / (np.max(np.abs(v)) + 1e-9) * db(level)


def glide(t0, t1, f0, f1, level=-28):
    tt = np.arange(int((t1 - t0 + 0.4) * SR)) / SR
    fr = f0 * (f1 / f0) ** np.clip(tt / (t1 - t0), 0, 1)
    return np.sin(np.cumsum(2 * math.pi * fr / SR)) * np.minimum(1, tt / 0.1) * np.minimum(1, (tt[-1] - tt) / 0.3) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(41)
    mix = np.zeros((NS, 2))
    env = np.minimum(1, np.maximum(0, (TT - 1.0) / 2.5)) * (1 - np.clip((TT - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]), 0, 1))
    for base, pan, lv in ((110.0, -0.4, -34), (164.81, 0.1, -36), (220.0, -0.1, -38), (277.18, 0.4, -40)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # one tick per nudged number (two runs each), found by scanning the nudge clock
    ts = np.arange(NUDGE[0], NUDGE[1], 1 / 600)
    ks = np.array([nudge_index(t) for t in ts])
    for k in range(1, NP + 1):
        i = int(np.searchsorted(ks, k))
        if i < len(ts):
            place(mix, tick(rng, -34 + 6 * k / NP), float(ts[i]))
    place(mix, glide(BP_FWD[0], BP_FWD[1], 262, 523), BP_FWD[0])
    place(mix, glide(BP_BWD[0], BP_BWD[1], 523, 262), BP_BWD[0])
    for k, fr in enumerate((659.25, 783.99, 987.77)):
        place(mix, A.pluck(fr, 0.6, -0.2 + 0.2 * k) * 0.6, SEG[5] + 0.3 + 0.08 * k)
    place(mix, glide(SEG[6] + 0.6, SEG[6] + 3.6, 98, 98, -26), SEG[6] + 0.6)
    place(mix, A.pluck(1046.5, 0.8, 0.0) * 0.8, SEG[7] + 0.4)
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
