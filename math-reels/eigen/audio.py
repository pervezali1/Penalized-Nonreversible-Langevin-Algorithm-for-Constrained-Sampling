"""Soundtrack for the eigenvector reel: Kokoro narration (voicekit) over a gentle bed in the register of reels
23-38: a soft chord; a rising glissando as the ball becomes the egg; three plucks as the eigenvectors appear (one
per eigenvalue, pitched by it); a tick for every whole application in the power steps, rising as the arrows
gather; bells at the headings. Circular over the reel."""

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
from scene import APPLY, FPS, K_MAX, LAM, LOOP_FADE, N_FRAMES, POWER, SEG, VOICE_AT, Scene, power_stats  # noqa: E402

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


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.minimum(1, np.maximum(0, (TT - 1.0) / 2.5)) * (1 - np.clip((TT - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]), 0, 1))
    env = np.maximum(env, np.clip((TT - LOOP_FADE[1]) / 0.2, 0, 1) * 0)
    for base, pan, lv in ((110.0, -0.4, -34), (164.81, 0.1, -36), (220.0, -0.1, -38), (277.18, 0.4, -40)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # the ball becoming the egg: a soft rising tone
    t0, t1 = APPLY
    tt = np.arange(int((t1 - t0 + 0.6) * SR)) / SR
    fr = 220 * 2 ** (np.clip(tt / (t1 - t0), 0, 1) * 1.0)
    v = np.sin(np.cumsum(2 * math.pi * fr / SR)) * np.minimum(1, tt / 0.3) * np.minimum(1, (tt[-1] - tt) / 0.5) * db(-30)
    place(mix, v, t0)
    for i, lam in enumerate(LAM):
        place(mix, A.pluck(261.63 * lam, 0.7, -0.4 + 0.4 * i) * 0.7, SEG[3] + 0.3 + 0.35 * i)
    # the power steps: one tick per whole application, pitch rising as the arrows gather
    p0, p1 = POWER
    ks = np.linspace(0, 1, 4000)
    smooth_k = ks * ks * (3 - 2 * ks) * K_MAX
    for k in range(1, K_MAX + 1):
        tk = p0 + (p1 - p0) * float(ks[np.argmin(np.abs(smooth_k - k))])
        _, ang = power_stats(k)
        frac = 1 - min(1.0, ang.mean() / 57.4)
        place(mix, A.pluck(392.0 * 2 ** frac, 0.5, 0.0) * 0.5, tk)
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
