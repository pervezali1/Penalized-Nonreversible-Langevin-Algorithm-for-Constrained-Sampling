"""Soundtrack for the determinant reel (50 s), in the gentle register of reels 23-37: a soft chord bed; a
glissando under each morph whose pitch follows the live volume (|det|, on a log scale: it rises as the solid
grows, sinks into a low hum as it flattens, and comes back up through the mirror); a low thud at det = 0, a
bright chime at the million-point check; bells at the headings. Circular over 50 s."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from scene import FPS, LOOP_FADE, MORPHS, N_FRAMES, Scene, matrix_at  # noqa: E402

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


def land(freq, level=-26):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)) * np.exp(-t / 0.16) * np.minimum(1, t / 0.004) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        env[f] = min(1.0, max(0.0, (t - 1.0) / 2.5)) * (1 - min(1.0, max(0.0, (t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))))
    env_s = np.interp(TT, np.append(np.arange(N_FRAMES) / FPS, DUR), np.append(env, env[0]))
    for base, pan, lv in ((110.0, -0.4, -34), (164.81, 0.1, -36), (220.0, -0.1, -38), (277.18, 0.4, -40)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env_s
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # the volume voice: a soft sine whose pitch follows |det| during each morph, faded in and out at its ends
    for t0, t1, _, _ in MORPHS:
        i0, i1 = int((t0 - 0.2) * SR), int((t1 + 0.4) * SR)
        tt = np.arange(i0, i1) / SR
        dets = np.array([abs(float(np.linalg.det(matrix_at(t)))) for t in np.linspace(tt[0], tt[-1], 200)])
        dets = np.interp(tt, np.linspace(tt[0], tt[-1], 200), dets)
        freq = 330.0 * 2 ** (np.log2(np.maximum(dets, 0.02)) * 0.75)
        ph = np.cumsum(2 * math.pi * freq / SR)
        envm = np.minimum(1, (tt - tt[0]) / 0.25) * np.minimum(1, (tt[-1] - tt) / 0.4)
        v = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * envm * db(-30)
        place(mix, v, tt[0])
    place(mix, land(73.42, -20), 32.4)                        # det = 0
    for k, fr in enumerate((659.25, 783.99, 987.77)):          # the million-point check
        place(mix, A.pluck(fr, 0.6, -0.2 + 0.2 * k) * 0.6, 24.4 + 0.08 * k)
    bell = A.bell_e6() * 0.45
    for c in sc.cues[1:-1]:
        place(mix, bell, c[0])
    out = A.circ_reverb(mix, rt60=3.0, wet=0.3)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    out *= db(-16.0 - meter.integrated_loudness(out))
    for _ in range(3):
        if np.max(np.abs(resample_poly(out, 4, 1, axis=0))) <= db(-1.0):
            break
        out = np.tanh(out / db(-1.0) * 0.9) * db(-1.0) / 0.9 * 0.98
    loud = meter.integrated_loudness(out)
    tp = 20 * math.log10(np.max(np.abs(resample_poly(out, 4, 1, axis=0))))
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
