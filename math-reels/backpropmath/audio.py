"""Soundtrack for the maths-only backpropagation reel (60 s), in the gentle register of reels 23-33: a soft
chord bed; a soft pluck as each line of maths appears (rising through the forward and update pages, a
darker falling voice through the backward pages); a faint high note when a result is revealed; a bell at
each new page; a chime when the finite-difference check agrees with backpropagation. Circular over 60 s."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from scene import FPS, LOOP_FADE, N_FRAMES, Scene  # noqa: E402

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
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)) * np.exp(-t / 0.12) * np.minimum(1, t / 0.004) * db(level)


def dark_pluck(freq, vel, pan):
    return sosfilt(butter(2, 900, "low", fs=SR, output="sos"), A.pluck(freq, vel, pan), axis=0)


UP = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99]
DOWN = [440.0, 392.0, 329.63, 293.66, 261.63, 220.0, 196.0, 164.81]


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        env[f] = min(1.0, max(0.0, (t - 1.0) / 2.5)) * (1 - min(1.0, max(0.0, (t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))))
    env_s = np.interp(TT, np.append(np.arange(N_FRAMES) / FPS, DUR), np.append(env, env[0]))
    for base, pan, lv in ((130.81, -0.4, -34), (196.0, 0.1, -36), (293.66, -0.1, -38), (329.63, 0.4, -39)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env_s
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    bell = A.bell_e6() * 0.4
    n_rows = 0
    for pg in sc.pages:
        place(mix, bell, pg["t0"])
        backward = pg["label"].startswith("BACKWARD")
        for i, row in enumerate(pg["rows"]):
            pan = -0.4 + 0.8 * (i / max(1, len(pg["rows"]) - 1))
            if backward:
                place(mix, dark_pluck(DOWN[i % len(DOWN)], 0.8, pan) * 0.9, row["t"])
            else:
                place(mix, A.pluck(UP[i % len(UP)], 0.6, pan) * 0.6, row["t"])
            if any(op[7] for op in row["Rr"][0]):
                place(mix, A.pluck(UP[(i + 4) % len(UP)] * 2, 0.35, -pan) * 0.35, row["t"] + 0.45)
            n_rows += 1
        if pg["label"].startswith("CHECK"):
            for k, fr in enumerate((659.25, 783.99, 1046.5)):
                place(mix, A.pluck(fr, 0.6, -0.2 + 0.2 * k) * 0.6, pg["rows"][-1]["t"] + 0.45 + 0.08 * k)
    place(mix, land(130.81, -22), sc.pages[-1]["t0"])
    out = A.circ_reverb(mix, rt60=2.8, wet=0.28)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    out *= db(-16.0 - meter.integrated_loudness(out))
    for _ in range(3):
        if np.max(np.abs(resample_poly(out, 4, 1, axis=0))) <= db(-1.0):
            break
        out = np.tanh(out / db(-1.0) * 0.9) * db(-1.0) / 0.9 * 0.98
    loud = meter.integrated_loudness(out)
    tp = 20 * math.log10(np.max(np.abs(resample_poly(out, 4, 1, axis=0))))
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}, rows {n_rows}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
