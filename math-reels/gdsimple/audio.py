"""A gentler soundtrack for the white-background gradient descent reel: a soft sustained chord, one
plucked note per step whose pitch falls with the ball's height (so the descent is audible), a quiet
brush as each arrow grows, a low tone when the ball lands, and a bell at each heading. Circular over
30 s, slower and quieter than the dark reels."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scene import FPS, LOOP_FADE, N_FRAMES, N_STEPS, STEP_DT, T_BALL, T_FIRST, T_SURF, Scene  # noqa: E402

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


def brush(rng, dur=0.25, level=-34):
    n = int(dur * SR)
    t = np.arange(n) / SR
    v = sosfilt(butter(2, [1200, 5000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    env = np.sin(math.pi * np.minimum(1.0, t / dur)) ** 2
    return v / (np.max(np.abs(v)) + 1e-9) * env * db(level)


def land(freq, level=-26):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)) * np.exp(-t / 0.12) * np.minimum(1, t / 0.004) * db(level)


# a descending major scale the plucks walk down as the height falls
SCALE = [392.0, 349.23, 329.63, 293.66, 261.63, 246.94, 220.0, 196.0, 174.61, 164.81, 146.83, 130.81]


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(7)
    mix = np.zeros((NS, 2))
    # soft chord bed (C major add9), very quiet, swells in with the surface and out at the loop
    env = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        env[f] = min(1.0, max(0.0, (t - T_SURF + 0.5) / 2.5)) * (1 - min(1.0, max(0.0, (t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))))
    env_s = np.interp(TT, np.append(np.arange(N_FRAMES) / FPS, DUR), np.append(env, env[0]))
    for base, pan, lv in ((130.81, -0.4, -34), (196.0, 0.1, -36), (293.66, -0.1, -38), (329.63, 0.4, -39)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env_s
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # the ball appears
    place(mix, land(196.0, -24), T_BALL)
    # per step: a brush as the arrow grows, a pluck when the ball lands, pitched by its new height
    h0 = sc.heights[0]
    for k in range(N_STEPS):
        t_arrow = T_FIRST + k * STEP_DT
        t_land = t_arrow + STEP_DT * 0.84
        place(mix, brush(rng, 0.22, -36 - 0.3 * k), t_arrow)
        frac = sc.heights[k + 1] / h0
        idx = int(round((1 - frac ** 0.35) * (len(SCALE) - 1)))
        freq = SCALE[min(len(SCALE) - 1, max(0, idx))]
        vel = 0.9 - 0.5 * (k / N_STEPS)
        pan = 0.3 * math.sin(k * 0.9)
        place(mix, A.pluck(freq, vel, pan) * 0.9, t_land)
    place(mix, land(130.81, -22), T_FIRST + N_STEPS * STEP_DT)
    # bells at the headings
    bell = A.bell_e6() * 0.45
    for t_b in (2.8, 9.6, 13.8, 17.8, 21.8, 25.6):
        place(mix, bell, t_b)
    out = A.circ_reverb(mix, rt60=2.8, wet=0.28)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    out *= db(-16.0 - meter.integrated_loudness(out))             # a little quieter than the dark reels
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
            json.dump(dict(heights=sc.heights, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
