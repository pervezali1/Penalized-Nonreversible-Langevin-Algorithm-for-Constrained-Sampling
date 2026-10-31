"""Soundtrack for the backpropagation reel (46 s), in the gentle register of reels 23-32: a soft chord bed;
for each network a rising pluck as the signal crosses each layer and a high note at the guess, a low thud
at the error, a falling darker pluck as the blame crosses each layer back, a quick shimmer when the weights
move and a chime at the end of the nudge pitched by the new accuracy. Bells at the headings. Circular over
46 s."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import FPS, LOOP_FADE, N_FRAMES, T_CLOSE, Scene  # noqa: E402

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
    """A pluck with its upper partials rolled off: the blame's voice."""
    v = A.pluck(freq, vel, pan)
    return sosfilt(butter(2, 900, "low", fs=SR, output="sos"), v, axis=0)


SCALE = [392.0, 349.23, 329.63, 293.66, 261.63, 246.94, 220.0, 196.0, 174.61, 164.81, 146.83, 130.81]


def scale_note(frac):
    idx = int(round((1 - max(0.0, min(1.0, frac)) ** 0.35) * (len(SCALE) - 1)))
    return SCALE[min(len(SCALE) - 1, max(0, idx))]


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        env[f] = min(1.0, max(0.0, (t - 1.5) / 2.5)) * (1 - min(1.0, max(0.0, (t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))))
    env_s = np.interp(TT, np.append(np.arange(N_FRAMES) / FPS, DUR), np.append(env, env[0]))
    for base, pan, lv in ((130.81, -0.4, -34), (196.0, 0.1, -36), (293.66, -0.1, -38), (329.63, 0.4, -39)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env_s
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    up = [261.63, 329.63, 392.0, 523.25]
    down = [220.0, 174.61, 146.83, 110.0]
    for s in sc.stages:
        for li, (fa, fb) in enumerate(s["fwd"]):
            place(mix, A.pluck(up[li], 0.7, -0.4 + 0.8 * li / max(1, len(s["fwd"]) - 1)) * 0.7, fa)
        place(mix, A.pluck(up[3], 0.8, 0.3) * 0.8, s["guess"])
        place(mix, land(98.0, -24), s["error"])
        for kk, (ba, bb) in enumerate(s["bwd"]):
            place(mix, dark_pluck(down[kk], 0.9, 0.4 - 0.8 * kk / max(1, len(s["bwd"]) - 1)) * 1.1, ba)
        n0, n1 = s["nudge"]
        for i, fr in enumerate((523.25, 659.25, 783.99, 1046.5)):
            place(mix, A.pluck(fr, 0.45, -0.3 + 0.2 * i) * 0.5, n0 + 0.09 * i)
        place(mix, A.pluck(scale_note(1 - s["acc1"]) * 2, 0.9, 0.0) * 0.8, n1 + 0.1)
    place(mix, land(130.81, -22), T_CLOSE)
    bell = A.bell_e6() * 0.45
    for c in sc.cues[1:-1]:
        place(mix, bell, c[0])
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
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
