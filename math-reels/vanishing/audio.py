"""Soundtrack for the vanishing-gradient reel (50 s), in the gentle register of reels 23-34: a soft chord bed;
as the blame reaches each layer, two plucks, one per network: the sigmoid's falls in loudness with the
logarithm of its gradient (it fades out as the blame vanishes), the ReLU's stays; a soft sweep as each
slope curve draws; a quiet tick every few training snapshots; a chime at the end of training pitched by
each network's accuracy; bells at the headings. Circular over 50 s."""

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
from scene import FPS, LOOP_FADE, N_FRAMES, T_CLOSE, T_WHY, T_WHY2, TRAIN_WIN, Scene  # noqa: E402

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


def tick(rng, level=-36):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    v = sosfilt(butter(2, [1500, 6000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.006)
    return v / (np.max(np.abs(v)) + 1e-9) * db(level)


def land(freq, level=-26):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)) * np.exp(-t / 0.12) * np.minimum(1, t / 0.004) * db(level)


SCALE = [392.0, 349.23, 329.63, 293.66, 261.63, 246.94, 220.0, 196.0, 174.61, 164.81, 146.83, 130.81]


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(35)
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
    L = sc.n_layers
    g_sig, g_relu = sc.n["sigmoid"]["g"], sc.n["relu"]["g"]
    top_sig, top_relu = math.log10(g_sig[-1]), math.log10(g_relu[-1])
    for li in range(L - 1, -1, -1):
        t0 = sc.row_time(li)
        fr = SCALE[(L - 1 - li) % len(SCALE)]
        # loudness follows the gradient: -3 dB per factor of 10 below the top layer (a fade, not silence)
        lv_sig = -3.0 * (top_sig - math.log10(g_sig[li]))
        lv_relu = -3.0 * (top_relu - math.log10(g_relu[li]))
        place(mix, A.pluck(fr, 0.75, -0.55) * db(lv_sig) * 0.8, t0)
        place(mix, A.pluck(fr * 1.5, 0.75, 0.55) * db(lv_relu) * 0.8, t0 + 0.04)
    for t0 in (T_WHY + 0.2, T_WHY2 + 0.2):
        n = int(1.2 * SR)
        noise = rng.standard_normal(n)
        sw = A.bandpass_sweep(noise, 300, 2400, 1.2) * np.sin(np.linspace(0, math.pi, n)) * db(-34)
        place(mix, sw, t0)
    n_snap = sc.n_snap
    a, b = TRAIN_WIN
    for j in range(0, n_snap, 4):
        prog = j / (n_snap - 1)
        tl = a + (b - a) * (1 - (1 - prog) ** (1 / 2.2))
        place(mix, tick(rng, -40 + 4 * prog), tl)
    for act, pan in (("sigmoid", -0.4), ("relu", 0.4)):
        acc = float(sc.n[act]["acc"][-1])
        idx = int(round((1 - max(0.0, min(1.0, (acc - 0.5) * 2)) ** 0.35) * (len(SCALE) - 1)))
        place(mix, A.pluck(SCALE[idx] * 2, 0.9, pan) * 0.8, b + 0.3)
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
