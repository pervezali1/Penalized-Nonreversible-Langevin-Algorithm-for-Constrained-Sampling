"""Sonification for the flipped-labels reel: three voices, one per network, each a chord whose pitch
climbs with that network's clean-test accuracy and whose detune grows with the number of lies it
believes (the plain net drifts out of tune as it memorises); a short sour tone each time the plain
net adopts another lie, a soft tick per training step, bells at the reveals. Circular over 23 s."""

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
from scene import F0, F1, FPS, LOOP_FADE, N_FRAMES, NAMES, Scene, frame_step  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS


def db(x):
    return 10 ** (x / 20)


def circ_interp(v):
    return np.interp(TT, np.append(FRAME_T, DUR), np.append(v, v[0]))


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
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, [2500, 9000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.004)
    return noise / (np.max(np.abs(noise)) + 1e-9) * db(level)


def sour(level=-27):
    """A short two-note clash: the sound of a lie being accepted."""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * 466.16 * t) + np.sin(2 * math.pi * 493.88 * t)) * np.exp(-t / 0.05) * db(level)


def chord(f_ctrl, detune_ctrl, rng, pan, level):
    out = np.zeros((NS, 2))
    L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    rp = rng.standard_normal(8)
    for k, ratio in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)):
        cents = rp[k] * 35 * detune_ctrl
        ph = np.cumsum(integer_cycles(2 * math.pi * f_ctrl * ratio * 2 ** (cents / 1200) / SR))
        v = np.sin(ph) * (0.9 / (k + 1)) * db(level)
        out[:, 0] += v * L
        out[:, 1] += v * R
    return out


VOICE = {"plain": (146.83, -0.5), "dropout": (196.0, 0.0), "l2": (220.0, 0.5)}


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(123)
    mix = np.zeros((NS, 2))
    steps = np.array([frame_step(f) for f in range(N_FRAMES)])
    recs = np.array([sc.rec(s) for s in steps])
    fade = np.array([1 - min(1.0, max(0.0, (f / FPS - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))) for f in range(N_FRAMES)])
    for name in NAMES:
        base, pan = VOICE[name]
        te = sc.acc_test[name][recs]
        te = te * fade + sc.acc_test[name][0] * (1 - fade)
        lies = sc.lies[name][recs] / sc.n_flip
        lies = lies * fade + sc.lies[name][0] / sc.n_flip * (1 - fade)
        u = A.onepole_circ_fast(circ_interp(np.clip((te - 0.5) / 0.4, 0, 1)), 0.08)
        det = A.onepole_circ_fast(circ_interp(np.clip((lies - 0.3) / 0.6, 0, 1)), 0.08)    # in tune below 30% lies, sour at 90%
        mix += chord(base * 2 ** (u * 0.4), det, rng, pan, -27)
    # a sour clash each time the plain network adopts another lie (after it is above half)
    prev = int(sc.lies["plain"][recs[F0]])
    for f in range(F0, F1 + 1):
        cur = int(sc.lies["plain"][recs[f]])
        if cur > prev and cur > sc.n_flip // 2:
            place(mix, sour(-29), f / FPS)
        prev = max(prev, cur)
    prev = 0
    for f in range(F0, F1 + 1):
        s = steps[f]
        for st in range(prev + 1, s + 1):
            if st % 12 == 0:
                place(mix, tick(rng, -38), f / FPS + (st - prev - 1) / max(1, s - prev) / FPS)
        prev = s
    bell = A.bell_e6()
    t_end = F1 / FPS
    place(mix, A.impact(rng) * 0.5, t_end + 0.3)
    place(mix, bell * 0.8, t_end + 0.3)
    place(mix, bell * 0.6, t_end + 2.8)
    place(mix, bell * 0.8, t_end + 5.4)
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts_ = np.arange(n_s) / SR
    env = db(-46) * (db(-22) / db(-46)) ** (ts_ / ts_[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, sw * env, LOOP_FADE[0] - 0.3)
    place(mix, sw * env * 0.7, F0 / FPS - 0.5)
    drone = np.zeros(NS)
    for fr in (49.0, 73.42):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-29)
    mix[:, 1] += drone * db(-29)
    out = A.circ_reverb(mix, rt60=2.4, wet=0.22)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    out *= db(-14.0 - meter.integrated_loudness(out))
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
            json.dump(dict(final=sc.final, n_flip=sc.n_flip, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
