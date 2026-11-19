"""Sonification for the medical test reel: ten low plucks as the ten sick people light up, a rising tone while the
test sweeps the crowd with a plink for every positive (lower for the sick), a whoosh as the positives gather, a bell
and impact on "only 8%", a soft falling tick for each false alarm that tests negative the second time, a bright bell
on "positive twice". Circular."""

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
import importlib.util  # noqa: E402

from scene import (FPS, GATHER0, GATHER1, LOOP0, N_FRAMES, RETEST0, SCAN0, SCAN1, T_SICK, Scene,  # noqa: E402
                   smooth)

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


def plink(freq, level, tau=0.04):
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * 2.01 * freq * t)) * np.exp(-t / tau)
    return out * np.minimum(1, t / 0.001) * db(level)


def whoosh(rng, seconds, lo, hi, level, rising=True):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), rng.standard_normal(n))
    x /= np.std(x) * 3
    env = np.sin(np.pi * t / seconds) ** 2 if not rising else (t / seconds) ** 2 * np.exp(-((t / seconds - 1) ** 2) * 0)
    env[-int(0.04 * SR):] *= np.linspace(1, 0, int(0.04 * SR))
    return x * env * db(level)


def scan_time(row):
    """When the sweep reaches a row (inverse of Scene.scan_row)."""
    lo, hi = SCAN0, SCAN1
    for _ in range(40):
        mid = (lo + hi) / 2
        if 100 * smooth((mid - SCAN0) / (SCAN1 - SCAN0)) <= row:
            lo = mid
        else:
            hi = mid
    return hi


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(99)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # the ten sick light up
    for k, p in enumerate(np.nonzero(sc.sick)[0]):
        pan = (sc.home[p, 0] - 470) / 420
        place(mix, A.pluck(110 * 2 ** (k % 5 * 2 / 12), 0.6, pan), T_SICK + 0.06 * k)
    # the sweep: a rising tone, and a plink per positive
    n = int((SCAN1 - SCAN0) * SR)
    t = np.arange(n) / SR
    f = 300 * 2 ** (1.5 * t / t[-1])
    tone = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 0.5 * db(-30)
    place(mix, tone, SCAN0)
    for p in np.nonzero(sc.pos1)[0]:
        t0 = scan_time(sc.row[p]) + rng.random() * 0.01
        pan = (sc.home[p, 0] - 470) / 420
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        c = plink(392.0 if sc.sick[p] else 1046.5 * 2 ** (rng.integers(0, 5) * 2 / 12), -24 if not sc.sick[p] else -18)
        place(mix, np.stack([c * L, c * R], 1), t0)
    # the positives gather
    place(mix, whoosh(rng, GATHER1 - GATHER0, 400, 3000, -20, rising=False), GATHER0)
    place(mix, A.chime(), GATHER1)
    # only 8%
    place(mix, bell * 1.1, 10.8)
    place(mix, A.impact(rng) * 0.35, 10.8)
    place(mix, A.pluck(130.8, 0.5, 0.0), 13.6)
    # second test: each false alarm that clears goes dark with a falling tick
    darks = np.sort(sc.dark_t[np.isfinite(sc.dark_t)])
    for k, t0 in enumerate(darks):
        freq = 1400 * 2 ** (-1.5 * k / max(1, len(darks) - 1))
        place(mix, plink(freq, -30, 0.02), t0)
    place(mix, bell * 1.2, 18.5)
    place(mix, A.chime(), 18.5)
    # everyone flies home
    place(mix, whoosh(rng, 0.8, 600, 4000, -22), LOOP0)
    drone = np.zeros(NS)
    for fr_hz, lev in ((55.0, -28), (82.41, -30), (110.0, -34)):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr_hz / SR)))) * db(lev)
    mix += np.stack([drone, drone], 1)
    out = A.circ_reverb(mix, rt60=1.8, wet=0.18)
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
            json.dump(dict(lufs=loud, true_peak=tp, retest0=RETEST0), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
