"""Sonification for the prime race. Every prime up to 60 plinks as it drops into its lane (Team 3 low, Team 1 high);
a rattle follows the primes as the race speeds up to 1,000 and 30,000; a tone tracks Team 3's lead (its pitch rises
with the lead); a bright two-note chime each time Team 1 takes the lead (26,861, then the bursts near 616,841,
12.3 million and 951.8 million); a plink per square; a soft pad for the fair count. Circular.
    python audio.py <race.npz> <out.wav> [events.json]"""

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

from scene import DUR, FAIR, FPS, LOG, N_FRAMES, SQUARES, WIPE, Scene, time_of, x_of  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS
SCALE = (0, 2, 4, 7, 9)


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


def plink(freq, level, tau_=0.06, seconds=0.3):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * 2.01 * freq * t)) * np.exp(-t / tau_)
    return out * np.minimum(1, t / 0.001) * level


def tick(rng, level=-33):
    m = int(0.03 * SR)
    tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
    tap = sosfilt(butter(2, [2000, 7000], "band", fs=SR, output="sos"), tap)
    return tap / (np.max(np.abs(tap)) + 1e-9) * db(level)


def swoosh(rng, seconds, level):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = sosfilt(butter(2, [500, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    x /= np.std(x) * 3
    return x * np.sin(np.pi * t / seconds) ** 2 * db(level)


def render(data_path, out_wav, events_json=None):
    sc = Scene(data_path)
    rng = np.random.default_rng(4)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # the rules: a plink per prime as it drops into its lane
    k3 = k1 = 0
    for p in sc.primes:
        if p == 2:
            continue
        t0 = sc.t_prime[p]
        if p % 4 == 3:
            f, k3, pan = 196.0 * 2 ** (SCALE[k3 % 5] / 12 + k3 // 5), k3 + 1, -0.4
        else:
            f, k1, pan = 392.0 * 2 ** (SCALE[k1 % 5] / 12 + k1 // 5), k1 + 1, 0.4
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        c = plink(f, db(-17), 0.1, 0.6)
        place(mix, np.stack([c * L, c * R], 1), t0)
    # the race speeds up: a tick for every frame with new primes, louder the more there are
    prev = 60
    for f in range(int(4.8 * FPS), int(15.2 * FPS)):
        xx = x_of(f / FPS)
        _, _, pi = sc.at(xx)
        _, _, pi0 = sc.at(prev)
        if pi > pi0:
            place(mix, tick(rng) * min(1.0, 0.25 + math.log10(1 + pi - pi0) / 2.5), f / FPS)
        prev = xx
    # a tone that follows Team 3's lead (scaled to the chart), from the race to the end of the sweep
    lead = np.zeros(N_FRAMES)
    gate = np.zeros(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        te = t + DUR if t < WIPE[0] else t
        if 4.8 <= te <= LOG[1] + 0.3 or te >= DUR:
            xx = x_of(min(te, DUR))
            _, D, pi = sc.at(xx)
            scale = 34.0 if xx <= 30000 else max(40.0, float(sc.run_max[int(np.searchsorted(sc.want, xx, side="right")) - 1]))
            lead[f] = D / scale
            gate[f] = 1.0 if te < DUR else 0.5
    ld = A.onepole_circ_fast(circ_interp(lead), 0.05)
    gt = A.onepole_circ_fast(circ_interp(gate), 0.12)
    ph = np.cumsum(integer_cycles(2 * math.pi * 220 * 2 ** (np.clip(ld, -0.5, 1.2)) / SR))
    v = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * gt * db(-26)
    mix += np.stack([v * 0.9, v], 1)
    # Team 1 takes the lead: a bright two-note chime
    for b0, _ in sc.bursts:
        if b0 <= 30000:
            t0 = time_of(b0)
        else:
            lo, hi = LOG
            for _ in range(50):
                mid = (lo + hi) / 2
                if x_of(mid) < b0:
                    lo = mid
                else:
                    hi = mid
            t0 = hi
        place(mix, plink(1046.5, db(-18), 0.15, 0.8), t0)
        place(mix, plink(1318.5, db(-19), 0.15, 0.8), t0 + 0.09)
    place(mix, bell * 0.9, 12.6)
    place(mix, A.impact(rng) * 0.28, 12.6)
    # the squares, and the fair count
    for i in range(6):
        place(mix, plink(784.0 * 2 ** (SCALE[i % 5] / 12), db(-19), 0.08, 0.4), SQUARES[0] + 0.2 + 0.22 * i)
    n = int((FAIR[1] - FAIR[0] + 1.2) * SR)
    t = np.arange(n) / SR
    pad = sum(np.sin(2 * math.pi * f0 * t) for f0 in (261.6, 329.6, 392.0)) * np.sin(np.pi * t / t[-1]) ** 2 * db(-30)
    place(mix, pad, FAIR[0])
    # the moments
    place(mix, swoosh(rng, 0.6, -20), WIPE[0] - 0.05)
    place(mix, A.chime(), 2.6)
    place(mix, A.pluck(98.0, 0.7, 0.0), 9.6)
    place(mix, swoosh(rng, 0.5, -21), LOG[0] - 0.1)
    place(mix, bell * 1.2, 18.4)
    place(mix, A.pluck(130.8, 0.7, 0.0), 21.3)
    place(mix, bell * 0.8, 24.8)
    place(mix, A.chime(), 27.5)
    drone = np.zeros(NS)
    for fr_hz, lev in ((55.0, -29), (82.41, -31)):
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
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
