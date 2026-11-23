"""Sonification for the Riemann reel. A plink for every prime as the number line is swept; a bell for each of the
first three zeros, then a tick for every zero found (softer as they come faster); and a hum that is the zeros
themselves: the first eight heights (14.13, 21.02, 25.01, ...) played as pitches at ten times their value in hertz
(141 Hz, 210 Hz, 250 Hz, ...), each one entering when its zero is found. A bell and impact when the primes appear, a
chime for the hypothesis. The first 2.4 s continue the end of the loop. Circular.
    python audio.py <data.npz> <out.wav> [events.json]"""

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

from rsim import DUR, FPS, N_FRAMES, SWEEP, WIPE, X_MAX  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
NS = int(SR * DUR)
TT = np.arange(NS) / SR
SCALE = (0, 2, 4, 7, 9)
HUM = 8                                          # how many zeros are played as pitches


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


def plink(freq, level, tau_=0.05, seconds=0.25):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * 2.01 * freq * t)) * np.exp(-t / tau_)
    return out * np.minimum(1, t / 0.001) * level


def swoosh(rng, seconds, level):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = sosfilt(butter(2, [500, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    x /= np.std(x) * 3
    return x * np.sin(np.pi * t / seconds) ** 2 * db(level)


def render(data_path, out_wav, events_json=None):
    d = np.load(data_path)
    gamma, hit, primes = d["gamma"], d["hit"], d["primes"]
    rng = np.random.default_rng(1859)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # the primes, as the number line is swept
    for k, p in enumerate(primes):
        t0 = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * p / X_MAX
        pan = -0.6 + 1.2 * p / X_MAX
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        c = plink(392.0 * 2 ** (k // 5 + SCALE[k % 5] / 12), db(-17), 0.09, 0.5)
        place(mix, np.stack([c * L, c * R], 1), t0)
    # the zeros: bells for the first three, ticks after (softer the faster they come)
    vt = hit % DUR                                   # zeros reached after the loop point sound at the start
    for k in range(len(hit)):
        if k < 3:
            place(mix, bell * (1.0 if k == 0 else 0.8), vt[k])
            place(mix, A.pluck(110.0 * 2 ** (SCALE[k] / 12), 0.7, 0.0), vt[k])
            continue
        if WIPE[0] <= vt[k] < WIPE[1] + 0.2:
            continue
        near = np.count_nonzero(np.abs(hit - hit[k]) < 0.5)                   # zeros per second around this one
        i = int(gamma[k] * 1.7) % 15
        c = plink(784.0 * 2 ** (i // 5 + SCALE[i % 5] / 12), db(-23) / math.sqrt(1 + near / 6), 0.03, 0.15)
        pan = rng.uniform(-0.5, 0.5)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        place(mix, np.stack([c * L, c * R], 1), vt[k])
    # the hum: the first eight zeros as pitches, 10 Hz per unit of height, each entering when it is found
    for k in range(HUM):
        env = np.zeros(NS)
        on = np.clip((TT - hit[k]) / 0.6, 0, 1)                               # in, from its zero to the end
        pre = np.clip((WIPE[0] + 0.4 - TT) / 0.4, 0, 1)                       # and through the loop point, until the wipe
        env = np.maximum(on, pre)
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * 10 * gamma[k] / SR)))
        trem = 1 + 0.15 * np.sin(2 * math.pi * (k + 2) * TT / DUR * 3)
        v = np.sin(ph) * env * trem * db(-27 - 1.2 * k)
        pan = -0.5 + k / (HUM - 1)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # the moments
    place(mix, swoosh(rng, 0.6, -20), WIPE[0] - 0.05)
    place(mix, A.chime(), 7.1)
    place(mix, bell * 1.25, 14.2)
    place(mix, A.impact(rng) * 0.4, 14.2)
    place(mix, A.chime(), 17.2)
    place(mix, A.pluck(98.0, 0.7, 0.0), 20.0)
    place(mix, A.pluck(130.8, 0.7, 0.0), 22.8)
    place(mix, bell * 0.7, 25.8)
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
            json.dump(dict(lufs=loud, true_peak=tp, hum_hz=[round(10 * g, 1) for g in gamma[:HUM]]), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
