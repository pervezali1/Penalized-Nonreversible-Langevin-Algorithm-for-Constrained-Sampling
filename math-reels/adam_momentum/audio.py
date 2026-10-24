"""Sonification for the Adam-momentum reel: pad pitch follows the mean loss, harsh jitter texture under the
bouncing plain-descent run, smooth flow under Adam, a pop per 100 particles that reach the answer."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
sys.path.insert(0, os.path.join(ROOT, "descent"))
import importlib.util  # noqa: E402

from amsim import FPS, LOOP, N_FRAMES, RESETS, RUNS  # noqa: E402
from landscape import F_MIN, found, loss  # noqa: E402

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


def render(data_path, out_wav, events_json=None):
    pos = np.load(data_path)["pos"]
    rng = np.random.default_rng(17)
    mean_loss = np.array([float(np.mean(loss(pos[f].astype(np.float64)) - F_MIN)) for f in range(N_FRAMES)])
    n_found = np.array([found(pos[f]).sum() for f in range(N_FRAMES)])
    speed = np.linalg.norm(pos - np.roll(pos, 1, axis=0), axis=2).mean(axis=1)
    speed[0] = speed[1]
    mix = np.zeros((NS, 2))
    u = A.onepole_circ_fast(circ_interp(np.clip(mean_loss / 1.2, 0, 1)), 0.05)
    trem = A.onepole_circ_fast(circ_interp(np.clip(speed / 0.02, 0, 1)), 0.02)
    for base, pan in ((220.0, -0.5), (277.18, -0.15), (329.63, 0.15), (440.0, 0.5)):
        ph = np.cumsum(integer_cycles(2 * math.pi * base * 2 ** (u - 1) / SR))
        lfo = 1 - 0.6 * trem * (0.5 + 0.5 * np.sin(2 * math.pi * 9 * TT))
        v = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * db(-27) * lfo
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    noise = sosfilt(butter(2, [300, 3000], "band", fs=SR, output="sos"), rng.standard_normal(NS))
    noise /= np.std(noise) * 3
    flow = circ_interp(np.clip(speed / 0.01, 0, 1)) * db(-28)
    mix[:, 0] += noise * flow
    mix[:, 1] += np.roll(noise, 1201) * flow
    acc_count = 0
    for f in range(1, N_FRAMES):
        if any(a <= f < b for a, b in RESETS) or f >= LOOP[0]:
            acc_count = 0
            continue
        acc_count += max(0, int(n_found[f]) - int(n_found[f - 1]))
        pops = acc_count // 100
        acc_count -= pops * 100
        for j in range(min(pops, 6)):
            nn = int(0.06 * SR)
            tt = np.arange(nn) / SR
            pop = np.sin(2 * math.pi * (1200 + 600 * rng.random()) * tt) * np.exp(-tt / 0.012) * db(-23)
            place(mix, pop, f / FPS + j * 0.004)
    bell = A.bell_e6()
    sw_n = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(sw_n))
    sw /= np.std(sw) * 3
    ts = np.arange(sw_n) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, bell * 0.6, 4.6)
    place(mix, sw * env, RESETS[0][0] / FPS)
    place(mix, A.chime(), RUNS[1][4] / FPS)
    place(mix, bell * 1.2, 13.0)
    place(mix, A.impact(rng) * 0.5, 13.0)
    place(mix, bell * 0.7, 17.6)
    place(mix, sw * env * 0.8, LOOP[0] / FPS)
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-28)
    mix[:, 1] += drone * db(-28)
    out = A.circ_reverb(mix, rt60=2.0, wet=0.2)
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
