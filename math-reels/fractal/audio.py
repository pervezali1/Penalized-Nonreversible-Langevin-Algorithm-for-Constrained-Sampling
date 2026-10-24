"""Sonification for the chaos-game reel: a pluck per point while the points are slow (note chosen
by the rule that was rolled), a granular rain whose density follows points-per-frame once they
speed up, a pad that resolves as the fern fills in, bells at the two reveals."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

from scene import (FERN_SLOW_DT, FERN_SLOW_N, FERN_SLOW_T0, FPS, LOOP_FADE, N_FERN, N_FRAMES,  # noqa: E402
                   N_TRI, TRI_SLOW_DT, TRI_SLOW_N, TRI_SLOW_T0, WIPE, Scene)

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS
TRI_NOTES = [523.25, 659.26, 783.99]                 # C5 E5 G5 for corners A B C
FERN_NOTES = [220.0, 440.0, 523.25, 659.26]          # stem, main, left, right
DISSONANT = [220.0, 233.08, 246.94, 261.63]
RESOLVED = [220.0, 277.18, 329.63, 440.0]


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


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(31)
    mix = np.zeros((NS, 2))
    # slow-phase plucks
    for i in range(TRI_SLOW_N):
        k = int(sc.tri_which[i])
        place(mix, A.pluck(TRI_NOTES[k], 0.8, (k - 1) * 0.6) * 0.8, TRI_SLOW_T0 + i * TRI_SLOW_DT)
    for i in range(FERN_SLOW_N):
        k = int(sc.fern_which[i])
        place(mix, A.pluck(FERN_NOTES[k], 0.8, (k - 1.5) * 0.4) * 0.8, FERN_SLOW_T0 + i * FERN_SLOW_DT)
    # granular rain: one grain per frame, level from points added that frame, pitch from fill
    counts = np.array([sc.tri_count(f / FPS) if f / FPS < WIPE[0] else sc.fern_count(f / FPS) for f in range(N_FRAMES)])
    for f in range(1, N_FRAMES):
        t = f / FPS
        new = counts[f] - counts[f - 1] if counts[f] >= counts[f - 1] else 0
        if new <= 0 or t >= LOOP_FADE[0]:
            continue
        total = N_TRI if t < WIPE[0] else N_FERN
        fill = counts[f] / total
        m = int(0.03 * SR)
        tt = np.arange(m) / SR
        freq = 600 + 1800 * fill
        grain = np.sin(2 * math.pi * freq * tt) * np.exp(-tt / 0.006) * db(-34) * min(1.0, 0.3 + math.log10(1 + new) / 2.5)
        place(mix, grain, t + rng.uniform(0, 1 / FPS))
    # pad resolves with the fern's fill
    fill_f = np.array([(sc.fern_count(f / FPS) / N_FERN if f / FPS >= WIPE[0] else 0.0) for f in range(N_FRAMES)])
    a0, a1 = LOOP_FADE
    mixf = np.clip((np.arange(N_FRAMES) / FPS - a0) / (a1 - a0), 0, 1)
    fill_f = fill_f * (1 - mixf * mixf * (3 - 2 * mixf))
    s = A.onepole_circ_fast(circ_interp(fill_f), 0.1)
    s = np.clip(s, 0, 1) ** 0.7
    gate = A.onepole_circ_fast(circ_interp((np.arange(N_FRAMES) / FPS >= WIPE[1]).astype(float) * (1 - mixf)), 0.3)
    for k, (fd, fr) in enumerate(zip(DISSONANT, RESOLVED)):
        ph = np.cumsum(integer_cycles(2 * math.pi * fd * (fr / fd) ** s / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-27) * gate
        pan = (-0.5, -0.17, 0.17, 0.5)[k]
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    bell = A.bell_e6()
    place(mix, bell, 5.2)
    place(mix, A.chime(), 5.2)
    n_s = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, WIPE[0])
    place(mix, bell * 1.2, 14.6)
    place(mix, A.impact(rng) * 0.4, 14.6)
    place(mix, bell * 0.6, 17.6)
    place(mix, sw * env * 0.8, LOOP_FADE[0])
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-28)
    mix[:, 1] += drone * db(-28)
    out = A.circ_reverb(mix, rt60=2.2, wet=0.22)
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
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
