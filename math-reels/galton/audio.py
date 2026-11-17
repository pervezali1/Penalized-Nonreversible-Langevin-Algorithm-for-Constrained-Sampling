"""Sonification for the Galton board reel: every peg hit can plink (pitched by where the peg is, left low, right
high; up to three per frame are heard, softer as the stream thickens), a rain texture follows the number of hits per
frame, a soft sand hiss follows the landings, a pad rises as the bins fill; a bell and chime when the prediction is
drawn, a low blip for the empty edges, a bell for the last line, a rumble and swoosh as the floor opens. Circular."""

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

from scene import CX, DRAIN0, DX, EDGE_T, FPS, LOOP_FADE, N, N_FRAMES, REVEAL, ROWS, Scene  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
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


def plink(freq, level):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * 2.01 * freq * t)) * np.exp(-t / 0.035)
    return out * np.minimum(1, t / 0.001) * level


def blip(freq=180.0, seconds=0.6, level=-15):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = freq * 0.55 ** (t / seconds)
    return np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) * db(level)


def noise_bed(rng, env_f, lo, hi, level):
    x = sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), rng.standard_normal(NS))
    x /= np.std(x) * 3
    return x * A.onepole_circ_fast(circ_interp(env_f), 0.03) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(16)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # plinks: up to three peg hits per frame are heard, softer as the stream thickens
    hf = sc.hit_frames.ravel()
    hx = sc.hit_x.ravel()
    order = np.argsort(hf, kind="stable")
    hf, hx = hf[order], hx[order]
    starts = np.searchsorted(hf, np.arange(N_FRAMES + 1))
    hits_pf = np.diff(starts)
    for f in range(N_FRAMES):
        n = hits_pf[f]
        if n == 0:
            continue
        pick = rng.choice(np.arange(starts[f], starts[f + 1]), size=min(3, n), replace=False)
        lvl = db(-20) / math.sqrt(1 + n / 8)
        for p in pick:
            u = (hx[p] - (CX - ROWS / 2 * DX)) / (ROWS * DX)                  # 0 left .. 1 right
            i = int(u * 9.999)
            freq = 523.25 * 2 ** (i // 5 + SCALE[i % 5] / 12)
            pan = (u - 0.5) * 1.2
            L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
            c = plink(freq, lvl)
            place(mix, np.stack([c * L, c * R], 1), (f + rng.random()) / FPS)
    # rain: the number of hits per frame (log); sand: landings per frame
    rain_f = np.log1p(hits_pf[:N_FRAMES]) / math.log1p(hits_pf.max())
    mix += np.stack([noise_bed(rng, rain_f, 3000, 9000, -24)] * 2, 1) * np.array([1.0, 0.9])
    land_f = np.zeros(N_FRAMES)
    np.add.at(land_f, np.floor(sc.t_land * FPS).astype(int) % N_FRAMES, 1.0)
    land_f = np.log1p(land_f) / math.log1p(land_f.max())
    mix += np.stack([noise_bed(rng, land_f, 300, 1500, -27)] * 2, 1)
    # a pad that rises as the bins fill, silent once the floor opens
    filled = np.array([np.count_nonzero(sc.t_land <= f / FPS) / N if f / FPS < DRAIN0 else 0.0 for f in range(N_FRAMES)])
    gate = np.array([1.0 if 0.0 < filled[f] else 0.0 for f in range(N_FRAMES)])
    fr = A.onepole_circ_fast(circ_interp(filled), 0.2)
    g = A.onepole_circ_fast(circ_interp(gate), 0.3)
    for mult, lev in ((1.0, -26), (1.5, -30), (2.0, -32)):
        ph = np.cumsum(integer_cycles(2 * math.pi * mult * 110 * 2 ** fr / SR))
        v = np.sin(ph) * db(lev) * g
        mix += np.stack([v, v], 1)
    # the moments
    place(mix, bell * 1.2, REVEAL)
    place(mix, A.chime(), REVEAL)
    place(mix, A.impact(rng) * 0.3, REVEAL)
    place(mix, np.stack([blip(), blip()], 1), EDGE_T)
    place(mix, np.stack([blip(), blip()], 1), EDGE_T + 0.25)
    place(mix, bell * 0.8, 18.3)
    n_s = int(0.8 * SR)
    ts = np.arange(n_s) / SR
    rumble = sosfilt(butter(2, 200, "low", fs=SR, output="sos"), rng.standard_normal(n_s))
    rumble = rumble / (np.std(rumble) * 3) * np.exp(-ts / 0.3) * db(-14)
    place(mix, rumble, DRAIN0)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    env = db(-44) * (db(-20) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, LOOP_FADE[0] - 0.3)
    drone = np.zeros(NS)
    for fr_hz in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr_hz / SR))))
    mix += np.stack([drone, drone], 1) * db(-30)
    out = A.circ_reverb(mix, rt60=1.6, wet=0.16)
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
            json.dump(dict(reveal=REVEAL, edge=EDGE_T, lufs=loud, true_peak=tp, hits=int(hits_pf.sum())), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
