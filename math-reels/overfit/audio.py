"""Sonification for the shuffled-labels reel: a chord that resolves with training accuracy (so it
resolves in BOTH runs), static that clears as the loss falls, ticks per 50 steps, a shuffle
rattle, a bright bell for the real test and a dissonant stinger for the coin-flip test."""

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

from ovtrain import FPS, LOOP, N_FRAMES, SHUFFLE, T_TEST_A, T_TEST_B  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS
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


def render(data_path, out_wav, events_json=None):
    with open(data_path.replace(".npz", ".json")) as fh:
        meta = json.load(fh)
    acc_f = np.array(meta["train"])
    frames = meta["frames"]
    rng = np.random.default_rng(23)
    a0, a1 = LOOP
    mixf = np.clip((np.arange(N_FRAMES) - a0) / (a1 - a0), 0, 1)
    mixf = mixf * mixf * (3 - 2 * mixf)
    acc_ctrl = (1 - mixf) * acc_f + mixf * acc_f[0]
    acc_ctrl[SHUFFLE[0]:SHUFFLE[1]] = acc_f[0]
    acc = A.onepole_circ_fast(circ_interp(acc_ctrl), 0.08)
    s = np.clip((acc - 0.5) / 0.5, 0, 1)
    s = s * s * (3 - 2 * s)
    mix = np.zeros((NS, 2))
    for k, (fd, fr) in enumerate(zip(DISSONANT, RESOLVED)):
        ph = np.cumsum(integer_cycles(2 * math.pi * fd * (fr / fd) ** s / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-26)
        pan = (-0.5, -0.17, 0.17, 0.5)[k]
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    noise = sosfilt(butter(2, 1500, "low", fs=SR, output="sos"), rng.standard_normal(NS))
    noise /= np.std(noise) * 3
    static_amp = db(-26) * (1 - s) ** 1.5
    mix[:, 0] += noise * static_amp
    mix[:, 1] += np.roll(noise, 997) * static_amp
    last = -1
    for f in range(N_FRAMES):
        si, step = frames[f]
        key = (si, step // 50)
        if step > 0 and key != last:
            freq = 800 + 1600 * np.clip((acc_f[f] - 0.5) / 0.5, 0, 1)
            n = int(0.05 * SR)
            tt = np.arange(n) / SR
            place(mix, np.sin(2 * math.pi * freq * tt) * np.exp(-tt / 0.008) * db(-22), f / FPS)
            last = key
    bell = A.bell_e6()
    place(mix, bell, T_TEST_A)
    place(mix, A.chime(), T_TEST_A)
    # shuffle: crushed-noise rattle
    n_g = int((SHUFFLE[1] - SHUFFLE[0]) / FPS * SR)
    gl = rng.standard_normal(n_g)
    gl = np.round(gl / np.max(np.abs(gl)) * 7) / 7 * db(-22) * np.hanning(n_g) ** 0.3
    place(mix, gl, SHUFFLE[0] / FPS)
    # coin-flip test: dissonant stinger + impact
    n_s = int(1.6 * SR)
    ts = np.arange(n_s) / SR
    sting = sum(np.sin(2 * math.pi * fr * ts) for fr in (440.0, 466.16, 622.25)) * np.exp(-ts / 0.7) * db(-16) / 3
    place(mix, sting, T_TEST_B)
    place(mix, A.impact(rng) * 0.8, T_TEST_B)
    place(mix, bell * 0.6, 18.2)
    n_w = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_w))
    sw /= np.std(sw) * 3
    tw = np.arange(n_w) / SR
    env = db(-44) * (db(-20) / db(-44)) ** (tw / tw[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, a0 / FPS)
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
