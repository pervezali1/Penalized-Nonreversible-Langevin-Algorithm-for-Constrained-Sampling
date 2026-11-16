"""Sonification for the 3D neurons reel (no narration): a four-note chord that glides from a cluster to a major
chord as the network's accuracy rises (training in act A, the neurons being added in act D), a pluck as each
layer-1 sheet rises (act B), a brighter one for each layer-2 sheet (act C), a bell for every neuron added to the
output, pitched by the accuracy it reaches, and a soft swell for the loop. Circular over 30.000 s, -14 LUFS."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pendulum"))
import audio as A  # noqa: E402  (pendulum sonification helpers)

sys.path.insert(0, HERE)
from scene import (ADD_DT, ADD_T0, BACK, FPS, N_FRAMES, T_B, T_C, T_D, TILE_DT, TILE_RISE0, Scene)  # noqa: E402

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS
DISSONANT = [220.0, 233.08, 246.94, 261.63]
RESOLVED = [220.0, 277.18, 329.63, 440.0]
PENTA = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25]


def db(x):
    return 10 ** (x / 20)


def circ_interp(frame_values):
    xp = np.append(FRAME_T, DUR)
    return np.interp(TT, xp, np.append(frame_values, frame_values[0]))


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
    rng = np.random.default_rng(65)
    events = []
    acc_f = np.array([sc.accuracy(t)[0] for t in FRAME_T])
    acc = A.onepole_circ_fast(circ_interp(acc_f), 0.12)
    s = np.clip((acc - 0.5) / 0.46, 0, 1)
    s = s * s * (3 - 2 * s)
    # the chord sits back while the neurons are on show (acts B and C)
    lvl_f = np.where((FRAME_T >= T_B) & (FRAME_T < T_D), -6.0, 0.0)
    lvl = db(A.onepole_circ_fast(circ_interp(lvl_f), 0.3))

    mix = np.zeros((NS, 2))
    for k, (fd, fr) in enumerate(zip(DISSONANT, RESOLVED)):
        f_inst = fd * (fr / fd) ** s
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-27) * lvl
        pan = (-0.5, -0.17, 0.17, 0.5)[k]
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # a breath of noise that clears as the accuracy rises
    noise = A.circ_filter(butter(2, 1400, "low", fs=SR, output="sos"), rng.standard_normal(NS))
    noise /= np.std(noise) * 3
    amp = db(-30) * (1 - s) ** 1.5
    mix[:, 0] += noise * amp
    mix[:, 1] += np.roll(noise, 1013) * amp

    # act B: a pluck per layer-1 sheet; act C: a brighter one per layer-2 sheet
    for j in range(8):
        tb = T_B - 0.6 + TILE_RISE0 + TILE_DT * j
        place(mix, A.pluck(PENTA[j], 0.55, -0.6 + 0.17 * j) * 0.55, tb)
        events.append((f"layer 1 sheet {j + 1}", round(tb, 3)))
        tc = T_C - 0.6 + TILE_RISE0 + TILE_DT * j
        place(mix, A.pluck(PENTA[j] * 2, 0.5, 0.6 - 0.17 * j) * 0.45, tc)
        events.append((f"layer 2 sheet {j + 1}", round(tc, 3)))
    # act D: a bell per neuron added, pitched by the accuracy it brings
    bell = A.bell_e6()
    for k in range(1, 9):
        td = ADD_T0 + (k - 1) * ADD_DT
        a_k = sc.partial_acc[k]
        semis = round(14 * (a_k - 0.5) / 0.46)
        clip = bell if semis == 0 else resample_poly(bell, 100, int(round(100 * 2 ** (semis / 12))))
        place(mix, clip * (0.35 + 0.35 * (k == 8)), td)
        events.append((f"neuron {k} added: {a_k:.3f}", round(td, 3)))
    place(mix, A.chime() * 0.8, ADD_T0 + 7 * ADD_DT + 0.5)
    # the loop: a short swell as the trained surface melts back
    n_s = int((BACK[1] - BACK[0]) * SR)
    sw = sosfilt(butter(2, 900, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-22) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, BACK[0])

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
            json.dump(dict(events=events, lufs=loud, true_peak=tp), fh, indent=1)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
