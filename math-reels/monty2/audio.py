"""Sonification for the Monty Hall v2 reel: a door creak for every door that opens (a resonant
filtered-noise sweep), a goat bleat (a short FM wobble) for each goat revealed, a bright ding chord
for the cars, a rain whose density follows games per frame during the 10,000-game race with two
tones that settle on the stay and switch percentages, 98 accelerating creak-clicks for the 100-door
kicker, bells at the reveals. Circular over 25 s."""

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
from scene import (FPS, LOOP_FADE, N_FRAMES, T_H0, T_H1, T_H_PICK, T_HOST, T_PICK, T_R0, T_R1, T_W_HOST, T_W_PICK,  # noqa: E402
                   T_W_REVEAL, T_W_TAG, T_W_TALLY, Scene)

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


def creak(rng, dur=0.45, level=-22, f0=380.0, f1=220.0):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    out = A.bandpass_sweep(noise, f0, f1, dur, q=9.0)
    out /= np.max(np.abs(out)) + 1e-9
    t = np.arange(n) / SR
    env = np.minimum(1.0, t / 0.03) * np.exp(-t / (dur * 0.45))
    return out * env * db(level)


def bleat(level=-24):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = 330.0 * (1 + 0.08 * np.sin(2 * math.pi * 22 * t))
    ph = 2 * math.pi * np.cumsum(f) / SR
    v = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) * np.exp(-t / 0.12) * np.minimum(1.0, t / 0.02)
    return v * db(level)


def ding(freq, level=-20, dur=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2.0 * t) * np.exp(-t / 0.3)) * np.exp(-t / (dur / 3)) * db(level)


def click(rng, level=-30):
    n = int(0.04 * SR)
    t = np.arange(n) / SR
    v = sosfilt(butter(2, [900, 5000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.006)
    return v / (np.max(np.abs(v)) + 1e-9) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(31)
    mix = np.zeros((NS, 2))
    # hook
    place(mix, click(rng, -24), T_PICK)
    place(mix, creak(rng), T_HOST)
    place(mix, bleat(-24), T_HOST + 0.35)
    place(mix, A.bell_e6() * 0.5, T_HOST + 0.8)
    # worlds
    place(mix, click(rng, -24), T_W_PICK)
    for k in range(3):
        place(mix, creak(rng, 0.4, -24), T_W_HOST + k * 0.04)
        place(mix, bleat(-27), T_W_HOST + 0.35 + k * 0.08)
    for k, fr in enumerate((523.25, 659.26, 783.99)):
        place(mix, creak(rng, 0.4, -26), T_W_REVEAL + k * 0.05)
        place(mix, ding(fr, -22, 1.2), T_W_REVEAL + 0.25 + k * 0.08)
    place(mix, A.impact(rng) * 0.4, T_W_TAG)
    place(mix, A.bell_e6() * 0.8, T_W_TALLY)
    # race: rain + two tones settling on the percentages
    games = np.array([sc.games_done(f / FPS) for f in range(N_FRAMES)])
    prev = 0
    for f in range(N_FRAMES):
        t = f / FPS
        if not (T_R0 <= t < T_R1):
            continue
        new = games[f] - prev
        prev = games[f]
        if new > 0:
            m = int(0.03 * SR)
            tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
            tap = sosfilt(butter(2, [2000, 8000], "band", fs=SR, output="sos"), tap)
            amp = db(-34) * min(1.0, 0.25 + math.log10(1 + new) / 3.5)
            place(mix, tap / (np.max(np.abs(tap)) + 1e-9) * amp, t)
    gate = np.zeros(N_FRAMES)
    p_stay = np.full(N_FRAMES, 1 / 3)
    p_sw = np.full(N_FRAMES, 2 / 3)
    for f in range(N_FRAMES):
        t = f / FPS
        n = games[f]
        if T_R0 - 0.3 <= t < 18.2 and n > 0:
            gate[f] = 1.0
            p_stay[f] = sc.cum_stay[n - 1] / n
            p_sw[f] = sc.cum_switch[n - 1] / n
    g = A.onepole_circ_fast(circ_interp(gate), 0.15)
    for p_ctrl, base, pan in ((p_stay, 196.0, -0.5), (p_sw, 196.0, 0.5)):
        f_inst = base * 2 ** (circ_interp(p_ctrl) * 1.2)         # 1/3 -> ~261 Hz, 2/3 -> ~343 Hz
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph)) * db(-27) * g
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    place(mix, A.impact(rng) * 0.5, 16.8)
    place(mix, A.bell_e6() * 0.8, 16.8)
    # hundred doors: a click per opened door, accelerating
    place(mix, click(rng, -24), T_H_PICK)
    for k in range(98):
        t_k = T_H0 + (T_H1 - T_H0) * ((k + 1) / 98)
        place(mix, click(rng, -28 + 4 * (k / 98)), t_k - 0.1)
        if k % 7 == 0:
            place(mix, bleat(-33), t_k)
    place(mix, A.bell_e6() * 0.8, 21.2)
    place(mix, A.impact(rng) * 0.4, 21.2)
    place(mix, ding(659.26, -20, 1.5), 23.5)
    place(mix, A.bell_e6() * 0.7, 23.5)
    # loop swoosh
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts_ = np.arange(n_s) / SR
    env = db(-46) * (db(-22) / db(-46)) ** (ts_ / ts_[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, sw * env, LOOP_FADE[0] - 0.3)
    # pad: a soft minor chord throughout
    for base, pan in ((110.0, -0.4), (130.81, 0.0), (164.81, 0.4)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.15 * np.sin(2 * ph)) * db(-31)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
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
            json.dump(dict(p_stay=sc.p_stay, p_switch=sc.p_switch, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
