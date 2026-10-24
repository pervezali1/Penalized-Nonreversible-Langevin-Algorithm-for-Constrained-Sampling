"""Sonification for the Martingale reel: chip clicks per spin, a soft coin for each round the solo
gambler wins, ten escalating hits for the losing streak (one per doubled bet) ending in an impact,
a chip-click rain and a thud per 25 busts for the crowd, a pad that sinks with the players' net,
bells at the reveals. Circular over 24 s."""

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
from scene import (BANK, CROWD_T0, CROWD_T1, FPS, LOOP_FADE, N, N_FRAMES, SOLO_IDX, SOLO_T0, SPINS, STREAK,  # noqa: E402
                   STREAK_DT, STREAK_T0, Scene)

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


def thud(level=-20, f0=70.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * f0 * t * (1 - 0.3 * t)) * np.exp(-t / 0.07) * db(level)


def hit(k, rng):
    """Escalating streak hit k (0..9): a pitched drum whose pitch and level rise with the doubled bet."""
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f0 = 90.0 * 2 ** (k / 6.0)
    body = np.sin(2 * math.pi * f0 * t * (1 - 0.25 * np.minimum(t, 0.2))) * np.exp(-t / (0.05 + 0.012 * k))
    noise = sosfilt(butter(2, [1500 + 400 * k, 9000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.02)
    noise /= np.max(np.abs(noise)) + 1e-9
    return (body + 0.45 * noise) * db(-20 + 1.4 * k)


def coin(rng):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    f = 2200.0 * 2 ** (rng.integers(0, 5) / 12)
    return (np.sin(2 * math.pi * f * t) + 0.4 * np.sin(2 * math.pi * f * 2.76 * t)) * np.exp(-t / 0.025) * db(-30)


def chip(rng, level):
    m = int(0.025 * SR)
    tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.004 * SR))
    tap = sosfilt(butter(2, [2500, 7000], "band", fs=SR, output="sos"), tap)
    return tap / (np.max(np.abs(tap)) + 1e-9) * level


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(44)
    mix = np.zeros((NS, 2))
    # pad control: 1.0 = everyone at the start, falls with the players' net (crowd) or follows the solo bankroll
    ctrl = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        if SOLO_T0 <= t < CROWD_T0:
            ctrl[f] = sc.solo[sc.solo_spins(t)] / BANK
        elif t >= CROWD_T0 and t < LOOP_FADE[0]:
            n = sc.crowd_spins(t)
            ctrl[f] = sc.paths_full[:, n].mean() / BANK * 0.5 + 0.5 * (N - int((sc.bust <= n).sum())) / N
    pitch_ctrl = A.onepole_circ_fast(circ_interp(np.clip(ctrl, 0, 1.5)), 0.12)
    for base, pan in ((220.0, -0.5), (261.63, -0.15), (329.63, 0.15), (440.0, 0.5)):
        f_inst = base * 2 ** ((pitch_ctrl - 1) * 0.8)
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-28)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # solo: chip clicks per frame of new spins, a coin per won round (thinned), the streak hits
    prev = 0
    for f in range(N_FRAMES):
        t = f / FPS
        if not (SOLO_T0 <= t < STREAK_T0):
            continue
        n = sc.solo_spins(t)
        new = n - prev
        if new > 0:
            place(mix, chip(rng, db(-34) * min(1.0, 0.3 + math.log10(1 + new) / 2.5)), t)
            wins = int(sc.win[SOLO_IDX, prev:n].sum())
            if wins and f % 2 == 0:
                place(mix, coin(rng) * min(1.0, 0.4 + wins / 6), t + 0.004)
        prev = n
    for k in range(STREAK):
        place(mix, hit(k, rng), STREAK_T0 + k * STREAK_DT)
    t_bust = STREAK_T0 + (STREAK - 1) * STREAK_DT
    place(mix, A.impact(rng) * 0.9, t_bust)
    place(mix, thud(-14, 50.0), t_bust + 0.03)
    place(mix, A.bell_e6() * 0.6, 10.5)
    # crowd: chip rain by spins per frame, a thud per 25 busts
    prev = 0
    for f in range(N_FRAMES):
        t = f / FPS
        if not (CROWD_T0 <= t < CROWD_T1):
            continue
        n = sc.crowd_spins(t)
        new = n - prev
        prev = n
        if new > 0:
            place(mix, chip(rng, db(-33) * min(1.0, 0.2 + math.log10(1 + new) / 3)), t)
    bust_spins = np.sort(sc.bust[sc.busted])
    for k in range(0, len(bust_spins), 25):
        b = bust_spins[k]
        t_k = CROWD_T0 + (CROWD_T1 - CROWD_T0) * (b / SPINS) ** (1 / 1.3)
        place(mix, thud(-24), t_k)
    bell = A.bell_e6()
    place(mix, bell * 0.9, 15.8)
    place(mix, A.impact(rng) * 0.5, 15.8)
    place(mix, bell * 0.7, 17.8)
    place(mix, bell * 0.8, 20.2)
    place(mix, thud(-16), 22.4)
    place(mix, bell * 0.6, 22.4)
    # loop swoosh into the fade
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env * 0.6, LOOP_FADE[0])
    drone = np.zeros(NS)
    for fr in (55.0, 65.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-28)
    mix[:, 1] += drone * db(-28)
    out = A.circ_reverb(mix, rt60=2.2, wet=0.2)
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
            json.dump(dict(busted=sc.n_bust, alive=sc.n_alive, net=sc.net_final, solo_bust=sc.solo_bust,
                           lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
