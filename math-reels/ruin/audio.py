"""Sonification for the gambler's ruin reel: a chip-click rain whose density follows bets per
frame, a sinking pad whose pitch tracks the average bankroll, a thud for every 25 ruins, one
big flip and a split chord for the all-in, bells at the reveals. Circular over 22 s."""

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

from scene import ALLIN_HIT, ALLIN_T0, FPS, LOOP_FADE, MAX_BETS, N, N_FRAMES, SIM_T0, SIM_T1, Scene  # noqa: E402

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


def thud(level=-20):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * 70 * t * (1 - 0.3 * t)) * np.exp(-t / 0.07) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(33)
    mix = np.zeros((NS, 2))
    # mean bankroll per frame of the alive+resolved population (0..200)
    mean_bank = []
    for f in range(N_FRAMES):
        t = f / FPS
        if t < SIM_T0 or t >= ALLIN_T0:
            mean_bank.append(100.0)
        else:
            b = sc.bets_shown(t)
            s = min(sc.paths.shape[1] - 1, b // 7)
            mean_bank.append(float(sc.paths[:, s].mean()))
    mean_bank = np.array(mean_bank)
    loopf = np.clip((np.arange(N_FRAMES) / FPS - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]), 0, 1)
    pitch_ctrl = A.onepole_circ_fast(circ_interp(np.clip(mean_bank / 100.0, 0, 1.2)), 0.1)
    # sinking pad: minor chord whose root glides down as the mean bankroll falls
    for k, (base, pan) in enumerate(((220.0, -0.5), (261.63, -0.15), (329.63, 0.15), (440.0, 0.5))):
        f_inst = base * 2 ** ((pitch_ctrl - 1) * 1.0)
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-27)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # chip clicks: density follows bets per frame
    prev = 0
    for f in range(N_FRAMES):
        t = f / FPS
        if not (SIM_T0 <= t < SIM_T1):
            continue
        b = sc.bets_shown(t)
        new = b - prev
        prev = b
        if new > 0:
            m = int(0.025 * SR)
            tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.004 * SR))
            tap = sosfilt(butter(2, [2500, 7000], "band", fs=SR, output="sos"), tap)
            amp = db(-32) * min(1.0, 0.2 + math.log10(1 + new) / 3)
            place(mix, tap / (np.max(np.abs(tap)) + 1e-9) * amp, t)
    # a thud for every 25 ruins
    ruin_bets = np.sort(sc.end_bet[sc.broke])
    for k in range(0, len(ruin_bets), 25):
        b = ruin_bets[k]
        # invert bets_shown: t = SIM_T0 + (SIM_T1-SIM_T0) * (b/MAX)^(1/1.4)
        t_k = SIM_T0 + (SIM_T1 - SIM_T0) * (b / MAX_BETS) ** (1 / 1.4)
        place(mix, thud(-24), t_k)
    bell = A.bell_e6()
    place(mix, bell * 0.8, 11.0)
    place(mix, A.impact(rng) * 0.5, 11.0)
    # all in: a swoosh, one big flip, split chord
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-16) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, ALLIN_HIT - 0.6)
    place(mix, A.impact(rng) * 0.9, ALLIN_HIT)
    place(mix, bell * 1.0, ALLIN_HIT)
    place(mix, np.stack([bell * 0.5, bell * 0.5], 1), 15.0)
    place(mix, bell * 0.8, 17.0)
    place(mix, thud(-16), 19.4)
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
            json.dump(dict(broke=sc.n_broke_final, doubled=sc.n_dbl_final, allin=sc.n_allin, p_slow=sc.p_slow,
                           lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
