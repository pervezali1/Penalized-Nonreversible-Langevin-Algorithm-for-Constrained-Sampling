"""Sonification for the Monty Hall reel: knocks and door clicks for the slow games, a bell when
switching wins and a low blip when staying wins, a rattle whose density follows games-per-frame,
two tones tracking the two win rates, 98 rising clicks for the 100-door host, bells at reveals."""

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

from scene import (B_OPEN0, B_OPEN1, B_REVEAL, FAST_T0, FAST_T1, FPS, GAMES, LOOP_FADE, N_FRAMES,  # noqa: E402
                   SLOW_DT, SLOW_N, SLOW_T0, Scene, smooth)

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


def knock(rng, freq=180.0, seconds=0.12, level=-20):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * math.pi * freq * t) * np.exp(-t / 0.03)
    tap = sosfilt(butter(2, [1500, 5000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.006)
    out = body + tap / (np.max(np.abs(tap)) + 1e-9) * 0.5
    return out / (np.max(np.abs(out)) + 1e-9) * db(level)


def blip(freq=330.0, seconds=0.35, level=-18):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = freq * (0.6 / 1.0) ** (t / seconds)
    return np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(29)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # slow games: pick knock, host door click, reveal: bell (switch won) + blip (stay lost) or reverse
    for g in range(SLOW_N):
        t0 = SLOW_T0 + g * SLOW_DT
        place(mix, knock(rng), t0)
        place(mix, A.click(rng) * 2.5, t0 + 0.12)
        place(mix, knock(rng, 240, 0.08, -24), t0 + 0.27)
        if sc.switch_win[g]:
            place(mix, A.pan(bell * 0.8, 0.5) if hasattr(A, "pan") else np.stack([bell * 0.4, bell * 0.8], 1), t0 + 0.45)
            place(mix, np.stack([blip() * 1.0, blip() * 0.4], 1), t0 + 0.45)
        else:
            place(mix, np.stack([bell * 0.8, bell * 0.4], 1), t0 + 0.45)
            place(mix, np.stack([blip() * 0.4, blip() * 1.0], 1), t0 + 0.45)
    # fast games: rattle density from games per frame; two tones track the win rates
    prev = SLOW_N
    for f in range(int(FAST_T0 * FPS), int(FAST_T1 * FPS) + 1):
        t = f / FPS
        done = sc.games_done(t)
        new = done - prev
        prev = done
        if new > 0:
            m = int(0.03 * SR)
            tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
            tap = sosfilt(butter(2, [2000, 7000], "band", fs=SR, output="sos"), tap)
            amp = db(-30) * min(1.0, 0.2 + math.log10(1 + new) / 2.5)
            place(mix, tap / (np.max(np.abs(tap)) + 1e-9) * amp, t)
    done_f = np.array([sc.games_done(f / FPS) for f in range(N_FRAMES)])
    gate_f = ((done_f > 0) & (np.arange(N_FRAMES) / FPS < LOOP_FADE[0]) & (np.arange(N_FRAMES) / FPS < 13.6)).astype(float)
    gate = A.onepole_circ_fast(circ_interp(gate_f), 0.1)
    for cum, pan in ((sc.cum_stay, -0.5), (sc.cum_switch, 0.5)):
        fr_f = np.array([cum[d] / d if d else 0.5 for d in done_f])
        fr = A.onepole_circ_fast(circ_interp(fr_f), 0.05)
        freq = 220 * 2 ** ((fr - 0.5) * 2)            # 33% -> 175 Hz, 67% -> 277 Hz
        ph = np.cumsum(integer_cycles(2 * math.pi * freq / SR))
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph)) * db(-26) * gate
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    place(mix, bell * 1.2, 11.6)
    place(mix, A.impact(rng) * 0.4, 11.6)
    # 100 doors: a swoosh in, then 98 rising clicks as the host opens goats
    n_s = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, 13.6)
    for i in range(98):
        u = i / 97
        t_i = B_OPEN0 + (B_OPEN1 - B_OPEN0) * (0.5 - 0.5 * math.cos(math.pi * u)) if False else None
        # invert smooth(): find t where 98*smooth(u) == i (monotone), by bisection
        lo, hi = B_OPEN0, B_OPEN1
        for _ in range(30):
            mid = (lo + hi) / 2
            if 98 * smooth((mid - B_OPEN0) / (B_OPEN1 - B_OPEN0)) < i + 0.5:
                lo = mid
            else:
                hi = mid
        nn = int(0.04 * SR)
        tt = np.arange(nn) / SR
        click = np.sin(2 * math.pi * (500 + 1500 * u) * tt) * np.exp(-tt / 0.006) * db(-22)
        place(mix, click, lo)
    place(mix, bell * 1.2, B_REVEAL)
    place(mix, A.chime(), B_REVEAL)
    place(mix, bell * 0.6, 17.8)
    place(mix, sw * env * 0.8, LOOP_FADE[0])
    drone = np.zeros(NS)
    for fr_hz in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr_hz / SR))))
    mix[:, 0] += drone * db(-28)
    mix[:, 1] += drone * db(-28)
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
            json.dump(dict(final=sc.final, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
