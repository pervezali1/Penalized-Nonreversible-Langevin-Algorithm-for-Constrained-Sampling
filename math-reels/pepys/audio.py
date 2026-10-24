"""Sonification for the Newton-Pepys reel: dice clatter per roll, a ding per six in the slow
rolls, a rattle whose density follows rolls-per-frame in the fast phase, three tones whose
pitches track the three running win rates, bells at the reveal. Circular over 20.000 s."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

from scene import BETS, FAST_T0, FAST_T1, FPS, LOOP_FADE, N_FRAMES, SLOW_DT, SLOW_N, SLOW_T0, TRIALS, TUMBLE, Scene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
import importlib.util  # noqa: E402

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


def clatter(rng, n_dice, seconds=0.3):
    """n_dice short filtered noise taps spread over the tumble."""
    n = int(seconds * SR) + int(0.05 * SR)
    out = np.zeros(n)
    for _ in range(n_dice):
        t0 = int(rng.uniform(0, seconds) * SR)
        m = int(0.02 * SR)
        tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.004 * SR))
        out[t0:t0 + m] += tap
    out = sosfilt(butter(2, [1500, 6000], "band", fs=SR, output="sos"), out)
    return out / (np.max(np.abs(out)) + 1e-9) * db(-20)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(21)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # slow rolls
    for k in range(SLOW_N):
        t0 = SLOW_T0 + k * SLOW_DT
        place(mix, clatter(rng, 36), t0)
        for bi, (name, n, kk, label) in enumerate(BETS):
            sixes = int((sc.faces[bi][k] == 6).sum())
            for j in range(sixes):
                place(mix, A.pluck(A.PENTA[(bi * 3 + j) % 10] * 2, 0.5, (bi - 1) * 0.5) * 0.5, t0 + TUMBLE + 0.03 * j)
            if sc.wins[bi][k]:
                place(mix, bell * 0.35, t0 + TUMBLE + 0.1)
    # fast phase: rattle density follows trials per frame; three tones track the win rates
    prev = SLOW_N
    for f in range(int(FAST_T0 * FPS), int(FAST_T1 * FPS) + 1):
        t = f / FPS
        done = sc.trials_done(t)
        new = done - prev
        prev = done
        if new > 0:
            m = int(0.03 * SR)
            tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
            tap = sosfilt(butter(2, [2000, 7000], "band", fs=SR, output="sos"), tap)
            amp = db(-30) * min(1.0, 0.2 + math.log10(1 + new) / 2.5)
            place(mix, tap / (np.max(np.abs(tap)) + 1e-9) * amp, t)
    done_f = np.array([sc.trials_done(f / FPS) for f in range(N_FRAMES)])
    gate_f = ((done_f > 0) & (np.arange(N_FRAMES) / FPS < LOOP_FADE[0])).astype(float)
    gate = A.onepole_circ_fast(circ_interp(gate_f), 0.1)
    for bi, pan in ((0, -0.5), (1, 0.0), (2, 0.5)):
        fr_f = np.array([sc.cum[bi][d] / d if d else 0.6 for d in done_f])
        fr = A.onepole_circ_fast(circ_interp(fr_f), 0.05)
        freq = 220 * 2 ** ((fr - 0.5) * 4)            # 50% -> 220 Hz, 66.5% -> 347 Hz
        ph = np.cumsum(integer_cycles(2 * math.pi * freq / SR))
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph)) * db(-27) * gate
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    place(mix, bell * 1.2, 15.2)
    place(mix, A.impact(rng) * 0.5, 15.2)
    place(mix, A.chime(), 2.6)
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-20) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, LOOP_FADE[0])
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
            json.dump(dict(final=sc.final, exact=sc.exact, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
