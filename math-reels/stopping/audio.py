"""Sonification for the 37% rule reel: every person of the example life plucks a note as they appear, pitched by how
good a match they are (quiet while just looking, louder once ready), a bell on the yes and a bell and impact when it
turns out to be the best; then each of the first six lives ends on its outcome (bell: the best, pluck: someone else,
low blip: alone), a rattle follows lives per frame, a tone tracks the share finding the best; while the curve draws,
a tone follows it (pitch = chance of finding the best) and chimes at 37; a falling tone for the catch. Circular."""

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

from scene import A0, A1, C0, C1, FPS, LIVES0, LOOP_FADE, N, N_FRAMES, R, SLOW_DT, SLOW_N, Scene, smooth  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS
SCALE = (0, 2, 4, 7, 9)                         # major pentatonic


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


def note(rank):
    """Rank 1..100 -> a pentatonic note over two octaves from A3."""
    i = min(10, int(rank / N * 10.999))
    return 220.0 * 2 ** (i // 5 + SCALE[i % 5] / 12)


def blip(freq=200.0, seconds=0.45, level=-17):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = freq * 0.55 ** (t / seconds)
    return np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.15) * db(level)


def tick(rng, level=-30):
    m = int(0.03 * SR)
    tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
    tap = sosfilt(butter(2, [2000, 7000], "band", fs=SR, output="sos"), tap)
    return tap / (np.max(np.abs(tap)) + 1e-9) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(37)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    ranks = sc.ranks[sc.demo]
    # the example life: a pluck per person, panned by position
    for k in range(N):
        t0 = sc.reveal_time(k)
        pan = -0.6 + 1.2 * k / (N - 1)
        if k < R:
            place(mix, A.pluck(note(ranks[k]), 0.35, pan), t0)
        elif k <= sc.dpick:
            place(mix, A.pluck(note(ranks[k]), 0.6, pan), t0)
        else:
            place(mix, tick(rng, -34), t0)
    place(mix, bell * 0.9, sc.T_YES)
    place(mix, A.chime(), sc.T_YES)
    place(mix, bell * 1.2, sc.T_BEST)
    place(mix, A.impact(rng) * 0.35, sc.T_BEST)
    # the first lives, one by one, end on their outcome
    for i in range(SLOW_N):
        t0 = LIVES0 + i * SLOW_DT
        c = int(sc.code[i])
        if c == 0:
            place(mix, bell * 0.8, t0)
        elif c == 1:
            place(mix, A.pluck(note(int(sc.picked_rank[i])), 0.7, 0.0), t0)
        else:
            place(mix, np.stack([blip(), blip()], 1), t0)
    # then the rest: a rattle whose density follows lives per frame
    prev = SLOW_N
    t_fast = LIVES0 + SLOW_N * SLOW_DT
    for f in range(int(t_fast * FPS), int(C0 * FPS)):
        done = sc.lives_done(f / FPS)
        new = done - prev
        prev = done
        if new > 0:
            place(mix, tick(rng) * min(1.0, 0.2 + math.log10(1 + new) / 2.5), f / FPS)
    # a tone tracks the share finding the best while the lives run
    done_f = np.array([sc.lives_done(f / FPS) if LIVES0 <= f / FPS < C0 else 0 for f in range(N_FRAMES)])
    share_f = np.array([sc.cum[0][d] / d if d else 0.37 for d in done_f])
    gate = A.onepole_circ_fast(circ_interp((done_f > 0).astype(float)), 0.1)
    fr = A.onepole_circ_fast(circ_interp(share_f), 0.05)
    ph = np.cumsum(integer_cycles(2 * math.pi * 220 * 2 ** (2 * fr) / SR))
    v = (np.sin(ph) + 0.2 * np.sin(2 * ph)) * db(-27) * gate
    mix += np.stack([v, v], 1)
    place(mix, bell * 1.2, 14.6)
    place(mix, A.impact(rng) * 0.4, 14.6)
    # the curve: pitch follows the chance of finding the best as the cutoff sweeps 0 -> 99
    p_f = np.zeros(N_FRAMES)
    g_f = np.zeros(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        if C0 <= t < C1 + 0.3:
            r = (N - 1) * smooth((t - C0) / (C1 - C0))
            p_f[f] = np.interp(r, np.arange(N), sc.curve)
            g_f[f] = 1.0
    p = A.onepole_circ_fast(circ_interp(p_f), 0.03)
    gate_c = A.onepole_circ_fast(circ_interp(g_f), 0.08)
    ph = np.cumsum(integer_cycles(2 * math.pi * 196 * 2 ** (2.5 * p) / SR))
    v = (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)) * db(-24) * gate_c
    mix += np.stack([v * 0.9, v], 1)
    lo, hi = C0, C1                                  # when the sweep passes 37
    for _ in range(40):
        mid = (lo + hi) / 2
        if (N - 1) * smooth((mid - C0) / (C1 - C0)) < R:
            lo = mid
        else:
            hi = mid
    place(mix, A.chime(), lo)
    # the catch: a falling tone as the "alone" line draws, a low blip at the reveal
    n = int((A1 - A0 + 0.6) * SR)
    t = np.arange(n) / SR
    f = 330 * 0.5 ** (t / (A1 - A0 + 0.6))
    fall = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.05) * np.exp(-t / 0.9) * db(-20)
    place(mix, np.stack([fall * 0.8, fall], 1), A0)
    place(mix, np.stack([blip(150, 0.7, -14), blip(150, 0.7, -14)], 1), A0)
    place(mix, bell * 0.6, 22.45)
    # a swoosh back to the start
    n_s = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env * 0.8, LOOP_FADE[0])
    drone = np.zeros(NS)
    for fr_hz in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr_hz / SR))))
    mix += np.stack([drone, drone], 1) * db(-28)
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
            json.dump(dict(t_yes=sc.T_YES, t_best=sc.T_BEST, chime_37=lo, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
