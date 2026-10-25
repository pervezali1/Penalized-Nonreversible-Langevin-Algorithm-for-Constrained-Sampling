"""Sonification for the initialisation-race reel: three voices, one per network. The random net's
voice is a chord of 8 detuned partials that climbs and resolves as its accuracy rises; the all-zero
net's voice is a single flat tone that never moves; the all-same net's voice is 8 partials locked
to exactly the same pitch (one sound pretending to be many). A soft tick per training step, a ding
when the random net hits 100%, bells at the reveals. Circular over 23 s."""

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
from scene import F0, F1, FPS, LOOP_FADE, N_FRAMES, NAMES, Scene, frame_step, step_frame  # noqa: E402

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


def tick(rng, level=-36):
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, [2500, 9000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.004)
    return noise / (np.max(np.abs(noise)) + 1e-9) * db(level)


def ding(freq, level=-20):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2.0 * t) * np.exp(-t / 0.3)) * np.exp(-t / 0.4) * db(level)


def voice(f_ctrl, partials, pan, level):
    """Sum of partials (ratio, detune in cents, gain) following the pitch control f_ctrl (Hz per sample)."""
    out = np.zeros((NS, 2))
    L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    for ratio, cents, gain in partials:
        ph = np.cumsum(integer_cycles(2 * math.pi * f_ctrl * ratio * 2 ** (cents / 1200) / SR))
        v = np.sin(ph) * gain * db(level)
        out[:, 0] += v * L
        out[:, 1] += v * R
    return out


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(99)
    mix = np.zeros((NS, 2))
    steps = np.array([frame_step(f) for f in range(N_FRAMES)])
    fade = np.array([1 - min(1.0, max(0.0, (f / FPS - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))) for f in range(N_FRAMES)])
    # random: 8 detuned partials whose spread collapses into a clean chord as accuracy rises
    acc_r = sc.acc["random"][steps]
    acc_r = acc_r * fade + sc.acc["random"][0] * (1 - fade)
    u_r = A.onepole_circ_fast(circ_interp(np.clip((acc_r - 0.5) / 0.5, 0, 1) ** 1.5), 0.08)
    f_r = 196.0 * 2 ** (u_r * 0.4)
    rp = rng.standard_normal(8)
    for k in range(8):
        ratio = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)[k]
        cents = circ_interp(np.zeros(N_FRAMES)) + rp[k] * 35 * (1 - u_r)      # detune shrinks to 0 at 100%
        ph = np.cumsum(integer_cycles(2 * math.pi * f_r * ratio * 2 ** (cents / 1200) / SR))
        v = np.sin(ph) * (0.9 / (k + 1)) * db(-26)
        mix[:, 0] += v * math.cos((-0.5 + 1) * math.pi / 4)
        mix[:, 1] += v * math.sin((-0.5 + 1) * math.pi / 4)
    # zeros: one flat tone, never moves
    mix += voice(np.full(NS, 146.83), [(1.0, 0.0, 0.9), (2.0, 0.0, 0.25)], 0.0, -29)
    # same: 8 partials all at exactly the same pitch (phase-aligned, so it is just one louder tone)
    mix += voice(np.full(NS, 220.0), [(1.0, 0.0, 0.14)] * 8 + [(2.0, 0.0, 0.2)], 0.5, -27)
    prev = 0
    for f in range(F0, F1 + 1):
        s = steps[f]
        for st in range(prev + 1, s + 1):
            if st % 4 == 0:
                place(mix, tick(rng, -38), f / FPS + (st - prev - 1) / max(1, s - prev) / FPS)
        prev = s
    s100 = sc.first100["random"]
    if s100 is not None:
        place(mix, ding(523.25, -18), step_frame(s100) / FPS)
    bell = A.bell_e6()
    t_end = F1 / FPS
    place(mix, A.impact(rng) * 0.5, t_end + 0.3)
    place(mix, bell * 0.8, t_end + 0.3)
    place(mix, bell * 0.6, t_end + 2.6)
    place(mix, bell * 0.8, t_end + 5.2)
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts_ = np.arange(n_s) / SR
    env = db(-46) * (db(-22) / db(-46)) ** (ts_ / ts_[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, sw * env, LOOP_FADE[0] - 0.3)
    place(mix, sw * env * 0.7, F0 / FPS - 0.5)
    drone = np.zeros(NS)
    for fr in (49.0, 73.42):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-29)
    mix[:, 1] += drone * db(-29)
    out = A.circ_reverb(mix, rt60=2.4, wet=0.22)
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
            json.dump(dict(first100=sc.first100, final=sc.final, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
