"""Sonification for the Newton's method reel: a tick when each tangent lands, a sparkle burst with
one coin per newly correct digit (1, 2, 3, 6, 12, 24 new digits), a pad whose pitch climbs with the
digits; a swell into the complex plane, a click and a chord tone per iteration, a long filtered
riser under the zoom, bells at the reveals. Circular over 24 s."""

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
from scene import (FPS, ITER_DT, ITER_MAX, ITER_T0, LOOP_FADE, N_FRAMES, PLANE_T0, STEP_T, TANGENT_DT,  # noqa: E402
                   ZOOM_T0, ZOOM_T1, Scene)

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


def coin(freq, level=-30, dur=0.12):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.35 * np.sin(2 * math.pi * freq * 2.01 * t)) * np.exp(-t / (dur / 4)) * db(level)


def tick(rng, level=-22):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * math.pi * 180 * t) * np.exp(-t / 0.012)
    noise = sosfilt(butter(2, [2000, 9000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.006)
    noise /= np.max(np.abs(noise)) + 1e-9
    return (body + 0.6 * noise) * db(level)


def thud(level=-20, f0=70.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * f0 * t * (1 - 0.3 * t)) * np.exp(-t / 0.07) * db(level)


# a pentatonic ladder for the digit sparkles and the iteration tones
LADDER = [440.0 * 2 ** (k / 12) for k in (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33, 36)]


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(55)
    mix = np.zeros((NS, 2))
    # pad control: digits (0..48) on the real line, then iteration depth, then zoom depth
    ctrl = np.zeros(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        if t < PLANE_T0:
            n, phase, u = sc.step_state(t)
            ctrl[f] = math.log2(1 + sc.digits[n]) / math.log2(49)
        elif t < ZOOM_T0:
            k = 0 if t < ITER_T0 else min(ITER_MAX, 1 + int((t - ITER_T0) / ITER_DT))
            ctrl[f] = 0.4 + 0.3 * k / ITER_MAX
        elif t < LOOP_FADE[0]:
            ctrl[f] = 0.7 + 0.3 * min(1.0, (t - ZOOM_T0) / (ZOOM_T1 - ZOOM_T0))
        else:
            ctrl[f] = 0.0
    pitch_ctrl = A.onepole_circ_fast(circ_interp(ctrl), 0.15)
    for base, pan in ((110.0, -0.5), (164.81, -0.15), (220.0, 0.15), (329.63, 0.5)):
        f_inst = base * 2 ** (pitch_ctrl * 1.0)
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)) * db(-29)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # real line: tick when each tangent lands, then one coin per newly correct digit
    for k, ts in enumerate(STEP_T):
        t_land = ts + TANGENT_DT
        place(mix, tick(rng, -20), t_land)
        new = sc.digits[k + 1] - sc.digits[k]
        for j in range(min(new, 24)):
            freq = LADDER[min(len(LADDER) - 1, sc.digits[k] // 3 + j % 8)]
            place(mix, coin(freq, -31 + 2 * min(j, 3) / 3, 0.14), t_land + 0.03 + j * 0.028)
    place(mix, A.bell_e6() * 0.7, 7.2)
    # into the plane: a swell
    n_s = int(0.8 * SR)
    sw = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts_ = np.arange(n_s) / SR
    env = db(-46) * (db(-20) / db(-46)) ** (ts_ / ts_[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, sw * env, PLANE_T0 - 0.8)
    place(mix, A.impact(rng) * 0.5, PLANE_T0)
    place(mix, A.bell_e6() * 0.5, 8.8)
    # iterations: a click and a rising chord tone each
    for k in range(1, ITER_MAX + 1):
        t_k = ITER_T0 + (k - 1) * ITER_DT
        place(mix, tick(rng, -24), t_k)
        place(mix, coin(LADDER[min(len(LADDER) - 1, 3 + k)], -30, 0.25), t_k)
    # zoom: a long riser (filtered noise whose cutoff climbs) under the zoom
    n_z = int((ZOOM_T1 - ZOOM_T0) * SR)
    noise = rng.standard_normal(n_z)
    riser = A.bandpass_sweep(noise, 120.0, 2400.0, ZOOM_T1 - ZOOM_T0, q=3.0)
    riser /= np.max(np.abs(riser)) + 1e-9
    tz = np.arange(n_z) / SR
    envz = db(-40) * (db(-22) / db(-40)) ** (tz / tz[-1])
    envz[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    place(mix, riser * envz, ZOOM_T0)
    bell = A.bell_e6()
    place(mix, bell * 0.8, 14.6)
    place(mix, bell * 0.7, 17.2)
    place(mix, bell * 0.8, 20.0)
    place(mix, A.impact(rng) * 0.6, ZOOM_T1)
    place(mix, thud(-18), 21.8)
    place(mix, bell * 0.6, 21.8)
    place(mix, sw * env * 0.5, LOOP_FADE[0] - 0.3)
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
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
            json.dump(dict(digits=sc.digits, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
