"""Sonification for the Monte Carlo pi reel: a dart tick per throw while they can be heard one by
one, then a rain whose density follows darts per frame; a coin when a new digit locks in and a dull
knock when one unlocks; a pad whose pitch climbs with the correct digits; a swell into the funnel,
a descending filtered sweep as the band narrows, bells at the reveals. Circular over 23 s."""

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
from scene import FPS, FUNNEL_T0, FUNNEL_T1, LOOP_FADE, N_FRAMES, RAIN_T0, RAIN_T1, Scene, digits_correct  # noqa: E402

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


def dart(rng, inside, level=-24):
    """A short tick: brighter, higher for a dart inside the circle."""
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    f0 = 1400.0 if inside else 700.0
    body = np.sin(2 * math.pi * f0 * t) * np.exp(-t / 0.008)
    noise = sosfilt(butter(2, [3000, 10000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.004)
    noise /= np.max(np.abs(noise)) + 1e-9
    return (body + 0.5 * noise) * db(level)


def coin(freq, level=-28, dur=0.18):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.35 * np.sin(2 * math.pi * freq * 2.01 * t)) * np.exp(-t / (dur / 4)) * db(level)


def knock(level=-26):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * 160 * t * (1 - 0.4 * t)) * np.exp(-t / 0.03) * db(level)


def thud(level=-20, f0=70.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * f0 * t * (1 - 0.3 * t)) * np.exp(-t / 0.07) * db(level)


LADDER = [523.25 * 2 ** (k / 12) for k in (0, 4, 7, 12, 16, 19, 24)]


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(66)
    mix = np.zeros((NS, 2))
    # per-frame digits and darts
    digits = np.zeros(N_FRAMES, int)
    darts = np.zeros(N_FRAMES, int)
    for f in range(N_FRAMES):
        t = f / FPS
        n = sc.darts_shown(t)
        darts[f] = n
        digits[f] = digits_correct(4 * sc.cum_in[n - 1] / n) if n else 0
    ctrl = np.zeros(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        if t < FUNNEL_T0 - 0.3:
            ctrl[f] = digits[f] / 4.0
        elif t < LOOP_FADE[0]:
            u = min(1.0, max(0.0, (t - FUNNEL_T0) / (FUNNEL_T1 - FUNNEL_T0)))
            ctrl[f] = 0.75 + 0.25 * u
    pitch_ctrl = A.onepole_circ_fast(circ_interp(ctrl), 0.15)
    for base, pan in ((130.81, -0.5), (196.0, -0.15), (261.63, 0.15), (392.0, 0.5)):
        f_inst = base * 2 ** (pitch_ctrl * 0.6)
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)) * db(-29)
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # darts: one tick each while few per frame, then a rain
    prev = 0
    for f in range(N_FRAMES):
        t = f / FPS
        if not (RAIN_T0 <= t < RAIN_T1):
            continue
        n = darts[f]
        new = n - prev
        if new <= 0:
            prev = n
            continue
        if new <= 3:
            for j in range(prev, n):
                place(mix, dart(rng, sc.inside[j], -24), t + j * 0.004)
        else:
            m = int(0.03 * SR)
            tap = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SR))
            tap = sosfilt(butter(2, [2500, 9000], "band", fs=SR, output="sos"), tap)
            amp = db(-33) * min(1.0, 0.25 + math.log10(1 + new) / 4)
            place(mix, tap / (np.max(np.abs(tap)) + 1e-9) * amp, t)
        prev = n
    # digit events: coin on lock-in, knock on loss
    n_lock = n_unlock = 0
    for f in range(1, N_FRAMES):
        t = f / FPS
        if not (RAIN_T0 <= t < RAIN_T1):
            continue
        if digits[f] > digits[f - 1]:
            place(mix, coin(LADDER[min(len(LADDER) - 1, digits[f])], -27), t)
            n_lock += 1
        elif digits[f] < digits[f - 1]:
            place(mix, knock(-28), t)
            n_unlock += 1
    bell = A.bell_e6()
    place(mix, bell * 0.8, 11.2)
    place(mix, A.impact(rng) * 0.4, 11.2)
    # into the funnel: a swell, then a descending sweep as the band narrows
    n_s = int(0.7 * SR)
    sw = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts_ = np.arange(n_s) / SR
    env = db(-46) * (db(-20) / db(-46)) ** (ts_ / ts_[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, sw * env, FUNNEL_T0 - 0.7)
    n_f = int((FUNNEL_T1 - FUNNEL_T0) * SR)
    sweep = A.bandpass_sweep(rng.standard_normal(n_f), 3000.0, 150.0, FUNNEL_T1 - FUNNEL_T0, q=3.0)
    sweep /= np.max(np.abs(sweep)) + 1e-9
    tz = np.arange(n_f) / SR
    envf = db(-24) * (db(-38) / db(-24)) ** (tz / tz[-1])
    envf[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    place(mix, sweep * envf, FUNNEL_T0)
    place(mix, bell * 0.7, 16.0)
    place(mix, thud(-18), 18.8)
    place(mix, bell * 0.6, 18.8)
    place(mix, bell * 0.8, 21.0)
    place(mix, sw * env * 0.5, LOOP_FADE[0] - 0.3)
    drone = np.zeros(NS)
    for fr in (65.41, 98.0):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-29)
    mix[:, 1] += drone * db(-29)
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
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}, "
          f"digit lock-ins {n_lock}, losses {n_unlock}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(est=sc.est_final, digits=sc.digits_final, lock_ins=n_lock, losses=n_unlock,
                           lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
