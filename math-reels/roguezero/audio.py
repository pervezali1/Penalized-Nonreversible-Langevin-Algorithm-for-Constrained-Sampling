"""Sonification for the rogue-zero reel. Recap: a run of rising ticks as the formula is rebuilt, a plink for each
zero placed on the line. The first zero's wave is a steady tone; as the zero is pushed off the line a second tone
slides up away from it (220 Hz to 440 Hz following the real part), so the pair goes out of tune. In the wide view a
low growl and a rising tone grow with the wave and carry on through the loop point. A bell and impact for the two
numbers, a bell for "as small as it can be", a chime for the last line. Circular.
    python audio.py <out.wav> [events.json]"""

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
sys.path.insert(0, os.path.join(ROOT, "riemann"))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402

from wsim import BUILD, DUR, FPS, N_FRAMES, PUSH, SWEEP, WAVE_IN, WIPE, ZOOM, sigma  # noqa: E402,F401

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
NS = int(SR * DUR)
TT = np.arange(NS) / SR
SCALE = (0, 2, 4, 7, 9)


def db(x):
    return 10 ** (x / 20)


def integer_cycles(inc):
    C = inc.sum() / (2 * math.pi)
    return inc + 2 * math.pi * ((round(C) - C) / DUR) / SR


def place(buf, clip, t0):
    i0 = int(round(t0 * SR)) % NS
    idx = (i0 + np.arange(len(clip))) % NS
    if buf.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    np.add.at(buf, idx, clip)


def plink(freq, level, tau_=0.05, seconds=0.25):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * 2.01 * freq * t)) * np.exp(-t / tau_)
    return out * np.minimum(1, t / 0.001) * level


def swoosh(rng, seconds, level):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = sosfilt(butter(2, [500, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    x /= np.std(x) * 3
    return x * np.sin(np.pi * t / seconds) ** 2 * db(level)


def ramp(t0, t1):
    return np.clip((TT - t0) / (t1 - t0), 0, 1)


def render(out_wav, events_json=None):
    rng = np.random.default_rng(80)
    mix = np.zeros((NS, 2))
    bell = A.bell_e6()
    # recap: rising ticks while the formula is rebuilt, a plink per zero placed on the line
    for k in range(36):
        u = k / 35
        place(mix, plink(600 * 2 ** (1.5 * u), db(-27), 0.02, 0.1), BUILD[0] + (BUILD[1] - BUILD[0]) * u ** 0.7)
    for k in range(8):
        place(mix, plink(523.25 * 2 ** (SCALE[k % 5] / 12 + k // 5), db(-19), 0.12, 0.6), BUILD[0] + 0.2 * k)
    # the first zero's wave: a steady tone from when it is drawn, until the picture clears
    on = ramp(WAVE_IN[0], WAVE_IN[1]) * (1 - ramp(WIPE[0], WIPE[1])) + (TT < WIPE[0]) * 1.0
    on = np.clip(on, 0, 1)
    base = np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * 220.0 / SR))))
    mix += np.stack([base, base], 1) * (on * db(-27))[:, None]
    # the pushed zero: a second tone that slides away from it, 220 Hz at 0.5 up to 440 Hz at 0.8
    sig = np.array([sigma(t if t >= WIPE[0] else t + DUR) for t in np.arange(N_FRAMES + 1) / FPS])
    sig = np.interp(TT, np.arange(N_FRAMES + 1) / FPS, sig)
    off = (sig - 0.5) / 0.3
    off = off * (1 - ramp(WIPE[0], WIPE[1]) * (TT < PUSH[0]))        # silent between the wipe and the push
    f2 = 220.0 * 2 ** np.clip(off, 0, 1)
    rogue = np.sin(np.cumsum(integer_cycles(2 * math.pi * f2 / SR)))
    mix += np.stack([rogue * 0.8, rogue], 1) * (np.clip(off * 3, 0, 1) * db(-23))[:, None]
    # the wide view: a growl that grows with the wave and carries through the loop point
    grow = np.where(TT < WIPE[0], 1.0, np.where(TT < SWEEP[0], 0.0, 0.25 + 0.75 * ramp(SWEEP[0], SWEEP[1] + 6)))
    grow = grow * (1 - ramp(WIPE[0], WIPE[1]) * (TT < SWEEP[0]))
    grow = A.onepole_circ_fast(grow, 0.15)
    noise = A.circ_filter(butter(2, [40, 220], "band", fs=SR, output="sos"), rng.standard_normal((NS, 2)))   # circular: no click at the loop
    noise /= np.std(noise) * 3
    trem = 1 + 0.5 * np.sin(2 * math.pi * 9 * TT)
    mix += noise * (grow * trem * db(-13))[:, None]
    n = int((SWEEP[1] - SWEEP[0]) * SR)
    t = np.arange(n) / SR
    rise = np.sin(2 * math.pi * np.cumsum(110 * 2 ** (2.0 * t / t[-1])) / SR) * (t / t[-1]) ** 1.5 * db(-22)
    rise[-int(0.2 * SR):] *= np.linspace(1, 0, int(0.2 * SR))
    place(mix, rise, SWEEP[0])
    # the moments
    place(mix, swoosh(rng, 0.6, -20), WIPE[0] - 0.05)
    place(mix, A.chime(), 5.2)
    place(mix, A.pluck(98.0, 0.8, 0.0), 7.6)
    place(mix, A.pluck(130.8, 0.7, 0.0), 10.6)
    place(mix, swoosh(rng, 0.5, -20), ZOOM[0] - 0.1)
    place(mix, bell * 1.25, 19.8)
    place(mix, A.impact(rng) * 0.45, 19.8)
    place(mix, bell * 0.7, 23.2)
    place(mix, A.chime(), 26.4)
    drone = np.zeros(NS)
    for fr_hz, lev in ((55.0, -29), (82.41, -31)):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr_hz / SR)))) * db(lev)
    mix += np.stack([drone, drone], 1)
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
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
