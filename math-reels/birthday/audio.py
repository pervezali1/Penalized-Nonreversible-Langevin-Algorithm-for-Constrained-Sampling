"""Sonification for the birthday reel: a pluck per person (note from the birthday), bells on
matches, a ticking rain while 10,000 rooms fill with a tone converging on the match rate,
a swell under the 253 chords, and a resolved chord at 99.9%. Circular over 20.000 s."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pendulum"))
import audio as A  # noqa: E402
from scene import (CHORD_T0, CHORD_T1, GRID_T0, GRID_T1, LOOP_FADE, N_FRAMES, FPS, Scene,  # noqa: E402
                   arrival, smooth)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
PENTA = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.26, 783.99]


def db(x):
    return 10 ** (x / 20)


def place(buf, clip, t0):
    i0 = int(round(t0 * SR)) % NS
    idx = (i0 + np.arange(len(clip))) % NS
    if buf.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    np.add.at(buf, idx, clip)


def integer_cycles(inc):
    C = inc.sum() / (2 * math.pi)
    return inc + 2 * math.pi * ((round(C) - C) / DUR) / SR


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(5)
    mix = np.zeros((NS, 2))
    events = []
    bell = A.bell_e6()

    # --- people: pluck on landing, bell on a match
    for k in range(1, 71):
        t0, fl = arrival(k)
        tl = t0 + fl
        day = int(sc.room[k - 1])
        ang = 2 * math.pi * day / 365 - math.pi / 2
        pan = 0.8 * math.cos(ang)
        vel = 0.9 if k <= 23 else 0.55
        place(mix, A.pluck(PENTA[day % 10] * (1 if k <= 23 else 2), vel, pan) * 0.9, tl)
        if sc.stack[k - 1] > 0:
            place(mix, bell * (1.0 if k <= 23 else 0.6), tl)
            place(mix, A.impact(rng) * (0.5 if k <= 23 else 0.25), tl)
            events.append(("match", round(tl, 3), k))
    # reveal at 23
    place(mix, A.chime(), 7.45)

    # --- act B: ticking rain + tone converging on the running match rate
    n_prev = 0
    tick_i = 0
    for f in range(int(GRID_T0 * FPS), int(GRID_T1 * FPS) + 1):
        t = f / FPS
        n = 10000 if t >= GRID_T1 else int(10000 * smooth((t - GRID_T0) / (GRID_T1 - GRID_T0)))
        for cell in range(n_prev, n):
            if cell % 100 == 0:
                rate = sc.grid_match[:cell + 1].mean() if cell else 0.5
                freq = 300 + 900 * rate
                nn = int(0.04 * SR)
                tt = np.arange(nn) / SR
                tick = np.sin(2 * math.pi * freq * tt) * np.exp(-tt / 0.006) * db(-20)
                place(mix, tick, t + (cell - n_prev) / max(1, n - n_prev) / FPS)
                tick_i += 1
        n_prev = n
    # rising noise bed under the grid
    n_b = int((GRID_T1 - GRID_T0) * SR)
    bed = sosfilt(butter(2, [800, 4000], "band", fs=SR, output="sos"), rng.standard_normal(n_b))
    bed /= np.std(bed) * 3
    tb = np.arange(n_b) / SR
    env = db(-40) * (db(-24) / db(-40)) ** (tb / tb[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, bed * env, GRID_T0)
    place(mix, bell, 13.9)
    events.append(("count", 13.9, sc.n_match))

    # --- act C: swell under the chords + ascending plucks
    n_c = int((CHORD_T1 - CHORD_T0 + 0.2) * SR)
    cy = sosfilt(butter(2, 3000, "high", fs=SR, output="sos"), rng.standard_normal(n_c))
    cy /= np.std(cy) * 3
    tc = np.arange(n_c) / SR
    env = db(-46) * (db(-20) / db(-46)) ** (tc / tc[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, cy * env, CHORD_T0 - 0.2)
    for i in range(12):
        place(mix, A.pluck(PENTA[i % 10] * (2 if i >= 10 else 1), 0.5, (i / 11 - 0.5) * 1.2) * 0.6,
              CHORD_T0 + (CHORD_T1 - CHORD_T0) * i / 12)

    # --- act D: resolved A-major chord at 99.9%, then fades with the loop dissolve
    t_hit = 17.85 + 0.025 * 46 + 0.18
    for fr, pan in ((220.0, -0.4), (277.18, -0.1), (329.63, 0.1), (440.0, 0.4)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR)))
        amp = np.zeros(NS)
        a0, a1 = int(t_hit * SR), int(LOOP_FADE[1] * SR)
        ramp = np.clip((TT - t_hit) / 0.15, 0, 1)
        fade = 1 - np.clip((TT - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]), 0, 1)
        amp = ramp * fade * db(-24)
        amp[: int((t_hit - 0.2) * SR)] = 0
        v = (np.sin(ph) + 0.2 * np.sin(2 * ph)) * amp
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    place(mix, bell * 1.2, t_hit)
    events.append(("99.9", round(t_hit, 3)))
    # loop dissolve: short reversed swell
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-20) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, LOOP_FADE[0])

    # --- drone
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    drone *= db(-28)
    mix[:, 0] += drone
    mix[:, 1] += drone

    # --- master
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
    print(f"ticks {tick_i}, loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, "
          f"seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(events=events, lufs=loud, true_peak=tp, first_match=sc.first_match,
                           rooms_with_match=sc.n_match), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
