"""Sonification for the descent reel: a pad whose pitch falls with the mean loss, a pop for every
batch of particles that reaches the best answer, a tremolo driven by the particles' speed in the
momentum run, bells at the reveals. Circular over 20.000 s."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
import importlib.util  # noqa: E402

from landscape import FPS, LOOP, N_FRAMES, RESET, RUN_A, RUN_B, F_MIN, found, loss  # noqa: E402

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


def render(data_path, out_wav, events_json=None):
    d = np.load(data_path)
    pos = d["pos"]
    rng = np.random.default_rng(9)
    events = []
    mean_loss = np.array([loss(pos[f]).mean() for f in range(N_FRAMES)])
    n_found = np.array([found(pos[f]).sum() for f in range(N_FRAMES)])
    speed = np.linalg.norm(pos - np.roll(pos, 1, axis=0), axis=2).mean(axis=1)
    speed[0] = speed[1]

    mix = np.zeros((NS, 2))
    # --- pad: pitch falls as the mean loss falls (one octave over the full range)
    u = circ_interp(np.clip((mean_loss - F_MIN) / 1.2, 0, 1))
    u = A.onepole_circ_fast(u, 0.05)
    trem = circ_interp(np.clip(speed / 0.02, 0, 1))
    trem = A.onepole_circ_fast(trem, 0.02)
    for k, (base, pan) in enumerate(((220.0, -0.5), (277.18, -0.15), (329.63, 0.15), (440.0, 0.5))):
        f_inst = base * 2 ** (u - 1)
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        lfo = 1 - 0.6 * trem * (0.5 + 0.5 * np.sin(2 * math.pi * 7 * TT))
        v = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * db(-27) * lfo
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # --- flowing texture: filtered noise whose level follows the particles' speed
    noise = sosfilt(butter(2, [300, 2500], "band", fs=SR, output="sos"), rng.standard_normal(NS))
    noise /= np.std(noise) * 3
    flow = circ_interp(np.clip(speed / 0.01, 0, 1)) * db(-30)
    mix[:, 0] += noise * flow
    mix[:, 1] += np.roll(noise, 1201) * flow
    # --- a pop for every 100 particles that newly reach the best answer
    acc_count = 0
    for f in range(1, N_FRAMES):
        if f in range(RESET[0], RESET[1]) or f in range(LOOP[0], N_FRAMES):
            acc_count = 0
            continue
        gained = max(0, int(n_found[f]) - int(n_found[f - 1]))
        acc_count += gained
        pops = acc_count // 100
        acc_count -= pops * 100
        for j in range(min(pops, 6)):
            nn = int(0.06 * SR)
            tt = np.arange(nn) / SR
            freq = 1200 + 600 * rng.random()
            pop = np.sin(2 * math.pi * freq * tt) * np.exp(-tt / 0.012) * db(-23)
            place(mix, pop, f / FPS + j * 0.004)
    # --- reveals, reset, final
    bell = A.bell_e6()
    place(mix, bell, 7.6)
    place(mix, A.impact(rng) * 0.4, 7.6)
    events.append(("reveal_a", 7.6))
    n_s = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, RESET[0] / FPS)
    place(mix, A.chime(), RUN_B[0] / FPS)
    place(mix, bell * 1.2, 16.6)
    place(mix, A.impact(rng) * 0.6, 16.6)
    events.append(("reveal_b", 16.6))
    place(mix, sw * env * 0.8, LOOP[0] / FPS)
    # --- drone
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    drone *= db(-28)
    mix[:, 0] += drone
    mix[:, 1] += drone

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
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(events=events, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
