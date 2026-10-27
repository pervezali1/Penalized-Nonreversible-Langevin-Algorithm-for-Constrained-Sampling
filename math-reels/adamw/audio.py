"""Soundtrack for the Adam reel (three acts), in the gentle register of reels 23-25: a soft chord bed; a
low tone as each act's balls drop; at every landing one sound per ball. Adam: a plucked note walking
down the scale as its height falls, the same tune in all three acts because its path is the same. Plain
descent: a quiet tick and a faint pluck while it is on the hill, a rising whoosh when the steep hill
throws it off. Bells at the headings. Circular over 30 s."""

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
from scene import FPS, LOOP_FADE, N_FRAMES, N_STEPS, STEP_DT, T_SURF, Scene  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "pendulum"))
_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR


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


def tick(rng, level=-34):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    v = sosfilt(butter(2, [1500, 6000], "band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.006)
    return v / (np.max(np.abs(v)) + 1e-9) * db(level)


def whoosh(level=-20, t_len=0.75):
    n = int(t_len * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(5)
    v = A.bandpass_sweep(rng.standard_normal(n), 400.0, 5200.0, t_len, q=3.0)
    env = np.sin(np.pi * np.minimum(1.0, t / t_len)) ** 1.5 * (1 - 0.5 * t / t_len)
    v = v / (np.max(np.abs(v)) + 1e-9) * env * db(level)
    return np.stack([v, v * 0.45], 1)


def land(freq, level=-26):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)) * np.exp(-t / 0.12) * np.minimum(1, t / 0.004) * db(level)


SCALE = [392.0, 349.23, 329.63, 293.66, 261.63, 246.94, 220.0, 196.0, 174.61, 164.81, 146.83, 130.81]


def scale_note(frac):
    idx = int(round((1 - max(0.0, min(1.0, frac)) ** 0.35) * (len(SCALE) - 1)))
    return SCALE[min(len(SCALE) - 1, max(0, idx))]


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(11)
    mix = np.zeros((NS, 2))
    env = np.ones(N_FRAMES)
    for f in range(N_FRAMES):
        t = f / FPS
        env[f] = min(1.0, max(0.0, (t - T_SURF + 0.5) / 2.5)) * (1 - min(1.0, max(0.0, (t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))))
    env_s = np.interp(TT, np.append(np.arange(N_FRAMES) / FPS, DUR), np.append(env, env[0]))
    for base, pan, lv in ((130.81, -0.4, -34), (196.0, 0.1, -36), (293.66, -0.1, -38), (329.63, 0.4, -39)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env_s
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    for act in sc.acts:
        place(mix, land(196.0, -26), act["t_drop"])
        plain, adam = act["balls"]
        for k in range(N_STEPS):
            t_land = act["t_first"] + k * STEP_DT + STEP_DT * 0.84
            fa = scale_note(adam["heights"][k + 1] / adam["heights"][0])
            place(mix, A.pluck(fa, 0.8 - 0.4 * (k / N_STEPS), 0.35) * 0.9, t_land)
            g = plain["vis_gone"]
            if g is None or k + 1 < g:
                fp = scale_note(plain["heights"][k + 1] / plain["heights"][0])
                place(mix, A.pluck(fp, 0.25, -0.5) * 0.35, t_land + 0.012)
                place(mix, tick(rng, -36), t_land + 0.012)
        if plain["vis_gone"] is not None:
            place(mix, whoosh(-21), act["t_first"] + (plain["vis_gone"] - 1) * STEP_DT + 0.42 * STEP_DT)
        place(mix, land(130.81, -24), act["t_end"] + 0.1)
    bell = A.bell_e6() * 0.45
    for c in sc.cues[1:-1]:
        place(mix, bell, c[0])
    out = A.circ_reverb(mix, rt60=2.8, wet=0.28)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    out *= db(-16.0 - meter.integrated_loudness(out))
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
