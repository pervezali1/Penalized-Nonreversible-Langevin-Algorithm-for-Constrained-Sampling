"""Sonification for the initialisation-scale reel: three voices, one per network. The He net's voice
is a chord that pulls into tune as its accuracy rises; the too-small net's voice is the same chord
turned down to a whisper (its level follows the size of its layer-8 signal); the too-big net's voice
is a distorted blast that overloads for the first two steps and then dies to a dull hum. A soft tick
per training step, a ding when the He net passes 90%, bells at the reveals. Circular over 23 s."""

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
from scene import F0, F1, FPS, LOOP_FADE, N_FRAMES, Scene, frame_step, step_frame  # noqa: E402

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


def chord(f_ctrl, detune_ctrl, rng, pan, level):
    """Eight partials whose random detune (in cents) shrinks with detune_ctrl (1 = full detune, 0 = in tune)."""
    out = np.zeros((NS, 2))
    L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    rp = rng.standard_normal(8)
    for k, ratio in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)):
        cents = rp[k] * 35 * detune_ctrl
        ph = np.cumsum(integer_cycles(2 * math.pi * f_ctrl * ratio * 2 ** (cents / 1200) / SR))
        v = np.sin(ph) * (0.9 / (k + 1)) * db(level)
        out[:, 0] += v * L
        out[:, 1] += v * R
    return out


def render(out_wav, events_json=None):
    sc = Scene()
    rng = np.random.default_rng(111)
    mix = np.zeros((NS, 2))
    steps = np.array([frame_step(f) for f in range(N_FRAMES)])
    fade = np.array([1 - min(1.0, max(0.0, (f / FPS - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0]))) for f in range(N_FRAMES)])

    def acc_ctrl(name):
        a = sc.acc[name][steps]
        a = a * fade + sc.acc[name][0] * (1 - fade)
        return A.onepole_circ_fast(circ_interp(np.clip((a - 0.5) / 0.5, 0, 1) ** 1.5), 0.08)

    # He: chord that climbs and tunes with accuracy
    u_he = acc_ctrl("he")
    mix += chord(196.0 * 2 ** (u_he * 0.4), 1 - u_he, rng, 0.0, -26)
    # too small: the same chord, level following its layer-8 signal (a whisper that barely grows)
    s8 = np.array([float(sc.sig["small"][s][-1]) for s in steps])
    lvl = np.clip((np.log10(np.maximum(s8, 1e-14)) + 14) / 18, 0, 1)      # 0 at 1e-14, 1 at 1e4
    lvl = lvl * fade + lvl[0] * (1 - fade)
    gain_small = circ_interp(0.08 + 0.6 * lvl)
    u_sm = acc_ctrl("small")
    mix += chord(146.83 * 2 ** (u_sm * 0.4), 1 - u_sm, rng, -0.5, -26) * gain_small[:, None]
    # too big: a two-step overload blast, then a dull dead hum
    t_blast = F0 / FPS
    n_b = int(0.9 * SR)
    tb = np.arange(n_b) / SR
    blast = np.tanh(6 * np.sin(2 * math.pi * 110 * tb * (1 + 3 * tb)) + 2 * rng.standard_normal(n_b) * np.exp(-tb / 0.2))
    blast *= np.exp(-tb / 0.35) * db(-16)
    place(mix, np.stack([blast * 0.6, blast], 1), t_blast)
    hum_gate = circ_interp(np.where(np.arange(N_FRAMES) >= F0 + 6, 1.0, 0.0) * fade)
    hum = np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * 55.0 / SR))))
    hum = hum * (1 + 0.3 * np.sin(2 * math.pi * 0.7 * TT)) * db(-30) * hum_gate
    mix[:, 0] += hum * 0.5
    mix[:, 1] += hum * 0.9
    prev = 0
    for f in range(F0, F1 + 1):
        s = steps[f]
        for st in range(prev + 1, s + 1):
            if st % 4 == 0:
                place(mix, tick(rng, -38), f / FPS + (st - prev - 1) / max(1, s - prev) / FPS)
        prev = s
    s90 = sc.first90["he"]
    if s90 is not None:
        place(mix, ding(523.25, -18), max(step_frame(s90) / FPS, 9.0))
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
            json.dump(dict(first90=sc.first90, final=sc.final, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
