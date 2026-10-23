"""Sonification for the neurons reel: a chord that resolves as accuracy rises, static that clears
as the loss falls, a tick per 50 training steps, chimes on milestones. Circular over 20.000 s."""

import json
import math
import os
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pendulum"))
import audio as A  # noqa: E402  (pendulum sonification helpers)
from train import FPS, N_FRAMES, RESETS, STAGES  # noqa: E402

SR = 48000
DUR = N_FRAMES / FPS
NS = int(SR * DUR)
TT = np.arange(NS) / SR
FRAME_T = np.arange(N_FRAMES) / FPS

DISSONANT = [220.0, 233.08, 246.94, 261.63]      # A, Bb, B, C: a tight cluster
RESOLVED = [220.0, 277.18, 329.63, 440.0]        # A, C#, E, A: a major chord


def db(x):
    return 10 ** (x / 20)


def circ_interp(frame_values):
    xp = np.append(FRAME_T, DUR)
    return np.interp(TT, xp, np.append(frame_values, frame_values[0]))


def integer_cycles(inc):
    C = inc.sum() / (2 * math.pi)
    return inc + 2 * math.pi * ((round(C) - C) / DUR) / SR


def place(buf, clip, t0):
    i0 = int(round(t0 * SR)) % NS
    idx = (i0 + np.arange(len(clip))) % NS
    if buf.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    np.add.at(buf, idx, clip)


def render(train_path, out_wav, events_json=None):
    with open(train_path.replace(".npz", ".json")) as fh:
        meta = json.load(fh)
    acc_f = np.array(meta["acc"])
    frames = meta["frames"]
    first100 = meta["first100_frame"]
    rng = np.random.default_rng(11)
    events = []

    # loop crossfade of the control signal mirrors the picture
    a0, a1 = 1158, 1200
    mixf = np.clip((np.arange(N_FRAMES) - a0) / (a1 - a0), 0, 1)
    mixf = mixf * mixf * (3 - 2 * mixf)
    acc_ctrl = (1 - mixf) * acc_f + mixf * acc_f[0]
    acc = circ_interp(acc_ctrl)
    acc = A.onepole_circ_fast(acc, 0.08)
    s = np.clip((acc - 0.5) / 0.5, 0, 1)
    s = s * s * (3 - 2 * s)

    mix = np.zeros((NS, 2))

    # --- chord: pitches glide from the cluster to the major chord with accuracy
    for k, (fd, fr) in enumerate(zip(DISSONANT, RESOLVED)):
        f_inst = fd * (fr / fd) ** s
        ph = np.cumsum(integer_cycles(2 * math.pi * f_inst / SR))
        v = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)) * db(-26)
        pan = (-0.5, -0.17, 0.17, 0.5)[k]
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # bright shimmer that only exists near 100%
    shimmer_amp = np.clip((acc - 0.97) / 0.03, 0, 1) * db(-30)
    ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * 880 / SR)))
    mix[:, 0] += np.sin(ph) * shimmer_amp
    mix[:, 1] += np.sin(ph * 1.0) * shimmer_amp

    # --- static that clears as the network learns
    noise = rng.standard_normal(NS)
    noise = sosfilt(butter(2, 1500, "low", fs=SR, output="sos"), noise)
    noise /= np.std(noise) * 3
    static_amp = db(-26) * (1 - s) ** 1.5
    mix[:, 0] += noise * static_amp
    mix[:, 1] += np.roll(noise, 997) * static_amp

    # --- a tick every 50 training steps, pitched by accuracy
    last_tick = -1
    for f in range(N_FRAMES):
        si, step = frames[f]
        key = (si, step // 50)
        if step > 0 and key != last_tick and step % 50 < 50:
            if step // 50 != (frames[f - 1][1] // 50 if f > 0 else -1) or si != frames[f - 1][0]:
                freq = 800 + 1600 * np.clip((acc_f[f] - 0.5) / 0.5, 0, 1)
                n = int(0.05 * SR)
                t = np.arange(n) / SR
                tick = np.sin(2 * math.pi * freq * t) * np.exp(-t / 0.008) * db(-22)
                place(mix, tick, f / FPS)
                last_tick = key

    # --- milestones
    bell = A.bell_e6()
    if first100 is not None:
        tf = first100 / FPS
        place(mix, bell, tf)
        place(mix, A.impact(rng) * 0.6, tf)
        events.append(("100%", tf))
    for si, (hidden, f0, f1, steps, expo) in enumerate(STAGES):
        seg = acc_f[f0:f1 + 1]
        for thr in (0.9,):
            if seg.max() >= thr and si == 2:
                fk = f0 + int(np.argmax(seg >= thr))
                place(mix, bell * 0.6, fk / FPS)
                events.append(("90%", fk / FPS))
    # stage resets: crushed-noise glitch
    for (a, b) in RESETS:
        n_g = int((b - a) / FPS * SR)
        gl = rng.standard_normal(n_g)
        gl = np.round(gl / np.max(np.abs(gl)) * 7) / 7 * db(-22) * np.hanning(n_g) ** 0.3
        place(mix, gl, a / FPS)
        place(mix, A.click(rng) * 2, a / FPS)
        events.append(("reset", a / FPS))
    # 86-billion payoff: reverse-cymbal swell into a chime
    n_c = int(1.7 * SR)
    cy = sosfilt(butter(2, 3000, "high", fs=SR, output="sos"), rng.standard_normal(n_c))
    cy /= np.std(cy) * 3
    tc = np.arange(n_c) / SR
    env = db(-46) * (db(-20) / db(-46)) ** (tc / tc[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, cy * env, 15.2)
    place(mix, A.chime(), 16.9)
    # loop reset: short reversed swell as the field dissolves
    n_s = int(0.6 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-44) * (db(-20) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    place(mix, sw * env, a0 / FPS)

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
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(events=events, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
