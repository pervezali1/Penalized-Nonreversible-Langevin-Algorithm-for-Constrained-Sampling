"""Sonification for the learning-rate reel: the descent pad (pitch follows average loss), a
jitter texture driven by particle speed, a rising scream and a detonation for the explosion."""

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

from lrsim import FPS, LOOP, N_FRAMES, RESETS, RUNS  # noqa: E402

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
    pos, mean_loss = d["pos"], d["mean_loss"]
    rng = np.random.default_rng(13)
    speed = np.linalg.norm(pos - np.roll(pos, 1, axis=0), axis=2)
    speed = np.where(np.isfinite(speed), np.minimum(speed, 10), 10).mean(axis=1)
    speed[0] = speed[1]
    mix = np.zeros((NS, 2))

    # pad: pitch follows log average loss (clipped), tremolo follows speed
    u = circ_interp(np.clip(np.log10(np.maximum(mean_loss, 1e-3) + 1) / 1.2, 0, 1))
    u = A.onepole_circ_fast(u, 0.05)
    trem = A.onepole_circ_fast(circ_interp(np.clip(speed / 0.02, 0, 1)), 0.02)
    for base, pan in ((220.0, -0.5), (277.18, -0.15), (329.63, 0.15), (440.0, 0.5)):
        ph = np.cumsum(integer_cycles(2 * math.pi * base * 2 ** (u - 0.5) / SR))
        lfo = 1 - 0.6 * trem * (0.5 + 0.5 * np.sin(2 * math.pi * 9 * TT))
        v = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * db(-27) * lfo
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    # jitter texture: band noise whose level follows speed (loud and harsh in the 0.3 run)
    noise = sosfilt(butter(2, [300, 3000], "band", fs=SR, output="sos"), rng.standard_normal(NS))
    noise /= np.std(noise) * 3
    flow = circ_interp(np.clip(speed / 0.01, 0, 1)) * db(-28)
    mix[:, 0] += noise * flow
    mix[:, 1] += np.roll(noise, 1201) * flow
    # explosion: riser from the start of run 3 into a detonation when the loss passes 1e3
    f0 = RUNS[2][3]
    t_boom = next(f for f in range(f0, N_FRAMES) if mean_loss[f] > 100) / FPS
    n_r = int((t_boom - f0 / FPS) * SR)
    rise = A.bandpass_sweep(rng.standard_normal(n_r), 300, 6000, t_boom - f0 / FPS)
    rise /= np.std(rise) * 3
    tr = np.arange(n_r) / SR
    rise *= db(-38) * (db(-16) / db(-38)) ** (tr / tr[-1])
    place(mix, rise, f0 / FPS)
    place(mix, A.impact(rng) * 1.2, t_boom)
    place(mix, A.bell_e6() * 0.5, t_boom)
    # resets and reveals
    sw_n = int(0.5 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(sw_n))
    sw /= np.std(sw) * 3
    ts = np.arange(sw_n) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (ts / ts[-1])
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    for a, b in RESETS:
        place(mix, sw * env, a / FPS)
    place(mix, A.chime(), 4.4)
    place(mix, sw * env * 0.8, LOOP[0] / FPS)
    place(mix, A.bell_e6() * 0.8, 16.8)
    drone = np.zeros(NS)
    for fr in (55.0, 82.41):
        drone += np.sin(np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * fr / SR))))
    mix[:, 0] += drone * db(-28)
    mix[:, 1] += drone * db(-28)

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
    print(f"boom at {t_boom:.2f}s, loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(boom=t_boom, lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
