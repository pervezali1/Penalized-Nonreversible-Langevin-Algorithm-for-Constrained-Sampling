"""Soundtrack for the adversarial reel: Kokoro narration (voicekit) over the gentle bed of reels 38-49: a soft chord;
a soft rising glide as the direction appears on the pixels; a pluck at one shade, a low pluck and a minor chord at
two shades (the 3 becomes "a 9"); a rising pluck for each of the ten targeted digits; a falling glide as the
gradient curve drops; bells at the headings. Circular."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import voicekit  # noqa: E402
from scene import FPS, N_FRAMES, SEG, T_ALL0, T_ALL1, T_END, T_ONE, T_TG0, T_TG1, T_TWO, VOICE_AT, Scene, n_targets  # noqa: E402

_spec = importlib.util.spec_from_file_location("pend_audio", os.path.join(ROOT, "pendulum", "audio.py"))
A = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
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


def glide(t0, t1, f0, f1, level=-31):
    tt = np.arange(int((t1 - t0 + 0.6) * SR)) / SR
    fr = f0 * (f1 / f0) ** np.clip(tt / (t1 - t0), 0, 1)
    return np.sin(np.cumsum(2 * math.pi * fr / SR)) * np.minimum(1, tt / 0.3) * np.minimum(1, (tt[-1] - tt) / 0.5) * db(level)


def render(out_wav, events_json=None):
    sc = Scene()
    mix = np.zeros((NS, 2))
    env = np.minimum(1, np.maximum(0, (TT - 1.0) / 2.5)) * (1 - np.clip((TT - T_END) / 0.55, 0, 1))
    for base, pan, lv in ((110.0, -0.4, -34), (164.81, 0.1, -36), (220.0, -0.1, -38), (277.18, 0.4, -40)):
        ph = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * base / SR)))
        v = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * db(lv) * env
        L, R = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        mix[:, 0] += v * L
        mix[:, 1] += v * R
    place(mix, glide(SEG[2] + 0.6, SEG[2] + 2.0, 330.0, 523.25, -33), SEG[2] + 0.6)
    place(mix, A.pluck(392.0, 0.5, -0.2) * 0.5, T_ONE)
    place(mix, A.pluck(146.83, 0.7, 0.0) * 0.7, T_TWO)
    for i, f in enumerate((220.0, 261.63, 329.63)):
        place(mix, A.pluck(f, 0.6, -0.4 + 0.4 * i) * 0.5, T_TWO + 0.12 + 0.08 * i)
    last = 0
    for t in np.arange(T_TG0 - 0.5, T_TG1 + 1.0, 1 / 240):
        nt = n_targets(t)
        if nt > last:
            place(mix, A.pluck(261.63 * 2 ** ((nt - 1) * 2 / 12), 0.35, -0.5 + nt / 10) * 0.4, t)
            last = nt
    place(mix, glide(T_ALL0 - 0.3, T_ALL1 + 0.6, 523.25, 196.0, -32), T_ALL0 - 0.3)
    bell = A.bell_e6() * 0.4
    for c in sc.cues[1:-1]:
        place(mix, bell, c[0])
    out = A.circ_reverb(mix, rt60=3.0, wet=0.3)
    out = A.circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    out = voicekit.mix(out, SR, os.path.dirname(os.path.abspath(out_wav)), VOICE_AT, NS)
    out, loud, tp = voicekit.finish(out, SR)
    print(f"loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam {np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(lufs=loud, true_peak=tp), fh)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
