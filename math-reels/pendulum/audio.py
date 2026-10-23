"""Sonification: every sound is derived from the simulation. Rendered circularly over exactly 18.000 s."""

import json
import math
import sys

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

import timemap as tm
from sim import PIVOT

SR = 48000
DUR = 18.0
NS = int(SR * DUR)                       # 864,000 samples
F = tm.N_FRAMES
TT = np.arange(NS) / SR
FRAME_T = np.arange(F) / tm.FPS

PENTA = [220.0, 261.6256, 293.6648, 329.6276, 391.9954, 440.0, 523.2511, 587.3295, 659.2551, 783.9909]
PENTA = [round(p * DUR) / DUR for p in PENTA]           # integer cycles over the loop


def db(x):
    return 10 ** (x / 20)


def circ_interp(frame_values):
    """Linear interpolation of per-frame control data to audio rate, wrapping at the loop."""
    fv = np.asarray(frame_values, dtype=np.float64)
    xp = np.append(FRAME_T, DUR)
    if fv.ndim == 1:
        return np.interp(TT, xp, np.append(fv, fv[0]))
    return np.stack([np.interp(TT, xp, np.append(fv[:, j], fv[0, j])) for j in range(fv.shape[1])], 1)


def onepole_circ_fast(x, tau):
    """One-pole smoother whose state wraps around the loop (second pass starts from the end state)."""
    from scipy.signal import lfilter
    a = math.exp(-1 / (tau * SR))
    y, zf = lfilter([1 - a], [1, -a], x, zi=np.zeros(1))
    y, _ = lfilter([1 - a], [1, -a], x, zi=zf)
    return y


def place(buf, clip, t0):
    """Overlap-add a (n,) or (n,2) clip at time t0, wrapping modulo the loop."""
    i0 = int(round(t0 * SR)) % NS
    n = len(clip)
    idx = (i0 + np.arange(n)) % NS
    if buf.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    np.add.at(buf, idx, clip)


def pan_lr(p):
    a = (np.clip(p, -1, 1) + 1) * math.pi / 4
    return np.cos(a), np.sin(a)


def integer_cycles(phase_inc):
    """Add a constant offset so the accumulated phase is an integer number of cycles over the loop."""
    C = phase_inc.sum() / (2 * math.pi)
    delta = (round(C) - C) / DUR
    return phase_inc + 2 * math.pi * delta / SR


def bandpass_sweep(noise, f0, f1, t_len, q=4.0, chunk=256):
    out = np.zeros_like(noise)
    n = len(noise)
    zi = None
    for i in range(0, n, chunk):
        fc = f0 * (f1 / f0) ** (i / n)
        sos = butter(2, [fc / math.sqrt(1 + 1 / (2 * q)), min(fc * math.sqrt(1 + 1 / (2 * q)), SR / 2 - 1)],
                     "band", fs=SR, output="sos")
        if zi is None:
            zi = sosfilt_zi(sos) * 0
        seg, zi = sosfilt(sos, noise[i:i + chunk], zi=zi)
        out[i:i + chunk] = seg
    return out


def circ_filter(sos, x):
    """Apply an IIR filter with loop-consistent state by filtering the loop twice."""
    if x.ndim == 1:
        y, zf = sosfilt(sos, x, zi=sosfilt_zi(sos) * 0)
        y, _ = sosfilt(sos, x, zi=zf)
        return y
    return np.stack([circ_filter(sos, x[:, c]) for c in range(x.shape[1])], 1)


def circ_reverb(stereo, rt60=2.2, pre=0.015, wet=0.18, seed=3):
    rng = np.random.default_rng(seed)
    n_ir = int(SR * 3.0)
    t = np.arange(n_ir) / SR
    env = np.exp(-6.91 * t / rt60)
    lp = butter(1, 6000, "low", fs=SR, output="sos")
    out = np.empty_like(stereo)
    for c in range(2):
        ir = sosfilt(lp, rng.standard_normal(n_ir) * env)
        ir = np.concatenate([np.zeros(int(pre * SR)), ir])
        ir /= np.sqrt(np.sum(ir ** 2))
        X = np.fft.rfft(stereo[:, c])
        Hh = np.fft.rfft(ir, NS)
        out[:, c] = np.fft.irfft(X * Hh, NS)
    return (1 - wet) * stereo + wet * out * 0.5


# ---------------------------------------------------------------- control data

def harp_control(tips):
    """Per-frame bin populations p[f, b], mean tip x per bin, and per-frame mean tip speed."""
    y_rel = PIVOT[1] - tips[:, :, 1].astype(np.float64)
    x = tips[:, :, 0].astype(np.float64)
    raw = np.clip(np.floor((y_rel + 330) / 66), 0, 9).astype(np.int64)
    prev = raw[-1].copy()
    bins = np.empty_like(raw)
    for _pass in range(2):
        for f in range(F):
            b = raw[f].copy()
            moved = b != prev
            upper = np.maximum(b, prev)
            boundary = -330 + 66 * upper
            near = np.abs(y_rel[f] - boundary) < 8
            keep = moved & near & (np.abs(b - prev) == 1)
            b[keep] = prev[keep]
            bins[f] = b
            prev = b
    p = np.zeros((F, 10))
    mx = np.zeros((F, 10))
    for b in range(10):
        m = bins == b
        p[:, b] = m.mean(axis=1)
        with np.errstate(invalid="ignore"):
            mx[:, b] = np.where(m.sum(1) > 0, (x * m).sum(1) / np.maximum(m.sum(1), 1), PIVOT[0])
    speed_px = np.linalg.norm(tips - np.roll(tips, 1, axis=0), axis=2).mean(axis=1) * tm.FPS
    return p, mx, speed_px


# ---------------------------------------------------------------- voices

def pluck(freq, vel, pan):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for mult, a, tau in ((1, 1.0, 0.60), (2, 0.5, 0.35), (3, 0.25, 0.20), (4, 0.12, 0.12)):
        out += a * np.sin(2 * math.pi * freq * mult * t) * np.exp(-t / tau)
    out *= np.minimum(1, t / 0.003)
    out *= db(-12) * vel / 1.87
    L, R = pan_lr(pan)
    return np.stack([out * L, out * R], 1)


def bell_e6():
    n = int(SR * 2.5)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for mult, a in ((1, 1.0), (2.76, 0.4), (5.40, 0.15)):
        out += a * np.sin(2 * math.pi * 1318.5 * mult * t) * np.exp(-t / 0.5)
    return out * np.minimum(1, t / 0.002) * db(-16) / 1.55


def woodblock():
    n = int(SR * 0.08)
    t = np.arange(n) / SR
    out = (np.sin(2 * math.pi * 1800 * t) + np.sin(2 * math.pi * 3200 * t)) * np.exp(-t / 0.012)
    return out * db(-22) / 2


def chime(reverse=False):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    out = sum(np.sin(2 * math.pi * fr * t) for fr in (1568, 2352, 3136)) * np.exp(-t / 0.4)
    out *= np.minimum(1, t / 0.003) * db(-18) / 3
    return out[::-1] if reverse else out


def click(rng):
    n = int(SR * 0.03)
    x = rng.standard_normal(n) * np.hanning(n)
    sos = butter(2, [2500, 3600], "band", fs=SR, output="sos")
    y = sosfilt(sos, x)
    return y / (np.max(np.abs(y)) + 1e-9) * db(-24)


def impact(rng):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    f_inst = np.where(t < 0.25, 150 * (38 / 150) ** (t / 0.25), 38.0)
    ph = 2 * math.pi * np.cumsum(f_inst) / SR
    sweep = np.sin(ph) * np.exp(-t / 0.35) * db(-6)
    noise = sosfilt(butter(2, 2000, "low", fs=SR, output="sos"), rng.standard_normal(n))
    noise = noise / (np.std(noise) * 3) * np.exp(-t / 0.08) * db(-14)
    return sweep + noise


# ---------------------------------------------------------------- main render

def render(frames_path, out_wav, events_json=None):
    fr = np.load(frames_path)
    with open(frames_path.replace(".npz", ".json")) as fh:
        meta = json.load(fh)
    d, spd, tips, M = fr["d"], fr["speed"], fr["tips"], fr["M"]
    gap_str, lens_k = meta["gap_str"], meta["lens_k"]
    t_chaos = meta["t_chaos"]
    rng = np.random.default_rng(42)
    events = []

    # --- global playback-speed warps (tape stop / freeze / rewind fade) at audio rate
    t = TT
    r = np.ones(NS)
    m1 = (t >= 9.8) & (t < 10.2)
    r[m1] = np.maximum(np.abs(np.interp(t[m1], FRAME_T, spd)), 0.02)
    r[(t >= 10.2) & (t < 10.3)] = 0.02
    g_warp = np.sqrt(r)
    g_warp[(t >= 10.2) & (t < 10.3)] = 0.0
    m2 = (t >= 10.3) & (t < 10.7)
    g_warp[m2] = (t[m2] - 10.3) / 0.4
    g_warp = onepole_circ_fast(g_warp, 0.005)
    r = onepole_circ_fast(r, 0.005)

    mix = np.zeros((NS, 2))

    # --- L1 harp
    p, mx, speed_px = harp_control(tips)
    a_frame = np.sqrt(p)
    a = circ_interp(a_frame)
    pan_frame = 0.8 * (mx - PIVOT[0]) / 330
    pans = circ_interp(pan_frame)
    sustain = np.zeros((NS, 2))
    for b in range(10):
        amp = onepole_circ_fast(a[:, b], 0.020) * db(-24)
        inc = 2 * math.pi * PENTA[b] * r / SR
        ph = np.cumsum(integer_cycles(inc))
        v = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * amp * g_warp
        L, R = pan_lr(pans[:, b])
        sustain[:, 0] += v * L
        sustain[:, 1] += v * R
    mix += sustain
    n_pluck = 0
    for f in range(F):
        dp = p[f] - p[(f - 1) % F]
        for b in np.where(dp >= 0.25)[0]:
            vel = 0.5 + 0.5 * min(1.0, speed_px[f] / 1200) ** 0.5
            tf = f / tm.FPS
            if 9.8 <= tf < 10.7:
                continue
            place(mix, pluck(PENTA[b], vel, pan_frame[f, b]), tf)
            n_pluck += 1
            events.append(("pluck", round(tf, 3), int(b)))

    # --- L2 gap beat: 220 Hz against 220 + f_b(M)
    Mi = circ_interp(M)
    f_b = 0.25 * (Mi / 1e-6) ** 0.2519
    lvl = db(-24 + 10 * np.clip((np.log10(Mi) + 6) / (np.log10(180) + 6), 0, 1))
    gate = np.ones(NS)
    t_regate = 11.1 + (8.7 - meta["t180"])
    gate[(t >= t_chaos) & (t < t_regate)] = 0
    gate = onepole_circ_fast(gate, 0.05)
    gate[(t >= t_chaos) & (t < t_chaos + 0.01)] *= np.linspace(1, 0, int(0.01 * SR))[: np.sum((t >= t_chaos) & (t < t_chaos + 0.01))]
    ph1 = np.cumsum(integer_cycles(np.full(NS, 2 * math.pi * 220 / SR)))
    ph2 = np.cumsum(integer_cycles(2 * math.pi * (220 + f_b) / SR))

    def tone(ph):
        return np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    beat = (tone(ph1) + tone(ph2)) / 3.5 * lvl * gate
    mix[:, 0] += beat * 0.92
    mix[:, 1] += beat * 0.92

    # --- L3 countdown: ticks and zero dings
    wb, bl = woodblock(), bell_e6()

    def exponent(s):
        v = float(s.rstrip("°+"))
        return math.floor(math.log10(v))
    for f in range(F):
        tf = f / tm.FPS
        d0, d1 = d[(f - 1) % F], d[f]
        tick = math.floor(d0) != math.floor(d1)
        if tf < 9.8 and 6.0 <= d1 <= meta["t180"] + 0.01:
            tick = tick or math.floor(2 * d0) != math.floor(2 * d1)
        if tick:
            place(mix, wb, tf)
            events.append(("tick", round(tf, 3), round(float(d1), 2)))
        if exponent(gap_str[f]) != exponent(gap_str[(f - 1) % F]):
            place(mix, bl, tf)
            events.append(("ding", round(tf, 3), gap_str[f]))
        k0, k1 = int(round(lens_k[(f - 1) % F])), int(round(lens_k[f]))
        if k0 != k1 and tm.lens_presence(d1)[1] > 0:
            place(mix, click(rng), tf)
            events.append(("snap", round(tf, 3), k1))

    # --- L4 riser
    n_r = int((t_chaos - 4.8) * SR)
    noise = rng.standard_normal(n_r)
    rise = bandpass_sweep(noise, 400, 5000, t_chaos - 4.8)
    rise /= np.std(rise) * 3
    tr = np.arange(n_r) / SR
    rise *= db(-40) * (db(-18) / db(-40)) ** (tr / tr[-1])
    sub_f = 45 + 15 * tr / tr[-1]
    sub = np.sin(2 * math.pi * np.cumsum(sub_f) / SR) * db(-22) * np.minimum(1, tr / 0.5)
    cut = np.ones(n_r)
    cut[-int(0.005 * SR):] = np.linspace(1, 0, int(0.005 * SR))
    place(mix, (rise + sub) * cut, 4.8)

    # --- L5 impact
    place(mix, impact(rng), t_chaos)

    # --- L6 tape/rewind FX
    n_g = int(0.1 * SR)
    gl = rng.standard_normal(n_g)
    gl = np.round(gl / np.max(np.abs(gl)) * 7) / 7 * db(-20) * np.hanning(n_g) ** 0.3
    place(mix, gl, 10.2)
    n_s = int(0.4 * SR)
    sw = sosfilt(butter(2, 800, "high", fs=SR, output="sos"), rng.standard_normal(n_s))
    sw /= np.std(sw) * 3
    ts = np.arange(n_s) / SR
    env = db(-40) * (db(-14) / db(-40)) ** (ts / ts[-1])
    env[-int(0.04 * SR):] *= np.linspace(1, 0, int(0.04 * SR))
    place(mix, sw * env, 10.3)
    n_c = int(1.6 * SR)
    cy = sosfilt(butter(2, 4000, "high", fs=SR, output="sos"), rng.standard_normal(n_c))
    cy /= np.std(cy) * 3
    tc = np.arange(n_c) / SR
    env = db(-44) * (db(-18) / db(-44)) ** (tc / tc[-1])
    env[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    place(mix, cy * env, 11.0)
    place(mix, chime(), 0.5)
    place(mix, chime(reverse=True), 16.93 - 2.0 + 0.17)

    # --- L7 drone
    drone = np.zeros(NS)
    for fr_hz in (55.0, 82.389):
        ph = np.cumsum(integer_cycles(2 * math.pi * fr_hz * r / SR))
        drone += np.sin(ph)
    drone *= db(-28) * g_warp
    mix[:, 0] += drone
    mix[:, 1] += drone

    # --- master
    out = circ_reverb(mix)
    out = circ_filter(butter(2, 30, "high", fs=SR, output="sos"), out)
    meter = pyloudnorm.Meter(SR)
    loud = meter.integrated_loudness(out)
    out *= db(-14.0 - loud)
    for _ in range(3):
        tp = np.max(np.abs(resample_poly(out, 4, 1, axis=0)))
        if tp <= db(-1.0):
            break
        out = np.tanh(out / db(-1.0) * 0.9) * db(-1.0) / 0.9 * 0.98
    loud = meter.integrated_loudness(out)
    tp = 20 * math.log10(np.max(np.abs(resample_poly(out, 4, 1, axis=0))))
    print(f"plucks {n_pluck}, loudness {loud:.2f} LUFS, true peak {tp:.2f} dBTP, seam |x[0]-x[-1]| = "
          f"{np.abs(out[0] - out[-1]).max():.4f}")
    wavfile.write(out_wav, SR, out.astype(np.float32))
    if events_json:
        with open(events_json, "w") as fh:
            json.dump(dict(events=events, lufs=loud, true_peak=tp), fh)
    return out


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
