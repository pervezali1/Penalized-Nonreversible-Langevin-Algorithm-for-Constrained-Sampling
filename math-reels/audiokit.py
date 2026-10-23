"""Procedural audio helpers: additive voices, envelopes, convolution reverb, loudness."""

import numpy as np
import pyloudnorm
from scipy.io import wavfile
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000


def tvec(seconds):
    return np.arange(int(round(seconds * SR))) / SR


def adsr(n, a=0.01, d=0.1, s=0.6, r=0.3, sr=SR):
    """Envelope of n samples; release occupies the tail."""
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    sus = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, s, d, endpoint=False),
        np.full(sus, s),
        np.linspace(s, 0, r),
    ])
    return np.pad(env, (0, max(0, n - len(env))))[:n]


def bell(freq, seconds, amp=0.3, decay=2.5, partials=((1, 1.0), (2.0, 0.35), (3.01, 0.12), (4.2, 0.06))):
    """Soft struck-bell / glass tone with slightly inharmonic partials."""
    t = tvec(seconds)
    out = np.zeros_like(t)
    for k, (mult, a) in enumerate(partials):
        out += a * np.sin(2 * np.pi * freq * mult * t) * np.exp(-t * decay * (1 + 0.6 * k))
    attack = np.minimum(1, t / 0.004)
    return amp * out * attack


def pad_voice(freq_fn, t, amp_fn, detune_cents=6.0, harmonics=((1, 1.0), (2, 0.18), (3, 0.06))):
    """Warm sustained voice whose frequency can vary per-sample (freq_fn(t) -> Hz array)."""
    f = freq_fn(t) if callable(freq_fn) else np.full_like(t, freq_fn)
    out = np.zeros_like(t)
    for cents in (-detune_cents, 0.0, detune_cents):
        ph = 2 * np.pi * np.cumsum(f * 2 ** (cents / 1200)) / SR
        for mult, a in harmonics:
            out += a * np.sin(mult * ph)
    amp = amp_fn(t) if callable(amp_fn) else amp_fn
    return out * amp / 3.0


def place(buf, clip, t0):
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    j = min(len(buf), i + len(clip))
    buf[i:j] += clip[: j - i]


def lowpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def highpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def reverb_ir(seconds=3.2, decay=3.0, seed=1):
    """Stereo exponentially-decaying filtered noise impulse response."""
    rng = np.random.default_rng(seed)
    t = tvec(seconds)
    env = np.exp(-t * decay)
    ir = []
    for _ in range(2):
        n = rng.standard_normal(len(t)) * env
        n = lowpass(n, 6000)
        ir.append(n / np.sqrt(np.sum(n ** 2)))
    return np.stack(ir, 1)


def reverb(stereo, wet=0.3, seconds=3.2, decay=3.0):
    ir = reverb_ir(seconds, decay)
    out = np.empty_like(stereo)
    for c in range(2):
        out[:, c] = fftconvolve(stereo[:, c], ir[:, c])[: len(stereo)]
    return (1 - wet) * stereo + wet * out


def pan(mono, p):
    """p in [-1, 1] (scalar or array), equal-power."""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], 1)


def normalize_lufs(stereo, target=-14.0, peak_db=-1.0):
    meter = pyloudnorm.Meter(SR)
    loud = meter.integrated_loudness(stereo)
    out = stereo * 10 ** ((target - loud) / 20)
    pk = np.max(np.abs(out))
    lim = 10 ** (peak_db / 20)
    if pk > lim:  # gentle soft-clip limiter for the few peaks above ceiling
        out = np.tanh(out / lim) * lim
    return out


def write_wav(path, stereo):
    wavfile.write(path, SR, (np.clip(stereo, -1, 1) * 32767).astype(np.int16))
