"""What if one zero left the line? The numbers behind the reel.

In Riemann's explicit formula each zero rho = sigma + i*gamma (with its mirror image below the axis) adds the wave
    W(x) = sum_n mu(n)/n * ( -2 Re Ei(rho * ln(x) / n) )
to the prime count; its size grows like 2 x^sigma / (|rho| ln x). On the critical line sigma = 1/2, so the wave grows
like the square root of x. This script takes the first zero (gamma = 14.134725...) and slides its real part from 0.5
to 0.8, leaving every other zero alone:

* on the reel's top chart (x up to 50) for 121 values of sigma, for the slide;
* on a wide range (x from 10 to 10^13) at sigma = 0.5 and at sigma = 0.8.

It is a what-if, not a possible world: no zeta function has that zero (the first 12,363,153,437,138 zeros are known to
be on the line), and a real zero off the line would come with a mirror twin at 1 - sigma, whose wave (it grows like
x^0.2) is left out.

For comparison it also counts the real primes up to 10^9 (segmented sieve, a checkpoint every 10,000) and measures
how far the count really strays from Riemann's smooth estimate R(x).
    python wsim.py <riemann.npz from ../riemann/rsim.py> <out.npz>  -> out.npz and out.json"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "riemann"))
from rsim import FPS, G3, N_SET, NX, X_MAX, X_MIN, primes_upto  # noqa: E402,F401

N_FRAMES = 1800
DUR = N_FRAMES / FPS                              # 30 s
GAMMA1 = G3[0]
SIGMA_OFF = 0.8
N_SIG = 121
WIPE = (2.4, 2.9)                                 # the finished picture clears; before it the loop's end continues
BUILD = (3.0, 5.0)                                # recap: Riemann's formula rebuilt from all the zeros
WAVE_IN = (5.3, 6.9)                              # the first zero's own wave is drawn
PUSH = (8.0, 9.8)                                 # the zero slides off the line
ZOOM = (13.2, 13.5)                               # the wave panel switches to the wide view
SWEEP = (13.5, 19.2)                              # its right edge runs from 50 to a billion
T_WRAP = DUR + WIPE[0]                            # and keeps going through the loop point, to 10^13
X_A, X_B, X_C = 50.0, 1e9, 1e13
X_LEFT = 10.0
BIG_N = ((1, 1), (2, -1), (3, -1), (5, -1), (6, 1), (7, -1), (10, 1))
BLOCK = 10 ** 4


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def effective_time(t):
    return t + DUR if t < WIPE[0] else t


def sigma(te):
    """Real part of the first zero at effective time te."""
    return 0.5 + (SIGMA_OFF - 0.5) * smooth((te - PUSH[0]) / (PUSH[1] - PUSH[0]))


def x_edge(te):
    """Right edge of the wide view."""
    if te <= SWEEP[0]:
        return X_A
    if te <= SWEEP[1]:
        u = (te - SWEEP[0]) / (SWEEP[1] - SWEEP[0])
        return math.exp(math.log(X_A) + (math.log(X_B) - math.log(X_A)) * u)
    u = min(1.0, (te - SWEEP[1]) / (T_WRAP - SWEEP[1]))
    return math.exp(math.log(X_B) + (math.log(X_C) - math.log(X_B)) * u)


def wave(sig, x, nset):
    """The wave a zero at sig + i*GAMMA1 (and its mirror image below the axis) adds to the prime count."""
    from scipy.special import exp1
    out = np.zeros_like(x, dtype=np.float64)
    for n, mu in nset:
        z = (sig + 1j * GAMMA1) * np.log(x) / n
        out += mu / n * (-2 * (-exp1(-z) + 1j * math.pi).real)      # Ei(z) = -E1(-z) + i*pi for Im z > 0
    return out


def riemann_r(x, zetas):
    """Riemann's smooth prime-count estimate, by the Gram series."""
    L = np.log(np.asarray(x, dtype=np.float64))
    out = np.ones_like(L)
    term = np.ones_like(L)
    for k in range(1, len(zetas) + 1):
        term = term * L / k                                          # L^k / k!
        out += term / (k * zetas[k - 1])
    return out


def prime_counts(limit, seg=10 ** 7):
    """pi(j * BLOCK) for j = 0 .. limit / BLOCK, by a segmented sieve."""
    small = primes_upto(int(limit ** 0.5) + 1)
    out = np.zeros(limit // BLOCK + 1, np.int64)
    total, j = 0, 0
    for lo in range(0, limit, seg):
        s = np.ones(seg, bool)
        if lo == 0:
            s[:2] = False
        for p in small:
            start = max(p * p, (lo + p - 1) // p * p)
            if start < lo + seg:
                s[start - lo::p] = False
        c = np.cumsum(s.reshape(-1, BLOCK).sum(1))
        out[j + 1:j + 1 + len(c)] = total + c
        total += int(c[-1])
        j += len(c)
    return out


def main(riemann_npz, out):
    import mpmath as mp
    mp.mp.dps = 20
    d = np.load(riemann_npz)
    assert abs(d["gamma"][0] - GAMMA1) < 1e-9
    # the Ei identity at the largest arguments used here, against mpmath
    for sig, xx in ((0.5, 1e13), (SIGMA_OFF, 1e9), (SIGMA_OFF, 1e13)):
        z = complex(sig, GAMMA1) * math.log(xx)
        ours = wave(sig, np.array([xx]), ((1, 1),))[0]
        ref = float(-2 * mp.ei(mp.mpc(z)).real)
        assert abs(ours - ref) < 1e-9 * abs(ref), (sig, xx, ours, ref)
    # 1. the slide, on the reel's top chart
    x = np.linspace(X_MIN, X_MAX, NX)
    assert np.allclose(x, d["x"])
    sigs = np.linspace(0.5, SIGMA_OFF, N_SIG)
    wtop = np.stack([wave(s, x, N_SET) for s in sigs])
    assert np.abs(wtop[0] - d["terms"][0]).max() < 1e-5            # sigma = 0.5 is the real first zero's wave
    primes = d["primes"]
    half = np.arange(2.5, X_MAX, 1.0)
    true_half = np.searchsorted(primes, half, side="right")
    full = d["base"] + d["terms"].astype(np.float64).sum(0)
    bent = full - wtop[0] + wtop[-1]
    # 2. the wide view
    xb = np.exp(np.linspace(math.log(X_LEFT), math.log(X_C), 9000))
    on_b, off_b = wave(0.5, xb, BIG_N), wave(SIGMA_OFF, xb, BIG_N)

    def swing(arr, x_hi):                                            # largest size over the last full oscillation
        m = (xb <= x_hi) & (xb >= x_hi * math.exp(-2 * math.pi / GAMMA1))
        return float(np.abs(arr[m]).max())

    # 3. the real primes up to a billion against Riemann's smooth estimate
    zetas = [float(mp.zeta(k + 1)) for k in range(1, 161)]
    for xx in (1e3, 1e6, 1e9):
        assert abs(riemann_r(xx, zetas) - float(mp.riemannr(xx))) < 1e-6 * xx ** 0.5
    pic = prime_counts(10 ** 9)
    xs = np.arange(1, len(pic)) * BLOCK
    dev = pic[1:] - riemann_r(xs, zetas)
    i_max = int(np.abs(dev).argmax())
    window = xs >= 1e9 * math.exp(-2 * math.pi / GAMMA1)
    facts = dict(gamma1=GAMMA1, sigma_off=SIGMA_OFF,
                 top_max_err_real=float(np.abs(np.interp(half, x, full) - true_half).max()),
                 top_max_err_bent=float(np.abs(np.interp(half, x, bent) - true_half).max()),
                 top_wave_on_max=float(np.abs(wtop[0][x >= 4]).max()), top_wave_off_max=float(np.abs(wtop[-1][x >= 4]).max()),
                 swing_on=dict((f"1e{k}", swing(on_b, 10.0 ** k)) for k in (3, 6, 9, 12, 13)),
                 swing_off=dict((f"1e{k}", swing(off_b, 10.0 ** k)) for k in (3, 6, 9, 12, 13)),
                 leading_term_off_1e9=2 * 1e9 ** SIGMA_OFF / (abs(complex(SIGMA_OFF, GAMMA1)) * math.log(1e9)),
                 pi_1e9=int(pic[-1]), r_1e9=float(riemann_r(1e9, zetas)), real_dev_at_1e9=float(dev[-1]),
                 real_dev_max=float(np.abs(dev).max()), real_dev_max_at=int(xs[i_max]),
                 real_dev_max_last_oscillation=float(np.abs(dev[window]).max()), checkpoints=int(len(xs)), block=BLOCK,
                 twin_swing_1e9=2 * 1e9 ** (1 - SIGMA_OFF) / (abs(complex(1 - SIGMA_OFF, GAMMA1)) * math.log(1e9)),
                 # a real zero off the line would sit above height 3,000,175,332,800 (Platt and Trudgian); at real part
                 # 0.8 its wave, x^0.8 / height, only overtakes the first zero's, x^0.5 / 14.13, past this power of ten
                 overtakes_first_wave_at_log10=math.log10(3000175332800 / GAMMA1) / (SIGMA_OFF - 0.5))
    np.savez(out, sigs=sigs, wtop=wtop.astype(np.float32), xb=xb, on_b=on_b, off_b=off_b)
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
