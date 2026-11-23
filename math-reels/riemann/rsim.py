"""The Riemann hypothesis reel, computed. Three things, all real:

1. Riemann's zeta function along the critical line, zeta(1/2 + i t), sampled every 0.02 from t = 0 to the height the
   reel reaches (mpmath, 15 digits): the glowing curve in the lower panel.
2. Its zeros on that line (mpmath.zetazero, indexed rigorously), checked two ways: the number of sign changes of the
   real function Z(t) on the sample grid equals the exact count N(T) of ALL zeros in the critical strip up to that
   height (mpmath.nzeros). So every zero up to the height shown is on the line and simple: a small, real verification
   of the hypothesis, as far as this reel goes.
3. Riemann's explicit formula for the prime count, one zero at a time:
       pi(x) = sum_n mu(n)/n J(x^(1/n)),   J(y) = li(y) - sum_rho Ei(rho ln y) - ln 2 + int_y^inf dt / (t (t^2-1) ln t)
   with n = 1, 2, 3, 5 (enough for x <= 50, since 2^6 > 50; mu(4) = 0), each pair of zeros rho, conj(rho) giving the
   wave -2 Re Ei(rho ln y). With no zeros it is a smooth curve; adding the zeros in order turns it into the staircase
   of the primes.

Also the timeline shared by scene.py and audio.py (how far up the line the reel has climbed at each instant), and
the facts quoted in the posting kit (pi(10^9) by sieve against Riemann's R, the count of zeros below 3 x 10^12).
    python rsim.py <out.npz>  -> out.npz and out.json"""

import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

FPS, N_FRAMES = 60, 1800
DUR = N_FRAMES / FPS                              # 30 s
WIPE = (2.4, 2.9)                                 # the finished picture clears; before it the loop's end state continues
SWEEP = (3.1, 6.6)                                # the number line is swept 0 -> 50, primes light up
T_ZETA = 7.0                                      # the climb up the line starts
HIT_T = (9.6, 10.5, 11.2)                         # when the first three zeros are reached
G3 = (14.134725141734695, 21.022039638771556, 25.01085758014569)   # asserted against mpmath in main()
V_FAST, V_CRUISE = 75.0, 30.0                     # climb speeds (height per second)
T_RAMP, T_FAST_END, T_SLOW = 12.7, 17.0, 18.0
T_WRAP = DUR + WIPE[0]                            # the climb continues through the loop point until the wipe
STEP = 0.02                                       # sampling of zeta along the line
X_MAX, NX = 50.0, 1600
X_MIN = 1.5
N_SET = ((1, 1), (2, -1), (3, -1), (5, -1))       # (n, mu(n)) with x^(1/n) >= 2 somewhere in [2, 50]


def tau(te):
    """Height on the critical line at effective time te (seconds; te in [T_ZETA, T_WRAP])."""
    if te <= T_ZETA:
        return 0.0
    if te <= HIT_T[0]:
        return G3[0] * ((te - T_ZETA) / (HIT_T[0] - T_ZETA)) ** 1.15
    if te <= HIT_T[1]:
        return G3[0] + (G3[1] - G3[0]) * (te - HIT_T[0]) / (HIT_T[1] - HIT_T[0])
    if te <= HIT_T[2]:
        return G3[1] + (G3[2] - G3[1]) * (te - HIT_T[1]) / (HIT_T[2] - HIT_T[1])
    v0 = (G3[2] - G3[1]) / (HIT_T[2] - HIT_T[1])
    d = min(te, T_RAMP) - HIT_T[2]
    h = G3[2] + v0 * d + (V_FAST - v0) * d * d / (2 * (T_RAMP - HIT_T[2]))
    if te <= T_RAMP:
        return h
    h += V_FAST * (min(te, T_FAST_END) - T_RAMP)
    if te <= T_FAST_END:
        return h
    d = min(te, T_SLOW) - T_FAST_END
    h += V_FAST * d - (V_FAST - V_CRUISE) * d * d / (2 * (T_SLOW - T_FAST_END))
    if te <= T_SLOW:
        return h
    return h + V_CRUISE * (te - T_SLOW)


T_END = tau(T_WRAP)


def effective_time(t):
    """Video time -> time on the climb: the first WIPE[0] seconds continue the end of the loop."""
    return t + DUR if t < WIPE[0] else t


def _zeta_chunk(rng_):
    import mpmath as mp
    mp.mp.dps = 15
    i0, i1 = rng_
    return [complex(mp.zeta(mp.mpc(0.5, i * STEP))) for i in range(i0, i1)]


def _zero_chunk(ns):
    import mpmath as mp
    mp.mp.dps = 15
    return [float(mp.zetazero(n).imag) for n in ns]


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.flatnonzero(s)


def count_primes(limit, seg=1 << 24):
    """pi(limit) by a segmented sieve."""
    small = primes_upto(int(limit ** 0.5) + 1)
    total = 0
    for lo in range(0, limit + 1, seg):
        hi = min(lo + seg, limit + 1)
        s = np.ones(hi - lo, bool)
        if lo == 0:
            s[:2] = False
        for p in small:
            start = max(p * p, (lo + p - 1) // p * p)
            if start < hi:
                s[start - lo::p] = False
        total += int(s.sum())
    return total


def main(out):
    import mpmath as mp
    from scipy.integrate import quad
    from scipy.special import exp1, expi
    mp.mp.dps = 15
    for g, n in zip(G3, (1, 2, 3)):
        assert abs(float(mp.zetazero(n).imag) - g) < 1e-9
    # 1. zeta along the line
    n_s = int(math.ceil(T_END / STEP)) + 3
    chunks = [(i, min(i + 400, n_s)) for i in range(0, n_s, 400)]
    with Pool(4) as pool:
        zs = np.array(sum(pool.map(_zeta_chunk, chunks), []), np.complex128)
        # 2. the zeros, by index, up to the height reached
        n_exact = int(mp.nzeros(T_END))
        gam = np.array(sum(pool.map(_zero_chunk, [list(range(i, min(i + 25, n_exact + 2))) for i in range(1, n_exact + 2, 25)]), []))
    ts = np.arange(n_s) * STEP
    assert gam[n_exact - 1] <= T_END < gam[n_exact], (gam[n_exact - 1], T_END, gam[n_exact])
    gam = gam[:n_exact]
    theta = np.array([float(mp.siegeltheta(t)) for t in ts])
    Z = (np.exp(1j * theta) * zs).real
    m = ts <= T_END
    changes = int(np.sum(np.sign(Z[m][1:]) != np.sign(Z[m][:-1])))
    assert changes == n_exact, (changes, n_exact)          # every zero up to T_END is on the line, and simple
    assert np.abs((np.exp(1j * theta) * zs).imag).max() < 1e-6
    # 3. the explicit formula, one zero at a time
    x = np.linspace(X_MIN, X_MAX, NX)
    base = np.zeros(NX)
    terms = np.zeros((n_exact, NX))
    for n, mu in N_SET:
        y = x ** (1.0 / n)
        L = np.log(y)
        tail = np.array([quad(lambda t: 1 / (t * (t * t - 1) * math.log(t)), yy, np.inf)[0] for yy in y])
        base += mu / n * (expi(L) - math.log(2) + tail)
        z = (0.5 + 1j * gam[:, None]) * L[None, :]
        ei = -exp1(-z) + 1j * math.pi                      # Ei(z) for Im z > 0
        terms += mu / n * (-2 * ei.real)
    for zz in (complex(0.5, gam[0]) * 0.7, complex(0.5, gam[-1]) * 3.9):     # the Ei identity, against mpmath
        assert abs(complex(mp.ei(zz)) - (-exp1(-zz) + 1j * math.pi)) < 1e-9 * max(1, abs(complex(mp.ei(zz))))
    primes = primes_upto(int(X_MAX))
    half = np.arange(2.5, X_MAX, 1.0)                      # midway between whole numbers: away from every step
    true_half = np.searchsorted(primes, half, side="right")
    cum = base[None, :] + np.concatenate([np.zeros((1, NX)), np.cumsum(terms, 0)])
    err = {}
    exact_from = None
    for n in range(n_exact + 1):
        e = np.abs(np.interp(half, x, cum[n]) - true_half).max()
        if n in (0, 1, 3, 10, 30, 100, 200, n_exact):
            err[n] = float(e)
        if e < 0.5 and exact_from is None:
            exact_from = n
        if e >= 0.5:
            exact_from = None
    # when each zero is reached
    te = np.linspace(T_ZETA, T_WRAP, int((T_WRAP - T_ZETA) * 2400) + 1)
    hs = np.array([tau(v) for v in te])
    assert np.all(np.diff(hs) >= 0)
    hit = np.interp(gam, hs, te)
    # facts for the posting kit
    pi9 = count_primes(10 ** 9)
    r9, li9 = float(mp.riemannr(10 ** 9)), float(mp.li(10 ** 9))
    # Platt and Trudgian (2021) checked every zero up to height 3,000,175,332,800: N(T) = theta(T)/pi + 1 + S(T), S small
    n_pt = float(mp.siegeltheta(mp.mpf(3000175332800)) / mp.pi + 1)
    facts = dict(t_end=T_END, zeros=n_exact, sign_changes=changes, gamma_first=gam[:5].tolist(), gamma_last=float(gam[-1]),
                 samples=int(n_s), step=STEP, primes=primes.tolist(), n_primes=int(len(primes)), x_max=X_MAX,
                 max_err_half_integers=err, exact_from_n_zeros=exact_from,
                 zeros_at=dict((f"{t:.1f}", int(np.searchsorted(gam, tau(t), side="right"))) for t in (12.0, 14.2, 17.0, 20.0, 25.0, 30.0)),
                 zeros_at_wrap=int(np.searchsorted(gam, T_END, side="right")),
                 zeta_half=[zs[0].real, zs[0].imag], max_abs_zeta=float(np.abs(zs).max()),
                 pi_1e9=pi9, riemann_R_1e9=r9, li_1e9=li9, R_minus_pi=r9 - pi9, li_minus_pi=li9 - pi9,
                 zeros_below_platt_trudgian_height_main_term=n_pt, years_open=2026 - 1859)
    np.savez(out, zeta=zs, ts=ts, gamma=gam, hit=hit, x=x, base=base, terms=terms.astype(np.float32), primes=primes)
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps({k: v for k, v in facts.items() if k != "primes"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
