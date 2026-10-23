"""10,000 frictionless double pendulums, RK4 at 1200 Hz, float64.

Angles are measured from the downward vertical and are never reduced mod 2*pi,
so the "largest gap so far" counter is a true running max.
"""

import math

import numpy as np
from numba import njit, prange

G = 9.81
DT = 1.0 / 1200.0
SNAP_EVERY = 10          # snapshots at 120 Hz
N = 10_000
T_END = 10.25
L_PX = 165.0             # rod length on screen
PIVOT = (470.0, 1150.0)

THETA1_0 = 140.0
THETA2_0 = 176.0
SPREAD_DEG = 0.99999e-6


@njit(cache=True, fastmath=False)
def accel(t1, t2, w1, w2):
    d = t1 - t2
    sd = math.sin(d)
    cd = math.cos(d)
    D = 3.0 - math.cos(2.0 * d)
    a1 = (-3.0 * G * math.sin(t1) - G * math.sin(t1 - 2.0 * t2)
          - 2.0 * sd * (w2 * w2 + w1 * w1 * cd)) / D
    a2 = (2.0 * sd * (2.0 * w1 * w1 + 2.0 * G * math.cos(t1) + w2 * w2 * cd)) / D
    return a1, a2


@njit(cache=True)
def rk4_one(t1, t2, w1, w2, h):
    k1a, k1b = accel(t1, t2, w1, w2)
    k2a, k2b = accel(t1 + 0.5 * h * w1, t2 + 0.5 * h * w2,
                     w1 + 0.5 * h * k1a, w2 + 0.5 * h * k1b)
    k3a, k3b = accel(t1 + 0.5 * h * (w1 + 0.5 * h * k1a), t2 + 0.5 * h * (w2 + 0.5 * h * k1b),
                     w1 + 0.5 * h * k2a, w2 + 0.5 * h * k2b)
    k4a, k4b = accel(t1 + h * (w1 + 0.5 * h * k2a), t2 + h * (w2 + 0.5 * h * k2b),
                     w1 + h * k3a, w2 + h * k3b)
    nt1 = t1 + h / 6.0 * (w1 + 2 * (w1 + 0.5 * h * k1a) + 2 * (w1 + 0.5 * h * k2a) + (w1 + h * k3a))
    nt2 = t2 + h / 6.0 * (w2 + 2 * (w2 + 0.5 * h * k1b) + 2 * (w2 + 0.5 * h * k2b) + (w2 + h * k3b))
    nw1 = w1 + h / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
    nw2 = w2 + h / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)
    return nt1, nt2, nw1, nw2


@njit(cache=True, parallel=True)
def step_all(state, h):
    for i in prange(state.shape[0]):
        state[i, 0], state[i, 1], state[i, 2], state[i, 3] = rk4_one(
            state[i, 0], state[i, 1], state[i, 2], state[i, 3], h)


@njit(cache=True)
def stats(state):
    """(gap in degrees, tip bbox diagonal in px, energy mean, energy max abs deviation)."""
    n = state.shape[0]
    mn1 = mx1 = state[0, 0]
    mn2 = mx2 = state[0, 1]
    xmin = ymin = 1e300
    xmax = ymax = -1e300
    for i in range(n):
        t1 = state[i, 0]
        t2 = state[i, 1]
        if t1 < mn1: mn1 = t1
        if t1 > mx1: mx1 = t1
        if t2 < mn2: mn2 = t2
        if t2 > mx2: mx2 = t2
        x = PIVOT[0] + L_PX * (math.sin(t1) + math.sin(t2))
        y = PIVOT[1] + L_PX * (math.cos(t1) + math.cos(t2))
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    gap = max(mx1 - mn1, mx2 - mn2) * 180.0 / math.pi
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    return gap, diag


@njit(cache=True)
def energy(state):
    n = state.shape[0]
    out = np.empty(n)
    for i in range(n):
        t1, t2, w1, w2 = state[i, 0], state[i, 1], state[i, 2], state[i, 3]
        out[i] = 0.5 * (2 * w1 * w1 + w2 * w2 + 2 * w1 * w2 * math.cos(t1 - t2)) \
            - G * (2 * math.cos(t1) + math.cos(t2))
    return out


def initial_state():
    i = np.arange(N, dtype=np.float64)
    st = np.zeros((N, 4))
    st[:, 0] = math.radians(THETA1_0)
    st[:, 1] = np.radians(THETA2_0 + (i / (N - 1) - 0.5) * SPREAD_DEG)
    return st


def run(t_end=T_END, dt=DT, snap_every=SNAP_EVERY):
    n_steps = int(round(t_end / dt))
    st = initial_state()
    spread = np.degrees(st[:, 1].max() - st[:, 1].min())
    assert spread <= 1e-6, spread
    assert len(np.unique(st[:, 1])) == N
    e0 = energy(st)
    snaps = np.empty((n_steps // snap_every + 1, N, 4))
    gap = np.empty(n_steps + 1)
    diag = np.empty(n_steps + 1)
    snaps[0] = st
    gap[0], diag[0] = stats(st)
    drift = 0.0
    for k in range(1, n_steps + 1):
        step_all(st, dt)
        gap[k], diag[k] = stats(st)
        if k % snap_every == 0:
            snaps[k // snap_every] = st
            e = energy(st)
            drift = max(drift, np.max(np.abs(e - e0)) / abs(e0.mean()))
    assert drift < 1e-6, drift
    M = np.maximum.accumulate(gap)
    E = np.maximum.accumulate(diag)
    return dict(snaps=snaps, gap=gap, M=M, E=E, drift=drift, dt=dt, snap_every=snap_every,
                e0=e0.mean(), spread=spread)


@njit(cache=True)
def state_at(snaps, s, dt, snap_every):
    """Exact RK4 state at sim time s >= 0 from the nearest earlier snapshot."""
    snap_dt = dt * snap_every
    n = int(math.floor(s / snap_dt + 1e-9))
    if n >= snaps.shape[0]:
        n = snaps.shape[0] - 1
    st = snaps[n].copy()
    rem = s - n * snap_dt
    full = int(math.floor(rem / dt + 1e-9))
    for _ in range(full):
        for i in range(st.shape[0]):
            st[i, 0], st[i, 1], st[i, 2], st[i, 3] = rk4_one(st[i, 0], st[i, 1], st[i, 2], st[i, 3], dt)
    h = rem - full * dt
    if h > 1e-12:
        for i in range(st.shape[0]):
            st[i, 0], st[i, 1], st[i, 2], st[i, 3] = rk4_one(st[i, 0], st[i, 1], st[i, 2], st[i, 3], h)
    return st


def first_crossing(arr, thr, dt=DT):
    idx = np.argmax(arr >= thr)
    return idx * dt if arr[idx] >= thr else None


if __name__ == "__main__":
    import sys
    import time

    out = sys.argv[1] if len(sys.argv) > 1 else "sim.npz"
    t0 = time.time()
    r = run()
    print(f"integrated in {time.time() - t0:.1f}s, drift {r['drift']:.2e}, spread {r['spread']:.6g} deg")
    for thr in (1.5e-6, 9.5e-6, 9.5e-5, 9.5e-4, 9.5e-3, 0.095, 0.95, 9.5, 100, 180):
        print(f"  M >= {thr:<8g} at s = {first_crossing(r['M'], thr):.4f}")
    for k in range(7, -1, -1):
        s = first_crossing(r["E"], 252.0 / 10 ** k)
        print(f"  lens x10^{k} ends at s = {s}")
    np.savez(out, snaps=r["snaps"], gap=r["gap"], M=r["M"], E=r["E"], dt=r["dt"],
             snap_every=r["snap_every"], drift=r["drift"], e0=r["e0"])
