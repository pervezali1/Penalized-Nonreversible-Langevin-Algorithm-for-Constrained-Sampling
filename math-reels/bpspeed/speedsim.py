"""Why backpropagation is fast, measured. (1) The 8 + 8 network of reels 31 and 33 (bpsim.train_to: the spiral, the
same seed, stopped at step 150): the gradient of the mean cross-entropy over all 400 points, once by backpropagation
(one forward and one backward pass) and once by central differences (nudge each of the 105 numbers by +-1e-5, two
forward passes each: 210 passes); the largest difference between the two. (2) A network with about a million
weights (2 -> 1000 -> 1000 -> 1, ReLU) on the same 400 points: the time of one forward pass and of one
backpropagation (forward + backward), median of repeated runs on this machine, central differences spot-checked
against backpropagation on 30 random weights, and the time 2,010,002 forward passes would take."""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backprop"))
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
from bpsim import DATA_SEED, grads, spiral, train_to  # noqa: E402
from nnsim import forward  # noqa: E402


def loss_of(ps, X, y):
    logit = forward(ps, X)[0]
    p = 1 / (1 + np.exp(-logit))
    return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))


def finite_diff(ps, X, y, eps=1e-5, idx=None):
    flat = [(li, i) for li, p in enumerate(ps) for i in range(p.size)]
    if idx is not None:
        flat = [flat[k] for k in idx]
    out, passes = [], 0
    for li, i in flat:
        a = [q.copy() for q in ps]
        b = [q.copy() for q in ps]
        a[li].flat[i] += eps
        b[li].flat[i] -= eps
        out.append((loss_of(a, X, y) - loss_of(b, X, y)) / (2 * eps))
        passes += 2
    return np.array(out), passes


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(out):
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    ps = train_to(16121, [8, 8], X, y, 150)
    n_par = sum(p.size for p in ps)
    _, gs, _ = grads(ps, X, y)
    g_bp = np.concatenate([g.ravel() for g in gs])
    g_fd, passes = finite_diff(ps, X, y)
    diff = float(np.abs(g_bp - g_fd).max())
    order = [li for li, p in enumerate(ps) for _ in range(p.size)]
    small = dict(n_par=int(n_par), passes_fd=int(passes), passes_bp=2, max_abs_diff=diff, g_bp=g_bp.tolist(), g_fd=g_fd.tolist(),
                 g_max=float(np.abs(g_bp).max()), layer_of=order,
                 t_forward=timed(lambda: loss_of(ps, X, y), 400), t_backprop=timed(lambda: grads(ps, X, y), 400))
    small["t_fd_all"] = timed(lambda: finite_diff(ps, X, y), 5)
    print(f"8+8: {n_par} numbers, {passes} passes vs 2; max |difference| {diff:.2e}; forward {small['t_forward'] * 1e6:.0f} us, backprop {small['t_backprop'] * 1e6:.0f} us, all nudges {small['t_fd_all'] * 1e3:.1f} ms")
    # the big network
    rng2 = np.random.default_rng(41)
    sizes = [2, 1000, 1000, 1]
    big = []
    for a_, b_ in zip(sizes[:-1], sizes[1:]):
        big += [rng2.normal(0, np.sqrt(2 / a_), (a_, b_)), np.zeros(b_)]
    n_big = sum(p.size for p in big)
    t_f = timed(lambda: loss_of(big, X, y), 30)
    t_b = timed(lambda: grads(big, X, y), 30)
    _, gb, _ = grads(big, X, y)
    gflat = np.concatenate([g.ravel() for g in gb])
    pick = np.random.default_rng(7).choice(n_big, 30, replace=False)
    gfd, _ = finite_diff(big, X, y, eps=1e-5, idx=pick)
    rel = float(np.max(np.abs(gfd - gflat[pick])) / (np.abs(gflat[pick]).max() + 1e-30))
    est = 2 * n_big * t_f
    bigd = dict(sizes=sizes, n_par=int(n_big), passes_fd=int(2 * n_big), t_forward=t_f, t_backprop=t_b, fd_estimate_s=est,
                spot_check_n=30, spot_check_rel_err=rel, ratio_bp_to_forward=t_b / t_f, speedup=est / t_b)
    print(f"big: {n_big:,} numbers; forward {t_f * 1e3:.1f} ms, backprop {t_b * 1e3:.1f} ms ({t_b / t_f:.1f} forward passes); "
          f"nudging all: {2 * n_big:,} passes = {est / 3600:.1f} h; spot check rel err {rel:.1e}; speed-up {est / t_b:,.0f}x")
    with open(out, "w") as fh:
        json.dump(dict(small=small, big=bigd), fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "speed_results.json"))
