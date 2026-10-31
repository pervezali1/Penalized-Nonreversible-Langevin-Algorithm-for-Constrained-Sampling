"""One training step in slow motion, for a one-neuron network and the 8 + 8 network of reel 31. Each network is
trained exactly as in nnbuild/nnsim.py (same data, same seeds, ReLU, Adam) up to a chosen step, then: one
example is sent forward (every activation recorded), the error at the output is computed, the blame is
sent backward (the delta at every neuron and the gradient on every weight recorded), and a plain gradient
step is taken on that one example so the field before and after can be compared. Nothing else."""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
from nnsim import GRID, LIM, LR, forward, grads, init  # noqa: E402
from train import spiral  # noqa: E402

DATA_SEED = 14
# the one-neuron net is taken after it has converged (step 300 of 1,500: its accuracy never changes after that);
# the 8 + 8 net in the middle of training (step 150, where it still gets a quarter of the points wrong).
# the step sizes are plain-gradient step sizes on one example, chosen so the change is visible but small
CASES = [dict(key="one", sizes=[1], seed=1110, stop=300, step=0.3),
         dict(key="deep", sizes=[8, 8], seed=16121, stop=150, step=0.02)]


def train_to(seed, sizes, X, y, stop):
    rng = np.random.default_rng(seed)
    ps = init(rng, [2] + sizes + [1])
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    b1, b2 = 0.9, 0.999
    for s in range(1, stop + 1):
        _, gs, _ = grads(ps, X, y)
        for i, g in enumerate(gs):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            ps[i] -= LR * (m[i] / (1 - b1 ** s)) / (np.sqrt(v[i] / (1 - b2 ** s)) + 1e-8)
    return ps


def one_example(ps, x, t):
    """Forward and backward for a single example: activations, pre-activations, output, deltas, weight gradients."""
    logit, acts, pre = forward(ps, x[None, :])
    p = float(1 / (1 + np.exp(-logit[0])))
    d = np.array([[p - t]])                                  # dLoss/dlogit for cross-entropy
    n_layers = len(ps) // 2
    deltas = [None] * n_layers                               # delta at the output of each weight layer
    gW = [None] * n_layers
    gb = [None] * n_layers
    for li in range(n_layers - 1, -1, -1):
        deltas[li] = d.copy()
        gW[li] = acts[li].T @ d
        gb[li] = d.sum(0)
        if li > 0:
            d = (d @ ps[2 * li].T) * (pre[li - 1] > 0)
    return dict(p=p, acts=[a[0] for a in acts], pre=[q[0] for q in pre], deltas=[q[0] for q in deltas], gW=gW, gb=gb)


def field(ps, GX):
    return (1 / (1 + np.exp(-forward(ps, GX)[0]))).reshape(GRID, GRID).astype(np.float32)


def main(out):
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    g = np.linspace(-LIM, LIM, GRID)
    GXX, GYY = np.meshgrid(g, g)
    GX = np.stack([GXX.ravel(), GYY.ravel()], 1)
    save = dict(X=X, y=y, grid=g)
    for c in CASES:
        ps = train_to(c["seed"], c["sizes"], X, y, c["stop"])
        _, _, p_all = grads(ps, X, y)
        acc0 = float(np.mean((p_all > 0.5) == (y > 0.5)))
        # the example: among the vermilion points the network gets wrong, not at the data's edge, and for which at
        # least one first-layer neuron is on (otherwise no blame reaches the input weights), the one whose blame on the
        # input weights is largest
        _, _, pre_all = forward(ps, X)
        wrong = np.where((y > 0.5) & (p_all < 0.5) & (np.abs(X).max(1) <= 0.9) & ((pre_all[0] > 0).sum(1) > 0))[0]
        assert len(wrong)
        g0 = [np.abs(one_example(ps, X[j], float(y[j]))["gW"][0]).max() for j in wrong]
        i = int(wrong[int(np.argmax(g0))])
        x, t = X[i], float(y[i])
        ex = one_example(ps, x, t)
        f0 = field(ps, GX)
        ps1 = [p - c["step"] * gg for p, gg in zip(ps, [v for pair in zip(ex["gW"], ex["gb"]) for v in pair])]
        ex1 = one_example(ps1, x, t)
        _, _, p_all1 = grads(ps1, X, y)
        acc1 = float(np.mean((p_all1 > 0.5) == (y > 0.5)))
        f1 = field(ps1, GX)
        k = c["key"]
        n_par = sum(int(p.size) for p in ps)
        n_w = sum(int(p.size) for p in ps[0::2])
        print(f"{k}: stop {c['stop']} acc {acc0:.3f}  example #{i} x={np.round(x, 3)} t={t}  p before {ex['p']:.3f} after {ex1['p']:.3f}  acc after {acc1:.3f}  step {c['step']}  params {n_par} (weights {n_w})")
        print(f"   loss on the example {-math.log(ex['p']):.3f} -> {-math.log(ex1['p']):.3f};  field change max {np.abs(f1 - f0).max():.3f} mean {np.abs(f1 - f0).mean():.3f}")
        save[f"{k}_n_par"], save[f"{k}_n_w"] = n_par, n_w
        for li, (gw, gb_) in enumerate(zip(ex["gW"], ex["gb"])):
            print(f"   layer {li}: |grad| max {np.abs(gw).max():.3f}  delta {np.round(ex['deltas'][li], 3)}")
        save[f"{k}_x"], save[f"{k}_t"], save[f"{k}_i"] = x, t, i
        save[f"{k}_p0"], save[f"{k}_p1"] = ex["p"], ex1["p"]
        save[f"{k}_acc0"], save[f"{k}_acc1"] = acc0, acc1
        save[f"{k}_f0"], save[f"{k}_f1"] = f0, f1
        save[f"{k}_stop"], save[f"{k}_step"] = c["stop"], c["step"]
        for li in range(len(ps) // 2):
            save[f"{k}_W{li}"] = ps[2 * li]
            save[f"{k}_b{li}"] = ps[2 * li + 1]
            save[f"{k}_W{li}_after"] = ps1[2 * li]
            save[f"{k}_gW{li}"] = ex["gW"][li]
            save[f"{k}_gb{li}"] = ex["gb"][li]
            save[f"{k}_delta{li}"] = ex["deltas"][li]
        for li, a in enumerate(ex["acts"]):
            save[f"{k}_act{li}"] = a
        for li, q in enumerate(ex["pre"]):
            save[f"{k}_pre{li}"] = q
    np.savez_compressed(out, **save)


if __name__ == "__main__":
    main(sys.argv[1])
