"""Ten layers without an activation are one neuron. The two-arm spiral of reels 31-47 (neurons/train.py, 400 points)
and a deep, narrow network: 2 inputs -> 10 layers of 8 neurons -> 1 output (681 numbers), trained with
backpropagation and Adam (step 0.01, 1,500 steps), binary cross-entropy, from the same random start twice:
  * with no activation (every layer just x W + b): it collapses, exactly, to one neuron (3 numbers);
  * with ReLU after every hidden neuron.
Also a single neuron trained the same way, the collapse layer by layer (the numbers left and the largest change in
the output), and the same two networks from 8 other random starts (the honesty notes).
    python relusim.py  -> relu_results.json, relu_fields.npz"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from nnsim import DATA_SEED, GRID, LIM, init  # noqa: E402
from train import spiral  # noqa: E402

SIZES = [2] + [8] * 10 + [1]
SEED, STEPS, LR, SNAP = 1, 1500, 0.01, 15
OTHER_SEEDS = (0, 2, 3, 4, 5, 6, 7, 8)


def forward(ps, X, relu):
    h = X
    acts, pre = [h], []
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        pre.append(z)
        h = (np.maximum(z, 0.0) if relu else z) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts, pre


def grads(ps, X, y, relu):
    logit, acts, pre = forward(ps, X, relu)
    p = 1 / (1 + np.exp(-logit))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    d = ((p - y) / len(y))[:, None]
    gs = [None] * len(ps)
    for i in range(len(ps) - 2, -1, -2):
        gs[i] = acts[i // 2].T @ d
        gs[i + 1] = d.sum(0)
        if i > 0:
            d = d @ ps[i].T
            if relu:
                d = d * (pre[i // 2 - 1] > 0)
    return loss, gs, p


def train(seed, sizes, relu, X, y, GX=None):
    ps = init(np.random.default_rng(seed), sizes)
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    accs, fields, snaps = [], [], []
    for s in range(STEPS + 1):
        loss, gs, p = grads(ps, X, y, relu)
        accs.append(float(np.mean((p > 0.5) == (y > 0.5))))
        if GX is not None and s % SNAP == 0:
            fields.append((1 / (1 + np.exp(-forward(ps, GX, relu)[0]))).reshape(GRID, GRID).astype(np.float32))
            snaps.append(np.concatenate([q.ravel() for q in ps]).astype(np.float32))
        if s == STEPS:
            break
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps, accs, fields, snaps


def collapse(ps, j):
    """Merge the first j weight layers of a linear network into one: (x W1 + b1) W2 + b2 = x (W1 W2) + (b1 W2 + b2)."""
    Wm, bm = ps[0], ps[1]
    for i in range(1, j):
        Wm, bm = Wm @ ps[2 * i], bm @ ps[2 * i] + ps[2 * i + 1]
    return [Wm, bm] + ps[2 * j:]


def main():
    X, y = spiral(np.random.default_rng(DATA_SEED))
    g = np.linspace(-LIM, LIM, GRID)
    GXX, GYY = np.meshgrid(g, g)
    GX = np.stack([GXX.ravel(), GYY.ravel()], 1)
    lin, acc_lin, f_lin, s_lin = train(SEED, SIZES, False, X, y, GX)
    rel, acc_rel, f_rel, s_rel = train(SEED, SIZES, True, X, y, GX)
    one, acc_one, f_one, _ = train(SEED, [2, 1], False, X, y, GX)
    n_params = sum(p.size for p in lin)
    both = np.concatenate([X, GX])
    z_full = forward(lin, both, False)[0]
    steps = []
    for j in range(1, len(lin) // 2 + 1):
        c = collapse(lin, j)
        steps.append(dict(merged=j, numbers=int(sum(p.size for p in c)), max_diff=float(np.abs(forward(c, both, False)[0] - z_full).max())))
    w_eff, b_eff = collapse(lin, len(lin) // 2)
    max_diff = max(s["max_diff"] for s in steps)
    decimals = int(np.floor(-np.log10(2 * max_diff)))
    others = dict(linear=[train(sd, SIZES, False, X, y)[1][-1] for sd in OTHER_SEEDS], relu=[train(sd, SIZES, True, X, y)[1][-1] for sd in OTHER_SEEDS])
    out = dict(sizes=SIZES, seed=SEED, steps=STEPS, lr=LR, snap=SNAP, n_params=int(n_params), acc_linear=acc_lin, acc_relu=acc_rel, acc_one=acc_one,
               w_eff=w_eff.ravel().tolist(), b_eff=float(b_eff[0]), w_one=one[0].ravel().tolist(), b_one=float(one[1][0]), collapse=steps,
               max_diff=max_diff, decimals=decimals, others=others, right_linear=int(round(acc_lin[-1] * 400)), right_relu=int(round(acc_rel[-1] * 400)),
               right_one=int(round(acc_one[-1] * 400)), relu_all=sorted([acc_rel[-1]] + others["relu"]))
    np.savez_compressed(os.path.join(HERE, "relu_fields.npz"), f_lin=np.array(f_lin), f_rel=np.array(f_rel), f_one=np.array(f_one), s_lin=np.array(s_lin), s_rel=np.array(s_rel))
    with open(os.path.join(HERE, "relu_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"{n_params} numbers. linear {acc_lin[-1]:.4f} ({out['right_linear']}/400), one neuron {acc_one[-1]:.4f}, relu {acc_rel[-1]:.4f} ({out['right_relu']}/400)")
    print("collapsed:", np.round(w_eff.ravel(), 4), round(float(b_eff[0]), 4), "one neuron:", np.round(one[0].ravel(), 4), round(float(one[1][0]), 4))
    print("collapse:", [(s["merged"], s["numbers"], f"{s['max_diff']:.1e}") for s in steps], "decimals", decimals)
    print("other starts: linear", [round(a, 3) for a in others["linear"]], "relu", [round(a, 3) for a in others["relu"]], "relu median of 9:", np.median(out["relu_all"]))


if __name__ == "__main__":
    main()
