"""Train the same two-class spiral (neurons/train.py) with five networks of growing size, ReLU hidden units,
Adam, binary cross-entropy: 1, 2, 3 and 8 hidden neurons in one layer, then two layers of 8. Record, at
every snapshot, the output probability on a grid (the field), the first-layer weights (each one is a
straight fold line in the input plane), the accuracy and the loss. Each stage is trained from scratch;
of six random seeds per stage the run with the median final accuracy is kept (the typical run)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from train import spiral  # noqa: E402

STAGES = [([1], "one"), ([2], "two"), ([3], "three"), ([8], "eight"), ([8, 8], "deep")]
STEPS = 1500
SNAP_EVERY = 15
LR = 0.01
GRID = 96
LIM = 1.05
N_SEEDS = 6
DATA_SEED = 14


def init(rng, sizes):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        ps.append(rng.normal(0, np.sqrt(2 / a), (a, b)))
        ps.append(np.zeros(b))
    return ps


def forward(ps, X):
    h = X
    acts, pre = [h], []
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        pre.append(z)
        h = np.maximum(z, 0.0) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts, pre


def grads(ps, X, y):
    logit, acts, pre = forward(ps, X)
    p = 1 / (1 + np.exp(-logit))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    d = ((p - y) / len(y))[:, None]
    gs = [None] * len(ps)
    for i in range(len(ps) - 2, -1, -2):
        gs[i] = acts[i // 2].T @ d
        gs[i + 1] = d.sum(0)
        if i > 0:
            d = (d @ ps[i].T) * (pre[i // 2 - 1] > 0)
    return loss, gs, p


def train(seed, sizes, X, y, GX):
    rng = np.random.default_rng(seed)
    ps = init(rng, [2] + sizes + [1])
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    b1, b2 = 0.9, 0.999
    fields, W1s, B1s, accs, losses, finals = [], [], [], [], [], None

    def snap(s):
        loss, _, p = grads(ps, X, y)
        fields.append((1 / (1 + np.exp(-forward(ps, GX)[0]))).reshape(GRID, GRID).astype(np.float32))
        W1s.append(ps[0].copy())
        B1s.append(ps[1].copy())
        accs.append(float(np.mean((p > 0.5) == (y > 0.5))))
        losses.append(float(loss))

    snap(0)
    for s in range(1, STEPS + 1):
        loss, gs, p = grads(ps, X, y)
        for i, g in enumerate(gs):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            ps[i] -= LR * (m[i] / (1 - b1 ** s)) / (np.sqrt(v[i] / (1 - b2 ** s)) + 1e-8)
        if s % SNAP_EVERY == 0:
            snap(s)
    return dict(fields=np.stack(fields), W1=np.stack(W1s), B1=np.stack(B1s), acc=np.array(accs), loss=np.array(losses),
                params=[p.copy() for p in ps], seed=seed)


def main(out):
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    g = np.linspace(-LIM, LIM, GRID)
    GXX, GYY = np.meshgrid(g, g)
    GX = np.stack([GXX.ravel(), GYY.ravel()], 1)
    save = dict(X=X, y=y, grid=g)
    for sizes, name in STAGES:
        runs = [train(100 + 10 * len(sizes) + 1000 * sum(sizes) + k, sizes, X, y, GX) for k in range(N_SEEDS)]
        finals = [r["acc"][-1] for r in runs]
        order = np.argsort(finals)
        r = runs[order[len(runs) // 2]]
        print(f"{name:6s} sizes {sizes}: final acc by seed {[round(f, 3) for f in finals]} -> kept seed {r['seed']} acc {r['acc'][-1]:.3f} loss {r['loss'][-1]:.3f}")
        save[f"{name}_fields"] = r["fields"]
        save[f"{name}_W1"] = r["W1"]
        save[f"{name}_B1"] = r["B1"]
        save[f"{name}_acc"] = r["acc"]
        save[f"{name}_loss"] = r["loss"]
        save[f"{name}_seed"] = r["seed"]
        save[f"{name}_finals"] = np.array(finals)
        for i, p in enumerate(r["params"]):
            save[f"{name}_p{i}"] = p
    np.savez_compressed(out, **save)


if __name__ == "__main__":
    main(sys.argv[1])
