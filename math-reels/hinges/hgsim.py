"""A network is a bunch of hinges. One hidden layer of ReLU neurons (1 -> N -> 1, linear output) fitted to one
curve, f(x) = sin(2 pi x) (0.55 + 0.45 x) on [-1, 1] (256 evenly spaced points), for N = 1, 2, 4, 8, 16, 32.
Each neuron computes max(0, w x + b): flat on one side of x = -b/w, a straight ramp on the other, one hinge.
The output is a weighted sum of these bent lines plus a constant. Adam, step size 0.01, 6,000 steps, mean
squared error; hinge positions start spread over [-1, 1]. Six seeds per N; the run with the (lower) median
final RMS error is kept, with its parameters every 60 steps."""

import os
import sys

import numpy as np

NS = (1, 2, 4, 8, 16, 32)
SEEDS = range(6)
STEPS, LR, SNAP = 6000, 0.01, 60
X = np.linspace(-1, 1, 256)


def target(x):
    return np.sin(2 * np.pi * x) * (0.55 + 0.45 * x)


def init(seed, N):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0), N)
    b1 = rng.uniform(-1, 1, N) * np.abs(W1)       # hinge at -b/w, spread over [-1, 1]
    W2 = rng.normal(0, np.sqrt(1 / N), N)
    return [W1, b1, W2, np.zeros(1)]


def forward(ps, x):
    W1, b1, W2, b2 = ps
    z = x[:, None] * W1 + b1
    h = np.maximum(z, 0)
    return h @ W2 + b2[0], z, h


def train(seed, N, snaps=False):
    ps = init(seed, N)
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    yt = target(X)
    out = []
    for s in range(STEPS + 1):
        y, z, h = forward(ps, X)
        if snaps and s % SNAP == 0:
            out.append(np.concatenate([p.ravel() for p in ps]))
        if s == STEPS:
            break
        d = 2 * (y - yt) / len(X)
        gW2 = h.T @ d
        gb2 = np.array([d.sum()])
        dh = d[:, None] * ps[2] * (z > 0)
        gW1 = (X[:, None] * dh).sum(0)
        gb1 = dh.sum(0)
        for i, g in enumerate((gW1, gb1, gW2, gb2)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    rms = float(np.sqrt(np.mean((forward(ps, X)[0] - yt) ** 2)))
    return rms, ps, (np.array(out) if snaps else None)


def main(out):
    save = dict(x=X, y=target(X))
    for N in NS:
        finals = [train(s, N)[0] for s in SEEDS]
        order = np.argsort(finals, kind="stable")
        seed = int(list(SEEDS)[order[(len(finals) - 1) // 2]])
        rms, ps, snaps = train(seed, N, snaps=True)
        save[f"n{N}_seed"] = seed
        save[f"n{N}_finals"] = np.array(finals)
        save[f"n{N}_snaps"] = snaps
        save[f"n{N}_rms"] = rms
        hx = -ps[1] / ps[0]
        inside = int(np.sum((hx > -1) & (hx < 1)))
        print(f"N {N:2d}: seed {seed} rms {rms:.4f} (seeds {np.round(finals, 3)}), hinges inside [-1, 1]: {inside}")
    np.savez_compressed(out, **save)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hinges_sim.npz"))
