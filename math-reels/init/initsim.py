"""Train the same 2-32-32-1 ReLU network on the same spiral three times with the same Adam settings,
changing only the starting weights: random (reel 18's seed), all zeros, and every weight the same
number (0.1). Records the weights after every step and, per step, how many distinct neurons the
first hidden layer contains (unique rows of weights+bias). Saves an .npz."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "neurons"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "activations"))
from actsim import ACTS  # noqa: E402
from train import init, spiral  # noqa: E402

DATA_SEED = 14
INIT_SEED = 2
SIZES = [2, 32, 32, 1]
STEPS = 1000
LR = 0.01
ACT = "relu"
SAME_VALUE = 0.1
NAMES = ["random", "zeros", "same"]


def start_weights(name):
    if name == "random":
        return init(np.random.default_rng(INIT_SEED), SIZES)
    ps = []
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        ps.append(np.zeros((a, b)) if name == "zeros" else np.full((a, b), SAME_VALUE))
        ps.append(np.zeros(b))
    return ps


def forward(ps, X):
    f = ACTS[ACT][0]
    h = X
    acts = [h]
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        h = f(z) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts


def distinct_neurons(ps):
    W1 = np.concatenate([ps[0].T, ps[1][:, None]], 1)
    return int(len(np.unique(np.round(W1, 9), axis=0)))


def train(name, X, y):
    ps = start_weights(name)
    df = ACTS[ACT][1]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    hist = [[p.copy() for p in ps]]
    acc, distinct = [], [distinct_neurons(ps)]
    for s in range(1, STEPS + 1):
        logit, acts = forward(ps, X)
        p = 1 / (1 + np.exp(-logit))
        acc.append(float(((p > 0.5) == y).mean()))
        g = (p - y)[:, None] / len(y)
        grads = []
        for i in range(len(ps) - 2, -1, -2):
            grads.insert(0, acts[i // 2].T @ g)
            grads.insert(1, g.sum(0))
            if i > 0:
                g = (g @ ps[i].T) * df(acts[i // 2])
        for k, (pk, gk) in enumerate(zip(ps, grads)):
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk * gk
            pk -= LR * (m[k] / (1 - 0.9 ** s)) / (np.sqrt(v[k] / (1 - 0.999 ** s)) + 1e-8)
        hist.append([p_.copy() for p_ in ps])
        distinct.append(distinct_neurons(ps))
    logit, _ = forward(ps, X)
    p = 1 / (1 + np.exp(-logit))
    acc.append(float(((p > 0.5) == y).mean()))
    return hist, np.array(acc), distinct


def main(out_path):
    X, y = spiral(np.random.default_rng(DATA_SEED))
    arrays = dict(X=X, y=y)
    meta = dict(steps=STEPS, lr=LR, sizes=SIZES, names=NAMES, same_value=SAME_VALUE, acc={}, distinct={}, first100={}, final={})
    for name in NAMES:
        hist, acc, distinct = train(name, X, y)
        for k in range(len(hist[0])):
            arrays[f"{name}_p{k}"] = np.stack([h[k] for h in hist])
        meta["acc"][name] = acc.tolist()
        meta["distinct"][name] = distinct
        hit = np.where(acc >= 1.0)[0]
        meta["first100"][name] = int(hit[0]) if len(hit) else None
        meta["final"][name] = float(acc[-1])
        print(f"{name:7s} final {acc[-1]:.4f} first 100% at {meta['first100'][name]} distinct neurons {distinct[0]} -> {distinct[-1]} (max {max(distinct)})")
    np.savez_compressed(out_path, **arrays)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "init_train.npz"))
