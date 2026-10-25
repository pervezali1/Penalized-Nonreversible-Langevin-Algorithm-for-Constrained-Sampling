"""Train the same 2-32-32-1 network on the same spiral three times with the same initial weights and
the same Adam settings, changing only the activation function (ReLU, tanh, sigmoid). Records the
weights after every step so the renderer shows the true decision field of each. Saves an .npz."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "neurons"))
from train import init, spiral  # noqa: E402

DATA_SEED = 14            # the spiral of reel 02
INIT_SEED = 2             # identical starting weights for all three nets
SIZES = [2, 32, 32, 1]
STEPS = 1000
LR = 0.01
NAMES = ["relu", "tanh", "sigmoid"]

ACTS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(np.float64)),
    "tanh": (np.tanh, lambda a: 1 - a * a),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),
}


def forward(ps, X, act):
    f = ACTS[act][0]
    h = X
    acts = [h]
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        h = f(z) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts


def train(act, X, y):
    ps = init(np.random.default_rng(INIT_SEED), SIZES)
    df = ACTS[act][1]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    hist = [[p.copy() for p in ps]]
    acc, loss = [], []
    for s in range(1, STEPS + 1):
        logit, acts = forward(ps, X, act)
        p = 1 / (1 + np.exp(-logit))
        acc.append(float(((p > 0.5) == y).mean()))
        loss.append(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
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
    logit, _ = forward(ps, X, act)
    p = 1 / (1 + np.exp(-logit))
    acc.append(float(((p > 0.5) == y).mean()))
    loss.append(loss[-1])
    return hist, np.array(acc), np.array(loss)


def main(out_path):
    X, y = spiral(np.random.default_rng(DATA_SEED))
    arrays = dict(X=X, y=y)
    meta = dict(steps=STEPS, lr=LR, sizes=SIZES, names=NAMES, acc={}, first100={}, final={})
    for name in NAMES:
        hist, acc, loss = train(name, X, y)
        for k in range(len(hist[0])):
            arrays[f"{name}_p{k}"] = np.stack([h[k] for h in hist])          # (STEPS+1, ...)
        meta["acc"][name] = acc.tolist()
        hit = np.where(acc >= 1.0)[0]
        meta["first100"][name] = int(hit[0]) if len(hit) else None
        meta["final"][name] = float(acc[-1])
        print(f"{name:8s} final {acc[-1]:.3f} first 100% at step {meta['first100'][name]} "
              f"acc@250 {acc[250]:.2f} @500 {acc[500]:.2f} @750 {acc[750]:.2f}")
    np.savez_compressed(out_path, **arrays)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "activations_train.npz"))
