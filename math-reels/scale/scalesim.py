"""Train the same 8-hidden-layer ReLU network (2 -> 32 x 8 -> 1) on the same spiral three times
with plain gradient descent, changing only the SIZE of the random starting weights: std 0.01 (too
small), He initialisation sqrt(2 / fan_in) = 0.25 (just right), std 1.0 (too big). Records, per step,
the decision field on a grid, which points are classified correctly, the accuracy and the standard
deviation of the activations at every hidden layer (how loud the signal is as it passes through).
Saves an .npz (weights are not stored: 8 layers x 1,000 steps x 3 nets would be 200 MB)."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "neurons"))
from train import spiral  # noqa: E402

DATA_SEED = 14
INIT_SEED = 3
DEPTH, WIDTH = 8, 32
SIZES = [2] + [WIDTH] * DEPTH + [1]
STEPS = 1000
LR = 0.1
NAMES = ["small", "he", "big"]
HE_STD = float(np.sqrt(2 / WIDTH))
STD = {"small": 0.01, "he": None, "big": 1.0}          # None: He, sqrt(2 / fan_in) per layer
GRID = 78                                            # field resolution per panel (310 px / 4)
GRID_EXTENT = 1.05


def start_weights(name):
    rng = np.random.default_rng(INIT_SEED)
    ps = []
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        std = np.sqrt(2 / a) if STD[name] is None else STD[name]
        ps.append(rng.normal(0, std, (a, b)))
        ps.append(np.zeros(b))
    return ps


def forward(ps, X):
    h = X
    acts = [h]
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        h = np.maximum(z, 0.0) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts


def grid_points():
    xs = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), -gy.ravel()], 1)


def train(name, X, y, G):
    ps = start_weights(name)
    fields = np.zeros((STEPS + 1, GRID, GRID), np.float16)
    right = np.zeros((STEPS + 1, len(y)), bool)
    sig = np.zeros((STEPS + 1, DEPTH), np.float32)
    acc = np.zeros(STEPS + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        for s in range(STEPS + 1):
            logit, acts = forward(ps, X)
            logit = np.nan_to_num(logit, nan=0.0, posinf=30, neginf=-30)
            p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
            right[s] = (p > 0.5) == (y > 0.5)
            acc[s] = right[s].mean()
            sig[s] = [np.nan_to_num(np.std(a)) for a in acts[1:-1]]
            gl = np.nan_to_num(forward(ps, G)[0], nan=0.0, posinf=30, neginf=-30)
            fields[s] = (1 / (1 + np.exp(-np.clip(gl, -30, 30)))).reshape(GRID, GRID)
            if s == STEPS:
                break
            g = (p - y)[:, None] / len(y)
            grads = []
            for i in range(len(ps) - 2, -1, -2):
                grads.insert(0, acts[i // 2].T @ g)
                grads.insert(1, g.sum(0))
                if i > 0:
                    g = (g @ ps[i].T) * (acts[i // 2] > 0)
            for pk, gk in zip(ps, grads):
                pk -= LR * np.nan_to_num(gk)
    return fields, right, sig, acc


def main(out_path):
    X, y = spiral(np.random.default_rng(DATA_SEED))
    G = grid_points()
    arrays = dict(X=X, y=y)
    meta = dict(steps=STEPS, lr=LR, sizes=SIZES, names=NAMES, he_std=HE_STD,
                std={n: (HE_STD if STD[n] is None else STD[n]) for n in NAMES}, acc={}, final={}, first90={}, sig_start={}, sig_end={})
    for name in NAMES:
        fields, right, sig, acc = train(name, X, y, G)
        arrays[f"{name}_fields"] = fields
        arrays[f"{name}_right"] = right
        arrays[f"{name}_sig"] = sig
        meta["acc"][name] = acc.tolist()
        meta["final"][name] = float(acc[-1])
        hit = np.where(acc >= 0.9)[0]
        meta["first90"][name] = int(hit[0]) if len(hit) else None
        meta["sig_start"][name] = sig[0].tolist()
        meta["sig_end"][name] = sig[-1].tolist()
        print(f"{name:5s} std {meta['std'][name]:.3f} final {acc[-1]:.3f} first 90% at {meta['first90'][name]} "
              f"signal layer1..8 at start: {' '.join('%.0e' % v for v in sig[0])}  | at end: {' '.join('%.0e' % v for v in sig[-1])}")
    np.savez_compressed(out_path, **arrays)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "scale_train.npz"))
