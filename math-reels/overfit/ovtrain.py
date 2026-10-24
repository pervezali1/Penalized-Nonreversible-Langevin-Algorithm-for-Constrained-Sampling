"""Reel 2's 64-neuron network trained twice on the same 400 spiral points: real labels, then the
same labels shuffled. A held-out test set exposes the difference. Weights recorded per frame."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neurons"))
from train import forward, init, predict, spiral  # noqa: E402

FPS = 60
N_FRAMES = 1200
SEED = 14
HIDDEN = [32, 32]
LR, STEPS = 0.02, 3000
STAGES = [(72, 372, 1.0), (456, 840, 1.0)]       # (first training frame, last, pacing exponent)
SHUFFLE = (396, 444)                              # labels flip to their shuffled values
T_TEST_A, T_TEST_B = 4.4, 14.0                    # test diamonds appear
LOOP = (1164, 1200)


def train(rng, X, y, steps, lr):
    ps = init(rng, [2] + HIDDEN + [1])
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    hist = [[p.copy() for p in ps]]
    acc = []
    for s in range(1, steps + 1):
        logit, acts = forward(ps, X)
        p = 1 / (1 + np.exp(-logit))
        acc.append(float(((p > 0.5) == y).mean()))
        g = (p - y)[:, None] / len(y)
        grads = []
        for i in range(len(ps) - 2, -1, -2):
            grads.insert(0, acts[i // 2].T @ g)
            grads.insert(1, g.sum(0))
            if i > 0:
                g = (g @ ps[i].T) * (1 - acts[i // 2] ** 2)
        for k, (pk, gk) in enumerate(zip(ps, grads)):
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk * gk
            pk -= lr * (m[k] / (1 - 0.9 ** s)) / (np.sqrt(v[k] / (1 - 0.999 ** s)) + 1e-8)
        hist.append([p_.copy() for p_ in ps])
    acc.append(float(((predict(ps, X) > 0.5) == y).mean()))
    return hist, np.array(acc)


def frame_step(f, f0, f1, expo):
    if f < f0:
        return 0
    if f >= f1:
        return STEPS
    return int(round(STEPS * ((f - f0) / (f1 - f0)) ** expo))


def main(out_path):
    rng = np.random.default_rng(SEED)
    X, y = spiral(rng)
    Xt, yt = spiral(np.random.default_rng(99))
    y_shuf = np.random.default_rng(5).permutation(y)
    yt_rand = np.random.default_rng(7).integers(0, 2, len(yt)).astype(np.float64)
    runs = []
    for labels, test_labels in ((y, yt), (y_shuf, yt_rand)):
        hist, acc = train(np.random.default_rng(SEED), X, labels, STEPS, LR)
        test = np.array([float(((predict(h, Xt) > 0.5) == test_labels).mean()) for h in hist])
        runs.append((hist, acc, test))
        print(f"train {acc[-1]:.3f}  test {test[-1]:.3f}")
    frames = []
    for f in range(N_FRAMES):
        if f >= LOOP[0]:
            frames.append((0, 0))                 # the loop tail shows the untrained network, like frame 0
            continue
        si = 1 if f >= SHUFFLE[0] else 0
        f0, f1, expo = STAGES[si]
        frames.append((si, frame_step(f, f0, f1, expo)))
    meta = dict(frames=frames,
                train=[[runs[si][1][s] for si, s in frames]][0],
                test=[[runs[si][2][s] for si, s in frames]][0],
                final=[dict(train=float(r[1][-1]), test=float(r[2][-1])) for r in runs])
    arrays = dict(X=X, y=y, y_shuf=y_shuf, Xt=Xt, yt=yt, yt_rand=yt_rand)
    for si, (hist, _a, _t) in enumerate(runs):
        for k in range(len(hist[0])):
            arrays[f"s{si}_p{k}"] = np.stack([h[k] for h in hist])
    np.savez(out_path, **arrays)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1])
