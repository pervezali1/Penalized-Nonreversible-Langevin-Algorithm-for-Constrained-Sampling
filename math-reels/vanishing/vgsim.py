"""The vanishing gradient, measured. Two deep networks on the two-arm spiral of reel 2 (neurons/train.py):
10 hidden layers of 16 neurons, one with sigmoid and one with ReLU, each with its standard random start
(sigmoid: weights ~ N(0, 1/fan_in), ReLU: N(0, 2/fan_in), biases 0). At the start, the gradient of the
mean cross-entropy loss over all 400 points is computed for every weight layer: the blame each layer
receives. Then both are trained with plain full-batch gradient descent, step size 0.1, 3,000 steps,
recording the loss, the accuracy, the output field and how far each layer's weights have moved
(|W - W0| / |W0|) every 25 steps. Of six seeds per activation the run with the (lower) median final
accuracy is kept. A ten-seed median of the starting gradients is recorded too."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from train import spiral  # noqa: E402

DATA_SEED = 14
DEPTH, WIDTH = 10, 16
LR, STEPS, SNAP = 0.1, 3000, 25
GRID, LIM = 96, 1.05
SEEDS = range(6)
ACTS = ("sigmoid", "relu")


def sig(z):
    return 1 / (1 + np.exp(-z))


def init(rng, act):
    sizes = [2] + [WIDTH] * DEPTH + [1]
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        std = np.sqrt(2 / a) if act == "relu" else np.sqrt(1 / a)
        ps += [rng.normal(0, std, (a, b)), np.zeros(b)]
    return ps


def forward(ps, X, act):
    h, acts, pres = X, [X], []
    nl = len(ps) // 2
    for li in range(nl):
        z = h @ ps[2 * li] + ps[2 * li + 1]
        pres.append(z)
        h = z if li == nl - 1 else (np.maximum(z, 0) if act == "relu" else sig(z))
        acts.append(h)
    return acts, pres


def slope(z, act):
    return (z > 0).astype(float) if act == "relu" else sig(z) * (1 - sig(z))


def grads(ps, X, y, act):
    acts, pres = forward(ps, X, act)
    p = sig(acts[-1][:, 0])
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    d = ((p - y) / len(y))[:, None]
    nl = len(ps) // 2
    gs, deltas = [None] * len(ps), [None] * nl
    for li in range(nl - 1, -1, -1):
        deltas[li] = d
        gs[2 * li] = acts[li].T @ d
        gs[2 * li + 1] = d.sum(0)
        if li > 0:
            d = (d @ ps[2 * li].T) * slope(pres[li - 1], act)
    return float(loss), gs, p, deltas, pres


def layer_rms(gs):
    return np.array([np.sqrt(np.mean(gs[2 * li] ** 2)) for li in range(len(gs) // 2)])


def train(seed, act, X, y, GX=None):
    ps = init(np.random.default_rng(seed), act)
    p0 = [q.copy() for q in ps]
    snaps = dict(loss=[], acc=[], rel=[], pmin=[], pmax=[], field=[])
    for s in range(STEPS + 1):
        loss, gs, p, _, _ = grads(ps, X, y, act)
        if s % SNAP == 0:
            snaps["loss"].append(loss)
            snaps["acc"].append(float(np.mean((p > 0.5) == (y > 0.5))))
            snaps["rel"].append([np.linalg.norm(ps[2 * li] - p0[2 * li]) / np.linalg.norm(p0[2 * li]) for li in range(DEPTH + 1)])
            snaps["pmin"].append(float(p.min()))
            snaps["pmax"].append(float(p.max()))
            if GX is not None:
                snaps["field"].append(sig(forward(ps, GX, act)[0][-1][:, 0]).reshape(GRID, GRID).astype(np.float32))
        if s < STEPS:
            for i in range(len(ps)):
                ps[i] -= LR * gs[i]
    return {k: np.array(v) for k, v in snaps.items()}


def main(out):
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    g = np.linspace(-LIM, LIM, GRID)
    gxx, gyy = np.meshgrid(g, g)
    GX = np.stack([gxx.ravel(), gyy.ravel()], 1)
    save = dict(X=X, y=y, grid=g)
    for act in ACTS:
        # the ten-seed median of the starting gradients
        G = np.array([layer_rms(grads(init(np.random.default_rng(s), act), X, y, act)[1]) for s in range(10)])
        save[f"{act}_g_median10"] = np.median(G, 0)
        finals = [float(train(s, act, X, y)["acc"][-1]) for s in SEEDS]
        order = np.argsort(finals, kind="stable")
        seed = int(list(SEEDS)[order[(len(finals) - 1) // 2]])
        ps = init(np.random.default_rng(seed), act)
        loss, gs, p, deltas, pres = grads(ps, X, y, act)
        g_rms = layer_rms(gs)
        save[f"{act}_seed"] = seed
        save[f"{act}_finals"] = np.array(finals)
        save[f"{act}_g"] = g_rms
        save[f"{act}_delta"] = np.array([np.sqrt(np.mean(d ** 2)) for d in deltas])
        save[f"{act}_slope_mean"] = np.array([float(np.mean(slope(z, act))) for z in pres[:-1]])
        save[f"{act}_loss0"] = loss
        sn = train(seed, act, X, y, GX)
        for k, v in sn.items():
            save[f"{act}_{k}"] = v
        print(f"{act}: seed {seed} (finals {np.round(finals, 3)}), start gradient per layer " + " ".join(f"{v:.1e}" for v in g_rms)
              + f", first/last {g_rms[0] / g_rms[-1]:.1e}; mean slope per hidden layer " + " ".join(f"{v:.3f}" for v in save[f"{act}_slope_mean"])
              + f"; final loss {sn['loss'][-1]:.4f} acc {sn['acc'][-1]:.3f} p range {sn['pmin'][-1]:.4f}-{sn['pmax'][-1]:.4f}; moved L1 {sn['rel'][-1][0]:.2e} L11 {sn['rel'][-1][-1]:.2e}")
    np.savez_compressed(out, **save)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "vanishing_sim.npz"))
