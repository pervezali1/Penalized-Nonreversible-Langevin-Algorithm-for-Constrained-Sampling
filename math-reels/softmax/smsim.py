"""Three classes and softmax. A three-arm spiral (3 x 150 points, 1.7 turns, a little noise) and a network 2 -> 16 -> 16
-> 3 with ReLU, a softmax output and cross-entropy loss, trained with Adam (step 0.005) for 1,500 steps on all points.
Every 15 steps: the three probabilities over a 96 x 96 grid, the accuracy, the loss, and for one ringed point its
three scores (logits), probabilities and the blame at the output p - y. Six seeds; the median final accuracy kept."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, SNAP, LR, GRID, LIM = 1500, 15, 0.005, 96, 1.05
SIZES = [2, 16, 16, 3]
TURNS, NOISE = 3.4, 0.06
SEEDS = range(6)


def spiral3(rng, n=150, noise=NOISE):
    X, y = [], []
    for c in range(3):
        r = np.linspace(0.12, 1.0, n)
        th = TURNS * r * np.pi + c * 2 * np.pi / 3
        pts = np.stack([r * np.cos(th), r * np.sin(th)], 1) + rng.normal(0, noise, (n, 2))
        X.append(pts)
        y += [c] * n
    return np.concatenate(X), np.array(y)


def init(rng):
    ps = []
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        ps += [rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)]
    return ps


def forward(ps, X):
    h, acts, pres = X, [X], []
    nl = len(ps) // 2
    for li in range(nl):
        z = h @ ps[2 * li] + ps[2 * li + 1]
        pres.append(z)
        h = z if li == nl - 1 else np.maximum(z, 0)
        acts.append(h)
    return h, acts, pres


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def grads(ps, X, y):
    z, acts, pres = forward(ps, X)
    p = softmax(z)
    Y = np.eye(3)[y]
    loss = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
    d = (p - Y) / len(y)
    nl = len(ps) // 2
    gs = [None] * len(ps)
    for li in range(nl - 1, -1, -1):
        gs[2 * li] = acts[li].T @ d
        gs[2 * li + 1] = d.sum(0)
        if li > 0:
            d = (d @ ps[2 * li].T) * (pres[li - 1] > 0)
    return loss, gs, p, z


def train(seed, X, y, GX=None, probe=None):
    ps = init(np.random.default_rng(seed))
    m = [np.zeros_like(q) for q in ps]
    v = [np.zeros_like(q) for q in ps]
    snaps = dict(acc=[], loss=[], field=[], z_all=[])
    for s in range(STEPS + 1):
        loss, gs, p, z = grads(ps, X, y)
        if s % SNAP == 0:
            snaps["acc"].append(float(np.mean(p.argmax(1) == y)))
            snaps["loss"].append(loss)
            if GX is not None:
                snaps["field"].append(softmax(forward(ps, GX)[0]).reshape(GRID, GRID, 3).astype(np.float16))
                snaps["z_all"].append(z.astype(np.float32))
        if s == STEPS:
            break
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return {k: np.array(v_) for k, v_ in snaps.items()}


def main(out):
    rng = np.random.default_rng(42)
    X, y = spiral3(rng)
    g = np.linspace(-LIM, LIM, GRID)
    gxx, gyy = np.meshgrid(g, g)
    GX = np.stack([gxx.ravel(), gyy.ravel()], 1)
    finals = [float(train(s, X, y)["acc"][-1]) for s in SEEDS]
    order = np.argsort(finals, kind="stable")
    seed = int(list(SEEDS)[order[(len(finals) - 1) // 2]])
    sn = train(seed, X, y, GX)
    # the ringed point: a blue (class 1) point, away from the centre, that the network gets right only after a while
    # and keeps right: the one whose last wrong snapshot comes latest
    P = np.exp(sn["z_all"] - sn["z_all"].max(2, keepdims=True))
    P /= P.sum(2, keepdims=True)
    right = P.argmax(2) == y[None, :]
    best, probe = -1, None
    for i in np.where((y == 1) & (np.hypot(*X.T) > 0.45))[0]:
        if not right[-1, i]:
            continue
        last_wrong = np.max(np.where(~right[:, i])[0]) if (~right[:, i]).any() else -1
        if last_wrong > best and last_wrong < 0.6 * len(right):
            best, probe = last_wrong, int(i)
    sn["pz"] = sn["z_all"][:, probe]
    sn["pp"] = P[:, probe]
    sn["right_from"] = best + 1
    del sn["z_all"]
    np.savez_compressed(out, X=X, y=y, grid=g, seed=seed, finals=np.array(finals), probe=probe, **sn)
    pp = sn["pp"]
    print("probe right from snapshot", best + 1, "(step", (best + 1) * SNAP, ")")
    print(f"seed {seed} (finals {np.round(finals, 3)}), final acc {sn['acc'][-1]:.3f} loss {sn['loss'][-1]:.3f}; probe #{probe} x={np.round(X[probe], 2)}")
    for k in (0, 5, 10, 20, 50, 100):
        print(f"  snap {k} (step {k * SNAP}): z {np.round(sn['pz'][k], 2)} p {np.round(pp[k], 3)} acc {sn['acc'][k]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "softmax_sim.npz"))
