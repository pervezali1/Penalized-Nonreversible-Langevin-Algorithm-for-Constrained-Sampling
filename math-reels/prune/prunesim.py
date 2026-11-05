"""Delete most of a network. The handwriting network of reel 49 (retrained exactly, see adversarial/advsim.py: 2,368
weights, 96.7% of the 540 unseen digits right) with a growing share of its weights set to zero, three ways:
  * at random (the median of 20 random choices);
  * the smallest first (the weights closest to zero, both layers together);
  * the smallest first, then 100 more Adam steps on the 1,257 training digits with the deleted weights held at zero.
Recorded: the accuracy for each share, the 32 neurons' weight pictures with the deleted weights, and what the network
reads for ten unseen digits.    python prunesim.py  -> prune_results.json"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
import dgsim  # noqa: E402
from advsim import trained  # noqa: E402

FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RETRAIN, N_RANDOM = 100, 20


def accuracy(ps, X, y):
    return float(np.mean(dgsim.forward(ps, X)[0].argmax(1) == y))


def smallest_first(ps, frac):
    w = np.abs(np.concatenate([ps[0].ravel(), ps[2].ravel()]))
    th = np.quantile(w, frac) if frac > 0 else -1.0
    m1, m2 = np.abs(ps[0]) > th, np.abs(ps[2]) > th
    return [ps[0] * m1, ps[1], ps[2] * m2, ps[3]], (m1, m2)


def retrain(ps, masks, Xtr, ytr, steps=RETRAIN):
    q = [a.copy() for a in ps]
    Y1 = np.eye(10)[ytr]
    m = [np.zeros_like(a) for a in q]
    v = [np.zeros_like(a) for a in q]
    for s in range(steps):
        _, gs = dgsim.grads(q, Xtr, Y1)
        gs[0], gs[2] = gs[0] * masks[0], gs[2] * masks[1]
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            q[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
        q[0], q[2] = q[0] * masks[0], q[2] * masks[1]
    return q


def main():
    ps, Xte, yte = trained()
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    Xtr, ytr = X[idx[540:]], y[idx[540:]]
    n_w = int(ps[0].size + ps[2].size)
    rng = np.random.default_rng(55)
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    rows = []
    for f in FRACS:
        rand = []
        for _ in range(N_RANDOM):
            m1, m2 = rng.random(ps[0].shape) >= f, rng.random(ps[2].shape) >= f
            rand.append(accuracy([ps[0] * m1, ps[1], ps[2] * m2, ps[3]], Xte, yte))
        q, masks = smallest_first(ps, f)
        r = retrain(q, masks, Xtr, ytr) if f > 0 else ps
        reads = lambda net: dgsim.forward(net, Xte[show])[0].argmax(1).tolist()  # noqa: E731
        rows.append(dict(frac=f, kept=int(masks[0].sum() + masks[1].sum()), random=float(np.median(rand)), random_range=[min(rand), max(rand)],
                         smallest=accuracy(q, Xte, yte), retrained=accuracy(r, Xte, yte),
                         reads_smallest=reads(q), reads_retrained=reads(r), mask1=masks[0].astype(int).tolist(),
                         w1_retrained=r[0].tolist(), w1_smallest=q[0].tolist()))
        print(f"delete {f:.0%}: {rows[-1]['kept']} kept; random {rows[-1]['random']:.3f} ({min(rand):.3f}-{max(rand):.3f}), smallest {rows[-1]['smallest']:.3f}, + retrain {rows[-1]['retrained']:.3f}")
    # one random 50% choice, for the pictures (the one closest to the median)
    best = None
    for _ in range(N_RANDOM):
        m1, m2 = rng.random(ps[0].shape) >= 0.5, rng.random(ps[2].shape) >= 0.5
        a = accuracy([ps[0] * m1, ps[1], ps[2] * m2, ps[3]], Xte, yte)
        if best is None or abs(a - rows[5]["random"]) < abs(best[0] - rows[5]["random"]):
            best = (a, m1, m2)
    rnet = [ps[0] * best[1], ps[1], ps[2] * best[2], ps[3]]
    out = dict(n_weights=n_w, acc=accuracy(ps, Xte, yte), retrain_steps=RETRAIN, n_random=N_RANDOM, rows=rows, W1=ps[0].tolist(), show=show,
               random50=dict(acc=best[0], kept=int(best[1].sum() + best[2].sum()), mask1=best[1].astype(int).tolist(), reads=dgsim.forward(rnet, Xte[show])[0].argmax(1).tolist()))
    with open(os.path.join(HERE, "prune_results.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
