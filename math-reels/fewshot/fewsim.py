"""How many examples does a network need? The handwriting network of reel 49 (64 -> 32 -> 10, the same start and the
same 300 Adam steps) trained on fewer of the 1,257 training digits: k of each digit, picked at random, for k = 1, 2, 3,
5, 10, 20, 50, and on all 1,257. Each size is trained on 10 different random picks (and starts); the median is
reported and its run shown. Always tested on the same 540 unseen digits.    python fewsim.py  -> few_results.json"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
import dgsim  # noqa: E402

KS = [1, 2, 3, 5, 10, 20, 50]
REPS = 10


def train(Xs, ys, seed):
    Y1 = np.eye(10)[ys]
    r = np.random.default_rng(seed)
    ps = [r.normal(0, np.sqrt(2 / 64), (64, dgsim.HIDDEN)), np.zeros(dgsim.HIDDEN), r.normal(0, np.sqrt(2 / dgsim.HIDDEN), (dgsim.HIDDEN, 10)), np.zeros(10)]
    m = [np.zeros_like(a) for a in ps]
    v = [np.zeros_like(a) for a in ps]
    for s in range(dgsim.STEPS):
        _, gs = dgsim.grads(ps, Xs, Y1)
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps


def main():
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    te, tr = idx[:540], idx[540:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    acc = lambda ps, A, b: float(np.mean(dgsim.forward(ps, A)[0].argmax(1) == b))  # noqa: E731
    sizes = []
    for k in KS:
        runs = []
        for rep in range(REPS):
            rng = np.random.default_rng(100 + rep)
            pick = np.concatenate([rng.choice(np.flatnonzero(ytr == d), k, replace=False) for d in range(10)])
            ps = train(Xtr[pick], ytr[pick], seed=rep)
            runs.append(dict(test=acc(ps, Xte, yte), train=acc(ps, Xtr[pick], ytr[pick]), pick=pick.tolist(), reads=dgsim.forward(ps, Xte[show])[0].argmax(1).tolist()))
        order = sorted(range(REPS), key=lambda i: runs[i]["test"])
        med = runs[order[REPS // 2]]
        tests = [r["test"] for r in runs]
        sizes.append(dict(k=k, n=10 * k, median=float(np.median(tests)), lo=min(tests), hi=max(tests), train=float(np.median([r["train"] for r in runs])),
                          shown=dict(test=med["test"], pick=med["pick"], reads=med["reads"])))
        print(f"{k} of each ({10 * k}): median {sizes[-1]['median']:.3f} ({min(tests):.3f}-{max(tests):.3f}); train {sizes[-1]['train']:.3f}; shown run {med['test']:.3f}")
    ps = train(Xtr, ytr, seed=dgsim.SEED)
    full = dict(k=None, n=len(tr), median=acc(ps, Xte, yte), lo=acc(ps, Xte, yte), hi=acc(ps, Xte, yte), train=acc(ps, Xtr, ytr),
                shown=dict(test=acc(ps, Xte, yte), pick=list(range(len(tr))), reads=dgsim.forward(ps, Xte[show])[0].argmax(1).tolist()))
    sizes.append(full)
    print(f"all {len(tr)}: {full['median']:.4f}")
    with open(os.path.join(HERE, "few_results.json"), "w") as fh:
        json.dump(dict(n_train=len(tr), n_test=len(te), reps=REPS, sizes=sizes, show=show, train_index=tr.tolist()), fh)


if __name__ == "__main__":
    main()
