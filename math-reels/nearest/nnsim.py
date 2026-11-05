"""The laziest algorithm vs the handwriting network. The network of reel 49 (64 -> 32 -> 10, 300 Adam steps; retrained
exactly, see adversarial/advsim.py) against nearest neighbour: keep all 1,257 training pictures and, for a new digit,
copy the label of the training picture closest to it, d = sum over the 64 pixels of (difference in ink)^2, ink in
shades 0..16. Nothing is trained. Recorded: both scores on the same 540 unseen digits, the nearest neighbour's
mistakes, the distances from ten shown digits to all 1,257 training pictures, the same race on six other random
splits, bigger networks trained longer, and the work per digit read.    python nnsim.py  -> nn_results.json"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
import dgsim  # noqa: E402
from advsim import trained  # noqa: E402

OTHER = ((1, 1), (2, 2), (3, 4), (4, 5), (5, 6), (6, 7))   # (split, start) pairs, as in digits/dgsim.py


def split(sp):
    X, y = dgsim.load()
    idx = np.random.default_rng(sp).permutation(len(X))
    te, tr = idx[:540], idx[540:]
    return X[tr], y[tr], X[te], y[te]


def dist(Xte, Xtr):
    """Squared ink differences, in whole shades (0..16 per pixel): exact integers."""
    a, b = np.rint(Xte * 16).astype(np.int64), np.rint(Xtr * 16).astype(np.int64)
    return ((a[:, None, :] - b[None]) ** 2).sum(-1)


def net(Xtr, ytr, seed, H=dgsim.HIDDEN, steps=dgsim.STEPS):
    Y1 = np.eye(10)[ytr]
    rng = np.random.default_rng(seed)
    ps = [rng.normal(0, np.sqrt(2 / 64), (64, H)), np.zeros(H), rng.normal(0, np.sqrt(2 / H), (H, 10)), np.zeros(10)]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for s in range(steps):
        _, gs = dgsim.grads(ps, Xtr, Y1)
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps


def main():
    ps, Xte, yte = trained()
    Xtr, ytr, Xte2, yte2 = split(dgsim.SPLIT_SEED)
    assert np.array_equal(Xte, Xte2) and np.array_equal(yte, yte2)
    pnet = dgsim.forward(ps, Xte)[0].argmax(1)
    D = dist(Xte, Xtr)
    near = D.argmin(1)
    assert all(len(set(ytr[D[i] == D[i, near[i]]])) == 1 for i in range(len(D))), "a tie between labels"   # 4 exact ties, same label
    pnn = ytr[near]
    knn = {}
    for k in (1, 3, 5, 7):
        lab = ytr[np.argsort(D, 1, kind="stable")[:, :k]]
        knn[k] = float(np.mean(np.array([np.bincount(r, minlength=10).argmax() for r in lab]) == yte))
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    wrong = np.flatnonzero(pnn != yte)
    mistakes = [dict(i=int(i), true=int(yte[i]), read=int(pnn[i]), d=int(D[i, near[i]]), nearest=int(near[i]),
                     d_same=int(D[i][ytr == yte[i]].min()), net_read=int(pnet[i])) for i in wrong]
    others = []
    for sp, sd in OTHER:
        a, b, c, d_ = split(sp)
        q = net(a, b, sd)
        others.append(dict(split=sp, start=sd, net=float(np.mean(dgsim.forward(q, c)[0].argmax(1) == d_)),
                           nearest=float(np.mean(b[dist(c, a).argmin(1)] == d_))))
    # bigger networks, longer training, same split and start seed (for the honesty notes)
    bigger = [dict(hidden=hn, steps=st, acc=float(np.mean(dgsim.forward(net(Xtr, ytr, dgsim.SEED, hn, st), Xte)[0].argmax(1) == yte)))
              for hn, st in ((128, 300), (128, 1000), (256, 1000))]
    n_tr = len(Xtr)
    out = dict(n_train=n_tr, n_test=len(Xte), acc_net=float(np.mean(pnet == yte)), right_net=int(np.sum(pnet == yte)),
               acc_nn=float(np.mean(pnn == yte)), right_nn=int(np.sum(pnn == yte)), knn=knn, mistakes=mistakes, others=others,
               show=show, show_near=[int(near[i]) for i in show], show_d=[D[i].tolist() for i in show], show_nn_read=[int(pnn[i]) for i in show],
               show_net_read=[int(pnet[i]) for i in show], ytr=ytr.tolist(),
               mults_net=int(ps[0].size + ps[2].size), numbers_net=int(sum(p.size for p in ps)), mults_nn=int(n_tr * 64), numbers_nn=int(n_tr * 64),
               net_wrong_nn_right=int(np.sum((pnet != yte) & (pnn == yte))), nn_wrong_net_right=int(np.sum((pnn != yte) & (pnet == yte))), bigger=bigger)
    with open(os.path.join(HERE, "nn_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"network {out['right_net']}/540 = {out['acc_net']:.4f}; nearest {out['right_nn']}/540 = {out['acc_nn']:.4f}; k-NN {knn}")
    print("mistakes", [(m['true'], m['read'], m['d'], m['d_same'], m['net_read']) for m in mistakes])
    print("others", [(o['net'], o['nearest']) for o in others])
    print("bigger networks", [(b['hidden'], b['steps'], round(b['acc'], 4)) for b in bigger])
    print("show reads", out["show_nn_read"], out["show_net_read"], "d", [out["show_d"][k][out["show_near"][k]] for k in range(10)])
    print("work", out["mults_net"], out["numbers_net"], out["mults_nn"], f"{out['mults_nn'] / out['mults_net']:.2f}x",
          "net wrong/nn right", out["net_wrong_nn_right"], "nn wrong/net right", out["nn_wrong_net_right"])


if __name__ == "__main__":
    main()
