"""Two numbers per digit. The UCI handwritten digits of reel 49 (the same 1,257 to train on, the same 540 unseen),
squeezed to 2 numbers and rebuilt, two ways, neither of which ever sees a label:
  * PCA (reel 43): the 2 directions of most spread;
  * an autoencoder: 64 pixels -> 64 ReLU -> 2 numbers -> 64 ReLU -> 64 pixels (sigmoid), trained with
    backpropagation and Adam (step 0.003, full batch, 3,000 steps) only to rebuild its input (mean squared error).
The labels are used afterwards, only to colour the map and to score it: for each unseen digit, do its 5 nearest
training digits on the map (a vote) name the right digit? Also recorded: the map at 100 moments of training, the
rebuilt digits, and the decoder's pictures for a 9 x 9 grid of points across the map.
    python aesim.py  -> ae_results.json, ae_results.npz"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
import dgsim  # noqa: E402

H, STEPS, LR, SNAPS = 64, 3000, 0.003, 100
SEEDS = tuple(range(1, 10))          # nine random starts; the one with the median map score is shown


def init(rng, sizes):
    ps = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        ps += [rng.normal(0, np.sqrt(2 / m), (m, n)), np.zeros(n)]
    return ps


def forward(ps, Z):
    """64 -> H (ReLU) -> 2 (linear: the code) -> H (ReLU) -> 64 (sigmoid)."""
    acts, pre, h = [Z], [], Z
    for i in range(4):
        z = h @ ps[2 * i] + ps[2 * i + 1]
        pre.append(z)
        h = 1 / (1 + np.exp(-z)) if i == 3 else (z if i == 1 else np.maximum(z, 0))
        acts.append(h)
    return h, acts, pre


def encode(ps, Z):
    return np.maximum(Z @ ps[0] + ps[1], 0) @ ps[2] + ps[3]


def decode(ps, C):
    return 1 / (1 + np.exp(-(np.maximum(C @ ps[4] + ps[5], 0) @ ps[6] + ps[7])))


def grads(ps, Z):
    out, acts, pre = forward(ps, Z)
    d = 2 * (out - Z) / Z.size * out * (1 - out)
    gs = [None] * 8
    for i in range(3, -1, -1):
        gs[2 * i], gs[2 * i + 1] = acts[i].T @ d, d.sum(0)
        if i > 0:
            d = d @ ps[2 * i].T
            if i - 1 != 1:
                d = d * (pre[i - 1] > 0)
    return float(np.mean((out - Z) ** 2)), gs


def knn(Ctr, ytr, Cte, yte, k=5):
    d = ((Cte[:, None, :] - Ctr[None]) ** 2).sum(-1)
    votes = ytr[np.argsort(d, 1)[:, :k]]
    return float(np.mean(np.array([np.bincount(v, minlength=10).argmax() for v in votes]) == yte))


def train(seed, Xtr, Xte=None):
    ps = init(np.random.default_rng(seed), [64, H, 2, H, 64])
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    snaps, losses = [], []
    for s in range(STEPS + 1):
        if Xte is not None and s % (STEPS // SNAPS) == 0:
            snaps.append(encode(ps, Xte).astype(np.float32))
        loss, gs = grads(ps, Xtr)
        losses.append(loss)
        if s == STEPS:
            break
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps, snaps, losses


def main():
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    te, tr = idx[:540], idx[540:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mu = Xtr.mean(0)
    _, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    P2 = Vt[:2]
    p_tr, p_te = (Xtr - mu) @ P2.T, (Xte - mu) @ P2.T
    p_rec = p_te @ P2 + mu
    scores = {}
    for sd in SEEDS:
        q = train(sd, Xtr)[0]
        scores[sd] = knn(encode(q, Xtr), ytr, encode(q, Xte), yte)
    SEED = sorted(scores, key=lambda k: scores[k])[len(SEEDS) // 2]
    ps, snaps, losses = train(SEED, Xtr, Xte)
    a_tr, a_te = encode(ps, Xtr), encode(ps, Xte)
    a_rec = decode(ps, a_te)
    mse = lambda A, B: float(np.mean((A - B) ** 2))  # noqa: E731
    # the decoder's pictures across the map: a 9 x 9 grid over the middle 96% of the unseen digits' codes
    lo, hi = np.percentile(a_te, 2, 0), np.percentile(a_te, 98, 0)
    gx, gy = np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 9)
    grid = np.array([[decode(ps, np.array([[x_, y_]]))[0] for x_ in gx] for y_ in gy[::-1]])
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    out = dict(n_train=len(tr), n_test=len(te), hidden=H, steps=STEPS, lr=LR, seed=SEED,
               pca=dict(var_kept=float((S[:2] ** 2).sum() / (S ** 2).sum()), mse=mse(p_rec, Xte), knn=knn(p_tr, ytr, p_te, yte)),
               ae=dict(mse=mse(a_rec, Xte), knn=knn(a_tr, ytr, a_te, yte), loss=losses[::10]),
               knn_by_seed={str(k): v for k, v in scores.items()}, show=show, grid_x=gx.tolist(), grid_y=gy.tolist())
    np.savez_compressed(os.path.join(HERE, "ae_results.npz"), yte=yte, Xte=Xte.astype(np.float32), pca_codes=p_te.astype(np.float32), pca_rec=p_rec.astype(np.float32),
                        ae_codes=a_te.astype(np.float32), ae_rec=a_rec.astype(np.float32), snaps=np.array(snaps), grid=grid.astype(np.float32))
    with open(os.path.join(HERE, "ae_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"PCA: keeps {out['pca']['var_kept']:.3f} of the spread, rebuild mse {out['pca']['mse']:.4f}, map 5-NN {out['pca']['knn']:.3f}")
    print(f"autoencoder: rebuild mse {out['ae']['mse']:.4f} ({1 - out['ae']['mse'] / out['pca']['mse']:.1%} less), map 5-NN {out['ae']['knn']:.3f} (seed {SEED}, the median of {sorted(round(v, 3) for v in scores.values())})")


if __name__ == "__main__":
    main()
