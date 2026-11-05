"""The network that is sure about static. The handwriting network of reel 49 (retrained exactly, see
adversarial/advsim.py: 96.7% of the 540 unseen digits) shown things that are not digits:
  * 1,000 images of random static (every pixel a random whole shade 0-16);
  * a letter A drawn in 8 x 8 pixels, a blank image and a fully inked one;
then the usual fix: an 11th answer, "not a digit", trained on 600 more static images alongside the same 1,257
digits (same start, same 300 Adam steps). Tested on 1,000 new static images, on the 540 unseen digits, on the
letter A, and on a different kind of noise (sparse dots: each pixel inked with chance 0.3) it never saw.
    python staticsim.py  -> static_results.json"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
import dgsim  # noqa: E402
from advsim import probs, trained  # noqa: E402

N_STATIC, N_TRAIN_STATIC = 1000, 600
LETTER_A = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0]], float).ravel()


def static(rng, n):
    return rng.integers(0, 17, (n, 64)) / 16.0


def dots(rng, n):
    return (rng.random((n, 64)) < 0.3) * rng.integers(4, 17, (n, 64)) / 16.0


def summary(P, none=None):
    mx = P.max(1)
    out = dict(mean_max=float(mx.mean()), median_max=float(np.median(mx)), over90=float((mx > 0.9).mean()), over99=float((mx > 0.99).mean()),
               classes=np.bincount(P.argmax(1), minlength=P.shape[1]).tolist())
    if none is not None:
        out["called_not_digit"] = float((P.argmax(1) == none).mean())
    return out


def main():
    ps, Xte, yte = trained()
    rng = np.random.default_rng(53)
    S = static(rng, N_STATIC)
    D = dots(rng, N_STATIC)
    P_S, P_D, P_T = probs(ps, S), probs(ps, D), probs(ps, Xte)
    # the static images shown: the first 10, and the one shown big (the first read as a 4 with 0.99 < p < 0.999)
    big = int(np.flatnonzero((P_S.argmax(1) == 4) & (P_S.max(1) > 0.99) & (P_S.max(1) < 0.999))[0])
    pa = probs(ps, LETTER_A)[0]
    # the fix: an 11th class, "not a digit", trained on 600 static images too
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    Xtr, ytr = X[idx[540:]], y[idx[540:]]
    Str = static(rng, N_TRAIN_STATIC)
    X2, y2 = np.vstack([Xtr, Str]), np.concatenate([ytr, np.full(N_TRAIN_STATIC, 10)])
    Y1 = np.eye(11)[y2]
    r = np.random.default_rng(dgsim.SEED)
    q = [r.normal(0, np.sqrt(2 / 64), (64, dgsim.HIDDEN)), np.zeros(dgsim.HIDDEN), r.normal(0, np.sqrt(2 / dgsim.HIDDEN), (dgsim.HIDDEN, 11)), np.zeros(11)]
    m = [np.zeros_like(a) for a in q]
    v = [np.zeros_like(a) for a in q]
    for s in range(dgsim.STEPS):
        _, gs = dgsim.grads(q, X2, Y1)
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            q[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    f = lambda Z: dgsim.forward(q, np.atleast_2d(Z))[0]  # noqa: E731
    S_new = static(rng, N_STATIC)
    Q_T, Q_S, Q_D, Q_A = f(Xte), f(S_new), f(D), f(LETTER_A)[0]
    # a dots image the 11-answer network misses (it reads a digit), shown as the example of the 27%
    Q_Dall = Q_D
    miss = int(np.flatnonzero(Q_Dall.argmax(1) != 10)[0])
    out = dict(n_static=N_STATIC, hist_static=np.histogram(P_S.max(1), bins=10, range=(0, 1))[0].tolist(),
               hist_real=np.histogram(P_T.max(1), bins=10, range=(0, 1))[0].tolist(),
               dot_example=dict(index=miss, x=D[miss].tolist(), p10=P_D[miss].tolist(), p11=Q_D[miss].tolist()), acc=float(np.mean(P_T.argmax(1) == yte)), real=summary(P_T), static=summary(P_S), dots=summary(P_D),
               blank=probs(ps, np.zeros(64))[0].tolist(), full=probs(ps, np.ones(64))[0].tolist(), letter_a=pa.tolist(), letter_a_x=LETTER_A.tolist(),
               shown=S[:10].tolist(), shown_p=P_S[:10].tolist(), big=S[big].tolist(), big_p=P_S[big].tolist(), big_index=big,
               fix=dict(n_train_static=N_TRAIN_STATIC, acc=float(np.mean(Q_T.argmax(1) == yte)), static=summary(Q_S, 10), dots=summary(Q_D, 10),
                        letter_a=Q_A.tolist(), shown_p=f(S[:10]).tolist(), big_p=f(S[big])[0].tolist()))
    with open(os.path.join(HERE, "static_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"real digits: {out['acc']:.4f} right, median max p {out['real']['median_max']:.3f}")
    print("static:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["static"].items()})
    print("dots:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["dots"].items()})
    print("big static image #", big, "reads", int(np.argmax(out["big_p"])), round(max(out["big_p"]), 4))
    print("letter A reads", int(pa.argmax()), round(float(pa.max()), 5), "blank", int(np.argmax(out["blank"])), round(max(out["blank"]), 3), "full", int(np.argmax(out["full"])), round(max(out["full"]), 4))
    print("fix:", out["fix"]["acc"], "new static ->", out["fix"]["static"]["called_not_digit"], "dots ->", round(out["fix"]["dots"]["called_not_digit"], 3),
          "letter A ->", int(np.argmax(out["fix"]["letter_a"])), round(max(out["fix"]["letter_a"]), 3), "big ->", int(np.argmax(out["fix"]["big_p"])), round(max(out["fix"]["big_p"]), 4))


if __name__ == "__main__":
    main()
