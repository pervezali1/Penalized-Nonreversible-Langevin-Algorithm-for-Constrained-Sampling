"""Move the digit one pixel. The handwriting network of reel 49 (64 pixels -> 32 neurons -> 10, retrained exactly as
digits/dgsim.py trains it: 96.7% of the 540 unseen digits right) and the same 540 unseen digits moved left or right
by one or two pixels (the columns that slide in are blank; the leftmost and rightmost columns hold almost no ink).
Against it, a small convolutional network trained on the same 1,257 digits, the same way (Adam, step 0.01, full
batch): 32 filters of 3 x 3 pixels slid over every position of the digit (zero padding), ReLU, the strongest
response of each filter anywhere in the image (global max pooling), then 32 -> 10 with softmax: 650 numbers.
Recorded: accuracies under every shift, for both networks; the filters; for one unseen digit, a filter's response
at every position before and after a shift. Vertical shifts are recorded too (the digits fill the box from top to
bottom, so they lose ink).    python cnnsim.py  -> cnn_results.json"""

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

K, CNN_STEPS, CNN_SEED = 32, 600, 3
SHIFTS = [-2, -1, 0, 1, 2]


def shift(Xf, dx, dy=0):
    """Move every 8 x 8 digit dx pixels right (negative: left) and dy down; what slides in is blank."""
    I = Xf.reshape(-1, 8, 8)
    O = np.zeros_like(I)
    ys, yd = slice(max(dy, 0), 8 + min(dy, 0)), slice(max(-dy, 0), 8 + min(-dy, 0))
    xs, xd = slice(max(dx, 0), 8 + min(dx, 0)), slice(max(-dx, 0), 8 + min(-dx, 0))
    O[:, ys, xs] = I[:, yd, xd]
    return O.reshape(len(Xf), 64)


def patches(Xf):
    """Every 3 x 3 window of every digit (zero padding): (n, 8, 8, 9)."""
    I = np.pad(Xf.reshape(-1, 8, 8), ((0, 0), (1, 1), (1, 1)))
    return np.stack([I[:, i:i + 8, j:j + 8] for i in range(3) for j in range(3)], -1)


def cnn_forward(ps, Xf):
    Wc, bc, W, b = ps
    z = patches(Xf) @ Wc + bc                     # (n, 8, 8, K): each filter's response at every position
    h = np.maximum(z, 0)
    f = h.reshape(len(h), 64, -1)
    am, g = f.argmax(1), f.max(1)                 # the strongest response of each filter, and where
    o = g @ W + b
    o = o - o.max(1, keepdims=True)
    e = np.exp(o)
    return e / e.sum(1, keepdims=True), (z, h, am, g)


def cnn_grads(ps, Xf, C, Y1):
    Wc, bc, W, b = ps
    p, (z, h, am, g) = cnn_forward(ps, Xf)
    n = len(Xf)
    d = (p - Y1) / n
    gW, gb, dg = g.T @ d, d.sum(0), d @ W.T
    dh = np.zeros((n, 64, Wc.shape[1]))
    np.put_along_axis(dh, am[:, None, :], dg[:, None, :], 1)      # the blame goes only to the strongest position
    dz = dh.reshape(h.shape) * (z > 0)
    return [np.einsum("nijc,nijk->ck", C, dz), dz.sum((0, 1, 2)), gW, gb]


def train_cnn(Xtr, ytr, seed=CNN_SEED):
    Y1 = np.eye(10)[ytr]
    C = patches(Xtr)
    rng = np.random.default_rng(seed)
    ps = [rng.normal(0, np.sqrt(2 / 9), (9, K)), np.zeros(K), rng.normal(0, np.sqrt(2 / K), (K, 10)), np.zeros(10)]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for s in range(CNN_STEPS):
        gs = cnn_grads(ps, Xtr, C, Y1)
        for i, g_ in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g_
            v[i] = 0.999 * v[i] + 0.001 * g_ * g_
            ps[i] -= 0.01 * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps


def main():
    mlp, Xte, yte = trained()
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    Xtr, ytr = X[idx[540:]], y[idx[540:]]
    assert np.allclose(X[idx[:540]], Xte)
    cnn = train_cnn(Xtr, ytr)
    acc = lambda P: float(np.mean(P.argmax(1) == yte))  # noqa: E731
    res = dict(mlp={}, cnn={}, mlp_v={}, cnn_v={})
    for dx in SHIFTS:
        Xs = shift(Xte, dx)
        res["mlp"][dx] = acc(probs(mlp, Xs))
        res["cnn"][dx] = acc(cnn_forward(cnn, Xs)[0])
    for dy in (-1, 1):
        Xs = shift(Xte, 0, dy)
        res["mlp_v"][dy] = acc(probs(mlp, Xs))
        res["cnn_v"][dy] = acc(cnn_forward(cnn, Xs)[0])
    I = Xte.reshape(-1, 8, 8)
    ink = dict(left_col=float(I[:, :, 0].sum() / I.sum()), right_col=float(I[:, :, 7].sum() / I.sum()), top_row=float(I[:, 0, :].sum() / I.sum()), bottom_row=float(I[:, 7, :].sum() / I.sum()))
    # the ten unseen digits of reel 49's strip, moved one pixel right: what each network reads
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    strip = []
    for d in range(10):
        x = Xte[show[d]][None]
        xr = shift(x, 1)
        strip.append(dict(true=d, mlp=int(probs(mlp, x)[0].argmax()), mlp_r=int(probs(mlp, xr)[0].argmax()), cnn=int(cnn_forward(cnn, x)[0][0].argmax()), cnn_r=int(cnn_forward(cnn, xr)[0][0].argmax()),
                          p_mlp_r=float(probs(mlp, xr)[0].max())))
    # the digit shown: of the 540 unseen digits, one the MLP reads right and misreads after one pixel either way, and the
    # CNN reads right at every shift; the one (not a 1) the CNN is surest of (its smallest probability over the five shifts)
    P_m = {dx: probs(mlp, shift(Xte, dx)) for dx in SHIFTS}
    P_c = {dx: cnn_forward(cnn, shift(Xte, dx))[0] for dx in SHIFTS}
    ok = (P_m[0].argmax(1) == yte) & (P_m[1].argmax(1) != yte) & (P_m[-1].argmax(1) != yte)
    for dx in SHIFTS:
        ok &= P_c[dx].argmax(1) == yte
    sure = np.min([P_c[dx][np.arange(540), yte] for dx in SHIFTS], 0)
    ok_shape = ok & (yte != 1)                          # a 1 is a plain stroke; show a digit with some shape to it
    i_show = int(np.flatnonzero(ok_shape)[np.argmax(sure[ok_shape])])
    print("surest candidate of each digit:", {int(d): round(float(sure[ok & (yte == d)].max()), 3) for d in range(10) if (ok & (yte == d)).any()})
    d_show = int(yte[i_show])
    x = Xte[i_show][None]
    per_shift = []
    for dx in SHIFTS:
        xs = shift(x, dx)
        pm, pc_, (zz, hh, aa, gg) = probs(mlp, xs)[0], *cnn_forward(cnn, xs)
        per_shift.append(dict(dx=dx, x=xs[0].tolist(), p_mlp=pm.tolist(), p_cnn=pc_[0].tolist(), maxes=gg[0].tolist(), argmax=aa[0].tolist(), fmaps=hh[0].transpose(2, 0, 1).tolist()))
    _, (z0, h0, am0, g0) = cnn_forward(cnn, x)
    _, (z1, h1, am1, g1) = cnn_forward(cnn, shift(x, 1))
    contrib = g0[0] * cnn[2][:, d_show]                 # how much each filter's strongest response votes for the digit
    k_show = int(np.argmax(contrib))
    n_ok = int(ok.sum())
    out = dict(n_test=540, k=K, cnn_steps=CNN_STEPS, n_mlp=int(sum(p.size for p in mlp)), n_cnn=int(sum(p.size for p in cnn)), acc=res, ink=ink, strip=strip,
               digit=d_show, index=int(i_show), per_shift=per_shift, filter=k_show, filters=cnn[0].T.tolist(),
               fmap0=h0[0, :, :, k_show].tolist(), fmap1=h1[0, :, :, k_show].tolist(), max0=float(g0[0, k_show]), max1=float(g1[0, k_show]),
               argmax0=int(am0[0, k_show]), argmax1=int(am1[0, k_show]), max_all_same=float(np.abs(g0 - g1).max()),
               W1=mlp[0].T.tolist(), n_candidates=n_ok, sure=float(sure[i_show]), Wout=cnn[2].tolist(), bc=cnn[1].tolist())
    # other random starts for the convolutional network (the honesty notes)
    others = []
    for sd in (1, 2, 4, 5, 6):
        c2 = train_cnn(Xtr, ytr, seed=sd)
        others.append(dict(seed=sd, clean=acc(cnn_forward(c2, Xte)[0]), right1=acc(cnn_forward(c2, shift(Xte, 1))[0]), left1=acc(cnn_forward(c2, shift(Xte, -1))[0])))
    out["cnn_other_starts"] = others
    with open(os.path.join(HERE, "cnn_results.json"), "w") as fh:
        json.dump(out, fh)
    print("numbers: mlp", out["n_mlp"], "cnn", out["n_cnn"])
    for key in ("mlp", "cnn"):
        print(key, {k: round(v, 4) for k, v in res[key].items()}, "vertical", {k: round(v, 4) for k, v in res[key + "_v"].items()})
    print("ink share in edge columns/rows:", {k: round(v, 4) for k, v in ink.items()})
    print("strip (true, mlp, mlp moved right, cnn, cnn moved right):", [(s["true"], s["mlp"], s["mlp_r"], s["cnn"], s["cnn_r"]) for s in strip])
    print("candidates", n_ok, "shown: test digit", i_show, "a", d_show, "cnn surest", round(float(sure[i_show]), 3))
    print("digit shown", d_show, "filter", k_show, "max before/after", round(out["max0"], 4), round(out["max1"], 4), "at", out["argmax0"], out["argmax1"], "all filters' max change", out["max_all_same"])
    print("per shift:", [(s["dx"], int(np.argmax(s["p_mlp"])), round(max(s["p_mlp"]), 3), int(np.argmax(s["p_cnn"])), round(max(s["p_cnn"]), 3)) for s in per_shift])
    print("other cnn starts:", [(o["seed"], round(o["clean"], 3), round(o["right1"], 3), round(o["left1"], 3)) for o in others])


if __name__ == "__main__":
    main()
