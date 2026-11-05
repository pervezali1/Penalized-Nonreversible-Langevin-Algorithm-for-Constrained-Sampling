"""How to fool a neural network. The handwriting network of reel 49, retrained exactly as digits/dgsim.py trains it
(same split, same start, 300 Adam steps: 96.7% of the 540 unseen digits right). Backpropagation carried to the
pixels gives, for every pixel, which way more ink pushes the network's answer; pushing every pixel a few shades of
ink (out of 16) the wrong way is the "fast gradient sign" attack (Goodfellow et al., 2014). Recorded:
  * the unseen 3 of reel 49: its probabilities, the gradient on its pixels, and the 1-shade nudge that fools it;
  * the same 3 nudged toward each of the ten digits (repeated small gradient steps toward the chosen answer, never
    more than k shades from the original in any pixel; the smallest k that works);
  * all 540 unseen digits nudged 0..4 shades along the gradient, against random nudges of the same size.
Every altered digit is still a valid scan: whole shades from 0 to 16.    python advsim.py  -> adv_results.json"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "digits"))
import dgsim  # noqa: E402

SHADE = 1 / 16


def trained():
    X, y = dgsim.load()
    idx = np.random.default_rng(dgsim.SPLIT_SEED).permutation(len(X))
    te, tr = idx[:540], idx[540:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    Y1 = np.eye(10)[ytr]
    rng = np.random.default_rng(dgsim.SEED)
    ps = [rng.normal(0, np.sqrt(2 / 64), (64, dgsim.HIDDEN)), np.zeros(dgsim.HIDDEN), rng.normal(0, np.sqrt(2 / dgsim.HIDDEN), (dgsim.HIDDEN, 10)), np.zeros(10)]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for s in range(dgsim.STEPS):
        _, gs = dgsim.grads(ps, Xtr, Y1)
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return ps, Xte, yte


def probs(ps, Xb):
    return dgsim.forward(ps, np.atleast_2d(Xb))[0]


def grad_loss(ps, Xb, lab):
    """d(-log p_lab)/d pixel for each digit in Xb: backpropagation through both layers down to the 64 inputs."""
    W1, b1, W2, b2 = ps
    p, h, z1 = dgsim.forward(ps, Xb)
    d2 = p - np.eye(10)[lab]
    d1 = (d2 @ W2.T) * (z1 > 0)
    return d1 @ W1.T


def nudge(x, g, k):
    """Every pixel k shades along the sign of g, kept inside the 0..16 scale (whole shades stay whole)."""
    return np.clip(np.round((x + k * SHADE * np.sign(g)) * 16) / 16, 0, 1)


def targeted(ps, x, target, k, iters=40):
    """40 one-shade steps that make `target` more likely, never more than k shades from x in any pixel."""
    lo, hi = np.maximum(x - k * SHADE, 0), np.minimum(x + k * SHADE, 1)
    xa = x.copy()
    for _ in range(iters):
        g = grad_loss(ps, xa[None], np.array([target]))[0]
        xa = np.clip(np.round((xa - SHADE * np.sign(g)) * 16) / 16, lo, hi)
    return xa if probs(ps, xa)[0].argmax() == target else None


def main():
    ps, Xte, yte = trained()
    p_all = probs(ps, Xte)
    pred = p_all.argmax(1)
    ref = json.load(open(os.path.join(os.path.dirname(HERE), "digits", "dg_results.json")))
    assert int((pred == yte).sum()) == ref["n_right"], "not the network of reel 49"
    show = ref["show"]
    i3 = show[3]
    x3 = Xte[i3]
    p3 = probs(ps, x3)[0]
    g3 = grad_loss(ps, x3[None], np.array([3]))[0]
    one = {}
    for k in (1, 2, 3):
        xa = nudge(x3, g3, k)
        pa = probs(ps, xa)[0]
        one[k] = dict(x=xa.tolist(), p=pa.tolist(), read=int(pa.argmax()), changed=int(np.sum(xa != x3)), max_shades=int(round(np.abs(xa - x3).max() * 16)))
    k1 = min(k for k in one if one[k]["read"] != 3)
    tg = {}
    for t in range(10):
        if t == 3:
            tg[t] = dict(k=0, x=x3.tolist(), p=p3.tolist())
            continue
        for k in range(1, 9):
            xa = targeted(ps, x3, t, k)
            if xa is not None:
                pa = probs(ps, xa)[0]
                tg[t] = dict(k=k, x=xa.tolist(), p=pa.tolist(), changed=int(np.sum(xa != x3)))
                break
    # all 540 unseen digits: along the gradient (each digit's own right answer) vs random signs, same size
    g_all = grad_loss(ps, Xte, yte)
    rng = np.random.default_rng(50)
    curve_g, curve_r = [], []
    for k in range(0, 5):
        curve_g.append(float(np.mean(probs(ps, nudge(Xte, g_all, k)).argmax(1) == yte)))
        rs = [float(np.mean(probs(ps, nudge(Xte, rng.choice([-1.0, 1.0], Xte.shape), k)).argmax(1) == yte)) for _ in range(20)]
        curve_r.append(float(np.median(rs)))
    # the smallest single-step nudge that fools each correctly read digit
    right = np.flatnonzero(pred == yte)
    kmin = []
    for i in right:
        kk = next((k for k in range(1, 9) if probs(ps, nudge(Xte[i], g_all[i], k))[0].argmax() != yte[i]), 99)
        kmin.append(kk)
    kmin = np.array(kmin)
    # the ten unseen digits of reel 49's strip, each nudged 3 shades along its own gradient
    strip = []
    for d in range(10):
        i = show[d]
        xa = nudge(Xte[i], g_all[i], 3)
        pa = probs(ps, xa)[0]
        strip.append(dict(true=d, x=xa.tolist(), read=int(pa.argmax()), p=float(pa.max()), clean_read=int(pred[i])))
    out = dict(n_test=540, n_right=int(len(right)), acc=float(np.mean(pred == yte)), shown_index=int(i3), x3=x3.tolist(), p3=p3.tolist(), grad3=g3.tolist(),
               one=one, k_first=k1, targeted={str(t): v for t, v in tg.items()}, k_targeted_max=max(v["k"] for v in tg.values()),
               curve_gradient=curve_g, curve_random=curve_r, kmin_hist={int(k): int((kmin == k).sum()) for k in sorted(set(kmin.tolist()))},
               right_after={int(k): int(round(c * 540)) for k, c in enumerate(curve_g)}, right_after_random={int(k): int(round(c * 540)) for k, c in enumerate(curve_r)}, strip3=strip,
               p3_by_k=[probs(ps, nudge(x3, g3, k))[0].tolist() for k in range(5)])
    with open(os.path.join(HERE, "adv_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"clean {out['acc']:.4f}; the 3: p(3) = {p3[3]:.4f}")
    for k, o in one.items():
        print(f"  {k} shade(s): read as {o['read']} p = {max(o['p']):.3f} (p3 {o['p'][3]:.3f}), {o['changed']} pixels changed")
    print("targeted:", {t: (v["k"], round(max(v["p"]), 3), v.get("changed")) for t, v in tg.items()})
    print("gradient:", [round(c, 4) for c in curve_g], "random:", [round(c, 4) for c in curve_r])
    print("strip, 3 shades:", [(o["true"], o["read"], round(o["p"], 2)) for o in strip])
    print("the 3 by shades:", [(int(np.argmax(p)), round(max(p), 3), round(p[3], 3)) for p in out["p3_by_k"]])
    print("smallest single nudge that fools each of the", len(right), "digits:", out["kmin_hist"])


if __name__ == "__main__":
    main()
