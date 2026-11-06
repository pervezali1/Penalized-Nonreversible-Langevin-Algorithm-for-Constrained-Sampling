"""Store a network's weights in fewer bits (quantization). The handwriting network of reel 49 (retrained exactly, see
adversarial/advsim.py: 2,368 weights in 32 bits each, 96.7% of the 540 unseen digits right) with every weight rounded:
  * b bits (b >= 2): to the nearest of the 2^(b-1) - 1 positive and negative multiples of s = max|w| / (2^(b-1) - 1),
    and 0, one s per layer (8 bits: 255 values, 4 bits: 15, 3 bits: 7);
  * 1 bit: only the sign of each weight is kept, times one size per layer, s = mean |w| (two values per layer);
  * the same, then 100 more Adam steps on the 1,257 training digits with the rounding inside the network (the
    gradient at the rounded weights moves hidden unrounded copies; the network used is only the rounded weights).
The 42 biases stay unrounded. Recorded: the accuracy for each, the 32 neurons' rounded weight pictures, the first
layer's weight values, what it reads for ten unseen digits, and the same on six other random splits and starts.
    python quantsim.py  -> quant_results.json"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digits"))
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
sys.path.insert(0, os.path.join(ROOT, "nearest"))
import dgsim  # noqa: E402
from advsim import trained  # noqa: E402
from nnsim import OTHER, net, split  # noqa: E402

BITS = [8, 6, 4, 3, 1]
RETRAIN = 100


def quant(w, bits):
    if bits == 1:
        return np.sign(w) * np.abs(w).mean()
    L = 2 ** (bits - 1) - 1
    s = np.abs(w).max() / L
    return np.round(w / s) * s


def qnet(ps, bits):
    return [quant(ps[0], bits), ps[1], quant(ps[2], bits), ps[3]]


def accuracy(ps, X, y):
    return float(np.mean(dgsim.forward(ps, X)[0].argmax(1) == y))


def retrain(ps, bits, Xtr, ytr, steps=RETRAIN):
    """Quantization-aware training: forward and backward through the rounded weights, steps on the unrounded copies."""
    lat = [p.copy() for p in ps]
    Y1 = np.eye(10)[ytr]
    m = [np.zeros_like(p) for p in lat]
    v = [np.zeros_like(p) for p in lat]
    for s in range(steps):
        _, gs = dgsim.grads(qnet(lat, bits), Xtr, Y1)
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            lat[i] -= dgsim.LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    return qnet(lat, bits)


def main():
    ps, Xte, yte = trained()
    Xtr, ytr, _, _ = split(dgsim.SPLIT_SEED)
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    reads = lambda q: dgsim.forward(q, Xte[show])[0].argmax(1).tolist()  # noqa: E731
    rows = [dict(bits=32, acc=accuracy(ps, Xte, yte), values1=None, W1=ps[0].tolist(), reads=reads(ps))]
    for b in BITS:
        q = qnet(ps, b)
        rows.append(dict(bits=b, acc=accuracy(q, Xte, yte), values1=int(len(np.unique(q[0]))), values2=int(len(np.unique(q[2]))),
                         W1=q[0].tolist(), reads=reads(q)))
    r1 = retrain(ps, 1, Xtr, ytr)
    r4 = retrain(ps, 4, Xtr, ytr)
    flips = int(np.sum(np.sign(r1[0]) != np.sign(ps[0])) + np.sum(np.sign(r1[2]) != np.sign(ps[2])))
    others = []
    for sp, sd in OTHER:
        a, b_, c, d = split(sp)
        q = net(a, b_, sd)
        others.append(dict(split=sp, start=sd, full=accuracy(q, c, d), one_bit=accuracy(qnet(q, 1), c, d), one_bit_retrained=accuracy(retrain(q, 1, a, b_), c, d)))
    n_w = int(ps[0].size + ps[2].size)
    out = dict(n_weights=n_w, n_biases=int(ps[1].size + ps[3].size), retrain_steps=RETRAIN, rows=rows,
               one_bit_retrained=dict(acc=accuracy(r1, Xte, yte), W1=r1[0].tolist(), reads=reads(r1), sign_flips=flips,
                                      scales=[float(np.abs(r1[0]).max()), float(np.abs(r1[2]).max())], values1=int(len(np.unique(r1[0])))),
               four_bit_retrained=accuracy(r4, Xte, yte), others=others, show=show,
               bytes={b: n_w * b / 8 for b in [32] + BITS})
    with open(os.path.join(HERE, "quant_results.json"), "w") as fh:
        json.dump(out, fh)
    for r in rows:
        print(f"{r['bits']:2d} bits: {r['acc']:.4f}", r.get("values1"), r["reads"])
    o = out["one_bit_retrained"]
    print(f"1 bit + {RETRAIN} steps: {o['acc']:.4f}, {o['sign_flips']} signs flipped, reads {o['reads']}; 4 bit + retrain {out['four_bit_retrained']:.4f}")
    print("others", [(round(x['full'], 4), round(x['one_bit'], 4), round(x['one_bit_retrained'], 4)) for x in others])


if __name__ == "__main__":
    main()
