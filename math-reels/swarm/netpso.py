"""The honest twist for the swarm reel: the same talking swarm (particle swarm optimisation, 10,000 particles, the
same settings as the 2-D reel) on a real network: the 8 + 8 network of reels 31, 33, 41 and 47 (2 -> 8 -> 8 -> 1, ReLU,
105 numbers) on the same 400-point spiral, minimising the same loss gradient descent minimises (cross-entropy).
Each particle starts as a network drawn the way the random-search reel drew them. 300 swarm steps = 3,000,000 runs
of the network, against gradient descent's 3,000 (1,500 steps x forward + backward) for 95.5%.
    python netpso.py -> netpso_results.json"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from nnsim import DATA_SEED  # noqa: E402
from train import spiral  # noqa: E402

N, STEPS, W_IN, C = 10_000, 300, 0.72, 1.49
SHAPES = [(2, 8), (8,), (8, 8), (8,), (8, 1), (1,)]
SIZES = [int(np.prod(s)) for s in SHAPES]
D = sum(SIZES)
CHUNK = 2_500


def unpack(Q):
    out, k = [], 0
    for s, n in zip(SHAPES, SIZES):
        out.append(Q[:, k:k + n].reshape((len(Q),) + s))
        k += n
    return out


def evaluate(Q, X, y):
    """Mean cross-entropy and accuracy of every network in Q (rows) on (X, y)."""
    loss = np.empty(len(Q))
    acc = np.empty(len(Q))
    for c0 in range(0, len(Q), CHUNK):
        W1, b1, W2, b2, W3, b3 = unpack(Q[c0:c0 + CHUNK])
        h = np.maximum(np.einsum("pi,nij->npj", X, W1) + b1[:, None, :], 0)
        h = np.maximum(np.einsum("npi,nij->npj", h, W2) + b2[:, None, :], 0)
        z = np.einsum("npi,nij->npj", h, W3)[..., 0] + b3
        # cross-entropy with logits, stable
        loss[c0:c0 + CHUNK] = np.mean(np.maximum(z, 0) - z * y[None, :] + np.log1p(np.exp(-np.abs(z))), 1)
        acc[c0:c0 + CHUNK] = ((z > 0) == (y[None, :] > 0.5)).mean(1)
    return loss, acc


def main():
    rng_d = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng_d)
    rng = np.random.default_rng(65)
    parts = [rng.normal(0, np.sqrt(2 / 2), (N, 16)), rng.normal(0, 1, (N, 8)), rng.normal(0, np.sqrt(2 / 8), (N, 64)),
             rng.normal(0, 1, (N, 8)), rng.normal(0, np.sqrt(2 / 8), (N, 8)), rng.normal(0, 1, (N, 1))]
    Q = np.concatenate(parts, 1)
    V = np.zeros_like(Q)
    pf, pa = evaluate(Q, X, y)
    pb = Q.copy()
    curve = []
    t0 = time.perf_counter()
    for s in range(1, STEPS + 1):
        g = pb[np.argmin(pf)]
        V = W_IN * V + C * rng.random((N, 1)) * (pb - Q) + C * rng.random((N, 1)) * (g - Q)
        Q = Q + V
        f, a = evaluate(Q, X, y)
        m = f < pf
        pb[m], pf[m], pa[m] = Q[m], f[m], a[m]
        if s % 10 == 0 or s == 1:
            i = int(np.argmin(pf))
            curve.append(dict(step=s, best_loss=float(pf[i]), best_acc=float(pa[i]), median_acc_now=float(np.median(a)),
                              share_90=float(np.mean(a >= 0.9))))
            print(curve[-1], f"{time.perf_counter() - t0:.0f}s", flush=True)
    i = int(np.argmin(pf))
    rs = json.load(open(os.path.join(ROOT, "randsearch", "rs_results.json")))
    out = dict(particles=N, steps=STEPS, dims=D, network_runs=N * (STEPS + 1), best_acc=float(pa[i]), best_loss=float(pf[i]),
               best_acc_any=float(pa.max()), curve=curve, gd_acc_final=float(rs["gd_acc"][-1]), gd_runs=rs["passes_gd"],
               random_best_of_1M=rs["best_random"], seconds=time.perf_counter() - t0)
    with open(os.path.join(HERE, "netpso_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print({k: v for k, v in out.items() if k != "curve"})


if __name__ == "__main__":
    main()
