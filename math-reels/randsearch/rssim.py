"""A million random networks against gradient descent. The 8 + 8 network of reels 31, 33 and 41 (2 -> 8 -> 8 -> 1,
ReLU, 105 numbers) on the same 400-point spiral. Random search: 1,000,000 networks with every number drawn at random
(weights as at the start of training, normal with variance 2 / fan-in; biases normal with sd 1, so the fold lines
don't all pass through the centre), each scored by how many of the 400 points it gets right; the best so far is
kept. Gradient descent: one random start (the seed of reel 33), Adam, step 0.01, 1,500 steps, exactly as reel 31
trained it. Also the work each needs (runs of the network) and the time each took on this computer.
    python rssim.py  -> rs_results.json, rs_fields.npz;  RS_IMPROVEMENTS=1 python rssim.py  -> rs_steps.npz (the record-setting networks' maps)"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from nnsim import DATA_SEED, GRID, LIM, LR, forward, grads, init  # noqa: E402
from train import spiral  # noqa: E402

N_RANDOM = 1_000_000
CHUNK = 10_000
GD_SEED, GD_STEPS = 16121, 1500
SPREADS = {"start": 1.0}           # the main search; other spreads are tried below for the honesty notes


def random_search(X, y, n, seed, w_scale=1.0, b_sd=1.0, keep_best=True):
    rng = np.random.default_rng(seed)
    best, best_ps, curve = -1.0, None, []
    t0 = time.perf_counter()
    for c0 in range(0, n, CHUNK):
        m = min(CHUNK, n - c0)
        W1 = rng.normal(0, w_scale * np.sqrt(2 / 2), (m, 2, 8))
        b1 = rng.normal(0, b_sd, (m, 8))
        W2 = rng.normal(0, w_scale * np.sqrt(2 / 8), (m, 8, 8))
        b2 = rng.normal(0, b_sd, (m, 8))
        W3 = rng.normal(0, w_scale * np.sqrt(2 / 8), (m, 8, 1))
        b3 = rng.normal(0, b_sd, (m, 1))
        h = np.maximum(np.einsum("pi,nij->npj", X, W1) + b1[:, None, :], 0)
        h = np.maximum(np.einsum("npi,nij->npj", h, W2) + b2[:, None, :], 0)
        z = np.einsum("npi,nij->npj", h, W3)[..., 0] + b3
        acc = ((z > 0) == (y[None, :] > 0.5)).mean(1)
        # the best so far after every network in this chunk
        run = np.maximum.accumulate(acc)
        run = np.maximum(run, best)
        for k in np.flatnonzero(np.diff(np.concatenate([[best], run])) > 0):
            curve.append((c0 + int(k) + 1, float(run[k])))
        if run[-1] > best:
            i = int(np.argmax(acc))
            best = float(acc[i])
            if keep_best:
                best_ps = [W1[i], b1[i], W2[i], b2[i], W3[i], b3[i]]
    return best, best_ps, curve, time.perf_counter() - t0


def field(ps, GX):
    return (1 / (1 + np.exp(-forward(ps, GX)[0]))).reshape(GRID, GRID).astype(np.float32)


def main():
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    g = np.linspace(-LIM, LIM, GRID)
    GXX, GYY = np.meshgrid(g, g)
    GX = np.stack([GXX.ravel(), GYY.ravel()], 1)
    best, best_ps, curve, t_rs = random_search(X, y, N_RANDOM, seed=2026)
    # the best-so-far networks along the way, for the map (re-run cheaply: record params at each improvement)
    print(f"random search: best {best:.4f} after {N_RANDOM:,} networks, {t_rs:.1f} s; improvements {len(curve)}")
    at = {n: max([a for k, a in curve if k <= n] + [0.0]) for n in (1, 10, 100, 1000, 10_000, 100_000, 1_000_000)}
    print("best after", {k: round(v, 4) for k, v in at.items()})
    # gradient descent, step by step
    rng = np.random.default_rng(GD_SEED)
    ps = init(rng, [2, 8, 8, 1])
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    accs, fields_gd = [], []
    t0 = time.perf_counter()
    for s in range(0, GD_STEPS + 1):
        loss, gs, p = grads(ps, X, y)
        accs.append(float(np.mean((p > 0.5) == (y > 0.5))))
        if s % 15 == 0:
            fields_gd.append(field(ps, GX))
        if s == GD_STEPS:
            break
        for i, gg in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    t_gd = time.perf_counter() - t0
    print(f"gradient descent: {accs[0]:.4f} -> {accs[-1]:.4f} in {GD_STEPS} steps, {t_gd:.2f} s; 90% at step {next(i for i, a in enumerate(accs) if a >= 0.9)}")
    beats = next(i for i, a in enumerate(accs) if a > best)
    print("gradient descent passes the best random network at step", beats)
    # other spreads, for the honesty notes (100,000 each)
    others = {}
    for ws, bs in ((0.5, 1.0), (2.0, 1.0), (1.0, 0.3), (1.0, 3.0), (3.0, 3.0)):
        b_, _, _, _ = random_search(X, y, 100_000, seed=7, w_scale=ws, b_sd=bs, keep_best=False)
        others[f"w{ws}_b{bs}"] = b_
        print(f"  spread weights x{ws}, biases sd {bs}: best of 100,000 = {b_:.4f}")
    rs_field = field([np.asarray(q) for q in best_ps], GX)
    np.savez_compressed(os.path.join(HERE, "rs_fields.npz"), X=X, y=y, grid=g, rs_field=rs_field, gd_fields=np.array(fields_gd))
    out = dict(n_random=N_RANDOM, best_random=best, curve=curve, best_after={str(k): v for k, v in at.items()}, t_random_s=t_rs,
               gd_steps=GD_STEPS, gd_acc=accs, t_gd_s=t_gd, gd_beats_random_at=beats, passes_random=N_RANDOM, passes_gd=2 * GD_STEPS,
               other_spreads_best_of_100k=others, n_par=105)
    with open(os.path.join(HERE, "rs_results.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__" and not os.environ.get("RS_IMPROVEMENTS"):
    main()


def improvements(n_upto=60_000):
    """Replay the search (same seed, same draws) up to just past its last improvement and return every record-setting
    network's map: the networks shown in the reel are exactly the ones the search found."""
    rng0 = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng0)
    g = np.linspace(-LIM, LIM, GRID)
    GXX, GYY = np.meshgrid(g, g)
    GX = np.stack([GXX.ravel(), GYY.ravel()], 1)
    rng = np.random.default_rng(2026)
    best, recs = -1.0, []
    for c0 in range(0, n_upto, CHUNK):
        m = CHUNK
        W1 = rng.normal(0, np.sqrt(2 / 2), (m, 2, 8))
        b1 = rng.normal(0, 1.0, (m, 8))
        W2 = rng.normal(0, np.sqrt(2 / 8), (m, 8, 8))
        b2 = rng.normal(0, 1.0, (m, 8))
        W3 = rng.normal(0, np.sqrt(2 / 8), (m, 8, 1))
        b3 = rng.normal(0, 1.0, (m, 1))
        h = np.maximum(np.einsum("pi,nij->npj", X, W1) + b1[:, None, :], 0)
        h = np.maximum(np.einsum("npi,nij->npj", h, W2) + b2[:, None, :], 0)
        z = np.einsum("npi,nij->npj", h, W3)[..., 0] + b3
        acc = ((z > 0) == (y[None, :] > 0.5)).mean(1)
        for k in range(m):
            if acc[k] > best:
                best = float(acc[k])
                ps = [W1[k], b1[k], W2[k], b2[k], W3[k], b3[k]]
                recs.append((c0 + k + 1, best, field(ps, GX)))
        if c0 == 0:
            # a sample of ordinary tries (the first 40 networks), to show what random networks look like
            sample = []
            for k in range(40):
                ps = [W1[k], b1[k], W2[k], b2[k], W3[k], b3[k]]
                sample.append((k + 1, float(acc[k]), field(ps, GX)))
    return recs, sample


def save_steps():
    recs, sample = improvements()
    print([(n, a) for n, a, _ in recs])
    np.savez_compressed(os.path.join(HERE, "rs_steps.npz"), rec_n=np.array([r[0] for r in recs]), rec_acc=np.array([r[1] for r in recs]),
                        rec_field=np.array([r[2] for r in recs]), sample_n=np.array([s[0] for s in sample]),
                        sample_acc=np.array([s[1] for s in sample]), sample_field=np.array([s[2] for s in sample]))


if __name__ == "__main__" and os.environ.get("RS_IMPROVEMENTS"):
    save_steps()
