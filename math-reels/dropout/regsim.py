"""Train the same wide ReLU network (2 -> 256 -> 256 -> 1) on the spiral with 20% of the training
labels flipped, three ways: plain, with dropout (p = 0.5 on both hidden layers, inverted, fresh mask
every step), and with weight decay (L2 1e-3). Same data, same flipped labels, same starting weights,
same Adam, same steps. Records every RECORD-th step: the decision field on a grid, the prediction on
every training point (so the renderer can show which flipped labels the network believes), training
accuracy on the noisy labels, accuracy on a clean held-out spiral. Saves an .npz."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "neurons"))
from train import init, spiral  # noqa: E402

DATA_SEED, TEST_SEED, FLIP_SEED, INIT_SEED, MASK_SEED = 14, 77, 1, 2, 5
SIZES = [2, 256, 256, 1]
STEPS = 3000
RECORD = 3                       # keep every 3rd step (1,001 snapshots)
LR = 0.01
FLIP_FRAC = 0.2
DROP_P = 0.5
L2 = 1e-3
NAMES = ["plain", "dropout", "l2"]
GRID, GRID_EXTENT = 78, 1.05


def forward(ps, X, rng=None, p=0.0):
    """Inverted dropout on hidden layers when rng is given; returns logits, activations, masks."""
    h = X
    acts = [h]
    masks = []
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        if i < len(ps) - 2:
            h = np.maximum(z, 0.0)
            if rng is not None and p > 0:
                mk = (rng.random(h.shape) >= p) / (1 - p)
                h = h * mk
                masks.append(mk)
            else:
                masks.append(None)
        else:
            h = z
        acts.append(h)
    return h[:, 0], acts, masks


def grid_points():
    xs = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), -gy.ravel()], 1)


def train(name, X, yn, y, Xt, yt, G):
    ps = init(np.random.default_rng(INIT_SEED), SIZES)
    rng = np.random.default_rng(MASK_SEED)
    m = [np.zeros_like(q) for q in ps]
    v = [np.zeros_like(q) for q in ps]
    n_rec = STEPS // RECORD + 1
    fields = np.zeros((n_rec, GRID, GRID), np.float16)
    pred = np.zeros((n_rec, len(y)), bool)
    acc_train = np.zeros(n_rec)
    acc_test = np.zeros(n_rec)
    for s in range(STEPS + 1):
        if s % RECORD == 0:
            r = s // RECORD
            lg = forward(ps, X)[0]
            pred[r] = lg > 0
            acc_train[r] = (pred[r] == (yn > 0.5)).mean()
            acc_test[r] = ((forward(ps, Xt)[0] > 0) == (yt > 0.5)).mean()
            gl = forward(ps, G)[0]
            fields[r] = (1 / (1 + np.exp(-np.clip(gl, -30, 30)))).reshape(GRID, GRID)
        if s == STEPS:
            break
        step = s + 1
        logit, acts, masks = forward(ps, X, rng if name == "dropout" else None, DROP_P)
        pr = 1 / (1 + np.exp(-logit))
        g = (pr - yn)[:, None] / len(yn)
        grads = []
        for i in range(len(ps) - 2, -1, -2):
            gw = acts[i // 2].T @ g
            if name == "l2":
                gw = gw + L2 * ps[i]
            grads.insert(0, gw)
            grads.insert(1, g.sum(0))
            if i > 0:
                g = (g @ ps[i].T) * (acts[i // 2] > 0)
                if masks[i // 2 - 1] is not None:
                    g = g * masks[i // 2 - 1]
        for k, (pk, gk) in enumerate(zip(ps, grads)):
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk * gk
            pk -= LR * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    return fields, pred, acc_train, acc_test


def main(out_path):
    X, y = spiral(np.random.default_rng(DATA_SEED))
    Xt, yt = spiral(np.random.default_rng(TEST_SEED))
    flip = np.random.default_rng(FLIP_SEED).random(len(y)) < FLIP_FRAC
    yn = y.copy()
    yn[flip] = 1 - yn[flip]
    G = grid_points()
    arrays = dict(X=X, y=y, yn=yn, flip=flip, Xt=Xt, yt=yt)
    meta = dict(steps=STEPS, record=RECORD, lr=LR, sizes=SIZES, names=NAMES, flip_frac=FLIP_FRAC, n_flip=int(flip.sum()),
                drop_p=DROP_P, l2=L2, acc_train={}, acc_test={}, lies={}, final={})
    for name in NAMES:
        fields, pred, acc_train, acc_test = train(name, X, yn, y, Xt, yt, G)
        lies = (pred[:, flip] == (yn[flip] > 0.5)).sum(1)          # flipped points predicted as their (wrong) label
        arrays[f"{name}_fields"] = fields
        arrays[f"{name}_pred"] = pred
        meta["acc_train"][name] = acc_train.tolist()
        meta["acc_test"][name] = acc_test.tolist()
        meta["lies"][name] = lies.tolist()
        meta["final"][name] = dict(train=float(acc_train[-1]), test=float(acc_test[-1]), lies=int(lies[-1]),
                                   test_peak=float(acc_test.max()), test_peak_step=int(acc_test.argmax() * RECORD))
        print(f"{name:8s} train(noisy) {acc_train[-1]:.3f} clean test {acc_test[-1]:.3f} (peak {acc_test.max():.3f} at step "
              f"{acc_test.argmax() * RECORD}) lies believed {lies[-1]}/{flip.sum()}  "
              f"lies@500 {lies[500 // RECORD]} @1000 {lies[1000 // RECORD]} @2000 {lies[2000 // RECORD]}")
    np.savez_compressed(out_path, **arrays)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dropout_train.npz"))
