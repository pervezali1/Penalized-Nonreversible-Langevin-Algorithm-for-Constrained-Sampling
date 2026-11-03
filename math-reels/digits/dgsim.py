"""A network learns to read handwritten digits. 1,797 real handwritten digits (digits.csv: 8 x 8 pixels, ink 0-16),
split at random into 1,257 to train on and 540 the network never trains on. Network: 64 pixels -> 32 ReLU neurons
-> 10 outputs with softmax (one probability per digit), cross-entropy, Adam (step 0.01, full batch), 300 steps.
Recorded: the accuracy on the 540 unseen digits at every step; the network's ten probabilities for a set of
shown digits at every step; then, for a few of the unseen digits, the blame backpropagated all the way to the
pixels (the gradient of the log-probability of the right digit with respect to each pixel), and the weights of
the 32 hidden neurons as 8 x 8 pictures.    python dgsim.py  -> dg_results.npz, dg_results.json"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, SPLIT_SEED = 3, 0
HIDDEN, STEPS, LR = 32, 300, 0.01


def load():
    rows = [ln.strip().split(",") for ln in open(os.path.join(HERE, "digits.csv")) if ln[0].isdigit()]
    A = np.array(rows, dtype=float)
    return A[:, :64] / 16.0, A[:, 64].astype(int)


def forward(ps, X):
    W1, b1, W2, b2 = ps
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    z2 = z2 - z2.max(1, keepdims=True)
    e = np.exp(z2)
    return e / e.sum(1, keepdims=True), h, z1


def grads(ps, X, Y1):
    W1, b1, W2, b2 = ps
    p, h, z1 = forward(ps, X)
    loss = -np.mean(np.log(p[Y1.astype(bool)] + 1e-12))
    d2 = (p - Y1) / len(X)
    gW2, gb2 = h.T @ d2, d2.sum(0)
    d1 = (d2 @ W2.T) * (z1 > 0)
    gW1, gb1 = X.T @ d1, d1.sum(0)
    return loss, [gW1, gb1, gW2, gb2]


def pixel_blame(ps, x, label):
    """d log p(label) / d pixel, for one digit: the blame carried back through the whole network to the input."""
    W1, b1, W2, b2 = ps
    p, h, z1 = forward(ps, x[None, :])
    d2 = -(np.eye(10)[label] - p[0])          # d(-log p_label)/dz2
    d1 = (d2 @ W2.T) * (z1[0] > 0)
    return -(d1 @ W1.T), p[0]                  # gradient of +log p_label


def main(seed=SEED, split_seed=SPLIT_SEED, save=True):
    X, y = load()
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(X))
    te, tr = idx[:540], idx[540:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    Y1 = np.eye(10)[ytr]
    rng = np.random.default_rng(seed)
    ps = [rng.normal(0, np.sqrt(2 / 64), (64, HIDDEN)), np.zeros(HIDDEN), rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)), np.zeros(10)]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    # the digits shown during training: the first unseen example of each digit 0..9
    show = np.array([np.flatnonzero(yte == d)[0] for d in range(10)])
    acc_te, acc_tr, probs_show, losses, W1s = [], [], [], [], []
    for s in range(STEPS + 1):
        pte = forward(ps, Xte)[0]
        acc_te.append(float(np.mean(pte.argmax(1) == yte)))
        acc_tr.append(float(np.mean(forward(ps, Xtr)[0].argmax(1) == ytr)))
        probs_show.append(pte[show].astype(np.float32))
        if s % 10 == 0:
            W1s.append(ps[0].astype(np.float32))
        loss, gs = grads(ps, Xtr, Y1)
        losses.append(float(loss))
        if s == STEPS:
            break
        for i, g in enumerate(gs):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= LR * (m[i] / (1 - 0.9 ** (s + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (s + 1))) + 1e-8)
    pte = forward(ps, Xte)[0]
    pred = pte.argmax(1)
    wrong = np.flatnonzero(pred != yte)
    blames = {int(i): pixel_blame(ps, Xte[i], int(yte[i]))[0].tolist() for i in list(show) + list(wrong)}
    out = dict(n=len(X), n_train=len(tr), n_test=len(te), hidden=HIDDEN, steps=STEPS, lr=LR, acc_test=acc_te, acc_train=acc_tr, loss=losses,
               show=show.tolist(), wrong=wrong.tolist(), wrong_pred=pred[wrong].tolist(), wrong_true=yte[wrong].tolist(),
               wrong_p=[float(pte[i, pred[i]]) for i in wrong], n_right=int(np.sum(pred == yte)), confusion=np.bincount(10 * yte + pred, minlength=100).reshape(10, 10).tolist(),
               first_90=next(i for i, a in enumerate(acc_te) if a >= 0.9))
    if not save:
        return out
    np.savez_compressed(os.path.join(HERE, "dg_results.npz"), Xte=Xte.astype(np.float32), yte=yte, probs_show=np.array(probs_show),
                        W1=ps[0].astype(np.float32), W1s=np.array(W1s), wrong_probs=pte[wrong].astype(np.float32), blame_idx=np.array(list(blames)), blame=np.array(list(blames.values()), np.float32), pred=pred)
    with open(os.path.join(HERE, "dg_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"test {acc_te[0]:.3f} -> {acc_te[-1]:.4f} ({out['n_right']}/{len(te)}), train {acc_tr[-1]:.4f}, 90% at step {out['first_90']}")
    print("wrong:", list(zip(out["wrong_true"], out["wrong_pred"], [round(p, 2) for p in out["wrong_p"]])))


if __name__ == "__main__":
    main()
    # the same experiment with other random splits and starts, for the honesty notes
    if os.environ.get("DG_QUICK"):
        raise SystemExit
    accs = [main(seed=sd, split_seed=sp, save=False)["acc_test"][-1] for sd, sp in ((1, 1), (2, 2), (4, 3), (5, 4), (6, 5), (7, 6))]
    print("other splits and starts:", [round(a, 4) for a in accs], "median", round(float(np.median(accs)), 4))
