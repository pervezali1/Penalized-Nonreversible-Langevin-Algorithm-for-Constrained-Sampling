"""The data behind the noisy slope: N points whose average squared error, as a function of the two weights
(u, v), is EXACTLY the bowl 0.5u^2 + 1.8v^2 plus a floor. Least squares: loss(w) = (1/2N) sum_i (x_i . w - e_i)^2.
The inputs are whitened so that (1/N) X^T X = diag(1, 3.6) exactly; the label noise e is made orthogonal to
the inputs and to the constant and scaled so that (1/2N) sum e^2 = FLOOR exactly. Then the full-batch slope
is exactly the bowl's gradient (Sigma w) and a mini-batch of B points gives that slope plus noise."""

import numpy as np

A_, B_ = 0.5, 1.8
N = 1000
FLOOR = 0.10                      # (1/2N) sum e^2: the error of the perfect weights, from the labels' own noise
SIGMA = np.diag([2 * A_, 2 * B_])  # second-moment matrix of the inputs: Hessian of the mean loss


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 2))
    M = X.T @ X / N
    # whiten to exact second moment Sigma
    w_, V = np.linalg.eigh(M)
    X = X @ V @ np.diag(w_ ** -0.5) @ V.T
    X = X @ np.diag(np.sqrt(np.diag(SIGMA)))
    e = rng.standard_normal(N)
    Q = np.column_stack([np.ones(N), X])
    e = e - Q @ np.linalg.lstsq(Q, e, rcond=None)[0]      # orthogonal to the inputs and to the constant
    e *= np.sqrt(2 * FLOOR * N / (e @ e))                  # (1/2N) sum e^2 = FLOOR exactly
    return X, e


def loss(X, e, w):
    r = X @ w - e
    return 0.5 * np.mean(r * r)


def true_grad(w):
    return SIGMA @ w


def batch_grad(X, e, w, idx):
    r = X[idx] @ w - e[idx]
    return X[idx].T @ r / len(idx)


def run(X, e, w0, lr, batch, n_steps, seed):
    """Plain descent on the mini-batch slope: w <- w - lr * g_B. Returns landings, measured slopes, batches."""
    rng = np.random.default_rng(seed)
    w = np.array(w0, float)
    pts, grads = [w.copy()], []
    for _ in range(n_steps):
        idx = rng.choice(N, size=batch, replace=False) if batch < N else np.arange(N)
        g = batch_grad(X, e, w, idx)
        grads.append(g)
        w = w - lr * g
        pts.append(w.copy())
    return pts, grads


if __name__ == "__main__":
    X, e = make_data()
    M = X.T @ X / N
    print("second moment", np.round(M, 12), "mean e", e.mean(), "X^T e", X.T @ e, "floor", 0.5 * np.mean(e * e))
    w0 = (2.85, -0.64)
    print("loss at start", loss(X, e, np.array(w0)), "bowl", A_ * w0[0] ** 2 + B_ * w0[1] ** 2 + FLOOR)
    for batch in (1, 10, 1000):
        for seed in (1, 2, 3, 4, 5):
            pts, grads = run(X, e, w0, 0.2, batch, 24, seed)
            hs = [loss(X, e, p) for p in pts]
            tail = hs[16:]
            vmax = max(abs(p[1]) for p in pts)
            print(f"B={batch:5d} seed {seed}: h[4,8,12,16,20,24] {[round(hs[k],3) for k in (4,8,12,16,20,24)]}  tail mean {np.mean(tail):.3f} std {np.std(tail):.3f}  max|v| {vmax:.2f} maxv {max(p[1] for p in pts):.2f}")
