"""The runs of the zig-zag reel. A bowl f(x, y) = (mu x^2 + L y^2) / 2 with L = 1 and mu = 1 / kappa (its Hessian is
diag(mu, L), eigenvalues mu and L, condition number kappa), started at (-2.8, 1.6). "At the bottom" means within
0.01 of the minimum. Each method's settings are the best on a grid (fewest steps to the bottom), searched per bowl:
gradient descent: step 0.01 ... 1.99; heavy-ball momentum: step 0.05 ... 3.95 and momentum 0 ... 0.98;
Adam (beta1 0.9, beta2 0.999): step 0.01 ... 1.99.    python zzsim.py  -> zz_results.json"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X0 = np.array([-2.8, 1.6])
TOL = 0.01
MAXIT = 3000
KAPPAS = (1, 10, 100)


def run(kappa, method, lr, beta=0.0, maxit=MAXIT):
    """The path [p0, p1, ...] until it is within TOL of the bottom (None if it never gets there)."""
    h = np.array([1.0 / kappa, 1.0])
    p, v, m, s = X0.copy(), np.zeros(2), np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for k in range(1, maxit + 1):
        g = h * p
        if method == "gd":
            p = p - lr * g
        elif method == "momentum":
            v = beta * v - lr * g
            p = p + v
        else:
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g * g
            p = p - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(s / (1 - 0.999 ** k)) + 1e-8)
        path.append(p.copy())
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6:
            return None
        if np.linalg.norm(p) < TOL:
            return np.array(path)
    return None


def best(kappa, method):
    grid = [(lr, 0.0) for lr in np.arange(0.01, 2.0, 0.01)] if method != "momentum" else \
        [(lr, b) for lr in np.arange(0.05, 4.0, 0.05) for b in np.arange(0.0, 0.99, 0.02)]
    top = None
    for lr, b in grid:
        p = run(kappa, method, lr, b)
        if p is not None and (top is None or len(p) < len(top[2])):
            top = (round(float(lr), 3), round(float(b), 3), p)
    return top


if __name__ == "__main__":
    out = {}
    for kappa in KAPPAS:
        for method in ("gd", "momentum", "adam"):
            lr, b, p = best(kappa, method)
            out[f"{method}_{kappa}"] = dict(lr=lr, beta=b, steps=len(p) - 1, path=p.round(6).tolist())
            print(f"kappa {kappa:3d} {method:8s} lr {lr:5.2f} beta {b:4.2f} steps {len(p) - 1}")
    with open(os.path.join(HERE, "zz_results.json"), "w") as fh:
        json.dump(out, fh)
