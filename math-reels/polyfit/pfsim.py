"""The fits of the overfitting reel. 20 measurements at evenly spaced x from -0.95 to 0.95, y = sin(2.6 x) + 0.3 x +
noise (sd 0.25), seed 0; 5,000 new points from the same source (x uniform over the same range, seed 999) to test on. Polynomials of degree 0 ... 19 fitted by least squares
(in the Legendre basis, which is the same polynomial as plain powers but better conditioned); degree 19 passes
through all 20 points. Then degree 19 fitted to 200 points from the same source (x uniform), and degree 19 on the 20 points
with a ridge penalty lambda * sum(c_k^2) on its Legendre coefficients, lambda chosen by leave-one-out
cross-validation on the 20 points (the 5,000 test points are never used to choose anything).
Errors are root-mean-square misses.    python pfsim.py  -> pf_results.json"""

import json
import os

import numpy as np
from numpy.polynomial import legendre as Lg

HERE = os.path.dirname(os.path.abspath(__file__))
SIG = 0.25
X_LO, X_HI = -0.95, 0.95
N, N_BIG, N_TEST, DMAX = 20, 200, 5000, 19
LAMS = 10.0 ** np.arange(-8, 1.01, 0.25)


def truth(x):
    return np.sin(2.6 * x) + 0.3 * x


def fit(x, y, d, lam=0.0):
    A = Lg.legvander(x, d)
    if lam == 0:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    return np.linalg.solve(A.T @ A + lam * np.eye(d + 1), A.T @ y)


def rmse(c, x, y):
    return float(np.sqrt(np.mean((Lg.legval(x, c) - y) ** 2)))


def loo_lambda(x, y, d):
    A = Lg.legvander(x, d)
    best = None
    for lam in LAMS:
        Hm = A @ np.linalg.solve(A.T @ A + lam * np.eye(d + 1), A.T)
        r = (y - Hm @ y) / (1 - np.diag(Hm))
        e = float(np.mean(r ** 2))
        if best is None or e < best[0]:
            best = (e, float(lam))
    return best[1]


def main(seed=0):
    rng_t = np.random.default_rng(999)
    xt = rng_t.uniform(X_LO, X_HI, N_TEST)
    yt = truth(xt) + SIG * rng_t.normal(size=N_TEST)
    rng = np.random.default_rng(seed)
    x = np.linspace(X_LO, X_HI, N)
    y = truth(x) + SIG * rng.normal(size=N)
    xb = rng.uniform(X_LO, X_HI, N_BIG)
    yb = truth(xb) + SIG * rng.normal(size=N_BIG)
    grid = np.linspace(-1.05, 1.05, 600)
    out = dict(x=x.tolist(), y=y.tolist(), xb=xb.tolist(), yb=yb.tolist(), xt=xt[:400].tolist(), yt=yt[:400].tolist(), grid=grid.tolist(),
               truth=truth(grid).tolist(), curves=[], train=[], test=[])
    for d in range(DMAX + 1):
        c = fit(x, y, d)
        out["curves"].append(Lg.legval(grid, c).tolist())
        out["train"].append(rmse(c, x, y))
        out["test"].append(rmse(c, xt, yt))
    cb = fit(xb, yb, DMAX)
    out["big_curve"] = Lg.legval(grid, cb).tolist()
    out["big_train"], out["big_test"] = rmse(cb, xb, yb), rmse(cb, xt, yt)
    lam = loo_lambda(x, y, DMAX)
    cr = fit(x, y, DMAX, lam)
    out["ridge_curve"] = Lg.legval(grid, cr).tolist()
    out["ridge_lambda"], out["ridge_train"], out["ridge_test"] = lam, rmse(cr, x, y), rmse(cr, xt, yt)
    out["best_degree"] = int(np.argmin(out["test"]))
    out["truth_test"] = float(np.sqrt(np.mean((truth(xt) - yt) ** 2)))     # the noise floor: the true curve's own error
    return out


if __name__ == "__main__":
    r = main()
    with open(os.path.join(HERE, "pf_results.json"), "w") as fh:
        json.dump(r, fh)
    for d in range(DMAX + 1):
        print(f"degree {d:2d}: train {r['train'][d]:.3g}  test {r['test'][d]:.4g}")
    print("best degree", r["best_degree"], "| 200 points, degree 19: test", round(r["big_test"], 4), "| ridge lambda", r["ridge_lambda"],
          "test", round(r["ridge_test"], 4), "train", round(r["ridge_train"], 4), "| noise floor", round(r["truth_test"], 4))
    # the U across other draws (for the honesty notes)
    for s in range(1, 8):
        q = main(s)
        print(f"seed {s}: best degree {q['best_degree']} test {min(q['test']):.3f}; degree 19 train {q['train'][19]:.1e} test {q['test'][19]:.4g}; "
              f"200 pts {q['big_test']:.3f}; ridge {q['ridge_test']:.3f}")
