"""Why AI runs on matrix multiplication, timed on this computer (run it on an otherwise idle machine).
One 512 x 512 matrix multiplication, C = A B (134,217,728 multiply-adds), three ways: plain Python loops (one
multiply-add at a time), NumPy restricted to one core (compiled, vectorized code: many multiply-adds per instruction)
and NumPy on every core. All three answers compared. Then the million-weight network of reel 41
(2 -> 1000 -> 1000 -> 1 on 400 points): its forward pass is three matrix multiplications; timed in NumPy, and in
plain Python estimated from the measured loop speed.    python mmsim.py  -> mm_results.json"""

import json
import os
import platform
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
N = 512


def py_matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    Bt = [list(col) for col in zip(*B)]
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for j in range(p):
            Bj = Bt[j]
            s = 0.0
            for k in range(m):
                s += Ai[k] * Bj[k]
            Ci[j] = s
    return C


def median_time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    Al, Bl = A.tolist(), B.tolist()
    t0 = time.perf_counter()
    Cp = py_matmul(Al, Bl)
    t_py = time.perf_counter() - t0
    Cp = np.array(Cp)
    with threadpool_limits(limits=1):
        t_np1 = median_time(lambda: A @ B, 30)
    t_npall = median_time(lambda: A @ B, 60)
    C = A @ B
    diff = float(np.abs(C - Cp).max())
    madds = N ** 3
    rate_py = madds / t_py
    # the network of reel 41
    X = rng.uniform(-1, 1, (400, 2))
    W1, W2, W3 = rng.normal(size=(2, 1000)), rng.normal(size=(1000, 1000)) * 0.03, rng.normal(size=(1000, 1)) * 0.03
    b1, b2 = np.zeros(1000), np.zeros(1000)

    def fwd():
        h = np.maximum(X @ W1 + b1, 0)
        h = np.maximum(h @ W2 + b2, 0)
        return h @ W3
    t_net = median_time(fwd, 40)
    net_madds = 400 * (2 * 1000 + 1000 * 1000 + 1000)
    t_net_py = net_madds / rate_py
    out = dict(n=N, madds=madds, t_python_s=t_py, t_numpy_1core_s=t_np1, t_numpy_all_s=t_npall, cores=os.cpu_count(), max_abs_diff=diff,
               speedup_1core=t_py / t_np1, speedup_all=t_py / t_npall, python_madds_per_s=rate_py, numpy_madds_per_s=madds / t_npall,
               net_madds=net_madds, t_net_numpy_s=t_net, t_net_python_est_s=t_net_py, net_speedup=t_net_py / t_net,
               blas=[{k: v for k, v in d.items() if k in ("internal_api", "num_threads", "version")} for d in threadpool_info()],
               python=platform.python_version(), numpy=np.__version__)
    with open(os.path.join(HERE, "mm_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
