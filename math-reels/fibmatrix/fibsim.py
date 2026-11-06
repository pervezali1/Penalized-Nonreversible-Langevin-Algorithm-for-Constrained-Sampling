"""Fibonacci numbers are powers of one matrix, and the golden ratio is its eigenvalue. A = [[1, 1], [1, 0]] sends the
pair (a, b) to (a + b, a), so A^n (1, 0) = (F(n+1), F(n)). Recorded, with exact integers (Python) and 80-digit
decimals:
  * the pairs from three starts, (1, 0) (Fibonacci), (2, -1) (Lucas numbers) and (1, -2), for 14 steps, and their
    directions; the eigenvectors (phi, 1) and (psi, 1), phi = (1 + sqrt 5) / 2, psi = (1 - sqrt 5) / 2;
  * the ratios F(n+1) / F(n) and how many digits of phi each gets right;
  * phi^100 / sqrt 5 against F(100);
  * F(1,000,000) two ways, timed on this machine (median of 3): a million additions, and A^1,000,000 by repeated
    squaring (counting the matrix multiplications).    python fibsim.py  -> fib_results.json"""

import json
import math
import os
import platform
import statistics
import sys
import time
from decimal import Decimal, getcontext

HERE = os.path.dirname(os.path.abspath(__file__))
getcontext().prec = 80
STARTS = [(1, 0), (2, -1), (1, -2)]
STEPS = 14
BIG = 1_000_000


def step(p):
    return (p[0] + p[1], p[0])


def matmul(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def matpow(n):
    """A^n by repeated squaring; returns it and the number of 2 x 2 matrix multiplications."""
    R, P, mults = None, [[1, 1], [1, 0]], 0
    while n:
        if n & 1:
            if R is None:
                R = P
            else:
                R = matmul(R, P)
                mults += 1
        n >>= 1
        if n:
            P = matmul(P, P)
            mults += 1
    return R, mults


def by_adding(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def digits_right(x, ref):
    """How many decimal places of ref the decimal x gets right (after the 1.)."""
    sx, sr = f"{x:.30f}", f"{ref:.30f}"
    k = 0
    while k < len(sx) and sx[k] == sr[k]:
        k += 1
    return max(0, k - 2)


def main():
    s5 = Decimal(5).sqrt()
    phi, psi = (1 + s5) / 2, (1 - s5) / 2
    paths = []
    for s in STARTS:
        p, path = s, [s]
        for _ in range(STEPS):
            p = step(p)
            path.append(p)
        paths.append(dict(start=list(s), pairs=[list(q) for q in path], angles=[math.degrees(math.atan2(q[1], q[0])) for q in path]))
    fib = [by_adding(n) for n in range(40)]
    ratios = [dict(n=n, num=fib[n + 1], den=fib[n], ratio=str(Decimal(fib[n + 1]) / Decimal(fib[n])), right=digits_right(Decimal(fib[n + 1]) / Decimal(fib[n]), phi))
              for n in range(1, 30)]
    F100 = fib[0] if False else by_adding(100)
    binet = phi ** 100 / s5
    sys.set_int_max_str_digits(0)
    t_add, t_mat = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        a = by_adding(BIG)
        t_add.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        R, mults = matpow(BIG)
        t_mat.append(time.perf_counter() - t0)
        assert R[0][1] == a
    s = str(a)
    m1000, mults1000 = matpow(1000)
    out = dict(starts=[list(s_) for s_ in STARTS], steps=STEPS, paths=paths, phi=str(phi), psi=str(psi),
               eig_angle=math.degrees(math.atan2(1, float(phi))), psi_angle=math.degrees(math.atan2(1, float(psi))),
               fib=[str(f) for f in fib], ratios=ratios, F100=str(F100), binet100=str(binet), binet_err=str(binet - F100),
               big=dict(n=BIG, digits=len(s), head=s[:12], tail=s[-12:], t_add=statistics.median(t_add), t_mat=statistics.median(t_mat),
                        t_add_all=t_add, t_mat_all=t_mat, mults=mults, squarings=BIG.bit_length() - 1, extra=bin(BIG).count("1") - 1),
               F1000_digits=len(str(m1000[0][1])), mults1000=mults1000, python=platform.python_version(), machine=platform.machine())
    with open(os.path.join(HERE, "fib_results.json"), "w") as fh:
        json.dump(out, fh)
    print("phi", str(phi)[:14], "eigenvector angle", round(out["eig_angle"], 2), "psi line", round(out["psi_angle"], 2))
    for p in paths:
        print(p["start"], p["pairs"][:8], [round(x, 1) for x in p["angles"][:8]])
    print("ratios", [(r["num"], r["den"], r["ratio"][:8], r["right"]) for r in ratios[:12]])
    print("F100", F100, "phi^100/sqrt5 - F100 =", out["binet_err"][:12])
    b = out["big"]
    print(f"F({BIG:,}): {b['digits']:,} digits, {b['head']}...{b['tail']}; adding {b['t_add']:.2f} s, {b['mults']} matrix multiplications {b['t_mat']:.3f} s ({b['squarings']} squarings + {b['extra']})")


if __name__ == "__main__":
    main()
