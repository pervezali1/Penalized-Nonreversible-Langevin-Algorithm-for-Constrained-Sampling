"""Every number shown on the Euler Sunday images is computed here.

Run:  python3 solutions.py   (prints each fact; asserts check the shortcuts
against brute force).
"""
from math import gcd, isqrt, log, lcm


# ---------- Problem 1: multiples of 3 or 5 ----------
def p1_loop(limit=1000):
    return sum(i for i in range(1, limit) if i % 3 == 0 or i % 5 == 0)


def p1_S(k, limit=1000):
    m = (limit - 1) // k
    return k * m * (m + 1) // 2


def p1_formula(limit=1000):
    return p1_S(3, limit) + p1_S(5, limit) - p1_S(15, limit)


# ---------- Problem 2: even Fibonacci ----------
def p2_terms(limit=4_000_000):
    f = [1, 2]
    while f[-1] + f[-2] <= limit:
        f.append(f[-1] + f[-2])
    return f


def p2_evens(limit=4_000_000):
    x, y, out = 2, 8, [2]
    while y <= limit:
        out.append(y)
        x, y = y, 4 * y + x
    return out


# ---------- Problem 3: largest prime factor ----------
def p3_factor(n=600851475143):
    """Trial division, dividing each factor out. Returns (factors, steps, last_d, chain)."""
    factors, chain, d, steps = [], [n], 2, 0
    while d * d <= n:
        steps += 1
        if n % d == 0:
            n //= d
            factors.append(d)
            chain.append(n)
        else:
            d += 1
    factors.append(n)
    return factors, steps, d, chain


def is_prime(n):
    if n < 2:
        return False
    return all(n % p for p in range(2, isqrt(n) + 1))


# ---------- Problem 4: palindrome products ----------
def is_pal(n):
    s = str(n)
    return s == s[::-1]


def p4_all():
    return [(a, b, a * b) for a in range(100, 1000) for b in range(a, 1000) if is_pal(a * b)]


def p4_smart():
    """Top-down search; one factor a multiple of 11; break early. Returns (best, a, b, checks)."""
    best, ba, bb, checks = 0, 0, 0, 0
    for a in range(999, 99, -1):
        if a * 999 <= best:
            break
        if a % 11 == 0:
            b_start, step = 999, 1
        else:
            b_start, step = 990, 11
        for b in range(b_start, a - 1, -step):
            checks += 1
            p = a * b
            if p <= best:
                break
            if is_pal(p):
                best, ba, bb = p, a, b
    return best, min(ba, bb), max(ba, bb), checks


# ---------- Problem 5: smallest multiple ----------
def p5_brute_count(n=20):
    L = lcm(*range(1, n + 1))
    return L // n  # number of multiples of n tried by "20, 40, 60, ..."


def prime_powers(n):
    out = []
    for p in range(2, n + 1):
        if is_prime(p):
            q = p
            while q * p <= n:
                q *= p
            out.append((p, q))
    return out


def exponent(p, k):
    e = 0
    while k % p == 0:
        k //= p
        e += 1
    return e


# ---------- Problem 6: sum square difference ----------
def p6(n=100):
    s = n * (n + 1) // 2
    sq = n * (n + 1) * (2 * n + 1) // 6
    return s, s * s, sq, s * s - sq


# ---------- Problem 7: 10001st prime ----------
def sieve(limit):
    is_p = bytearray([1]) * (limit + 1)
    is_p[0:2] = b"\x00\x00"
    for i in range(2, isqrt(limit) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(limit + 1) if is_p[i]]


def p7_bound(n=10001):
    return n * (log(n) + log(log(n)))


if __name__ == "__main__":
    # P1
    assert p1_loop() == p1_formula() == 233168
    print("P1", p1_loop(), [(k, (999 // k), p1_S(k)) for k in (3, 5, 15)])
    print("P1 1e6", p1_formula(10**6), p1_loop(10**6) == p1_formula(10**6))
    yt = sum(i for i in range(1, 1000) if i % 4 == 0 or i % 6 == 0)
    S = lambda k: k * (999 // k) * (999 // k + 1) // 2
    print("P1 YOUR TURN 4 or 6:", yt, S(4), S(6), S(12), S(4) + S(6) - S(12), "wrong w/24:", S(4) + S(6) - S(24))
    # P2
    t = p2_terms()
    ev = p2_evens()
    assert sum(x for x in t if x % 2 == 0) == sum(ev) == 4613732
    print("P2 terms", len(t), "last", t[-1], "evens", len(ev), ev, sum(ev))
    print("P2 100th term parity (PE indexing 1,2,3,5..):", "even" if (100 % 3 == 2) else "odd")
    f = [1, 2]
    while len(f) < 100:
        f.append(f[-1] + f[-2])
    print("   check 100th:", f[99] % 2, [i + 1 for i in range(12) if f[i] % 2 == 0])
    # P3
    fac, steps, d, chain = p3_factor()
    print("P3", fac, "steps", steps, "last d", d, "chain", chain, "sqrt", isqrt(600851475143), all(is_prime(x) for x in fac))
    for y in (2026, 2027):
        print("   ", y, p3_factor(y)[0], is_prime(y))
    # P4
    allp = p4_all()
    best = max(allp, key=lambda r: r[2])
    print("P4 palindrome pairs", len(allp), "best", best, "pairs total", 900 * 901 // 2)
    print("P4 smart", p4_smart())
    print("P4 smallest", min(allp, key=lambda r: r[2]))
    print("P4 all 6-digit pal div by 11:", all(int(f"{a}{b}{c}{c}{b}{a}") % 11 == 0 for a in range(1, 10) for b in range(10) for c in range(10)))
    print("P4 distinct palindromes", len({r[2] for r in allp}))
    # P5
    L = lcm(*range(1, 21))
    print("P5", L, "brute tries", p5_brute_count(), prime_powers(20), "1..25:", lcm(*range(1, 26)), prime_powers(25))
    # P6
    print("P6 n=100", p6(100), "n=10", p6(10), "n=20", p6(20))
    n = 100
    print("   closed form", (n - 1) * n * (n + 1) * (3 * n + 2) // 12)
    # P7
    b = p7_bound()
    pr = sieve(int(b) + 1)
    print("P7 bound", b, "primes<=bound", len(pr), "10001st", pr[10000], "100th", pr[99], "pi(1000)", sum(1 for p in pr if p < 1000))
