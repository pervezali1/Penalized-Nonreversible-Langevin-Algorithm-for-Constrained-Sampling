"""How long is a game of snakes and ladders? One matrix answers it exactly. A board made up for this reel: squares
1-36, ladders 3 -> 16, 8 -> 26, 20 -> 29, snakes 13 -> 2, 27 -> 10, 32 -> 6, 34 -> 23. You start off the board (square
0); each turn roll a fair die and move; land on a ladder's foot or a snake's head and you go to its other end; reach
or pass 36 and the game is over. Q[i, j] = the chance of going from square i to square j in one turn (squares 0-35).
Recorded:
  * the expected number of turns, t = (I - Q)^-1 1 (row for square 0), in floating point and as an exact fraction
    (Gaussian elimination in rational numbers); the expected visits to each square (row 0 of (I - Q)^-1);
  * the exact distribution of the game's length (repeated multiplication by Q);
  * 100,000 simulated games (seeded): their lengths, mean, longest, and one sample game for the animation;
  * the same with no snakes or ladders, and with each snake or ladder removed in turn.
    python boardsim.py  -> board_results.json"""

import json
import os
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 36
LADDERS = {3: 16, 8: 26, 20: 29}
SNAKES = {13: 2, 27: 10, 32: 6, 34: 23}
GAMES, SEED, KMAX = 100_000, 64, 200


def jumps_of(lad, sn):
    j = dict(lad)
    j.update(sn)
    return j


def chain(jumps):
    Q = np.zeros((N, N))
    for s in range(N):
        for d in range(1, 7):
            t = s + d
            if t >= N:
                continue
            Q[s, jumps.get(t, t)] += 1 / 6
    return Q


def expected(jumps):
    Fm = np.linalg.inv(np.eye(N) - chain(jumps))
    return float(Fm[0].sum()), Fm[0]


def exact_expected(jumps):
    A = [[Fraction(int(i == j)) for j in range(N)] + [Fraction(1)] for i in range(N)]
    for s in range(N):
        for d in range(1, 7):
            t = s + d
            if t >= N:
                continue
            A[s][jumps.get(t, t)] -= Fraction(1, 6)
    for c in range(N):
        p = next(r for r in range(c, N) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [v / piv for v in A[c]]
        for r in range(N):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return A[0][N]


def play(jumps, rng):
    s, path, rolls = 0, [0], []
    while True:
        d = int(rng.integers(1, 7))
        rolls.append(d)
        t = s + d
        if t >= N:
            path.append(N)
            return path, rolls
        s = jumps.get(t, t)
        path.append(s)


def main():
    J = jumps_of(LADDERS, SNAKES)
    Q = chain(J)
    e, visits = expected(J)
    ex = exact_expected(J)
    p = np.zeros(N)
    p[0] = 1
    alive = [1.0]
    for _ in range(KMAX):
        p = p @ Q
        alive.append(float(p.sum()))
    dist = [alive[k - 1] - alive[k] for k in range(1, KMAX + 1)]          # P(the game takes exactly k turns)
    rng = np.random.default_rng(SEED)
    lens, sample = [], None
    for g in range(GAMES):
        path, rolls = play(J, rng)
        lens.append(len(rolls))
        if sample is None and 12 <= len(rolls) <= 16:
            sample = dict(path=path, rolls=rolls, game=g + 1)
    lens = np.array(lens)
    hist = np.bincount(lens, minlength=KMAX + 1)[1:KMAX + 1] / GAMES
    e_plain, _ = expected({})
    without = {}
    for k, v in list(LADDERS.items()) + list(SNAKES.items()):
        j = dict(J)
        del j[k]
        without[f"{k}-{v}"] = expected(j)[0] - e
    out = dict(n=N, ladders=LADDERS, snakes=SNAKES, Q=Q.tolist(), expected=e, exact=f"{ex.numerator}/{ex.denominator}", exact_float=float(ex),
               visits=visits.tolist(), dist=dist, prob_done_by=[1 - a for a in alive], games=GAMES, sim_mean=float(lens.mean()),
               sim_longest=int(lens.max()), sim_shortest=int(lens.min()), hist=hist.tolist(), max_gap=float(np.abs(hist - np.array(dist)).max()),
               plain=e_plain, without=without, sample=sample, mode_turns=int(np.argmax(dist) + 1), median_turns=int(np.argmax(np.array([1 - a for a in alive]) >= 0.5)))
    with open(os.path.join(HERE, "board_results.json"), "w") as fh:
        json.dump(out, fh)
    print(f"expected {e:.4f} turns (exact {ex} = {float(ex):.10f}); no snakes or ladders {e_plain:.4f}")
    print(f"{GAMES:,} games: mean {lens.mean():.4f}, shortest {lens.min()}, longest {lens.max()}; largest gap between histogram and matrix {out['max_gap']:.4f}")
    print("most likely length", out["mode_turns"], "median", out["median_turns"], "P(done by 10)", round(out["prob_done_by"][10], 4))
    print("removing each:", {k: round(v, 3) for k, v in without.items()})
    print("most visited square", int(np.argmax(visits[1:]) + 1), round(float(visits[1:].max()), 3), "sample game", sample["game"], sample["path"])


if __name__ == "__main__":
    main()
