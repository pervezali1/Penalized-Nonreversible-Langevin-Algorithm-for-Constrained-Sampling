"""The 37% rule (optimal stopping, the "secretary problem"), simulated. 10,000 lives; in each, 100 people arrive in
a random order, each must get a yes or no on the spot, and a no is final. The rule: say no to the first R, remember
the best of them, then say yes to the first one better than all of those. If nobody is, you end up alone.

Every life is scored for every cutoff R = 0..99 at once: with b the position of the very best and m the position of
the best of the people before b, the rule with cutoff R finds the very best exactly when m < R <= b (R = 0 means
say yes to the first person, which finds the best only when b = 0). The exact chance is
P(R) = R/n * sum_{i=R}^{n-1} 1/i  (P(0) = 1/n), and the chance of ending up alone is R/n.
    python stopsim.py <out.json>"""

import json
import sys

import numpy as np

N, LIVES, R, SEED = 100, 10_000, 37, 67


def lives(rng=None):
    """ranks[l, i] = how good person i of life l is: 1 (worst) .. N (the very best), a random order per life."""
    rng = rng or np.random.default_rng(SEED)
    return np.argsort(rng.random((LIVES, N)), axis=1).argsort(axis=1) + 1


def outcome(ranks, r=R):
    """For each life: (index of the person you say yes to or -1 if alone, the threshold = best of the first r)."""
    thr = ranks[:, :r].max(1) if r > 0 else np.zeros(len(ranks), int)
    later = ranks[:, r:] > thr[:, None]
    has = later.any(1)
    pick = np.where(has, r + later.argmax(1), -1)
    return pick, thr


def exact(r, n=N):
    if r == 0:
        return 1 / n
    return r / n * sum(1 / i for i in range(r, n))


def per_cutoff(ranks):
    """Share of the lives in which cutoff r finds the very best, r = 0..N-1, all at once."""
    b = ranks.argmax(1)
    pref = np.maximum.accumulate(ranks, axis=1)
    m = np.array([int(np.argmax(ranks[l, :b[l]] == pref[l, b[l] - 1])) if b[l] > 0 else -1 for l in range(len(ranks))])
    hits = np.zeros(N + 1)
    for bl, ml in zip(b, m):
        if bl == 0:                      # only "yes to the first person" finds them
            hits[0] += 1
            hits[1] -= 1
        else:
            hits[ml + 1] += 1
            hits[bl + 1] -= 1
    return np.cumsum(hits)[:N] / len(ranks)


def main(out):
    ranks = lives()
    pick, thr = outcome(ranks)
    best = pick >= 0
    best &= np.take_along_axis(ranks, np.maximum(pick, 0)[:, None], 1)[:, 0] == N
    alone = pick < 0
    settled = ~best & ~alone
    curve = per_cutoff(ranks)
    # brute-force check of the all-cutoffs shortcut on a few cutoffs
    for r in (0, 1, 10, 37, 60, 99):
        p, _ = outcome(ranks, r)
        got = np.zeros(LIVES, bool)
        ok = p >= 0
        got[ok] = ranks[ok, p[ok]] == N
        assert abs(got.mean() - curve[r]) < 1e-12, (r, got.mean(), curve[r])
    ex = np.array([exact(r) for r in range(N)])
    r_best_exact = int(np.argmax(ex))
    picked_rank = np.take_along_axis(ranks, np.maximum(pick, 0)[:, None], 1)[:, 0]
    # the example life: the first life (in order) where the rule finds the very best
    demo = int(np.argmax(best))
    facts = dict(n=N, lives=LIVES, r=R, seed=SEED,
                 found_best=float(best.mean()), settled=float(settled.mean()), alone=float(alone.mean()),
                 exact_best=exact(R), exact_alone=R / N, r_best_exact=r_best_exact, p_best_exact_max=float(ex.max()),
                 r_best_sim=int(np.argmax(curve)), p_best_sim_max=float(curve.max()),
                 random_pick=float(curve[0]), exact_random=1 / N,
                 settled_median_rank_from_top=int(np.median(N + 1 - picked_rank[settled])),
                 settled_top3=float(np.mean(picked_rank[settled] >= N - 2)),
                 demo_life=demo, demo_pick=int(pick[demo]), demo_threshold=int(thr[demo]),
                 demo_best_pos=int(ranks[demo].argmax()),
                 curve=curve.tolist(), curve_exact=ex.tolist())
    with open(out, "w") as fh:
        json.dump(facts, fh, indent=1)
    print({k: v for k, v in facts.items() if not k.startswith("curve")})


if __name__ == "__main__":
    main(sys.argv[1])
