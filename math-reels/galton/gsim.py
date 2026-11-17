"""A Galton board, simulated: 10,000 balls, 16 rows of pegs, and at every peg a fair coin flip sends the ball half a
peg to the left or to the right. A ball's bin = the number of rights, 0..16. The exact chance of bin k is
C(16, k) / 2^16, so the expected count is 10,000 x C(16, k) / 65,536 (middle: 1,963.8; each edge: 0.15).
    python gsim.py <out.json>"""

import json
import math
import sys

import numpy as np

N, ROWS, SEED = 10_000, 16, 68


def flips(rng=None):
    """flips[i, j] = 1 if ball i goes right at row j."""
    rng = rng or np.random.default_rng(SEED)
    return (rng.random((N, ROWS)) < 0.5).astype(np.int8)


def expected():
    return np.array([N * math.comb(ROWS, k) / 2 ** ROWS for k in range(ROWS + 1)])


def main(out):
    fl = flips()
    bins = fl.sum(1)
    counts = np.bincount(bins, minlength=ROWS + 1)
    ex = expected()
    chi2 = float(((counts - ex) ** 2 / np.maximum(ex, 1e-9))[2:-2].sum())     # bins with expectation >= 5
    facts = dict(n=N, rows=ROWS, seed=SEED, counts=counts.tolist(), expected=ex.tolist(),
                 middle=int(counts[ROWS // 2]), middle_expected=float(ex[ROWS // 2]),
                 middle_share=float(counts[ROWS // 2] / N), middle_share_exact=math.comb(ROWS, ROWS // 2) / 2 ** ROWS,
                 edges=[int(counts[0]), int(counts[-1])], edge_odds=2 ** ROWS, edge_expected=float(ex[0]),
                 within_2=float(np.mean(np.abs(bins - ROWS // 2) <= 2)),
                 within_2_exact=sum(math.comb(ROWS, k) for k in range(6, 11)) / 2 ** ROWS,
                 max_bin=int(counts.max()), chi2_bins_2_to_14=chi2, chi2_dof=12)
    with open(out, "w") as fh:
        json.dump(facts, fh, indent=1)
    print(facts)


if __name__ == "__main__":
    main(sys.argv[1])
