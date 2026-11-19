"""The medical test paradox, simulated: 10,000 people, exactly 10 of them sick (1 in 1,000). A test that is right 99%
of the time (it catches 99% of sick people and clears 99% of healthy people) is given to everyone; everyone who tests
positive is tested again, and the second test errs independently of the first.
Exact answers: P(sick | positive) = 0.99 x 0.001 / (0.99 x 0.001 + 0.01 x 0.999) = 9.0%;
P(sick | positive twice) = 0.99^2 x 0.001 / (0.99^2 x 0.001 + 0.01^2 x 0.999) = 90.75%.
    python bsim.py <out.json>"""

import json
import sys

import numpy as np

N, SICK, ACC, SEED = 10_000, 10, 0.99, 69


def people(rng=None):
    """(sick, positive on test 1, positive on test 2 -- only meaningful for those positive on test 1)."""
    rng = rng or np.random.default_rng(SEED)
    sick = np.zeros(N, bool)
    sick[rng.choice(N, SICK, replace=False)] = True
    right1 = rng.random(N) < ACC
    right2 = rng.random(N) < ACC
    pos1 = np.where(sick, right1, ~right1)
    pos2 = np.where(sick, right2, ~right2)
    return sick, pos1, pos2


def exact(times):
    p = SICK / N
    return ACC ** times * p / (ACC ** times * p + (1 - ACC) ** times * (1 - p))


def main(out):
    sick, pos1, pos2 = people()
    both = pos1 & pos2
    facts = dict(n=N, sick=int(sick.sum()), healthy=int((~sick).sum()), acc=ACC, seed=SEED,
                 positive=int(pos1.sum()), true_pos=int((pos1 & sick).sum()), false_pos=int((pos1 & ~sick).sum()),
                 missed=int((~pos1 & sick).sum()),
                 share_sick_pos=float(sick[pos1].mean()), exact_pos=exact(1),
                 positive_twice=int(both.sum()), true_pos2=int((both & sick).sum()), false_pos2=int((both & ~sick).sum()),
                 share_sick_pos2=float(sick[both].mean()), exact_pos2=exact(2),
                 expected_false_alarms=(1 - ACC) * (N - SICK))
    with open(out, "w") as fh:
        json.dump(facts, fh, indent=1)
    print(facts)


if __name__ == "__main__":
    main(sys.argv[1])
