"""The data behind the shaken valleys: N examples whose AVERAGE error is exactly the two-valley hill f(u, v).
Each example's own error is the hill plus a tilt, l_i(w) = f(w) + tau_i . w, with the tilts summing to zero
exactly, so the full-batch slope is exactly grad f and a mini-batch of B examples feels grad f plus the mean
tilt of its batch: real sampling noise whose size shrinks like 1/sqrt(B)."""

import numpy as np

N = 1000
TILT = 7.0                      # per-example tilt scale (standard deviation of each tilt component)


def make_tilts(seed=3):
    rng = np.random.default_rng(seed)
    tau = rng.standard_normal((N, 2)) * TILT
    tau -= tau.mean(axis=0)     # the tilts sum to zero exactly: the average error is the hill itself
    tau *= TILT / tau.std(axis=0)
    tau -= tau.mean(axis=0)
    return tau


def batch_noise(tau, rng, batch):
    """Mean tilt of a fresh random batch (without replacement); zero for the full batch."""
    if batch >= N:
        return np.zeros(2)
    idx = rng.choice(N, size=batch, replace=False)
    return tau[idx].mean(axis=0)
