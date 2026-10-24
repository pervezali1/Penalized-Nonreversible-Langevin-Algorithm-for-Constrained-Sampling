"""Adam with momentum off (beta1 = 0) vs on (beta1 = 0.95) at the same step size on the descent
reel's landscape, from the same 10,000 random starts. Positions per video frame."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "descent"))
from landscape import F_MIN, N, SEED, at_step, found, frame_step, grad, loss  # noqa: E402

FPS = 60
N_FRAMES = 1200
# (label, beta1, learning rate, steps, first frame, last frame, pacing exponent)
LR = 0.05
RUNS = [("MOMENTUM OFF", 0.0, LR, 150, 72, 420, 1.6),
        ("MOMENTUM ON", 0.95, LR, 150, 468, 780, 1.6)]
RESETS = [(426, 462)]
LOOP = (1164, 1197)


def run_gd(P0, lr, steps):
    P = P0.copy()
    out = [P.copy()]
    for _ in range(steps):
        P = P - lr * grad(P)
        out.append(P.copy())
    return np.stack(out)


def run_adam(P0, lr, steps, b1, b2=0.999, eps=1e-8):
    P = P0.copy()
    m = np.zeros_like(P)
    v = np.zeros_like(P)
    out = [P.copy()]
    for s in range(1, steps + 1):
        g = grad(P)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        P = P - lr * (m / (1 - b1 ** s)) / (np.sqrt(v / (1 - b2 ** s)) + eps)
        out.append(P.copy())
    return np.stack(out)


def main(out_path):
    rng = np.random.default_rng(SEED)
    P0 = rng.uniform(-1, 1, (N, 2))
    trajs = [run_adam(P0, lr, steps, b1) for (_, b1, lr, steps, *_r) in RUNS]
    info = []
    for (label, b1, lr, steps, *_r), tr in zip(RUNS, trajs):
        fr = found(tr[-1]).mean()
        first90 = next((s for s in range(len(tr)) if found(tr[s]).mean() >= 0.9 * fr), steps)
        info.append(dict(label=label, beta1=b1, lr=lr, found=float(fr), settled_step=int(first90)))
        print(f"{label} lr {lr}: found {fr:.3f}, 90% of final by step {first90}")
    pos = np.empty((N_FRAMES, N, 2), np.float32)
    for f in range(N_FRAMES):
        if f < RESETS[0][0]:
            pos[f] = at_step(trajs[0], frame_step(f, RUNS[0][4], RUNS[0][5], RUNS[0][3], RUNS[0][6]))
        elif f < RESETS[0][1]:
            u = (f - RESETS[0][0]) / (RESETS[0][1] - RESETS[0][0])
            u = u * u * (3 - 2 * u)
            pos[f] = trajs[0][-1] * (1 - u) + P0 * u
        elif f < LOOP[0]:
            pos[f] = at_step(trajs[1], frame_step(f, RUNS[1][4], RUNS[1][5], RUNS[1][3], RUNS[1][6]))
        elif f < LOOP[1]:
            u = (f - LOOP[0]) / (LOOP[1] - LOOP[0])
            u = u * u * (3 - 2 * u)
            pos[f] = trajs[1][-1] * (1 - u) + P0 * u
        else:
            pos[f] = P0
    found_a, found_b = found(trajs[0][-1]), found(trajs[1][-1])
    np.savez(out_path, pos=pos, P0=P0.astype(np.float32), found_a=found_a, found_b=found_b)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(runs=info, lr=LR), fh)


if __name__ == "__main__":
    main(sys.argv[1])
