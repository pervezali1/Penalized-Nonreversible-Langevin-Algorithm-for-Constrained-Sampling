"""Three gradient-descent runs on the descent reel's landscape with different step sizes:
just right, 150x bigger (cannot settle), 1000x bigger (explodes). Positions per video frame."""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "descent"))
from landscape import F_MIN, N, SEED, at_step, found, frame_step, grad, loss  # noqa: E402

FPS = 60
N_FRAMES = 1200
# (label, learning rate, steps, first frame, last frame, pacing exponent)
RUNS = [("0.002", 0.002, 450, 72, 372, 1.2),
        ("0.3", 0.3, 300, 408, 708, 1.0),
        ("1.7", 1.7, 70, 744, 1000, 1.0)]
RESETS = [(378, 402), (714, 738)]
LOOP = (1020, 1080)          # exploded particles fly back to the starts during the payoff
CLIP = 1.6                   # positions are clipped here for the fly-back only


def run(P0, lr, steps):
    P = P0.copy()
    out = [P.copy()]
    for _ in range(steps):
        P = P - lr * grad(P)
        out.append(P.copy())
    return np.stack(out)


def main(out_path):
    rng = np.random.default_rng(SEED)
    P0 = rng.uniform(-1, 1, (N, 2))
    trajs = [run(P0, lr, steps) for (_, lr, steps, *_r) in RUNS]
    pos = np.empty((N_FRAMES, N, 2), np.float32)
    mean_loss = np.empty(N_FRAMES)
    info = []
    for (label, lr, steps, f0, f1, expo), tr in zip(RUNS, trajs):
        end = tr[-1]
        info.append(dict(label=label, lr=lr, found=float(found(end).mean()),
                         mean_loss=float(np.mean(loss(end) - F_MIN)), max_abs=float(np.abs(end).max())))
        print(f"lr {label}: found {info[-1]['found']:.3f}, mean loss {info[-1]['mean_loss']:.3g}, max |P| {info[-1]['max_abs']:.3g}")
    prev_end = None
    for f in range(N_FRAMES):
        placed = False
        for k, ((label, lr, steps, f0, f1, expo), tr) in enumerate(zip(RUNS, trajs)):
            r0 = RESETS[k - 1][1] if k > 0 else 0
            r1 = RESETS[k][0] if k < len(RESETS) else LOOP[0]
            if r0 <= f < r1:
                pos[f] = at_step(tr, frame_step(f, f0, f1, steps, expo))
                placed = True
                break
        if not placed:
            for k, (a, b) in enumerate(RESETS):
                if a <= f < b:
                    u = (f - a) / (b - a)
                    u = u * u * (3 - 2 * u)
                    pos[f] = trajs[k][-1] * (1 - u) + P0 * u
                    placed = True
        if not placed:
            if LOOP[0] <= f < LOOP[1]:
                u = (f - LOOP[0]) / (LOOP[1] - LOOP[0])
                u = u * u * (3 - 2 * u)
                start = np.clip(trajs[-1][-1], -CLIP, CLIP)
                pos[f] = start * (1 - u) + P0 * u
            else:
                pos[f] = P0
        mean_loss[f] = float(np.mean(loss(pos[f].astype(np.float64)) - F_MIN))
    np.savez(out_path, pos=pos, P0=P0.astype(np.float32), mean_loss=mean_loss)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(runs=info), fh)


if __name__ == "__main__":
    main(sys.argv[1])
