"""A 2-D loss landscape with one deep well and eight shallow traps, and gradient descent of
10,000 particles from the same random starts, plain and with momentum. Positions are
recorded at every video frame."""

import json
import math
import sys

import numpy as np

FPS = 60
N_FRAMES = 1200
N = 10_000
SEED = 3
LR, MOMENTUM, STEPS = 0.002, 0.97, 450
GLOBAL = (0.45, -0.30)
WELLS = [(GLOBAL, 1.2, 0.25)] + [(c, 0.3, 0.12) for c in
                                 [(-0.50, 0.40), (-0.40, -0.55), (0.30, 0.60), (0.0, 0.05),
                                  (-0.05, -0.85), (0.85, 0.35), (-0.85, -0.10), (0.60, 0.90)]]
BOWL = 0.6
FOUND_MARGIN = 0.25

# runs: (first frame, last frame) of each descent; reveals and resets live in scene.py
RUN_A = (60, 450)        # plain gradient descent, 1.0-7.5 s
RUN_B = (624, 984)       # with momentum, 10.4-16.4 s
RESET = (588, 618)       # fly back to the starts, 9.8-10.3 s
LOOP = (1164, 1197)      # fly back to the starts for the loop, 19.4-19.95 s
EXPO_A, EXPO_B = 1.2, 1.5  # step pacing per run (>1 spends fewer steps early so the motion lasts)


def loss(P):
    x, y = P[..., 0], P[..., 1]
    v = BOWL * (x * x + y * y)
    for (cx, cy), A, s in WELLS:
        v = v - A * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return v


def grad(P):
    x, y = P[..., 0], P[..., 1]
    gx, gy = 2 * BOWL * x, 2 * BOWL * y
    for (cx, cy), A, s in WELLS:
        e = A * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        gx = gx + e * (x - cx) / (s * s)
        gy = gy + e * (y - cy) / (s * s)
    return np.stack([gx, gy], -1)


F_MIN = float(loss(np.array(GLOBAL)))


def found(P):
    return loss(P) < F_MIN + FOUND_MARGIN


def frame_step(f, f0, f1, steps=STEPS, expo=1.0):
    """Fractional step index for a frame (positions are interpolated between steps)."""
    if f < f0:
        return 0.0
    if f >= f1:
        return float(steps)
    return steps * ((f - f0) / (f1 - f0)) ** expo


def at_step(traj, s):
    i = int(math.floor(s))
    if i >= len(traj) - 1:
        return traj[-1]
    u = s - i
    return traj[i] * (1 - u) + traj[i + 1] * u


def run(P0, momentum):
    P, V = P0.copy(), np.zeros_like(P0)
    out = [P.copy()]
    for _ in range(STEPS):
        V = momentum * V - LR * grad(P)
        P = P + V
        out.append(P.copy())
    return np.stack(out)


def main(out_path):
    rng = np.random.default_rng(SEED)
    P0 = rng.uniform(-1, 1, (N, 2))
    traj_a = run(P0, 0.0)
    traj_b = run(P0, MOMENTUM)
    fa, fb = found(traj_a[-1]).mean(), found(traj_b[-1]).mean()
    print(f"plain GD found {fa:.3f}, momentum found {fb:.3f}, max |P| {max(np.abs(traj_a).max(), np.abs(traj_b).max()):.2f}")
    pos = np.empty((N_FRAMES, N, 2), np.float32)
    for f in range(N_FRAMES):
        if f < RESET[0]:
            pos[f] = at_step(traj_a, frame_step(f, *RUN_A, expo=EXPO_A))
        elif f < RESET[1]:
            u = (f - RESET[0]) / (RESET[1] - RESET[0])
            u = u * u * (3 - 2 * u)
            pos[f] = traj_a[-1] * (1 - u) + P0 * u
        elif f < LOOP[0]:
            pos[f] = at_step(traj_b, frame_step(f, *RUN_B, expo=EXPO_B))
        elif f < LOOP[1]:
            u = (f - LOOP[0]) / (LOOP[1] - LOOP[0])
            u = u * u * (3 - 2 * u)
            pos[f] = traj_b[-1] * (1 - u) + P0 * u
        else:
            pos[f] = P0
    found_a = found(traj_a[-1])
    found_b = found(traj_b[-1])
    np.savez(out_path, pos=pos, P0=P0.astype(np.float32), found_a=found_a, found_b=found_b)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(found_a=float(fa), found_b=float(fb), f_min=F_MIN), fh)


if __name__ == "__main__":
    main(sys.argv[1])
