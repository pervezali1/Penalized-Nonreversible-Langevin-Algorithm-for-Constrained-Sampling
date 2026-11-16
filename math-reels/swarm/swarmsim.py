"""10,000 AIs, part 2: this time they talk. The landscape, the 10,000 random starts and the first run are exactly reel
04's (descent/landscape.py): plain gradient descent, each particle alone, 34% reach the best answer. The second run
starts from the same 10,000 points but the particles share what they find: particle swarm optimisation (Kennedy and
Eberhart 1995; inertia 0.72 and pulls 1.49 from Clerc's constriction), each particle steering toward the best spot it
has found itself and the best spot any of them has found, speed capped at 0.05 per step. They never use the slope,
only the heights they land on. Positions are recorded at every video frame.
    python swarmsim.py <out.npz>  -> out.npz (positions) and out.json (facts)"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "descent"))
from landscape import EXPO_A, F_MIN, GLOBAL, LOOP, N, N_FRAMES, RESET, RUN_A, RUN_B, SEED, at_step, found, frame_step, loss, run  # noqa: E402,F401

FPS = 60
PSO_STEPS = 80
W_IN, C1, C2, VMAX = 0.72, 1.49, 1.49, 0.05
PSO_SEED = 64
EXPO_B = 1.6


def swarm(P0):
    rng = np.random.default_rng(PSO_SEED)
    X, V = P0.copy(), np.zeros_like(P0)
    pb, pf = X.copy(), loss(X)
    traj, gbest = [X.copy()], []
    for _ in range(PSO_STEPS):
        g = pb[np.argmin(pf)]
        gbest.append(g.copy())
        V = W_IN * V + C1 * rng.random((len(X), 1)) * (pb - X) + C2 * rng.random((len(X), 1)) * (g - X)
        sp = np.linalg.norm(V, axis=1, keepdims=True)
        V = np.where(sp > VMAX, V / np.maximum(sp, 1e-12) * VMAX, V)
        X = X + V
        traj.append(X.copy())
        f = loss(X)
        m = f < pf
        pb[m], pf[m] = X[m], f[m]
    gbest.append(pb[np.argmin(pf)].copy())
    return np.stack(traj), np.stack(gbest)


def main(out_path):
    rng = np.random.default_rng(SEED)
    P0 = rng.uniform(-1, 1, (N, 2))
    traj_a = run(P0, 0.0)
    traj_b, gbest = swarm(P0)
    frac_b = np.array([found(x).mean() for x in traj_b])
    fa, fb = float(found(traj_a[-1]).mean()), float(frac_b[-1])
    starter = int(np.argmin(loss(P0)))
    print(f"alone (plain GD) {fa:.3f}, talking {fb:.3f}; 50% at step {int(np.argmax(frac_b >= 0.5))}, 90% at {int(np.argmax(frac_b >= 0.9))}, "
          f"all at {int(np.argmax(frac_b >= 1.0))}; max |P| {np.abs(traj_b).max():.3f}; starter {starter} at {P0[starter].round(3)}")
    pos = np.empty((N_FRAMES, N, 2), np.float32)
    gb = np.zeros((N_FRAMES, 2), np.float32)
    for f in range(N_FRAMES):
        if f < RESET[0]:
            pos[f] = at_step(traj_a, frame_step(f, *RUN_A, expo=EXPO_A))
        elif f < RESET[1]:
            u = (f - RESET[0]) / (RESET[1] - RESET[0])
            u = u * u * (3 - 2 * u)
            pos[f] = traj_a[-1] * (1 - u) + P0 * u
        elif f < LOOP[0]:
            s = frame_step(f, *RUN_B, steps=PSO_STEPS, expo=EXPO_B)
            pos[f] = at_step(traj_b, s)
            gb[f] = gbest[min(int(s), PSO_STEPS)]
        elif f < LOOP[1]:
            u = (f - LOOP[0]) / (LOOP[1] - LOOP[0])
            u = u * u * (3 - 2 * u)
            pos[f] = traj_b[-1] * (1 - u) + P0 * u
        else:
            pos[f] = P0
    np.savez(out_path, pos=pos, P0=P0.astype(np.float32), found_a=found(traj_a[-1]), found_b=found(traj_b[-1]), gbest=gb)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(found_a=fa, found_b=fb, f_min=F_MIN, pso_steps=PSO_STEPS, w=W_IN, c1=C1, c2=C2, vmax=VMAX,
                       step_50=int(np.argmax(frac_b >= 0.5)), step_90=int(np.argmax(frac_b >= 0.9)), step_all=int(np.argmax(frac_b >= 1.0)),
                       found_curve_b=frac_b.tolist(), starter=starter, starter_pos=P0[starter].tolist(),
                       found_at_start=float(found(P0).mean()), gbest_final=gbest[-1].tolist()), fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
