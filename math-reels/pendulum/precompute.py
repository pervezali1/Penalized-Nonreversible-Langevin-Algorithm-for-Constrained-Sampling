"""Per-frame data derived from the simulation: tips, HUD values, lens schedule, CHAOS frame."""

import json
import sys
import time

import numpy as np

import timemap as tm
from raster import tips_of
from sim import DT, N, state_at


def main(sim_path, out_path):
    sim = np.load(sim_path)
    snaps, M, E = sim["snaps"], sim["M"], sim["E"]
    dt, snap_every = float(sim["dt"]), int(sim["snap_every"])

    def M_at(d):
        return M[min(len(M) - 1, int(round(d / dt)))]

    def E_at(d):
        return E[min(len(E) - 1, int(round(d / dt)))]

    # lens change points: first s at which magnification 10^k would overflow 252 px
    change_points = []
    for k in range(7, 0, -1):
        idx = int(np.argmax(E >= 252.0 / 10 ** k))
        change_points.append(idx * dt)
    assert all(np.diff(change_points) > 0.2), change_points

    F = tm.N_FRAMES
    d = np.array([tm.d_of_frame(f) for f in range(F)])
    spd = np.array([tm.speed(f / tm.FPS) for f in range(F)])
    tips = np.empty((F, N, 2), np.float32)
    lens_pts = np.empty((F, 1001, 2), np.float64)
    mean_tip = np.empty((F, 2))
    gap_str, lens_k, lens_tween, Mv, Ev = [], [], [], [], []
    t0 = time.time()
    for f in range(F):
        st = state_at(snaps, d[f], dt, snap_every)
        tp = tips_of(st)
        tips[f] = tp
        mean_tip[f] = tp.mean(axis=0)
        centre = (tp.min(axis=0) + tp.max(axis=0)) / 2
        sel = np.r_[np.arange(0, N, 10), N - 1]
        lens_pts[f] = tp[sel] - centre
        Mv.append(float(M_at(d[f])))
        Ev.append(float(E_at(d[f])))
        gap_str.append(tm.gap_display(Mv[-1]))
        lk, tw = tm.lens_log_mag(d[f], change_points)
        lens_k.append(lk)
        lens_tween.append(tw)
        if f % 200 == 0:
            print(f"  frame {f}  d={d[f]:.4f}  M={Mv[-1]:.3g}  {gap_str[-1]}  lens={lk:.2f}", flush=True)
    print(f"states in {time.time() - t0:.1f}s")

    Mv = np.array(Mv)
    chaos = int(np.argmax(Mv >= 180))
    assert Mv[chaos] >= 180
    t180 = int(np.argmax(M >= 180)) * dt
    assert t180 < 8.3, t180
    assert M[int(round(t180 / dt))] / M[0] >= 1.8e8
    # frames since the counter string last changed (looping)
    since = np.zeros(F, int)
    for f in range(F):
        k = 0
        while k < F and gap_str[(f - k - 1) % F] == gap_str[f]:
            k += 1
        since[f] = k
    meta = dict(chaos_frame=chaos, t_chaos=chaos / tm.FPS, t180=t180, M0=float(M[0]),
                change_points=change_points, gap_str=gap_str, lens_k=lens_k, lens_tween=lens_tween,
                doubling=t180 / np.log2(M[int(round(t180 / dt))] / M[0]),
                first_fringe_frame=int(np.argmax(np.array(Ev) >= 1.0)))
    print(f"CHAOS frame {chaos} (t={chaos / 60:.4f}, timer {d[chaos]:.2f} s), T180={t180:.4f}, "
          f"doubling {meta['doubling']:.3f} s, fringe frame {meta['first_fringe_frame']}")
    print("lens change points", [round(c, 4) for c in change_points])
    np.savez(out_path, d=d, speed=spd, tips=tips, lens_pts=lens_pts, mean_tip=mean_tip, M=Mv,
             E=np.array(Ev), since=since)
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
