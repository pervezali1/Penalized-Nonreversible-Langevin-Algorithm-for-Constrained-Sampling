"""One-shot build: simulate -> precompute -> audio -> video -> cover + QA stills.

    python build.py <workdir> [--force]

Steps whose outputs already exist in <workdir> are skipped unless --force is given.
"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reelkit import render_video, save_png  # noqa: E402

QA_FRAMES = [0, 30, 79, 130, 250, 352, 384, 440, 485, 504, 560, 615, 650, 760, 900, 1000, 1042, 1079]


def main():
    work = os.path.abspath(sys.argv[1])
    force = "--force" in sys.argv
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    sim_p, fr_p, wav_p = f"{work}/sim.npz", f"{work}/frames.npz", f"{work}/audio.wav"
    mp4 = f"{work}/pendulums_reel.mp4"

    def need(p):
        return force or not os.path.exists(p)

    if need(sim_p):
        subprocess.run([py, f"{HERE}/sim.py", sim_p], check=True)
    if need(fr_p):
        subprocess.run([py, f"{HERE}/precompute.py", sim_p, fr_p], check=True)
    if need(wav_p):
        subprocess.run([py, f"{HERE}/audio.py", fr_p, wav_p, f"{work}/events.json"], check=True)

    from scene import Scene
    scene = Scene(sim_p, fr_p)
    meta = scene.meta
    print(f"CHAOS frame {scene.f_chaos}  T180 {meta['t180']:.4f}s  doubling {meta['doubling']:.3f}s")

    t0 = time.time()
    render_video(scene.frame, 1080, mp4, fps=60, crf=16, preset="slow", audio_path=wav_p)
    print(f"video in {time.time() - t0:.0f}s -> {mp4}")

    qa = f"{work}/qa"
    os.makedirs(qa, exist_ok=True)
    for f in QA_FRAMES:
        save_png(scene.frame(f), f"{qa}/f{f:04d}.png")
    cover = scene.frame(504)
    save_png(cover, f"{work}/cover.png")

    # loop-seam check: the 1079 -> 0 step must look like any other step
    a, b, c = (scene.frame(i).astype(np.float32) for i in (1079, 0, 1))
    seam, step = np.abs(b - a).mean(), np.abs(c - b).mean()
    print(f"seam diff {seam:.3f} vs ordinary step {step:.3f} (ratio {seam / step:.2f})")
    assert seam / step < 1.5, "loop seam is visible"

    ims = [cv2.resize(cv2.imread(f"{qa}/f{f:04d}.png"), (360, 640)) for f in QA_FRAMES]
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]
    cv2.imwrite(f"{work}/contact_sheet.png", np.vstack(rows))
    with open(f"{work}/build_info.json", "w") as fh:
        json.dump(dict(chaos_frame=scene.f_chaos, t180=meta["t180"], doubling=meta["doubling"],
                       change_points=meta["change_points"], seam_ratio=float(seam / step)), fh, indent=1)


if __name__ == "__main__":
    main()
