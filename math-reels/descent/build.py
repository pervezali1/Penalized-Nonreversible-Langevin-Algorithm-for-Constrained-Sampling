"""One-shot build for the descent reel: simulate -> audio -> video -> cover + QA stills.

    python build.py <workdir> [--force]
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

QA_FRAMES = [0, 90, 150, 230, 330, 420, 480, 560, 600, 640, 700, 800, 900, 1020, 1100, 1180, 1199]


def main():
    work = os.path.abspath(sys.argv[1])
    force = "--force" in sys.argv
    os.makedirs(work, exist_ok=True)
    data_p, wav_p, mp4 = f"{work}/descent.npz", f"{work}/audio.wav", f"{work}/descent_reel.mp4"
    if force or not os.path.exists(data_p):
        subprocess.run([sys.executable, f"{HERE}/landscape.py", data_p], check=True)
    if force or not os.path.exists(wav_p):
        subprocess.run([sys.executable, f"{HERE}/audio.py", data_p, wav_p, f"{work}/events.json"], check=True)

    from landscape import N_FRAMES
    from scene import Scene
    scene = Scene(data_p)
    print(f"plain GD {scene.pct_a}%, momentum {scene.pct_b}%")
    t0 = time.time()
    render_video(scene.frame, N_FRAMES, mp4, fps=60, crf=16, preset="slow", audio_path=wav_p)
    print(f"video in {time.time() - t0:.0f}s -> {mp4}")

    qa = f"{work}/qa"
    os.makedirs(qa, exist_ok=True)
    for f in QA_FRAMES:
        save_png(scene.frame(f), f"{qa}/f{f:04d}.png")
    save_png(scene.frame(1020), f"{work}/cover.png")
    a, b, c = (scene.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    seam, step = np.abs(b - a).mean(), np.abs(c - b).mean()
    print(f"seam diff {seam:.3f} vs ordinary step {step:.3f}")
    ims = [cv2.resize(cv2.imread(f"{qa}/f{f:04d}.png"), (360, 640)) for f in QA_FRAMES]
    ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{work}/contact_sheet.png", np.vstack(rows))
    with open(f"{work}/build_info.json", "w") as fh:
        json.dump(dict(found_plain=scene.meta["found_a"], found_momentum=scene.meta["found_b"],
                       seam=float(seam), step=float(step)), fh, indent=1)


if __name__ == "__main__":
    main()
