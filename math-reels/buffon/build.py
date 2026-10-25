"""One-shot build for the Buffon needle reel.    python build.py <workdir> [--force]"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from reelkit import render_video, save_png  # noqa: E402

QA_FRAMES = [0, 170, 220, 280, 340, 420, 500, 580, 660, 760, 800, 880, 960, 1010, 1080, 1150, 1260, 1380]


def main():
    work = os.path.abspath(sys.argv[1])
    force = "--force" in sys.argv
    os.makedirs(work, exist_ok=True)
    wav_p, mp4 = f"{work}/audio.wav", f"{work}/buffon_reel.mp4"
    if force or not os.path.exists(wav_p):
        subprocess.run([sys.executable, f"{HERE}/audio.py", wav_p, f"{work}/events.json"], check=True)
    from scene import N_FRAMES, Scene
    scene = Scene()
    print(f"100k needles: pi ~ {scene.est_final:.6f} ({scene.digits_final} digits); Lazzarini repeats hitting 1,808: {scene.laz_hits}/1000")
    t0 = time.time()
    render_video(scene.frame, N_FRAMES, mp4, fps=60, crf=16, preset="slow", audio_path=wav_p)
    print(f"video in {time.time() - t0:.0f}s -> {mp4}")
    qa = f"{work}/qa"
    os.makedirs(qa, exist_ok=True)
    for f in QA_FRAMES:
        save_png(scene.frame(f), f"{qa}/f{f:04d}.png")
    save_png(scene.frame(1400), f"{work}/cover.png")
    a, b, c = (scene.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    seam, step = np.abs(b - a).mean(), np.abs(c - b).mean()
    print(f"seam diff {seam:.3f} vs ordinary step {step:.3f}")
    ims = [cv2.resize(cv2.imread(f"{qa}/f{f:04d}.png"), (360, 640)) for f in QA_FRAMES]
    ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{work}/contact_sheet.png", np.vstack(rows))
    with open(f"{work}/build_info.json", "w") as fh:
        json.dump(dict(est=scene.est_final, digits=scene.digits_final, laz_hits=scene.laz_hits, laz_est=scene.laz_est,
                       laz_lo=scene.laz_lo, laz_hi=scene.laz_hi, seam=float(seam), step=float(step)), fh, indent=1)


if __name__ == "__main__":
    main()
