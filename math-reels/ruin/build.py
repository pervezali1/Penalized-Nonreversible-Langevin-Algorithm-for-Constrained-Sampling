"""One-shot build for the gambler's ruin reel.    python build.py <workdir> [--force]"""

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

QA_FRAMES = [0, 150, 320, 380, 450, 520, 600, 680, 720, 800, 880, 900, 960, 1050, 1150, 1250, 1319]


def main():
    work = os.path.abspath(sys.argv[1])
    force = "--force" in sys.argv
    os.makedirs(work, exist_ok=True)
    wav_p, mp4 = f"{work}/audio.wav", f"{work}/ruin_reel.mp4"
    if force or not os.path.exists(wav_p):
        subprocess.run([sys.executable, f"{HERE}/audio.py", wav_p, f"{work}/events.json"], check=True)
    from scene import N_FRAMES, Scene
    scene = Scene()
    print(f"slow: broke {scene.n_broke_final} doubled {scene.n_dbl_final}; all-in doubled {scene.n_allin}; P(slow) = {scene.p_slow:.3e}")
    t0 = time.time()
    render_video(scene.frame, N_FRAMES, mp4, fps=60, crf=16, preset="slow", audio_path=wav_p)
    print(f"video in {time.time() - t0:.0f}s -> {mp4}")
    qa = f"{work}/qa"
    os.makedirs(qa, exist_ok=True)
    for f in QA_FRAMES:
        save_png(scene.frame(f), f"{qa}/f{f:04d}.png")
    save_png(scene.frame(690), f"{work}/cover.png")
    a, b, c = (scene.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    seam, step = np.abs(b - a).mean(), np.abs(c - b).mean()
    print(f"seam diff {seam:.3f} vs ordinary step {step:.3f}")
    ims = [cv2.resize(cv2.imread(f"{qa}/f{f:04d}.png"), (360, 640)) for f in QA_FRAMES]
    ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{work}/contact_sheet.png", np.vstack(rows))
    with open(f"{work}/build_info.json", "w") as fh:
        json.dump(dict(broke=scene.n_broke_final, doubled=scene.n_dbl_final, allin=scene.n_allin,
                       p_slow=scene.p_slow, seam=float(seam), step=float(step)), fh, indent=1)


if __name__ == "__main__":
    main()
