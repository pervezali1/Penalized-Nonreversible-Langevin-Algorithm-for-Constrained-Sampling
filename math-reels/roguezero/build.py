"""One-shot build for the rogue-zero reel: zeros (../riemann/rsim.py) -> what-if numbers -> audio -> video -> stills.
    python build.py <workdir> [--force]"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from reelkit import render_video, save_png  # noqa: E402

QA_FRAMES = [0, 150, 170, 240, 380, 520, 560, 700, 830, 900, 1020, 1150, 1260, 1440, 1620, 1740, 1799]
COVER_FRAME = 1260


def main():
    work = os.path.abspath(sys.argv[1])
    force = "--force" in sys.argv
    os.makedirs(work, exist_ok=True)
    zeros, rogue = f"{work}/riemann.npz", f"{work}/rogue.npz"
    wav_p, mp4 = f"{work}/audio.wav", f"{work}/roguezero_reel.mp4"
    if force or not os.path.exists(zeros):
        subprocess.run([sys.executable, f"{ROOT}/riemann/rsim.py", zeros], check=True)
    if force or not os.path.exists(rogue):
        subprocess.run([sys.executable, f"{HERE}/wsim.py", zeros, rogue], check=True)
    if force or not os.path.exists(wav_p):
        subprocess.run([sys.executable, f"{HERE}/audio.py", wav_p, f"{work}/events.json"], check=True)
    from scene import N_FRAMES, Scene
    scene = Scene(zeros, rogue)
    print("real primes stray by", scene.real_max, "| rogue wave at a billion", scene.rogue_1e9)
    t0 = time.time()
    render_video(scene.frame, N_FRAMES, mp4, fps=60, crf=16, preset="slow", audio_path=wav_p)
    print(f"video in {time.time() - t0:.0f}s -> {mp4}")
    qa = f"{work}/qa"
    os.makedirs(qa, exist_ok=True)
    for f in QA_FRAMES:
        save_png(scene.frame(f), f"{qa}/f{f:04d}.png")
    save_png(scene.frame(COVER_FRAME), f"{work}/cover.png")
    a, b, c = (scene.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    seam, step = np.abs(b - a).mean(), np.abs(c - b).mean()
    print(f"seam diff {seam:.3f} vs ordinary step {step:.3f}")
    ims = [cv2.resize(cv2.imread(f"{qa}/f{f:04d}.png"), (360, 640)) for f in QA_FRAMES]
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]
    cv2.imwrite(f"{work}/contact_sheet.png", np.vstack(rows))
    with open(f"{work}/build_info.json", "w") as fh:
        json.dump(dict(real_max=scene.real_max, rogue_1e9=scene.rogue_1e9, seam=float(seam), step=float(step)), fh, indent=1)


if __name__ == "__main__":
    main()
