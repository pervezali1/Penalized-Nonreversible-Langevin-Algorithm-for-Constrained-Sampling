"""Frame renderer for the Adam-momentum reel: Adam with beta1 = 0 vs beta1 = 0.9 (reuses the descent
landscape, splats and HUD style)."""

import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
sys.path.insert(0, os.path.join(ROOT, "descent"))
from textkit import (H, W, cells_width, dither, font, layer, linear_to_srgb, over, rgb, show_cells,  # noqa: E402
                     show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)
from raster import splat_trails  # noqa: E402
from landscape import F_MIN, GLOBAL, found, frame_step, loss  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("descent_scene", os.path.join(ROOT, "descent", "scene.py"))
descent_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(descent_scene)
from amsim import FPS, LOOP, LR, N_FRAMES, RESETS, RUNS  # noqa: E402

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
WEIGHT = 0.05
to_px = descent_scene.to_px
splat_points = descent_scene.splat_points


class Scene(descent_scene.Scene):
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.pos, self.P0 = d["pos"], d["P0"]
        self.found_a, self.found_b = d["found_a"], d["found_b"]
        self.pct_a = int(round(100 * self.meta["runs"][0]["found"]))
        self.pct_b = int(round(100 * self.meta["runs"][1]["found"]))
        self.settled_b = self.meta["runs"][1]["settled_step"]
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_amber = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.bg = self._landscape()
        self.cues = self._cues()

    def _cues(self):
        hook = [("Same 10,000 AIs. Now Adam.", "dbold", 50, WHITE, 292, 0.85), ("Momentum: OFF", "black", 88, AMBER, 384, 1)]
        return [
            (0.0, 1.8, hook, 0, 6, "none"),
            (2.0, 3.8, [("Adam without momentum.", "black", 68, WHITE, 312, 1), ("Each step sized by the slope alone.", "dbold", 44, AMBER, 396, 1)], 8, 6, "pop"),
            (4.6, 6.9, [("It still stalls in the traps.", "black", 64, WHITE, 312, 1), (f"{self.pct_a}% find the best answer.", "dbold", 50, MAGENTA, 396, 1)], 8, 6, "pop"),
            (7.9, 9.6, [("Momentum: ON", "black", 88, CYAN, 320, 1), ("β₁ = 0.9, Adam's default.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (10.4, 12.6, [("It coasts through the dips.", "black", 68, WHITE, 350, 1)], 8, 6, "pop"),
            (13.0, 15.0, [(f"{self.pct_b}%", "black", 150, CYAN, 330, 1), (f"found the best answer by step {self.settled_b}.", "dbold", 46, WHITE, 404, 1)], 6, 6, "chaos"),
            (15.3, 17.4, [("Adam's momentum is the same idea:", "black", 54, WHITE, 312, 1), ("remember where you were heading.", "dbold", 46, AMBER, 396, 1)], 8, 6, "pop"),
            (17.6, 19.3, [("Last reel, plain descent + momentum: 85%.", "dbold", 44, WHITE, 306, 0.9), ("Momentum is the ingredient.", "black", 64, AMBER, 396, 1)], 8, 6, "pop"),
            (19.45, 20.2, hook, 10, 0, "fade"),
        ]

    def _run_index(self, f):
        if f < RESETS[0][0]:
            return 0
        if RESETS[0][1] <= f < LOOP[0]:
            return 1
        return None

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        P = self.pos[f]
        k = self._run_index(f)
        in_reset = any(a <= f < b for a, b in RESETS) or LOOP[0] <= f < LOOP[1]
        revealed = (k == 1 and t >= 13.0) or (k == 0 and 4.6 <= t < RESETS[0][0] / FPS)
        if revealed:
            fl = self.found_b if k == 1 else self.found_a
            cols = np.where(fl[:, None], self.c_cyan, self.c_mag)
        else:
            u = np.clip((loss(P.astype(np.float64)) - F_MIN) / 1.6, 0, 1) ** 0.6
            cols = self.c_cyan * (1 - u[:, None]) + self.c_amber * u[:, None]
        if in_reset:
            cols = cols * 0.5 + np.array([0.4, 0.4, 0.45], np.float32) * 0.5
        cols = cols.astype(np.float32)
        trail, wmul = (10, 1.0) if k == 1 else (3, 0.35)   # hook, run 1, resets and loop share one weight so the seam matches
        acc = np.zeros((H, W, 3), np.float32)
        pts = to_px(P).astype(np.float64)
        idx = [(f - i) % N_FRAMES for i in range(trail + 1)]
        trails = to_px(self.pos[idx]).astype(np.float32)
        if RUNS[0][4] <= f < LOOP[0]:        # no trails in the hook frames or during the loop fly-back
            splat_trails(acc, trails, cols, WEIGHT * wmul, 1.0, 0.0, 0.0)
        splat_points(acc, pts, cols, WEIGHT * 2.5 * max(wmul, 0.6), 1.7)
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud_adam(img, f, t, P, k, in_reset)
        return dither(img, f)

    def _hud_adam(self, img, f, t, P, k, in_reset):
        surf, ctx = layer()
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, "%": 60})
        if k is None or in_reset:
            step = 0
            label = RUNS[0][0] if (f < RUNS[0][4] or f >= LOOP[0]) else RUNS[k if k is not None else 0][0]
            pct = int(round(100 * found(P).mean()))
        else:
            lab, b1, lr, steps, f0, f1, expo = RUNS[k]
            step, label = int(frame_step(f, f0, f1, steps, expo)), lab
            pct = int(round(100 * found(P).mean()))
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "FOUND THE BEST ANSWER", 0.14 * 24)
        show_tracked(ctx, "FOUND THE BEST ANSWER", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{step:,}", 70, 545, big)
        s = f"{pct}%"
        ctx.set_source_rgb(*(rgb(CYAN) if pct >= 60 else rgb(MAGENTA) if pct >= 25 else rgb(AMBER)))
        tw = cells_width(ctx, s, big)
        show_cells(ctx, s, 910 - tw, 545, big)
        text = f"ADAM  ·  {label}  ·  STEP {LR}"
        font(ctx, "bold", 26)
        lw = tracked_width(ctx, text, 0.08 * 26)
        x0, y0, bw, bh, r = CX - lw / 2 - 18, 586, lw + 36, 44, 22
        ctx.new_sub_path()
        ctx.arc(x0 + bw - r, y0 + r, r, -math.pi / 2, 0)
        ctx.arc(x0 + bw - r, y0 + bh - r, r, 0, math.pi / 2)
        ctx.arc(x0 + r, y0 + bh - r, r, math.pi / 2, math.pi)
        ctx.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        pc = rgb(CYAN) if label == "MOMENTUM ON" else rgb(AMBER)
        ctx.set_source_rgba(*pc, 0.14)
        ctx.fill()
        ctx.set_source_rgba(*pc, 0.9)
        show_tracked(ctx, text, CX - lw / 2, y0 + 31, 0.08 * 26)
        gx, gy = to_px(np.array(GLOBAL))
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(*rgb(CYAN), 0.5)
        ctx.arc(gx, gy, 14, 0, 2 * math.pi)
        ctx.stroke()
        font(ctx, "bold", 20)
        ctx.set_source_rgba(*rgb(CYAN), 0.7)
        s = "BEST ANSWER"
        show_tracked(ctx, s, gx - tracked_width(ctx, s, 0.12 * 20) / 2, gy + 44, 0.12 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)
