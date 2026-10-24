"""Frame renderer for the learning-rate reel (reuses the descent landscape and splats)."""

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
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from raster import splat_trails  # noqa: E402
from landscape import F_MIN, GLOBAL, frame_step, loss  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("descent_scene", os.path.join(ROOT, "descent", "scene.py"))
descent_scene = importlib.util.module_from_spec(_spec)   # the descent reel's renderer: landscape + point splat
_spec.loader.exec_module(descent_scene)
from lrsim import FPS, LOOP, N_FRAMES, RESETS, RUNS  # noqa: E402

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
TRAIL = 10
WEIGHT = 0.05
MAX_TEXT_W = 820
to_px = descent_scene.to_px
splat_points = descent_scene.splat_points


class Scene(descent_scene.Scene):
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.pos, self.P0, self.mean_loss = d["pos"], d["P0"], d["mean_loss"]
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_amber = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.bg = self._landscape()
        self.cues = self._cues()

    def _cues(self):
        pa = int(round(100 * self.meta["runs"][0]["found"]))
        pb = int(round(100 * self.meta["runs"][1]["found"]))
        hook = [("Teaching an AI:", "dbold", 54, WHITE, 292, 0.85), ("how big a step?", "black", 88, WHITE, 384, 1)]
        return [
            (0.0, 1.8, hook, 0, 6, "none"),
            (2.0, 3.6, [("Step size: 0.002", "black", 80, CYAN, 350, 1)], 8, 6, "pop"),
            (4.4, 6.2, [("It settles. It learns.", "black", 70, WHITE, 312, 1), (f"{pa}% find the best answer.", "dbold", 50, CYAN, 396, 1)], 8, 6, "pop"),
            (7.0, 8.6, [("Step size: 0.3", "black", 80, AMBER, 312, 1), ("150× bigger.", "dbold", 54, WHITE, 396, 0.9)], 8, 6, "pop"),
            (9.4, 11.7, [("It overshoots every answer.", "black", 64, WHITE, 312, 1), (f"Can't settle. {pb}%.", "black", 72, MAGENTA, 400, 1)], 8, 6, "pop"),
            (12.6, 13.9, [("Step size: 1.7", "black", 80, MAGENTA, 312, 1), ("850× bigger.", "dbold", 54, WHITE, 396, 0.9)], 8, 6, "pop"),
            (14.1, 16.6, [("EXPLODED.", "black", 150, MAGENTA, 385, 1)], 6, 6, "chaos"),
            (16.8, 19.3, [("That dial is the learning rate.", "black", 64, WHITE, 312, 1), ("Too small: slow. Too big: chaos.", "dbold", 50, AMBER, 396, 1)], 8, 6, "pop"),
            (19.45, 20.2, hook, 10, 0, "fade"),
        ]

    def _run_index(self, f):
        for k, (label, lr, steps, f0, f1, expo) in enumerate(RUNS):
            r0 = RESETS[k - 1][1] if k > 0 else 0
            r1 = RESETS[k][0] if k < len(RESETS) else LOOP[0]
            if r0 <= f < r1:
                return k
        return None

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        P = self.pos[f]
        k = self._run_index(f)
        in_reset = any(a <= f < b for a, b in RESETS) or LOOP[0] <= f < LOOP[1]
        u = np.clip((loss(P.astype(np.float64)) - F_MIN) / 1.6, 0, 1) ** 0.6
        cols = self.c_cyan * (1 - u[:, None]) + self.c_amber * u[:, None]
        if k == 2:                                   # exploding run: white-hot as the loss blows up
            hot = np.clip(np.log10(np.maximum(np.abs(P).max(1), 1e-9)) / 2.0, 0, 1)[:, None]
            cols = cols * (1 - hot) + np.array([1.4, 1.1, 1.3], np.float32) * hot
        if in_reset:
            cols = cols * 0.5 + np.array([0.4, 0.4, 0.45], np.float32) * 0.5
        cols = cols.astype(np.float32)

        # per-run trail length and weight: the bouncing run and the explosion would otherwise
        # pile 10,000 long streaks into a white-out
        trail, wmul, lim = {None: (10, 1.0, 1.4), 0: (10, 1.0, 1.4), 1: (2, 0.25, 1.4), 2: (6, 0.5, 2.6)}[k]
        acc = np.zeros((H, W, 3), np.float32)
        pts = to_px(P).astype(np.float64)
        idx = [(f - i) % N_FRAMES for i in range(trail + 1)]
        raw = self.pos[idx]
        on = np.all(np.abs(raw) < lim, axis=(0, 2))          # keep trails that stay near the landscape
        trails = to_px(raw).astype(np.float32)
        if f >= RUNS[0][3] and on.any():
            splat_trails(acc, np.ascontiguousarray(trails[:, on]), np.ascontiguousarray(cols[on]), WEIGHT * wmul, 1.0, 0.0, 0.0)
        ok = np.all(np.abs(P) < lim, axis=1)
        if ok.any():
            splat_points(acc, np.ascontiguousarray(pts[ok]), np.ascontiguousarray(cols[ok]), WEIGHT * 2.5 * max(wmul, 0.6), 1.7)

        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud_lr(img, f, t, k, in_reset)
        return dither(img, f)

    def _hud_lr(self, img, f, t, k, in_reset):
        surf, ctx = layer()
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        if k is None or in_reset:
            step, label = 0, (RUNS[k][0] if k is not None else RUNS[0][0])
            if f >= LOOP[0] or f < RUNS[0][3]:
                label = RUNS[0][0]
        else:
            lab, lr, steps, f0, f1, expo = RUNS[k]
            step, label = int(frame_step(f, f0, f1, steps, expo)), lab
        ml = self.mean_loss[f]
        loss_s = "∞" if ml > 100 else f"{ml:.2f}"
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "AVERAGE LOSS", 0.14 * 24)
        show_tracked(ctx, "AVERAGE LOSS", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{step:,}", 70, 545, big)
        col = rgb(MAGENTA) if ml > 100 else rgb(CYAN) if ml < 0.8 else rgb(AMBER)
        ctx.set_source_rgb(*col)
        tw = cells_width(ctx, loss_s, big)
        show_cells(ctx, loss_s, 910 - tw, 545, big)
        # step-size pill
        text = f"STEP SIZE  {label}"
        font(ctx, "bold", 26)
        lw = tracked_width(ctx, text, 0.08 * 26)
        x0, y0, bw, bh, r = CX - lw / 2 - 18, 586, lw + 36, 44, 22
        ctx.new_sub_path()
        ctx.arc(x0 + bw - r, y0 + r, r, -math.pi / 2, 0)
        ctx.arc(x0 + bw - r, y0 + bh - r, r, 0, math.pi / 2)
        ctx.arc(x0 + r, y0 + bh - r, r, math.pi / 2, math.pi)
        ctx.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        pc = {"0.002": rgb(CYAN), "0.3": rgb(AMBER), "1.7": rgb(MAGENTA)}[label]
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
