"""Three identical 8-layer ReLU networks race on the spiral with plain gradient descent, differing
only in how big their random starting weights are: std 0.01 (too small), He initialisation (just
right), std 1.0 (too big). Three live decision fields, an accuracy under each, under that a bar per
layer showing how loud the signal is as it passes through the 8 layers (log scale), and a race
chart. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from textkit import (H, W, cells_width, dither, font, layer, linear_to_srgb, over, rgb, show_cells,  # noqa: E402
                     show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)
from scalesim import DEPTH, NAMES, STEPS  # noqa: E402

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1380                     # 23 s
F0, F1, EXPO = 30, 900, 1.3
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
NET_COL = {"small": CYAN, "he": AMBER, "big": MAGENTA}
LABEL = {"small": "TOO SMALL", "he": "HE INIT", "big": "TOO BIG"}
CX = 470
PANEL_Y0, PANEL_S = 690, 310
PANEL_X0 = [60, 385, 710]
BOARD_Y0, BOARD_H = 1112, 150
LOG_LO, LOG_HI = -14.0, 4.0         # bar scale: 1e-14 .. 1e4
CHART = (90, 1345, 990, 1455)
LOOP_FADE = (22.3, 22.85)
MAX_TEXT_W = 820
TRAIN_PATH = os.path.join(HERE, "scale_train.npz")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def frame_step(f):
    if f < F0:
        return 0
    if f >= F1:
        return STEPS
    u = (f - F0) / (F1 - F0)
    return int(round(STEPS * (1 - (1 - u) ** EXPO)))


def step_frame(s):
    u = 1 - (1 - s / STEPS) ** (1 / EXPO)
    return int(math.ceil(F0 + u * (F1 - F0)))


def fmt_signal(v):
    if v == 0:
        return "0"
    if math.isinf(v) or math.isnan(v):
        return "∞"
    if 0.01 <= v < 100:
        return f"{v:.2f}"
    e = int(math.floor(math.log10(v)))
    return "10" + str(e).translate(SUP)


class Scene:
    def __init__(self, train_path=TRAIN_PATH):
        if not os.path.exists(train_path):
            import scalesim
            scalesim.main(train_path)
        d = np.load(train_path)
        with open(train_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.X, self.y = d["X"], d["y"]
        self.fields = {n: d[f"{n}_fields"] for n in NAMES}
        self.right = {n: d[f"{n}_right"] for n in NAMES}
        self.sig = {n: d[f"{n}_sig"] for n in NAMES}
        self.acc = {n: np.array(self.meta["acc"][n]) for n in NAMES}
        self.final = self.meta["final"]
        self.first90 = self.meta["first90"]
        self.std = self.meta["std"]
        self.x_std = float(np.std(self.X))
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("How random should the start be?", "black", 54, WHITE, 300, 1), ("Same 8-layer network. Three sizes of random.", "dbold", 42, WHITE, 392, 0.9)]
        s_small0 = fmt_signal(float(self.sig["small"][0][-1]))
        s_big0 = fmt_signal(float(self.sig["big"][0][-1]))
        s_big1 = fmt_signal(float(self.sig["big"][1][-1]))
        cues = [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.8, 5.4, [(f"Weights: std {self.std['small']:.2f}, std {self.std['he']:.2f} (He), std {self.std['big']:.1f}.", "black", 42, WHITE, 312, 1), ("Below each: how loud the signal is at each of 8 layers.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
            (5.8, 8.6, [(f"Too small: the signal fades to {s_small0} by layer 8.", "black", 42, CYAN, 312, 1), (f"Too big: {s_big0} at layer 8. One step later: {s_big1}.", "dbold", 40, MAGENTA, 396, 0.9)], 8, 6, "pop"),
        ]
        t_end = F1 / FPS
        s_he = self.first90["he"]
        if s_he is not None:
            t_h = max(step_frame(s_he) / FPS + 0.1, 9.0)
            cues.append((t_h, t_h + 2.2, [(f"He init: 90% at step {s_he}.", "black", 60, AMBER, 350, 1)], 6, 6, "chaos"))
            t_mid0 = t_h + 2.5
        else:
            t_mid0 = 9.0
        if t_end - 0.4 - t_mid0 > 1.6:
            cues.append((t_mid0, t_end - 0.4, [("Too big overflowed at step 2. It's wreckage now.", "black", 42, MAGENTA, 312, 1), ("Too small is whispering. He init is learning.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"))
        cues += [
            (t_end + 0.3, t_end + 2.4, [(f"Small: {self.final['small'] * 100:.0f}%.  He: {self.final['he'] * 100:.0f}%.  Big: {self.final['big'] * 100:.0f}%.", "black", 50, WHITE, 312, 1), ("Only the right size ever learned.", "dbold", 42, AMBER, 396, 0.9)], 6, 6, "chaos"),
            (t_end + 2.6, t_end + 5.0, [("He init: std = √(2 ÷ inputs per neuron).", "black", 44, WHITE, 312, 1), ("Each layer hands the signal on at the same size.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (t_end + 5.2, 22.2, [("Not too quiet. Not too loud.", "black", 56, AMBER, 312, 1), ("Kaiming He, 2015. Every ReLU network starts this way.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
            (22.5, 23.1, hook, 10, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        step = frame_step(f)
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        lin = self.bg.copy()
        pts = np.zeros((H, W, 3), np.float32)
        for i, name in enumerate(NAMES):
            p_small = self.fields[name][step].astype(np.float32)
            if mix > 0:
                p_small = (1 - mix) * p_small + mix * self.fields[name][0].astype(np.float32)
            self._panel(lin, pts, i, name, p_small, 0 if mix > 0.5 else step)
        small = cv2.resize(pts, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.6 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.6 * (pts + glow))
        lin = 1 - (1 - np.clip(lin, 0, 1)) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, 0 if mix > 0.5 else step)
        _ruin.Scene._headlines(self, img, f, t)
        return dither(img, f)

    def _panel(self, lin, pts, i, name, p_small, step):
        x0, y0, s = PANEL_X0[i], PANEL_Y0, PANEL_S
        p = cv2.resize(p_small, (s, s), interpolation=cv2.INTER_CUBIC)
        p = np.clip(p, 0, 1)
        conf = np.abs(2 * p - 1)
        k = (conf ** 0.9 * 0.12)[..., None]
        tint = p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan
        region = self.c_base + k * tint
        edge = np.exp(-((p - 0.5) / 0.03) ** 2)
        gmag = np.hypot(cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3))
        edge *= np.clip(gmag / 0.02, 0, 1)
        region += (edge * 0.16)[..., None] * (0.55 + 0.45 * tint)
        lin[y0:y0 + s, x0:x0 + s] = region
        right = self.right[name][step]
        surf, ctx = layer()
        cx, cy = x0 + s / 2, y0 + s / 2
        sc = (s / 2) / 1.05
        for j in range(len(self.X)):
            col = rgb(MAGENTA) if self.y[j] > 0.5 else rgb(CYAN)
            x, y = cx + self.X[j, 0] * sc, cy - self.X[j, 1] * sc
            if right[j]:
                ctx.set_source_rgba(*col, 1)
                ctx.arc(x, y, 2.9, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.arc(x, y, 1.3, 0, 2 * math.pi)
                ctx.fill()
            else:
                ctx.set_source_rgba(*col, 0.6)
                ctx.set_line_width(1.6)
                ctx.arc(x, y, 3.7, 0, 2 * math.pi)
                ctx.stroke()
        prgb, _ = surf_rgba(surf)
        pts += (prgb ** 2.2 * 0.7).astype(np.float32)

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, step):
        surf, ctx = layer()
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, "%": 60, ".": 22})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "LEADER", 0.14 * 24)
        show_tracked(ctx, "LEADER", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{step:,}", 70, 545, big)
        accs = {n: self.acc[n][step] for n in NAMES}
        leader = max(NAMES, key=lambda n: (accs[n], -NAMES.index(n))) if step > 0 else None
        font(ctx, "dbold", 52)
        s = LABEL[leader] if leader else "—"
        ctx.set_source_rgb(*(rgb(NET_COL[leader]) if leader else (1, 1, 1)))
        ctx.move_to(910 - ctx.text_extents(s).x_advance, 545)
        ctx.show_text(s)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, f"2 → 32 ×{DEPTH} → 1   ·   ReLU   ·   PLAIN GRADIENT DESCENT   ·   SAME DATA", 70, 612, 0.08 * 22)
        acc_cells = {str(c): 40 for c in range(10)} | {"%": 46, ".": 18}
        for i, name in enumerate(NAMES):
            x0, y0, s_ = PANEL_X0[i], PANEL_Y0, PANEL_S
            col = NET_COL[name]
            ctx.set_line_width(1.5)
            ctx.set_source_rgba(*rgb(col), 0.5)
            ctx.rectangle(x0 + 0.5, y0 + 0.5, s_, s_)
            ctx.stroke()
            font(ctx, "black", 28)
            ctx.set_source_rgba(*rgb(col), 0.98)
            ctx.move_to(x0, y0 - 20)
            ctx.show_text(LABEL[name])
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.5)
            sub = f"random, std {self.std[name]:.2f}" + ("  = √(2/32)" if name == "he" else "")
            show_tracked(ctx, sub, x0, y0 - 50, 0.04 * 17)
            a = accs[name]
            font(ctx, "dbold", 56)
            s = f"{a * 100:.0f}%"
            ctx.set_source_rgb(*(rgb(GREEN) if a >= 1 else rgb(col)))
            tw = cells_width(ctx, s, acc_cells)
            show_cells(ctx, s, x0 + s_ / 2 - tw / 2, y0 + s_ + 66, acc_cells)
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.45)
            lab = "ACCURACY"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.12 * 17) / 2, y0 + s_ + 92, 0.12 * 17)
            self._bars(ctx, x0, name, step)
        # race chart
        cx0, cy0, cx1, cy1 = CHART

        def px_(st):
            return cx0 + (cx1 - cx0) * st / STEPS

        def py_(a):
            return cy1 - (cy1 - cy0) * (a - 0.5) / 0.5

        ctx.set_line_width(1)
        for a, lab in ((0.5, "50%"), (1.0, "100%")):
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.5 if a == 1 else 0.2)
            ctx.set_dash([5, 7] if a == 1 else [])
            ctx.move_to(cx0, py_(a) + 0.5)
            ctx.line_to(cx1, py_(a) + 0.5)
            ctx.stroke()
            ctx.set_dash([])
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.7 if a == 1 else 0.45)
            ctx.move_to(cx0 - ctx.text_extents(lab).x_advance - 10, py_(a) + 6)
            ctx.show_text(lab)
        font(ctx, "bold", 17)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        for st in (0, 500, 1000):
            lab = f"{st}"
            ctx.move_to(px_(st) - ctx.text_extents(lab).x_advance / 2, cy1 + 26)
            ctx.show_text(lab)
        lab = "STEPS"
        ctx.move_to(cx1 - ctx.text_extents(lab).x_advance, cy1 + 48)
        ctx.show_text(lab)
        for name in NAMES:
            acc = self.acc[name]
            ctx.set_line_width(3)
            ctx.set_source_rgba(*rgb(NET_COL[name]), 0.95)
            ctx.move_to(px_(0), py_(max(0.5, acc[0])))
            for st in range(1, step + 1, 2):
                ctx.line_to(px_(st), py_(max(0.5, acc[st])))
            if step > 0:
                ctx.line_to(px_(step), py_(max(0.5, acc[step])))
            ctx.stroke()
            if step > 0:
                ctx.arc(px_(step), py_(max(0.5, acc[step])), 6, 0, 2 * math.pi)
                ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def _bars(self, ctx, x0, name, step):
        """One bar per hidden layer: std of that layer's activations, log scale 1e-14 .. 1e4."""
        col = NET_COL[name]
        sig = self.sig[name][step]
        bw, gap = 30, 6
        total = DEPTH * bw + (DEPTH - 1) * gap
        bx0 = x0 + (PANEL_S - total) / 2
        by1 = BOARD_Y0 + BOARD_H - 26
        by0 = BOARD_Y0 + 4
        # input-level reference
        ref = (math.log10(self.x_std) - LOG_LO) / (LOG_HI - LOG_LO)
        yref = by1 - (by1 - by0) * ref
        ctx.set_line_width(1)
        ctx.set_dash([3, 5])
        ctx.set_source_rgba(1, 1, 1, 0.35)
        ctx.move_to(bx0 - 4, yref + 0.5)
        ctx.line_to(bx0 + total + 4, yref + 0.5)
        ctx.stroke()
        ctx.set_dash([])
        ctx.set_source_rgba(1, 1, 1, 0.2)
        ctx.move_to(bx0 - 4, by1 + 0.5)
        ctx.line_to(bx0 + total + 4, by1 + 0.5)
        ctx.stroke()
        for L in range(DEPTH):
            v = float(sig[L])
            if v > 0 and not math.isnan(v):
                frac = 1.0 if math.isinf(v) else min(1.0, max(0.0, (math.log10(v) - LOG_LO) / (LOG_HI - LOG_LO)))
            else:
                frac = 0.0
            hh = (by1 - by0) * frac
            x = bx0 + L * (bw + gap)
            ctx.set_source_rgba(*rgb(col), 0.16)
            ctx.rectangle(x, by0, bw, by1 - by0)
            ctx.fill()
            if hh > 0:
                ctx.set_source_rgba(*rgb(col), 0.95 if not math.isinf(v) else 1.0)
                ctx.rectangle(x, by1 - hh, bw, hh)
                ctx.fill()
            if math.isinf(v):
                font(ctx, "bold", 16)
                ctx.set_source_rgba(1, 1, 1, 0.95)
                s = "∞"
                ctx.move_to(x + bw / 2 - ctx.text_extents(s).x_advance / 2, by0 + 14)
                ctx.show_text(s)
            font(ctx, "bold", 13)
            ctx.set_source_rgba(1, 1, 1, 0.4)
            s = str(L + 1)
            ctx.move_to(x + bw / 2 - ctx.text_extents(s).x_advance / 2, by1 + 15)
            ctx.show_text(s)
        font(ctx, "bold", 15)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        show_tracked(ctx, "INPUT", bx0 + total + 8, yref + 5, 0.06 * 15)
        font(ctx, "bold", 18)
        ctx.set_source_rgba(*rgb(col), 0.95)
        lab = f"SIGNAL AT LAYER 8:  {fmt_signal(float(sig[-1]))}"
        lw = tracked_width(ctx, lab, 0.08 * 18)
        show_tracked(ctx, lab, x0 + PANEL_S / 2 - lw / 2, by1 + 40, 0.08 * 18)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s first90 {sc.first90} final {sc.final}")
    frames = [0, 120, 200, 260, 320, 380, 440, 500, 560, 640, 720, 800, 900, 980, 1080, 1180, 1280, 1379]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (440, 980):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
