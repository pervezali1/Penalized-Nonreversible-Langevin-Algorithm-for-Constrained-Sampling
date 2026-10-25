"""Three identical networks race on the spiral, differing only in the activation function: ReLU,
tanh, sigmoid. Three live decision fields side by side, an accuracy under each, and a race chart
beneath. frame(f) -> uint8 1080x1920 RGB."""

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
from actsim import ACTS, NAMES, STEPS, forward  # noqa: E402

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1380                     # 23 s
F0, F1, EXPO = 30, 960, 1.3         # training frames and front-loading exponent
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
NET_COL = {"relu": AMBER, "tanh": CYAN, "sigmoid": MAGENTA}
LABEL = {"relu": "ReLU", "tanh": "tanh", "sigmoid": "sigmoid"}
FORMULA = {"relu": "max(0, z)", "tanh": "tanh(z)", "sigmoid": "1 / (1 + e⁻ᶻ)"}
CX = 470
PANEL_Y0, PANEL_S = 700, 310
PANEL_X0 = [60, 385, 710]
CHART = (90, 1160, 990, 1440)
LOOP_FADE = (22.3, 22.85)
MAX_TEXT_W = 820
TRAIN_PATH = os.path.join(HERE, "activations_train.npz")


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


class Scene:
    def __init__(self, train_path=TRAIN_PATH):
        if not os.path.exists(train_path):
            import actsim
            actsim.main(train_path)
        d = np.load(train_path)
        with open(train_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.X, self.y = d["X"], d["y"]
        self.params = {}
        for name in NAMES:
            ps, k = [], 0
            while f"{name}_p{k}" in d:
                ps.append(d[f"{name}_p{k}"])
                k += 1
            self.params[name] = ps
        self.acc = {n: np.array(self.meta["acc"][n]) for n in NAMES}
        self.first100 = self.meta["first100"]
        self.final = self.meta["final"]
        self.n_pts = len(self.y)
        # per-panel evaluation grid (quarter resolution)
        g = PANEL_S // 4
        xs = (np.arange(g) * 4 + 2 - PANEL_S / 2) / (PANEL_S / 2) * 1.05
        gx, gy = np.meshgrid(xs, xs)
        self.grid = np.stack([gx.ravel(), -gy.ravel()], 1)
        self.gshape = (g, g)
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("Same AI. Same data. Three activations.", "black", 50, WHITE, 300, 1), ("ReLU vs tanh vs sigmoid. Which one learns?", "dbold", 42, WHITE, 392, 0.9)]
        cues = [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.8, 5.4, [("Same 64 neurons, same Adam, same 1,000 steps.", "black", 42, WHITE, 312, 1), ("Only the function inside each neuron differs.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
        ]
        t_end = F1 / FPS
        s_relu = self.first100["relu"]
        if s_relu is not None:
            t_r = step_frame(s_relu) / FPS + 0.1
            cues.append((t_r, t_r + 2.2, [(f"ReLU: 100% at step {s_relu}.", "black", 60, AMBER, 350, 1)], 6, 6, "chaos"))
        wrong_tanh = int(round((1 - self.final["tanh"]) * self.n_pts))
        s_tanh = self.first100["tanh"]
        if s_tanh is not None:
            t_t = step_frame(s_tanh) / FPS + 0.1
            cues.append((t_t, t_t + 2.0, [(f"tanh: 100% at step {s_tanh}.", "black", 60, CYAN, 350, 1)], 6, 6, "chaos"))
            tanh_line = f"tanh: 100% at step {s_tanh}."
        else:
            tanh_line = f"tanh: {self.final['tanh'] * 100:.1f}%, {wrong_tanh} point{'s' if wrong_tanh != 1 else ''} still wrong."
        mid = (cues[-1][1] + 0.3, t_end - 0.4)
        if mid[1] - mid[0] > 1.6:
            cues.append((mid[0], mid[1], [("Sigmoid is barely moving.", "black", 56, MAGENTA, 350, 1)], 8, 6, "pop"))
        cues += [
            (t_end + 0.3, t_end + 2.4, [(f"Sigmoid: {self.final['sigmoid'] * 100:.0f}% after 1,000 steps.", "black", 54, MAGENTA, 312, 1), (tanh_line, "dbold", 40, CYAN, 396, 0.9)], 6, 6, "chaos"),
            (t_end + 2.6, t_end + 4.9, [("Sigmoid's slope is at most ¼ per layer.", "black", 46, WHITE, 312, 1), ("Two layers: 1/16 of the error signal gets back.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (t_end + 5.1, 22.2, [("Gradients vanish. ReLU's slope is 1.", "black", 50, AMBER, 312, 1), ("That's why nobody puts sigmoid inside a network.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (22.5, 23.1, hook, 10, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ helpers

    def params_at(self, name, step):
        return [p[step] for p in self.params[name]]

    def field(self, name, step):
        logit = forward(self.params_at(name, step), self.grid, name)[0]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return p.reshape(self.gshape).astype(np.float32)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        step = frame_step(f)
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        lin = self.bg.copy()
        pts = np.zeros((H, W, 3), np.float32)
        for i, name in enumerate(NAMES):
            p_small = self.field(name, step)
            if mix > 0:
                p_small = (1 - mix) * p_small + mix * self.field(name, 0)
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
        region += (edge * 0.16)[..., None] * (0.55 + 0.45 * tint)
        lin[y0:y0 + s, x0:x0 + s] = region
        # points
        pred = forward(self.params_at(name, step), self.X, name)[0] > 0
        right = pred == (self.y > 0.5)
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
        leader = max(NAMES, key=lambda n: (accs[n], -NAMES.index(n)))
        if step == 0:
            leader = None
        font(ctx, "dbold", 64)
        s = LABEL[leader] if leader else "—"
        ctx.set_source_rgb(*(rgb(NET_COL[leader]) if leader else (1, 1, 1)))
        ctx.move_to(910 - ctx.text_extents(s).x_advance, 545)
        ctx.show_text(s)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, "2 → 32 → 32 → 1   ·   ADAM 0.01   ·   SAME START", 70, 612, 0.08 * 22)
        # panel labels, glyphs, frames and accuracies
        for i, name in enumerate(NAMES):
            x0, y0, s_ = PANEL_X0[i], PANEL_Y0, PANEL_S
            col = NET_COL[name]
            ctx.set_line_width(1.5)
            ctx.set_source_rgba(*rgb(col), 0.5)
            ctx.rectangle(x0 + 0.5, y0 + 0.5, s_, s_)
            ctx.stroke()
            font(ctx, "black", 30)
            ctx.set_source_rgba(*rgb(col), 0.98)
            ctx.move_to(x0, y0 - 22)
            ctx.show_text(LABEL[name])
            lw = ctx.text_extents(LABEL[name]).x_advance
            # tiny glyph of the function
            gx, gy, gw, gh = x0 + lw + 16, y0 - 44, 56, 34
            ctx.set_line_width(1)
            ctx.set_source_rgba(1, 1, 1, 0.18)
            ctx.move_to(gx, gy + gh / 2 + 0.5)
            ctx.line_to(gx + gw, gy + gh / 2 + 0.5)
            ctx.stroke()
            ctx.set_line_width(2.5)
            ctx.set_source_rgba(*rgb(col), 0.9)
            first = True
            for k in range(41):
                z = -3 + 6 * k / 40
                v = float(ACTS[name][0](np.array([z]))[0])
                frac = v / 3 if name == "relu" else (v + 1) / 2 if name == "tanh" else v     # 0..1 within the glyph
                px, py = gx + gw * k / 40, gy + gh - frac * gh
                if first:
                    ctx.move_to(px, py)
                    first = False
                else:
                    ctx.line_to(px, py)
            ctx.stroke()
            font(ctx, "bold", 18)
            ctx.set_source_rgba(1, 1, 1, 0.5)
            show_tracked(ctx, FORMULA[name], x0, y0 - 52, 0.04 * 18)
            # accuracy
            a = accs[name]
            font(ctx, "dbold", 60)
            s = f"{a * 100:.0f}%" if a < 0.995 or a >= 1 else f"{a * 100:.1f}%"
            ctx.set_source_rgb(*(rgb(GREEN) if a >= 1 else rgb(col)))
            tw = cells_width(ctx, s, {str(c): 40 for c in range(10)} | {"%": 46, ".": 18})
            show_cells(ctx, s, x0 + s_ / 2 - tw / 2, y0 + s_ + 72, {str(c): 40 for c in range(10)} | {"%": 46, ".": 18})
            font(ctx, "bold", 18)
            ctx.set_source_rgba(1, 1, 1, 0.45)
            lab = "ACCURACY"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.12 * 18) / 2, y0 + s_ + 100, 0.12 * 18)
        # race chart
        cx0, cy0, cx1, cy1 = CHART

        def px_(st):
            return cx0 + (cx1 - cx0) * st / STEPS

        def py_(a):
            return cy1 - (cy1 - cy0) * (a - 0.5) / 0.5

        ctx.set_line_width(1)
        for a, lab in ((0.5, "50%"), (0.75, "75%"), (1.0, "100%")):
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.5 if a == 1 else 0.12)
            ctx.set_dash([5, 7] if a == 1 else [])
            ctx.move_to(cx0, py_(a) + 0.5)
            ctx.line_to(cx1, py_(a) + 0.5)
            ctx.stroke()
            ctx.set_dash([])
            font(ctx, "bold", 18)
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.7 if a == 1 else 0.45)
            ctx.move_to(cx0 - ctx.text_extents(lab).x_advance - 10, py_(a) + 6)
            ctx.show_text(lab)
        font(ctx, "bold", 18)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        for st in (0, 250, 500, 750, 1000):
            lab = f"{st}"
            ctx.move_to(px_(st) - ctx.text_extents(lab).x_advance / 2, cy1 + 30)
            ctx.show_text(lab)
        lab = "STEPS"
        ctx.move_to(cx1 - ctx.text_extents(lab).x_advance, cy1 + 54)
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


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s first100 {sc.first100} final {sc.final}")
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
