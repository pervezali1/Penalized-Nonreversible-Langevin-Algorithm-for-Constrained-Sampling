"""Three identical ReLU networks race on the spiral, differing only in their starting weights:
random, all zeros, all the same number. Three live decision fields, an accuracy under each, a board
of the 32 first-layer neurons' response maps under that (32 different pictures vs 32 identical
ones), a distinct-neuron count, and a race chart. frame(f) -> uint8 1080x1920 RGB."""

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
from initsim import NAMES, SAME_VALUE, STEPS, forward  # noqa: E402

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1380                     # 23 s
F0, F1, EXPO = 30, 900, 1.3
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
NET_COL = {"random": AMBER, "zeros": MAGENTA, "same": CYAN}
LABEL = {"random": "RANDOM", "zeros": "ALL ZERO", "same": "ALL 0.1"}
SUB = {"random": "every weight different", "zeros": "every weight 0", "same": f"every weight {SAME_VALUE}"}
CX = 470
PANEL_Y0, PANEL_S = 690, 310
PANEL_X0 = [60, 385, 710]
TILE, TGAP, TCOLS, TROWS = 35, 3, 8, 4
BOARD_Y0 = 1112
CHART = (90, 1345, 990, 1455)
LOOP_FADE = (22.3, 22.85)
MAX_TEXT_W = 820
TRAIN_PATH = os.path.join(HERE, "init_train.npz")


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
            import initsim
            initsim.main(train_path)
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
        self.distinct = {n: self.meta["distinct"][n] for n in NAMES}
        self.first100 = self.meta["first100"]
        self.final = self.meta["final"]
        g = PANEL_S // 4
        xs = (np.arange(g) * 4 + 2 - PANEL_S / 2) / (PANEL_S / 2) * 1.05
        gx, gy = np.meshgrid(xs, xs)
        self.grid = np.stack([gx.ravel(), -gy.ravel()], 1)
        self.gshape = (g, g)
        tg = 12
        txs = np.linspace(-1.05, 1.05, tg)
        tgx, tgy = np.meshgrid(txs, txs)
        self.tgrid = np.stack([tgx.ravel(), -tgy.ravel()], 1)
        self.tgshape = (tg, tg)
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.c_tile = {n: np.array(srgb_to_linear(rgb(NET_COL[n])), np.float32) for n in NAMES}
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("What if every weight starts at zero?", "black", 52, WHITE, 300, 1), ("Same 64 neurons, same data. Three starts.", "dbold", 42, WHITE, 392, 0.9)]
        cues = [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.8, 5.4, [("Random start. All zeros. All the same number.", "black", 42, WHITE, 312, 1), ("Below each: its 32 first-layer neurons. Same count.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
        ]
        t_end = F1 / FPS
        s_r = self.first100["random"]
        if s_r is not None:
            t_r = step_frame(s_r) / FPS + 0.1
            cues.append((t_r, t_r + 2.2, [(f"Random: 100% at step {s_r}.", "black", 60, AMBER, 350, 1)], 6, 6, "chaos"))
        t_mid0 = (cues[-1][1] + 0.3) if s_r is not None else 6.0
        t_mid1 = t_end - 0.4
        if t_mid1 - t_mid0 > 1.6:
            cues.append((t_mid0, t_mid1, [("Zeros: nothing moves.", "black", 52, MAGENTA, 312, 1), ("Same: all 32 move together, as one.", "dbold", 42, CYAN, 396, 0.9)], 8, 6, "pop"))
        cues += [
            (t_end + 0.3, t_end + 2.4, [(f"Zeros: {self.final['zeros'] * 100:.0f}%.  Same: {self.final['same'] * 100:.0f}%.  Random: {self.final['random'] * 100:.0f}%.", "black", 48, WHITE, 312, 1), ("32 neurons each. Distinct ones: 1, 1 and 32.", "dbold", 40, AMBER, 396, 0.9)], 6, 6, "chaos"),
            (t_end + 2.6, t_end + 5.0, [("Identical neurons get identical gradients.", "black", 46, WHITE, 312, 1), ("So they stay identical. Forever. 64 neurons, one opinion.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
            (t_end + 5.2, 22.2, [("Randomness isn't noise.", "black", 56, AMBER, 312, 1), ("It's what makes 64 neurons 64 different neurons.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (22.5, 23.1, hook, 10, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ helpers

    def params_at(self, name, step):
        return [p[step] for p in self.params[name]]

    def field(self, name, step):
        logit = forward(self.params_at(name, step), self.grid)[0]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return p.reshape(self.gshape).astype(np.float32)

    def tiles(self, name, step):
        """(32, tg, tg) response of each first-layer neuron over the plane, each scaled to its own max."""
        ps = self.params_at(name, step)
        a = np.maximum(self.tgrid @ ps[0] + ps[1], 0.0)          # (tg*tg, 32)
        mx = a.max(0, keepdims=True)
        a = np.where(mx > 1e-9, a / np.maximum(mx, 1e-9), 0.0)
        return a.T.reshape(32, *self.tgshape).astype(np.float32)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        step = frame_step(f)
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        lin = self.bg.copy()
        pts = np.zeros((H, W, 3), np.float32)
        for i, name in enumerate(NAMES):
            p_small = self.field(name, step)
            tl = self.tiles(name, step)
            if mix > 0:
                p_small = (1 - mix) * p_small + mix * self.field(name, 0)
                tl = (1 - mix) * tl + mix * self.tiles(name, 0)
            self._panel(lin, pts, i, name, p_small, 0 if mix > 0.5 else step)
            self._board(lin, i, name, tl)
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
        edge *= np.clip(gmag / 0.02, 0, 1)                      # a flat field has no boundary to glow
        region += (edge * 0.16)[..., None] * (0.55 + 0.45 * tint)
        lin[y0:y0 + s, x0:x0 + s] = region
        pred = forward(self.params_at(name, step), self.X)[0] > 0
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

    def _board(self, lin, i, name, tl):
        """32 tiles: each neuron's response map, in the network's colour; dark means the neuron is silent."""
        x0 = PANEL_X0[i] + (PANEL_S - (TCOLS * TILE + (TCOLS - 1) * TGAP)) // 2
        col = self.c_tile[name]
        for n in range(32):
            r, c = divmod(n, TCOLS)
            tx, ty = x0 + c * (TILE + TGAP), BOARD_Y0 + r * (TILE + TGAP)
            im = cv2.resize(tl[n], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
            lin[ty:ty + TILE, tx:tx + TILE] = self.c_base * 0.6 + (0.03 + 0.5 * im[..., None]) * col

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
        font(ctx, "dbold", 56)
        s = LABEL[leader] if leader else "—"
        ctx.set_source_rgb(*(rgb(NET_COL[leader]) if leader else (1, 1, 1)))
        ctx.move_to(910 - ctx.text_extents(s).x_advance, 545)
        ctx.show_text(s)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, "2 → 32 → 32 → 1   ·   ReLU   ·   ADAM 0.01   ·   SAME DATA", 70, 612, 0.08 * 22)
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
            show_tracked(ctx, SUB[name], x0, y0 - 50, 0.04 * 17)
            a = accs[name]
            font(ctx, "dbold", 56)
            s = f"{a * 100:.0f}%" if a < 0.995 or a >= 1 else f"{a * 100:.1f}%"
            ctx.set_source_rgb(*(rgb(GREEN) if a >= 1 else rgb(col)))
            tw = cells_width(ctx, s, acc_cells)
            show_cells(ctx, s, x0 + s_ / 2 - tw / 2, y0 + s_ + 66, acc_cells)
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.45)
            lab = "ACCURACY"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.12 * 17) / 2, y0 + s_ + 92, 0.12 * 17)
            # distinct neurons under the board
            dn = self.distinct[name][step]
            by = BOARD_Y0 + TROWS * (TILE + TGAP)
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.5)
            lab = "32 NEURONS"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.1 * 17) / 2, by + 22, 0.1 * 17)
            font(ctx, "bold", 20)
            ctx.set_source_rgba(*rgb(col), 0.95)
            lab = "ALL DIFFERENT" if dn == 32 else "ALL IDENTICAL" if dn == 1 else f"{dn} DISTINCT"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.1 * 20) / 2, by + 48, 0.1 * 20)
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


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s first100 {sc.first100} final {sc.final} distinct end "
          f"{ {n: sc.distinct[n][-1] for n in NAMES} }")
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
