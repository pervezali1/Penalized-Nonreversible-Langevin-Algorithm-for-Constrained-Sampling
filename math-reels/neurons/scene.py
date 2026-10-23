"""Frame renderer for "This AI has 2 neurons."  frame(f) -> uint8 1080x1920 RGB.

The whole background is the network's live decision field p(x, y); the 400 spiral
points sit in a square below the HUD; a small network diagram shows the real weights.
"""

import json
import math
import sys
import os

import cairo
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from train import FPS, N_FRAMES, RESETS, STAGES, forward  # noqa: E402

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
SQ_C = (470, 1120)        # data square centre
SQ_H = 400                # half-size in px (data radius 0.9 -> 360 px)
NET_Y0, NET_Y1 = 600, 740
MAX_TEXT_W = 820
LOOP_FADE = (1158, 1200)  # crossfade back to the frame-0 state


def pct(a):
    return f"{int(math.floor(a * 100 + 1e-9))}%"


class Scene:
    def __init__(self, train_path):
        d = np.load(train_path)
        with open(train_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.X, self.y = d["X"], d["y"]
        self.params = []
        for si in range(len(STAGES)):
            ps = []
            k = 0
            while f"s{si}_p{k}" in d:
                ps.append(d[f"s{si}_p{k}"])
                k += 1
            self.params.append(ps)
        self.frames = self.meta["frames"]
        self.acc = np.array(self.meta["acc"])
        self.neurons = self.meta["neurons"]
        self.first100 = self.meta["first100_frame"]
        gw, gh = W // 4, H // 4
        xs = (np.arange(gw) * 4 + 2 - SQ_C[0]) / SQ_H
        ys = (np.arange(gh) * 4 + 2 - SQ_C[1]) / SQ_H
        gx, gy = np.meshgrid(xs, ys)
        self.grid = np.stack([gx.ravel(), gy.ravel()], 1)
        self.gshape = (gh, gw)
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.cues = self._cues()
        self.px = SQ_C[0] + self.X[:, 0] * SQ_H
        self.py = SQ_C[1] + self.X[:, 1] * SQ_H

    # ------------------------------------------------------------ text cues

    def _cues(self):
        a = self.meta["final_acc"]
        t100 = self.first100 / FPS if self.first100 is not None else 13.0
        hook = [("This AI has", "dbold", 54, WHITE, 292, 0.8), ("2 neurons.", "black", 96, WHITE, 392, 1)]
        return [
            (0.0, 1.6, hook, 0, 6, "none"),
            (1.8, 3.2, [("It can't bend", "black", 80, WHITE, 312, 1), ("enough.", "black", 80, WHITE, 396, 1)], 8, 6, "pop"),
            (3.3, 4.0, [(f"Stuck at {pct(a[0])}.", "black", 88, MAGENTA, 350, 1)], 8, 6, "pop"),
            (4.2, 5.4, [("Give it 8.", "black", 96, CYAN, 350, 1)], 8, 6, "pop"),
            (6.6, 7.6, [("Better.", "black", 80, WHITE, 312, 1), ("Still lost.", "black", 80, WHITE, 396, 1)], 8, 6, "pop"),
            (7.7, 8.5, [(f"Stuck at {pct(a[1])}.", "black", 88, MAGENTA, 350, 1)], 8, 6, "pop"),
            (8.7, 9.6, [("Give it 64.", "black", 96, CYAN, 350, 1)], 8, 6, "pop"),
            (9.7, t100 - 0.1, [("Watch it find", "black", 76, WHITE, 312, 1), ("the spiral.", "black", 76, WHITE, 396, 1)], 8, 6, "pop"),
            (t100, 15.0, [("100%.", "black", 170, CYAN, 385, 1)], 6, 6, "chaos"),
            (15.1, 17.4, [("Your brain has", "dbold", 54, WHITE, 292, 0.8), ("86,000,000,000.", "black", 88, AMBER, 384, 1)], 8, 6, "pop"),
            (17.5, 19.3, [("No pattern is too hard.", "black", 72, WHITE, 312, 1), ("Only too few neurons.", "dbold", 56, AMBER, 396, 1)], 8, 6, "pop"),
            (19.35, 20.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ model helpers

    def _params_at(self, f):
        si, step = self.frames[f]
        return si, [p[step] for p in self.params[si]]

    def _field(self, ps):
        logit = forward(ps, self.grid)[0]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return p.reshape(self.gshape).astype(np.float32)

    def _loop_mix(self, f):
        a, b = LOOP_FADE
        return smooth((f - a) / (b - a)) if f >= a else 0.0

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        si, ps = self._params_at(f)
        p_small = self._field(ps)
        mix = self._loop_mix(f)
        if mix > 0:
            p_small = (1 - mix) * p_small + mix * self._field(self._params_at(0)[1])
        acc = self.acc[f] if mix == 0 else (1 - mix) * self.acc[f] + mix * self.acc[0]
        step = self.frames[f][1]
        if mix > 0.5:                     # HUD and diagram snap back to the frame-0 network mid-fade
            si, ps = self._params_at(0)
            step = 0
        neurons = self.neurons[si]
        self._step = step

        # decision field in linear light: two-colour tint by p, white glow on the boundary
        p = cv2.resize(p_small, (W, H), interpolation=cv2.INTER_CUBIC)
        p = np.clip(p, 0, 1)
        conf = np.abs(2 * p - 1)
        k = (conf ** 0.9 * 0.04)[..., None]
        lin = self.c_base + k * (p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan)
        edge = np.exp(-((p - 0.5) / 0.03) ** 2)
        lin += (edge * 0.09)[..., None] * (0.55 + 0.45 * (p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan))
        # faint grid lines in the data square
        yy, xx = np.mgrid[0:H, 0:W]
        inside = (np.abs(xx - SQ_C[0]) <= SQ_H) & (np.abs(yy - SQ_C[1]) <= SQ_H)
        gridl = (((xx - SQ_C[0]) % 100 == 0) | ((yy - SQ_C[1]) % 100 == 0)) & inside
        lin[gridl] += 0.006
        # vignette
        lin *= self._vignette()[..., None]

        # data points (emissive, bloomed)
        pts = self._points(ps, f)
        small = cv2.resize(pts, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.6 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.6 * (pts + glow))
        lin = 1 - (1 - np.clip(lin, 0, 1)) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)

        self._hud(img, f, t, acc, neurons, si, ps)
        if any(a <= f < b for a, b in RESETS):
            img = self._glitch(img, f)
        return dither(img, f)

    _vig = None

    def _vignette(self):
        if Scene._vig is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - CX) / 700) ** 2 + ((yy - 1000) / 1100) ** 2)
            Scene._vig = (1 - 0.35 * np.clip(r - 0.6, 0, 1) ** 1.5).astype(np.float32)
        return Scene._vig

    def _points(self, ps, f):
        """Emissive layer: correctly classified points bright, wrong ones dim with a hollow ring."""
        pred = forward(ps, self.X)[0] > 0
        right = pred == (self.y > 0.5)
        surf, ctx = layer()
        for i in range(len(self.X)):
            col = rgb(MAGENTA) if self.y[i] > 0.5 else rgb(CYAN)
            x, y = self.px[i], self.py[i]
            if right[i]:
                ctx.set_source_rgba(*col, 1)
                ctx.arc(x, y, 6.5, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.arc(x, y, 2.6, 0, 2 * math.pi)
                ctx.fill()
            else:
                ctx.set_source_rgba(*col, 0.55)
                ctx.set_line_width(2.2)
                ctx.arc(x, y, 7.5, 0, 2 * math.pi)
                ctx.stroke()
        prgb, _ = surf_rgba(surf)
        return (prgb ** 2.2 * 0.9).astype(np.float32)

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, acc, neurons, si, ps):
        surf, ctx = layer()
        # labels
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "NEURONS", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "ACCURACY", 0.14 * 24)
        show_tracked(ctx, "ACCURACY", 910 - w, 470, 0.14 * 24)
        # neuron count (with the 86-billion payoff tick-up)
        n_str, n_col = f"{neurons:,}", rgb(WHITE)
        if 15.1 <= t < 19.35:
            u = smooth((t - 15.3) / 1.6)
            val = int(round(10 ** (math.log10(64) + (math.log10(86e9) - math.log10(64)) * u)))
            if u >= 1:
                val = 86_000_000_000
            n_str, n_col = f"{val:,}", rgb(AMBER)
        font(ctx, "dbold", 76 if len(n_str) <= 6 else 54)
        ctx.set_source_rgb(*n_col)
        widths = {str(k): (50 if len(n_str) <= 6 else 36) for k in range(10)}
        widths[","] = 22 if len(n_str) <= 6 else 16
        show_cells(ctx, n_str, 70, 545, widths)
        # accuracy
        s = pct(acc)
        col = rgb(CYAN) if acc >= 0.999 else (rgb(WHITE) if acc >= 0.9 else rgb(MAGENTA) if acc < 0.7 else rgb(AMBER))
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*col)
        widths = {str(k): 50 for k in range(10)}
        widths["%"] = 60
        tw = cells_width(ctx, s, widths)
        show_cells(ctx, s, 910 - tw, 545, widths)
        # step counter
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        show_tracked(ctx, f"STEP {self._step:,}", 70, 585, 0.1 * 22)
        self._network(ctx, si, ps, f)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

    def _network(self, ctx, si, ps, f):
        """Real weights: edge alpha = |w| / max|w| for that layer, node glow = bias sign."""
        sizes = [2] + STAGES[si][0] + [1]
        cols_x = np.linspace(CX - 230, CX + 230, len(sizes))
        pos = []
        for j, n in enumerate(sizes):
            if n == 1:
                ys = np.array([(NET_Y0 + NET_Y1) / 2])
            else:
                span = min(NET_Y1 - NET_Y0, 14 * (n - 1) if n > 8 else 36 * (n - 1))
                ys = np.linspace((NET_Y0 + NET_Y1) / 2 - span / 2, (NET_Y0 + NET_Y1) / 2 + span / 2, n)
            pos.append([(cols_x[j], yv) for yv in ys])
        for j in range(len(sizes) - 1):
            Wm = ps[2 * j]
            mx = np.abs(Wm).max() + 1e-9
            ctx.set_line_width(1.0 if Wm.size > 200 else 1.6)
            for a in range(Wm.shape[0]):
                for b in range(Wm.shape[1]):
                    wv = Wm[a, b] / mx
                    c = rgb(MAGENTA) if wv > 0 else rgb(CYAN)
                    ctx.set_source_rgba(*c, 0.08 + 0.6 * abs(wv) ** 1.5)
                    ctx.move_to(*pos[j][a])
                    ctx.line_to(*pos[j + 1][b])
                    ctx.stroke()
        for j, layer_pos in enumerate(pos):
            r = 7 if sizes[j] <= 8 else 3.2
            for (x, y) in layer_pos:
                ctx.set_source_rgba(1, 1, 1, 0.95)
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.fill()
        # pop when a stage starts
        f0 = STAGES[si][1] - 30
        k = f - f0
        if 0 <= k < 10:
            ctx.set_source_rgba(1, 1, 1, 0.5 * (1 - k / 10))
            ctx.set_line_width(2)
            ctx.rectangle(CX - 260, NET_Y0 - 20, 520, NET_Y1 - NET_Y0 + 40)
            ctx.stroke()

    def _headlines(self, img, f, t):
        for (t_in, t_out, lines, n_in, n_out, style) in self.cues:
            if not (t_in - 1e-6 <= t < t_out + n_out / FPS):
                continue
            k_in = (t - t_in) * FPS
            k_out = (t - t_out) * FPS
            alpha = 1.0
            if n_in and k_in < n_in:
                alpha = k_in / n_in if style == "fade" else ease_out_cubic(k_in / n_in)
            if k_out >= 0 and n_out:
                alpha = min(alpha, 1 - k_out / n_out)
            if alpha <= 0:
                continue
            sc, dy = 1.0, 0.0
            if style == "pop" and k_in < n_in:
                u = ease_out_cubic(k_in / n_in)
                sc, dy = 0.92 + 0.08 * u, 18 * (1 - u)
            if style == "chaos" and k_in < n_in:
                sc = 1.30 - 0.30 * ease_out_back(k_in / n_in)
            surf, ctx = layer()
            tops, bots = [], []
            for (s, kind, size, colr, base, la) in lines:
                font(ctx, kind, size)
                w = ctx.text_extents(s).x_advance
                fs = min(1.0, MAX_TEXT_W / w)
                cy = base - size * 0.36
                ctx.save()
                ctx.translate(CX, cy + dy)
                ctx.scale(sc * fs, sc * fs)
                ctx.translate(-CX, -cy)
                ctx.set_source_rgba(*rgb(colr), la)
                ctx.move_to(CX - w / 2, base)
                ctx.show_text(s)
                ctx.restore()
                tops.append(base - size)
                bots.append(base + size * 0.3)
            y0, y1 = max(0, int(min(tops) - 60)), min(H, int(max(bots) + 60))
            trgb, ta = surf_rgba(surf)
            band = slice(y0, y1)
            a_band = ta[band]
            if style == "chaos":
                g = cv2.GaussianBlur(a_band, (0, 0), 14) * 0.8 * alpha
                img[band] += g[..., None] * np.array(rgb(CYAN), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)

    def _glitch(self, img, f):
        rng = np.random.default_rng(1000 + f)
        out = img.copy()
        for _ in range(6):
            h = int(rng.integers(20, 50))
            y = int(rng.integers(600, H - h))
            sh = int(rng.integers(10, 26)) * (1 if rng.random() < 0.5 else -1)
            out[y:y + h] = np.roll(img[y:y + h], sh, axis=1)
        out[:, 3:, 0] = out[:, :-3, 0]
        out[::3] *= 0.88
        return out
