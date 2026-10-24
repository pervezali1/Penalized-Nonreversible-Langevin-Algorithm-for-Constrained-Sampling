"""Frame renderer for the shuffled-labels reel. Same look as the neurons reel: the background is
the network's decision field; dots are the training points (hollow ring = currently wrong);
diamonds are held-out test points shown during the test phases."""

import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from train import forward  # noqa: E402  (the neurons reel's forward pass)
from ovtrain import FPS, HIDDEN, LOOP, N_FRAMES, SHUFFLE, STAGES, T_TEST_A, T_TEST_B, frame_step  # noqa: E402

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
SQ_C = (470, 1120)
SQ_H = 400
NET_Y0, NET_Y1 = 600, 740
MAX_TEXT_W = 820


def pct(a):
    return f"{int(math.floor(a * 100 + 1e-9))}%"


class Scene:
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.X, self.y, self.y_shuf = d["X"], d["y"], d["y_shuf"]
        self.Xt, self.yt, self.yt_rand = d["Xt"], d["yt"], d["yt_rand"]
        self.params = []
        for si in range(2):
            ps, k = [], 0
            while f"s{si}_p{k}" in d:
                ps.append(d[f"s{si}_p{k}"])
                k += 1
            self.params.append(ps)
        self.frames = self.meta["frames"]
        self.train_acc = np.array(self.meta["train"])
        self.test_acc = np.array(self.meta["test"])
        self.final = self.meta["final"]
        gw, gh = W // 4, H // 4
        xs = (np.arange(gw) * 4 + 2 - SQ_C[0]) / SQ_H
        ys = (np.arange(gh) * 4 + 2 - SQ_C[1]) / SQ_H
        gx, gy = np.meshgrid(xs, ys)
        self.grid = np.stack([gx.ravel(), gy.ravel()], 1)
        self.gshape = (gh, gw)
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.px = SQ_C[0] + self.X[:, 0] * SQ_H
        self.py = SQ_C[1] + self.X[:, 1] * SQ_H
        self.tpx = SQ_C[0] + self.Xt[:, 0] * SQ_H
        self.tpy = SQ_C[1] + self.Xt[:, 1] * SQ_H
        self.shuffle_order = np.random.default_rng(3).permutation(len(self.X))   # stagger of the label flips
        self.cues = self._cues()

    def _cues(self):
        fa, fb = self.final
        hook = [("Can a neural network learn", "dbold", 50, WHITE, 292, 0.85), ("pure noise?", "black", 96, WHITE, 392, 1)]
        every = "every dot." if fb["train"] >= 0.995 else "almost every dot."
        return [
            (0.0, 2.0, hook, 0, 6, "none"),
            (2.0, 3.6, [("Real labels.", "black", 84, CYAN, 350, 1)], 8, 6, "pop"),
            (T_TEST_A, 6.4, [(f"Train {pct(fa['train'])}. Test {pct(fa['test'])}.", "black", 66, WHITE, 312, 1), ("It learned the pattern.", "dbold", 50, CYAN, 396, 1)], 8, 6, "pop"),
            (6.7, 8.6, [("Now I shuffle the labels.", "black", 66, WHITE, 312, 1), ("Pure noise. Nothing to learn.", "dbold", 48, AMBER, 396, 1)], 8, 6, "pop"),
            (9.2, 11.0, [("It keeps going…", "black", 76, WHITE, 350, 1)], 8, 6, "pop"),
            (11.4, 13.7, [(f"Train {pct(fb['train'])}.", "black", 92, WHITE, 316, 1), (f"It memorised {every}", "dbold", 48, AMBER, 400, 1)], 8, 6, "pop"),
            (T_TEST_B, 16.1, [(f"Test: {pct(fb['test'])}.", "black", 110, MAGENTA, 320, 1), ("A coin flip.", "dbold", 56, WHITE, 404, 1)], 6, 6, "chaos"),
            (16.3, 18.0, [("Training accuracy can lie.", "black", 62, WHITE, 312, 1), ("Only unseen data tells the truth.", "dbold", 46, AMBER, 396, 1)], 8, 6, "pop"),
            (18.2, 19.35, [("That's overfitting.", "black", 88, MAGENTA, 350, 1)], 8, 6, "pop"),
            (19.45, 20.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ helpers

    def _params_at(self, f):
        si, step = self.frames[f]
        return si, [p[step] for p in self.params[si]]

    def _field(self, ps):
        logit = forward(ps, self.grid)[0]
        return (1 / (1 + np.exp(-np.clip(logit, -30, 30)))).reshape(self.gshape).astype(np.float32)

    def _labels_at(self, f):
        """Per-point label colour mix (0 = real label, 1 = shuffled), staggered during the shuffle."""
        a, b = SHUFFLE
        if f < a:
            return np.zeros(len(self.X))
        if f >= LOOP[0]:
            u = 1 - smooth((f - LOOP[0]) / (LOOP[1] - LOOP[0]))
            return np.full(len(self.X), u)
        if f >= b:
            return np.ones(len(self.X))
        u = (f - a) / (b - a)
        return np.clip((u * 1.3 - self.shuffle_order / len(self.X) * 0.3) / 1.0, 0, 1) ** 0.5 * (1.0)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        si, ps = self._params_at(f)
        mix = smooth((f - LOOP[0]) / (LOOP[1] - LOOP[0])) if f >= LOOP[0] else 0.0
        p_small = self._field(ps)
        in_shuffle = SHUFFLE[0] <= f < SHUFFLE[1]
        if in_shuffle:
            u = smooth((f - SHUFFLE[0]) / (SHUFFLE[1] - SHUFFLE[0]))
            p_small = (1 - u) * self._field(self._params_at(SHUFFLE[0] - 1)[1]) + u * self._field(self._params_at(SHUFFLE[1])[1])
        if mix > 0:                                 # dissolve from the shuffled result back to the untrained field
            p_small = (1 - mix) * self._field(self._params_at(LOOP[0] - 1)[1]) + mix * self._field(ps)
        lab_mix = self._labels_at(f)
        labels = np.where(lab_mix > 0.5, self.y_shuf, self.y)

        p = np.clip(cv2.resize(p_small, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
        conf = np.abs(2 * p - 1)
        k = (conf ** 0.9 * 0.04)[..., None]
        lin = self.c_base + k * (p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan)
        edge = np.exp(-((p - 0.5) / 0.03) ** 2)
        lin += (edge * 0.09)[..., None] * (0.55 + 0.45 * (p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan))
        lin *= self._vignette()[..., None]

        testing = (T_TEST_A <= t < SHUFFLE[0] / FPS) or (t >= T_TEST_B and mix < 0.5)
        pts = self._points(ps, labels, lab_mix, si, testing)
        small = cv2.resize(pts, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.6 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.6 * (pts + glow))
        lin = 1 - (1 - np.clip(lin, 0, 1)) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, si, ps, labels, testing, mix, in_shuffle)
        return dither(img, f)

    _vig = None

    def _vignette(self):
        if Scene._vig is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - CX) / 700) ** 2 + ((yy - 1000) / 1100) ** 2)
            Scene._vig = (1 - 0.35 * np.clip(r - 0.6, 0, 1) ** 1.5).astype(np.float32)
        return Scene._vig

    def _points(self, ps, labels, lab_mix, si, testing):
        pred = forward(ps, self.X)[0] > 0
        right = pred == (labels > 0.5)
        surf, ctx = layer()
        for i in range(len(self.X)):
            col = rgb(MAGENTA) if labels[i] > 0.5 else rgb(CYAN)
            x, y = self.px[i], self.py[i]
            flip = 1 - abs(2 * lab_mix[i] - 1) if 0 < lab_mix[i] < 1 else 0.0   # white flash mid-flip
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
            if flip > 0:
                ctx.set_source_rgba(1, 1, 1, 0.8 * flip)
                ctx.arc(x, y, 9, 0, 2 * math.pi)
                ctx.fill()
        if testing:
            tl = self.yt if si == 0 else self.yt_rand
            tpred = forward(ps, self.Xt)[0] > 0
            for i in range(len(self.Xt)):
                col = rgb(MAGENTA) if tl[i] > 0.5 else rgb(CYAN)
                x, y = self.tpx[i], self.tpy[i]
                ok = tpred[i] == (tl[i] > 0.5)
                ctx.save()
                ctx.translate(x, y)
                ctx.rotate(math.pi / 4)
                if ok:
                    ctx.set_source_rgba(*col, 0.95)
                    ctx.rectangle(-5, -5, 10, 10)
                    ctx.fill()
                else:
                    ctx.set_source_rgba(1, 1, 1, 0.9)
                    ctx.set_line_width(2)
                    ctx.rectangle(-6, -6, 12, 12)
                    ctx.stroke()
                ctx.restore()
        prgb, _ = surf_rgba(surf)
        return (prgb ** 2.2 * 0.9).astype(np.float32)

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, si, ps, labels, testing, mix, in_shuffle):
        surf, ctx = layer()
        tr = float(((forward(ps, self.X)[0] > 0) == (labels > 0.5)).mean())
        if testing:
            tl = self.yt if si == 0 else self.yt_rand
            te = float(((forward(ps, self.Xt)[0] > 0) == (tl > 0.5)).mean())
        else:
            te = None
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "TRAIN ACCURACY", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "TEST ACCURACY", 0.14 * 24)
        show_tracked(ctx, "TEST ACCURACY", 910 - w, 470, 0.14 * 24)
        big = {str(k): 50 for k in range(10)}
        big["%"] = 60
        font(ctx, "dbold", 76)
        s = pct(tr)
        ctx.set_source_rgb(*(rgb(CYAN) if tr >= 0.999 else rgb(WHITE) if tr >= 0.9 else rgb(AMBER) if tr >= 0.7 else rgb(MAGENTA)))
        show_cells(ctx, s, 70, 545, big)
        if te is None:
            s, col = "—", (1, 1, 1)
            ctx.set_source_rgba(1, 1, 1, 0.35)
        else:
            s = pct(te)
            col = rgb(CYAN) if te >= 0.85 else rgb(MAGENTA) if te < 0.6 else rgb(AMBER)
            ctx.set_source_rgb(*col)
        tw = cells_width(ctx, s, big)
        show_cells(ctx, s, 910 - tw, 545, big)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        step = 0 if (mix > 0.5 or in_shuffle) else self.frames[f][1]
        label = "LABELS: SHUFFLED" if (si == 1 and mix <= 0.5) else "LABELS: REAL"
        show_tracked(ctx, f"STEP {step:,}   ·   {label}   ·   {sum(HIDDEN)} NEURONS", 70, 585, 0.1 * 22)
        self._network(ctx, ps)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

    def _network(self, ctx, ps):
        sizes = [2] + HIDDEN + [1]
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
                img[band] += g[..., None] * np.array(rgb(MAGENTA), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
