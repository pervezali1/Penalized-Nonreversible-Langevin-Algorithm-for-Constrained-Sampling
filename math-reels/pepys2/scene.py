"""Newton-Pepys, v2: three bet panels, each with its dice and a live histogram of the number of
sixes, magenta below the target and cyan at or above it. The win rate is literally the cyan area.

A: at least one 6 in 6 dice   B: at least two 6s in 12 dice   C: at least three 6s in 18 dice
"""

import math
import os
import sys
from math import comb

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)

FPS = 60
N_FRAMES = 1200
SEED = 6
TRIALS = 10_000
BETS = [("A", 6, 1, "ONE 6 in 6 dice", 7),
        ("B", 12, 2, "TWO 6s in 12 dice", 8),
        ("C", 18, 3, "THREE 6s in 18 dice", 10)]       # (name, dice, need, label, histogram bins)
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
IVORY = "#F2EFE6"
CX = 470
SLOW_T0, SLOW_DT, SLOW_N, TUMBLE = 5.4, 0.5, 6, 0.3
FAST_T0, FAST_T1 = 8.6, 14.4
T_REVEAL, T_WHY = 14.8, 17.2
LOOP_FADE = (19.4, 20.0)
MAX_TEXT_W = 820
PANEL_Y0, PANEL_H = 620, 290
DIE, DIE_GAP = 34, 5
HX0, HX1 = 70, 860
BAR_H = 140


def exact(n, k):
    return sum(comb(n, j) * (1 / 6) ** j * (5 / 6) ** (n - j) for j in range(k, n + 1))


def binom(n, j):
    return comb(n, j) * (1 / 6) ** j * (5 / 6) ** (n - j)


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.faces = [rng.integers(1, 7, (TRIALS, n)) for (_, n, _k, _l, _b) in BETS]
        self.sixes = [(fc == 6).sum(1) for fc in self.faces]
        self.wins = [s >= k for s, (_, n, k, _l, _b) in zip(self.sixes, BETS)]
        self.cum = [np.concatenate([[0], np.cumsum(w)]) for w in self.wins]
        # cumulative histogram of the number of sixes: hist[t, j] = count of trials < t with j sixes
        self.hist = []
        for s, (_, n, k, _l, nb) in zip(self.sixes, BETS):
            oh = np.zeros((TRIALS, nb), np.int32)
            oh[np.arange(TRIALS), np.minimum(s, nb - 1)] = 1
            self.hist.append(np.concatenate([np.zeros((1, nb), np.int32), np.cumsum(oh, axis=0)]))
        self.exact = [exact(n, k) for (_, n, k, _l, _b) in BETS]
        self.final = [float(w.mean()) for w in self.wins]
        assert self.final[0] > self.final[1] > self.final[2]
        self.cues = self._cues()
        self.bg = self._background()

    def _cues(self):
        hook = [("1693. A gambler asks Newton:", "dbold", 52, WHITE, 292, 0.85), ("which bet wins?", "black", 88, WHITE, 384, 1)]
        pa = f"{100 * self.final[0]:.1f}%"
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.6, 5.2, [("Pepys bet on C.", "black", 88, AMBER, 320, 1), ("More dice, more sixes… right?", "dbold", 48, WHITE, 400, 0.9)], 8, 6, "pop"),
            (8.9, 10.8, [("10,000 rolls each.", "black", 76, WHITE, 350, 1)], 8, 6, "pop"),
            (11.2, 14.4, [("Cyan = win. Magenta = lose.", "black", 60, WHITE, 312, 1), ("Which pile has the most cyan?", "dbold", 46, AMBER, 396, 1)], 8, 6, "pop"),
            (T_REVEAL, T_WHY - 0.1, [("Newton was right.", "black", 84, WHITE, 312, 1), (f"A wins: {pa}", "black", 72, CYAN, 404, 1)], 8, 6, "chaos"),
            (T_WHY, 19.3, [("The target is exactly the average.", "black", 54, WHITE, 312, 1), ("More dice: the pile hugs it. Less luck.", "dbold", 44, AMBER, 396, 1)], 8, 6, "pop"),
            (19.5, 20.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1000) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ trial schedule

    def trials_done(self, t):
        if t < SLOW_T0:
            return 0
        if t < FAST_T0:
            return min(SLOW_N, int((t - SLOW_T0) // SLOW_DT) + (1 if (t - SLOW_T0) % SLOW_DT >= TUMBLE else 0))
        if t >= FAST_T1:
            return TRIALS
        u = (t - FAST_T0) / (FAST_T1 - FAST_T0)
        return min(TRIALS, SLOW_N + int((TRIALS - SLOW_N) * u ** 2.2))

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        done = self.trials_done(t)
        if mix > 0.5:
            done = 0
        tumbling = SLOW_T0 <= t < FAST_T0 and (t - SLOW_T0) % SLOW_DT < TUMBLE
        just_landed = SLOW_T0 <= t < FAST_T0 and not tumbling
        cur = max(0, done - 1)
        rng = np.random.default_rng(10_000 + f)
        revealed = t >= T_REVEAL and mix < 0.5
        why = t >= T_WHY and mix < 0.5

        emis_s, ctx = layer()
        for bi, (name, n, k, label, nb) in enumerate(BETS):
            y0 = PANEL_Y0 + bi * PANEL_H
            # dice strip
            if done == 0 and not tumbling:
                faces = self.faces[bi][0]
            elif tumbling:
                faces = rng.integers(1, 7, n)
            elif FAST_T0 <= t < FAST_T1:
                faces = self.faces[bi][cur] if f % 2 == 0 else rng.integers(1, 7, n)
            else:
                faces = self.faces[bi][cur]
            total_w = n * DIE + (n - 1) * DIE_GAP
            x_start = HX0
            for i, face in enumerate(faces):
                self._die(ctx, x_start + i * (DIE + DIE_GAP), y0 + 42, DIE, int(face),
                          highlight=(face == 6 and not tumbling), jitter=tumbling, rng=rng)
            # histogram of sixes
            counts = self.hist[bi][done] if done else np.zeros(nb)
            scale_ref = max(binom(n, j) for j in range(nb)) if done >= 50 else max(1e-9, counts.max() / max(done, 1))
            bw = (HX1 - HX0) / nb
            base_y = y0 + 96 + BAR_H
            dim = 0.45 if (revealed and bi != 0) else 1.0
            for j in range(nb):
                frac = counts[j] / done if done else 0.0
                h = min(BAR_H, BAR_H * frac / scale_ref) if done else 0.0
                win = j >= k
                col = rgb(CYAN) if win else rgb(MAGENTA)
                x = HX0 + j * bw + 3
                if h > 0:
                    ctx.set_source_rgba(*col, (0.85 if win else 0.6) * dim)
                    ctx.rectangle(x, base_y - h, bw - 6, h)
                    ctx.fill()
                # just-landed bar flash
                if just_landed and done and j == min(int(self.sixes[bi][done - 1]), nb - 1):
                    age = ((t - SLOW_T0) % SLOW_DT - TUMBLE) / (SLOW_DT - TUMBLE)
                    ctx.set_source_rgba(1, 1, 1, 0.8 * (1 - age))
                    ctx.rectangle(x, base_y - max(h, 6) - 4, bw - 6, 4)
                    ctx.fill()
            # exact outline at the reveal
            if revealed:
                ctx.set_line_width(1.5)
                ctx.set_source_rgba(1, 1, 1, 0.55 * dim)
                for j in range(nb):
                    hx = BAR_H * binom(n, j) / scale_ref
                    x = HX0 + j * bw + 3
                    ctx.move_to(x, base_y - hx)
                    ctx.line_to(x + bw - 6, base_y - hx)
                ctx.stroke()
        emis_rgb, emis_a = surf_rgba(emis_s)
        emis = (emis_rgb ** 2.2).astype(np.float32)
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.35 * cv2.GaussianBlur(small, (0, 0), 3) + 0.25 * cv2.GaussianBlur(small, (0, 0), 10),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        lin = 1 - (1 - self.bg) * (1 - np.clip(glow * 0.6, 0, 1))
        img = linear_to_srgb(lin).astype(np.float32)
        over(img, emis_rgb, emis_a)
        self._hud(img, f, t, done, mix, revealed, why)
        return dither(img, f)

    def _die(self, ctx, x, y, s, face, highlight, jitter, rng):
        ctx.save()
        if jitter:
            ctx.translate(x + s / 2, y + s / 2)
            ctx.rotate(rng.uniform(-0.25, 0.25))
            ctx.scale(rng.uniform(0.92, 1.04), rng.uniform(0.92, 1.04))
            ctx.translate(-s / 2, -s / 2)
        else:
            ctx.translate(x, y)
        r = s * 0.18
        ctx.new_sub_path()
        ctx.arc(s - r, r, r, -math.pi / 2, 0)
        ctx.arc(s - r, s - r, r, 0, math.pi / 2)
        ctx.arc(r, s - r, r, math.pi / 2, math.pi)
        ctx.arc(r, r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        if highlight:
            ctx.set_source_rgb(*rgb(CYAN))
        else:
            ctx.set_source_rgba(*rgb(IVORY), 0.92 if not jitter else 0.75)
        ctx.fill()
        pip_r = s * 0.08
        pos = {1: [(0.5, 0.5)], 2: [(0.25, 0.25), (0.75, 0.75)], 3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
               4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
               5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
               6: [(0.25, 0.22), (0.75, 0.22), (0.25, 0.5), (0.75, 0.5), (0.25, 0.78), (0.75, 0.78)]}[face]
        ctx.set_source_rgb(*((1, 1, 1) if highlight else rgb("#141828")))
        for (px, py) in pos:
            ctx.arc(px * s, py * s, pip_r, 0, 2 * math.pi)
            ctx.fill()
        ctx.restore()

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, done, mix, revealed, why):
        surf, ctx = layer()
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "ROLLS", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "LEADING BET", 0.14 * 24)
        show_tracked(ctx, "LEADING BET", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{done:,}", 70, 545, big)
        fr = [c[done] / done if done else 0.0 for c in self.cum]
        if done == 0:
            lead_s, lead_c = "C?", rgb(AMBER)
        else:
            li = int(np.argmax(fr))
            lead_s, lead_c = BETS[li][0], (rgb(CYAN) if li == 0 else rgb(AMBER))
        ctx.set_source_rgb(*lead_c)
        tw = cells_width(ctx, lead_s, big)
        show_cells(ctx, lead_s, 910 - tw, 545, big)

        for bi, (name, n, k, label, nb) in enumerate(BETS):
            y0 = PANEL_Y0 + bi * PANEL_H
            dim = 0.45 if (revealed and bi != 0) else 1.0
            # label + win rate
            font(ctx, "black", 30)
            ctx.set_source_rgba(*(rgb(CYAN) if revealed and bi == 0 else rgb(WHITE)), 0.9 * dim)
            ctx.move_to(HX0, y0 + 26)
            ctx.show_text(name)
            font(ctx, "bold", 24)
            ctx.set_source_rgba(1, 1, 1, 0.7 * dim)
            show_tracked(ctx, f"at least {label}", HX0 + 40, y0 + 26, 0.04 * 24)
            if done:
                col = rgb(CYAN) if bi == 0 else rgb(WHITE) if bi == 1 else rgb(AMBER)
                font(ctx, "dbold", 44)
                s = f"{100 * fr[bi]:.1f}%"
                ctx.set_source_rgba(*col, dim)
                ctx.move_to(910 - ctx.text_extents(s).x_advance, y0 + 32)
                ctx.show_text(s)
            # win/lose flash during the slow rolls
            if SLOW_T0 <= t < FAST_T0 and done > 0:
                age = (t - SLOW_T0) % SLOW_DT
                if age >= TUMBLE:
                    won = bool(self.wins[bi][done - 1])
                    a = 1 - (age - TUMBLE) / (SLOW_DT - TUMBLE)
                    font(ctx, "black", 26)
                    s = "WIN" if won else "LOSE"
                    ctx.set_source_rgba(*(rgb(CYAN) if won else rgb(MAGENTA)), a)
                    ctx.move_to(910 - ctx.text_extents(s).x_advance, y0 + 62)
                    ctx.show_text(s)
            # histogram axis: bin labels, target line, average marker
            bw = (HX1 - HX0) / nb
            base_y = y0 + 96 + BAR_H
            ctx.set_line_width(1)
            ctx.set_source_rgba(1, 1, 1, 0.18 * dim)
            ctx.move_to(HX0, base_y + 0.5)
            ctx.line_to(HX1, base_y + 0.5)
            ctx.stroke()
            font(ctx, "bold", 18)
            for j in range(nb):
                s = f"{j}" if j < nb - 1 else f"{j}+"
                ctx.set_source_rgba(1, 1, 1, 0.45 * dim)
                ctx.move_to(HX0 + j * bw + bw / 2 - ctx.text_extents(s).x_advance / 2, base_y + 20)
                ctx.show_text(s)
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.4 * dim)
            s = "SIXES"
            ctx.move_to(HX1 - ctx.text_extents(s).x_advance, base_y + 38)
            ctx.show_text(s)
            # target line between bin k-1 and k
            tx = HX0 + k * bw
            ctx.set_line_width(2)
            ctx.set_source_rgba(*rgb(CYAN), 0.9 * dim)
            ctx.set_dash([6, 5])
            ctx.move_to(tx, y0 + 88)
            ctx.line_to(tx, base_y + 2)
            ctx.stroke()
            ctx.set_dash([])
            font(ctx, "bold", 18)
            ctx.set_source_rgba(*rgb(CYAN), 0.9 * dim)
            show_tracked(ctx, f"NEED {k}+", tx + 8, y0 + 100, 0.08 * 18)
            if why:
                ax = HX0 + k * bw + bw / 2
                ctx.set_source_rgba(*rgb(AMBER), 0.95)
                ctx.move_to(ax - 9, base_y + 4)
                ctx.line_to(ax + 9, base_y + 4)
                ctx.line_to(ax, base_y - 6)
                ctx.close_path()
                ctx.fill()
                font(ctx, "bold", 17)
                s = f"AVERAGE = {n}/6 = {k}"
                ctx.set_source_rgba(*rgb(AMBER), 0.95)
                show_tracked(ctx, s, ax + 14, y0 + 124, 0.06 * 17)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

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
