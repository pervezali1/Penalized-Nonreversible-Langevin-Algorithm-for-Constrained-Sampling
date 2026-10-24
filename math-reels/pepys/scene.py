"""Frame renderer for the Newton-Pepys reel: three bets rolled 10,000 times each.

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
BETS = [("A", 6, 1, "at least ONE 6 in 6 dice"),
        ("B", 12, 2, "at least TWO 6s in 12 dice"),
        ("C", 18, 3, "at least THREE 6s in 18 dice")]
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
IVORY = "#F2EFE6"
CX = 470
SLOW_T0, SLOW_DT, SLOW_N, TUMBLE = 6.0, 0.5, 6, 0.3
FAST_T0, FAST_T1 = 9.0, 15.0
LOOP_FADE = (19.4, 20.0)
MAX_TEXT_W = 820
# layout: (label baseline, dice size, lines of dice, first dice top y, bar y)
LAYOUT = [(742, 100, 1, 758, 882), (942, 64, 1, 958, 1046), (1106, 64, 2, 1122, 1282)]
BAR_X0, BAR_X1 = 70, 870


def exact(n, k):
    return sum(comb(n, j) * (1 / 6) ** j * (5 / 6) ** (n - j) for j in range(k, n + 1))


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.faces = [rng.integers(1, 7, (TRIALS, n)) for (_, n, _k, _l) in BETS]
        self.wins = [(fc == 6).sum(1) >= k for fc, (_, n, k, _l) in zip(self.faces, BETS)]
        self.cum = [np.concatenate([[0], np.cumsum(w)]) for w in self.wins]
        self.exact = [exact(n, k) for (_, n, k, _l) in BETS]
        self.final = [float(w.mean()) for w in self.wins]
        assert self.final[0] > self.final[1] > self.final[2]
        self.cues = self._cues()
        self.bg = self._background()
        self.hook_faces = [fc[0] for fc in self.faces]

    def _cues(self):
        hook = [("1693. Pepys asks Newton:", "dbold", 54, WHITE, 292, 0.85), ("which bet wins?", "black", 88, WHITE, 384, 1)]
        pa = f"{100 * self.final[0]:.1f}%"
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.6, 5.6, [("Pepys bet on C.", "black", 88, AMBER, 320, 1), ("18 dice, three sixes.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (9.3, 11.2, [("10,000 rolls each.", "black", 76, WHITE, 350, 1)], 8, 6, "pop"),
            (15.2, 17.4, [("Newton was right.", "black", 84, WHITE, 312, 1), (f"A wins: {pa}", "black", 72, CYAN, 404, 1)], 8, 6, "chaos"),
            (17.6, 19.3, [("More dice, fewer surprises.", "black", 64, WHITE, 312, 1), ("And you need one.", "dbold", 56, AMBER, 396, 1)], 8, 6, "pop"),
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
        """Number of completed trials at video time t (0 .. 10,000)."""
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
        cur = max(0, done - 1)
        rng = np.random.default_rng(10_000 + f)

        emis_s, ctx = layer()
        for bi, ((name, n, k, label), (ly, size, lines, y0, by)) in enumerate(zip(BETS, LAYOUT)):
            if done == 0 and not tumbling:
                faces = self.hook_faces[bi]
            elif tumbling or (FAST_T0 <= t < FAST_T1):
                faces = rng.integers(1, 7, n) if (tumbling or f % 2 == 0) else self.faces[bi][cur]
                if tumbling and (t - SLOW_T0) % SLOW_DT > TUMBLE - 0.03:
                    faces = self.faces[bi][done]
            else:
                faces = self.faces[bi][cur]
            settled = not tumbling and done > 0 and t < FAST_T0
            per_line = n // lines
            gap = 10 if size >= 64 else 6
            total_w = per_line * size + (per_line - 1) * gap
            x_start = CX - total_w / 2
            for i, face in enumerate(faces):
                line, col = divmod(i, per_line)
                x = x_start + col * (size + gap)
                y = y0 + line * (size + 10)
                jitter = tumbling
                self._die(ctx, x, y, size, int(face), highlight=(face == 6 and not tumbling), jitter=jitter, rng=rng)
        emis_rgb, emis_a = surf_rgba(emis_s)
        emis = (emis_rgb ** 2.2).astype(np.float32)
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.35 * cv2.GaussianBlur(small, (0, 0), 3) + 0.25 * cv2.GaussianBlur(small, (0, 0), 10),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        lin = self.bg.copy()
        lin = 1 - (1 - lin) * (1 - np.clip(glow * 0.6, 0, 1))
        img = linear_to_srgb(lin).astype(np.float32)
        over(img, emis_rgb, emis_a)
        self._hud(img, f, t, done, mix)
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
        pip_r = s * 0.075
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

    def _hud(self, img, f, t, done, mix):
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
        revealed = t >= 15.2 and mix < 0.5
        for bi, ((name, n, k, label), (ly, size, lines, y0, by)) in enumerate(zip(BETS, LAYOUT)):
            dim = 0.45 if (revealed and bi != 0) else 1.0
            font(ctx, "bold", 26)
            ctx.set_source_rgba(*(rgb(CYAN) if revealed and bi == 0 else (1, 1, 1)), 0.75 * dim)
            show_tracked(ctx, f"{name}   {label}", 70, ly, 0.06 * 26)
            # bar
            ctx.set_source_rgba(1, 1, 1, 0.10)
            ctx.rectangle(BAR_X0, by, BAR_X1 - BAR_X0, 10)
            ctx.fill()
            if done:
                col = rgb(CYAN) if bi == 0 else rgb(AMBER) if bi == 2 else rgb(WHITE)
                ctx.set_source_rgba(*col, 0.9 * dim)
                ctx.rectangle(BAR_X0, by, (BAR_X1 - BAR_X0) * fr[bi], 10)
                ctx.fill()
                font(ctx, "dbold", 30)
                s = f"{100 * fr[bi]:.1f}%"
                ctx.set_source_rgba(*col, dim)
                ctx.move_to(BAR_X1 + 12, by + 12)
                ctx.show_text(s)
            # exact-probability tick, shown from the reveal
            if revealed:
                ex = BAR_X0 + (BAR_X1 - BAR_X0) * self.exact[bi]
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.set_line_width(2)
                ctx.move_to(ex, by - 6)
                ctx.line_to(ex, by + 16)
                ctx.stroke()
                font(ctx, "bold", 20)
                ctx.set_source_rgba(1, 1, 1, 0.7)
                s = f"exact {100 * self.exact[bi]:.1f}%"
                ctx.move_to(ex - ctx.text_extents(s).x_advance / 2, by + 36)
                ctx.show_text(s)
            # win/lose flash during the slow rolls
            if SLOW_T0 <= t < FAST_T0 and done > 0:
                age = (t - SLOW_T0) % SLOW_DT
                if age >= TUMBLE:
                    won = bool(self.wins[bi][done - 1])
                    a = 1 - (age - TUMBLE) / (SLOW_DT - TUMBLE)
                    font(ctx, "black", 30)
                    s = "WIN" if won else "LOSE"
                    ctx.set_source_rgba(*(rgb(CYAN) if won else rgb(MAGENTA)), a)
                    ctx.move_to(910 - ctx.text_extents(s).x_advance, ly)
                    ctx.show_text(s)
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
