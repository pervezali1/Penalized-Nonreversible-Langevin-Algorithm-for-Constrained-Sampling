"""Monty Hall, the explanatory cut. Hook: three doors, you pick 1, the host opens a goat, switch or
stay? Proof: the three possible worlds (car behind 1, 2 or 3) side by side; the host opens a goat in
each; switching wins in two of them. Scoreboard: 10,000 games, each played both ways, live
percentages. Kicker: 100 doors, the host opens 98 goats. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, linear_to_srgb, over, rgb,  # noqa: E402
                     show_cells, show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1500                     # 25 s
SEED = 8
GAMES = 10_000
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
DOOR = "#D9CDB4"
CX = 470
# sections (seconds)
HOOK = (0.0, 3.4)
WORLDS = (3.6, 10.8)
RACE = (11.0, 18.2)
HUNDRED = (18.5, 23.4)
LOOP_FADE = (24.3, 24.85)
XFADE = 0.35
MAX_TEXT_W = 820
# hook door events
T_PICK, T_HOST = 1.0, 2.0
# worlds events
T_W_PICK, T_W_HOST, T_W_REVEAL, T_W_TAG, T_W_TALLY = 4.2, 5.4, 7.4, 7.9, 8.8
# race
T_R0, T_R1 = 11.3, 16.6
# hundred doors
T_H_PICK, T_H0, T_H1 = 18.9, 19.3, 21.6


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        car = rng.integers(0, 3, GAMES)
        pick = rng.integers(0, 3, GAMES)
        self.stay_win = car == pick
        self.cum_stay = np.cumsum(self.stay_win).astype(np.int64)
        self.cum_switch = np.cumsum(~self.stay_win).astype(np.int64)
        self.p_stay = self.cum_stay[-1] / GAMES
        self.p_switch = self.cum_switch[-1] / GAMES
        # the 100-door kicker: car behind door 61 (0-based 60), player picks door 23 (0-based 22)
        self.h_car, self.h_pick = 60, 22
        others = [i for i in range(100) if i not in (self.h_car, self.h_pick)]
        self.h_order = np.array(others)[np.random.default_rng(SEED + 1).permutation(len(others))]
        self.col_win = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.col_lose = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.rain = np.random.default_rng(SEED + 2).random((GAMES, 2))
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("3 doors. 1 car. 2 goats.", "black", 66, WHITE, 300, 1), ("You pick door 1. The host opens a goat. Switch?", "dbold", 40, WHITE, 392, 0.9)]
        ps, pw = self.p_stay * 100, self.p_switch * 100
        return [
            (0.0, 3.3, hook, 0, 6, "none"),
            (3.6, 5.2, [("Where's the car? Three worlds, equally likely.", "black", 44, WHITE, 350, 1)], 8, 6, "pop"),
            (5.4, 7.3, [("You always pick door 1.", "black", 48, AMBER, 312, 1), ("The host always opens a goat.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (7.9, 10.7, [("Stay wins in 1 world.", "black", 50, MAGENTA, 312, 1), ("Switch wins in 2.", "black", 60, CYAN, 404, 1)], 6, 6, "chaos"),
            (11.0, 13.4, [("Don't trust the argument? 10,000 games.", "black", 46, WHITE, 312, 1), ("Each game played both ways.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (16.8, 18.3, [(f"Stay: {ps:.1f}%.", "black", 60, MAGENTA, 312, 1), (f"Switch: {pw:.1f}%.", "black", 68, CYAN, 404, 1)], 6, 6, "chaos"),
            (18.6, 20.8, [("Still feels like 50/50? 100 doors.", "black", 48, WHITE, 312, 1), ("You pick one. The host opens 98 goats.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (21.2, 23.3, [("Your door: 1%.", "black", 52, MAGENTA, 312, 1), ("The one he skipped: 99%.", "black", 56, CYAN, 404, 1)], 6, 6, "chaos"),
            (23.5, 24.4, [("The host knows where the car is.", "black", 46, WHITE, 312, 1), ("His choice is information. Switch.", "black", 50, AMBER, 404, 1)], 8, 6, "pop"),
            (24.5, 25.1, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ drawing helpers

    def _door(self, ctx, x, y, w, h, number, state, open_u=0.0, pick=False, alpha=1.0, number_size=None):
        """Draw a door frame at (x, y). state: 'goat' | 'car' | None (what is behind). open_u swings the panel."""
        r = max(4, w * 0.06)
        # dark opening behind
        ctx.new_sub_path()
        ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        ctx.line_to(x + w, y + h)
        ctx.line_to(x, y + h)
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        ctx.set_source_rgba(0.05, 0.06, 0.11, 0.95 * alpha)
        ctx.fill()
        if state is not None and open_u > 0.02:
            self._prize(ctx, x + w / 2, y + h / 2, w, state, alpha * min(1.0, open_u * 1.5))
        # the panel, hinged on the left: width shrinks with the swing
        sw = max(0.04, math.cos(open_u * math.pi * 0.5 * 0.98))
        pw = w * sw
        ctx.new_sub_path()
        ctx.arc(x + pw - r * sw, y + r, r, -math.pi / 2, 0)
        ctx.line_to(x + pw, y + h)
        ctx.line_to(x, y + h)
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        shade = 0.9 - 0.45 * open_u
        ctx.set_source_rgba(*(c * shade for c in rgb(DOOR)), alpha)
        ctx.fill_preserve()
        if pick:
            ctx.set_source_rgba(*rgb(AMBER), alpha)
            ctx.set_line_width(max(3, w * 0.03))
            ctx.stroke()
        else:
            ctx.new_path()
        if sw > 0.3:
            ctx.set_source_rgba(0.1, 0.1, 0.15, 0.8 * alpha)
            ctx.arc(x + pw - w * 0.12, y + h / 2, max(2, w * 0.025), 0, 2 * math.pi)
            ctx.fill()
            font(ctx, "black", number_size or int(h * 0.26))
            s = str(number)
            ctx.set_source_rgba(0.12, 0.1, 0.08, 0.85 * alpha)
            ctx.move_to(x + pw / 2 - ctx.text_extents(s).x_advance / 2 * sw, y + h * 0.33)
            ctx.save()
            ctx.translate(x + pw / 2, 0)
            ctx.scale(sw, 1)
            ctx.translate(-(x + pw / 2), 0)
            ctx.show_text(s)
            ctx.restore()

    def _prize(self, ctx, cx, cy, w, state, alpha):
        k = w / 200.0
        if state == "car":
            ctx.set_source_rgba(*rgb(CYAN), alpha)
            ctx.rectangle(cx - 40 * k, cy - 6 * k, 80 * k, 30 * k)
            ctx.fill()
            ctx.rectangle(cx - 24 * k, cy - 28 * k, 48 * k, 24 * k)
            ctx.fill()
            ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
            for sx in (-1, 1):
                ctx.arc(cx + sx * 22 * k, cy + 26 * k, 9 * k, 0, 2 * math.pi)
                ctx.fill()
        else:
            ctx.set_source_rgba(*rgb(MAGENTA), alpha)
            ctx.arc(cx, cy + 6 * k, 26 * k, 0, 2 * math.pi)
            ctx.fill()
            for sx in (-1, 1):
                ctx.move_to(cx + sx * 14 * k, cy - 14 * k)
                ctx.line_to(cx + sx * 32 * k, cy - 44 * k)
                ctx.line_to(cx + sx * 5 * k, cy - 20 * k)
                ctx.close_path()
                ctx.fill()
            ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
            for sx in (-1, 1):
                ctx.arc(cx + sx * 9 * k, cy, 3.5 * k, 0, 2 * math.pi)
                ctx.fill()

    def _tag(self, ctx, x, y, text, col, alpha, size=22):
        font(ctx, "bold", size)
        tw = tracked_width(ctx, text, 0.08 * size)
        ctx.set_source_rgba(*rgb(col), 0.16 * alpha)
        ctx.rectangle(x - tw / 2 - 16, y - size, tw + 32, size * 1.6)
        ctx.fill()
        ctx.set_source_rgba(*rgb(col), 0.95 * alpha)
        show_tracked(ctx, text, x - tw / 2, y + size * 0.15, 0.08 * size)

    # ------------------------------------------------------------ sections

    def _hook(self, ctx, t, alpha):
        w, h, gap = 262, 390, 42
        x0 = CX - (3 * w + 2 * gap) / 2 + 70
        y = 740
        open3 = ease_out_cubic(seg(t, T_HOST, T_HOST + 0.6))
        for i in range(3):
            self._door(ctx, x0 + i * (w + gap), y, w, h, i + 1, "goat" if i == 2 else None, open3 if i == 2 else 0.0,
                       pick=(i == 0 and t >= T_PICK), alpha=alpha)
        if t >= T_PICK:
            a = alpha * ease_out_cubic(seg(t, T_PICK, T_PICK + 0.3))
            self._tag(ctx, x0 + w / 2, y + h + 50, "YOUR PICK", AMBER, a)
        if t >= T_HOST + 0.4:
            a = alpha * ease_out_cubic(seg(t, T_HOST + 0.4, T_HOST + 0.7))
            self._tag(ctx, x0 + 2 * (w + gap) + w / 2, y + h + 50, "HOST OPENS", MAGENTA, a)
        if t >= T_HOST + 0.8:
            a = alpha * ease_out_cubic(seg(t, T_HOST + 0.8, T_HOST + 1.1))
            self._tag(ctx, x0 + (w + gap) + w / 2, y + h + 50, "SWITCH HERE?", CYAN, a, 24)

    def _worlds(self, ctx, t, alpha):
        w, h, gap = 120, 170, 22
        rows_y = (720, 980, 1240)
        x0 = 300
        host_door = {0: 2, 1: 2, 2: 1}          # world index (car position) -> door the host opens
        u_host = ease_out_cubic(seg(t, T_W_HOST, T_W_HOST + 0.6))
        u_rev = ease_out_cubic(seg(t, T_W_REVEAL, T_W_REVEAL + 0.5))
        for wi in range(3):
            y = rows_y[wi]
            font(ctx, "bold", 20)
            ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
            show_tracked(ctx, "CAR BEHIND", 70, y + 60, 0.1 * 20)
            font(ctx, "dbold", 60)
            ctx.set_source_rgba(*rgb(CYAN), 0.95 * alpha)
            ctx.move_to(70, y + 122)
            ctx.show_text(str(wi + 1))
            for di in range(3):
                x = x0 + di * (w + gap)
                behind = "car" if di == wi else "goat"
                if di == host_door[wi]:
                    ou = u_host
                else:
                    ou = u_rev
                self._door(ctx, x, y, w, h, di + 1, behind, ou, pick=(di == 0 and t >= T_W_PICK), alpha=alpha)
            if t >= T_W_HOST + 0.5:
                a = alpha * ease_out_cubic(seg(t, T_W_HOST + 0.5, T_W_HOST + 0.8))
                self._tag(ctx, x0 + host_door[wi] * (w + gap) + w / 2, y + h + 36, "HOST", MAGENTA, a, 16)
            if t >= T_W_TAG:
                a = alpha * ease_out_cubic(seg(t, T_W_TAG, T_W_TAG + 0.35))
                win = "STAY WINS" if wi == 0 else "SWITCH WINS"
                self._tag(ctx, 880, y + h / 2 + 8, win, MAGENTA if wi == 0 else CYAN, a, 22)
        if t >= T_W_PICK:
            a = alpha * ease_out_cubic(seg(t, T_W_PICK, T_W_PICK + 0.3))
            self._tag(ctx, x0 + w / 2, 700 - 24, "YOUR PICK", AMBER, a, 16)
        if t >= T_W_TALLY:
            a = alpha * ease_out_cubic(seg(t, T_W_TALLY, T_W_TALLY + 0.4))
            font(ctx, "bold", 22)
            ctx.set_source_rgba(1, 1, 1, 0.55 * a)
            show_tracked(ctx, "STAY", 70, 470, 0.14 * 22)
            tw = tracked_width(ctx, "SWITCH", 0.14 * 22)
            show_tracked(ctx, "SWITCH", 910 - tw, 470, 0.14 * 22)
            font(ctx, "dbold", 76)
            ctx.set_source_rgba(*rgb(MAGENTA), a)
            ctx.move_to(70, 545)
            ctx.show_text("1/3")
            ctx.set_source_rgba(*rgb(CYAN), a)
            s = "2/3"
            ctx.move_to(910 - ctx.text_extents(s).x_advance, 545)
            ctx.show_text(s)

    def games_done(self, t):
        if t < T_R0:
            return 0
        if t >= T_R1:
            return GAMES
        u = (t - T_R0) / (T_R1 - T_R0)
        return max(1, int(GAMES * u ** 1.6))

    def _race(self, ctx, emis, t, alpha):
        n = self.games_done(t)
        ws, ww = (int(self.cum_stay[n - 1]), int(self.cum_switch[n - 1])) if n else (0, 0)
        cols = ((240, "STAY", MAGENTA, ws), (700, "SWITCH", CYAN, ww))
        bar_y0, bar_y1, bar_w = 760, 1420, 300
        for (cx, lab, col, wins) in cols:
            frac = wins / n if n else 0.0
            # frame
            ctx.set_line_width(2)
            ctx.set_source_rgba(1, 1, 1, 0.25 * alpha)
            ctx.rectangle(cx - bar_w / 2 + 0.5, bar_y0 + 0.5, bar_w, bar_y1 - bar_y0)
            ctx.stroke()
            # fill
            hh = (bar_y1 - bar_y0) * frac
            ctx.set_source_rgba(*rgb(col), 0.28 * alpha)
            ctx.rectangle(cx - bar_w / 2, bar_y1 - hh, bar_w, hh)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), 0.95 * alpha)
            ctx.rectangle(cx - bar_w / 2, bar_y1 - hh - 3, bar_w, 6)
            ctx.fill()
            # reference lines at 1/3 and 2/3
            for fr_, lab_ in ((1 / 3, "1/3"), (2 / 3, "2/3")):
                yy = bar_y1 - (bar_y1 - bar_y0) * fr_
                ctx.set_dash([4, 6])
                ctx.set_source_rgba(1, 1, 1, 0.3 * alpha)
                ctx.move_to(cx - bar_w / 2, yy + 0.5)
                ctx.line_to(cx + bar_w / 2, yy + 0.5)
                ctx.stroke()
                ctx.set_dash([])
                font(ctx, "bold", 16)
                ctx.set_source_rgba(1, 1, 1, 0.45 * alpha)
                ctx.move_to(cx + bar_w / 2 + 8, yy + 6)
                ctx.show_text(lab_)
            font(ctx, "black", 30)
            ctx.set_source_rgba(*rgb(col), 0.98 * alpha)
            ctx.move_to(cx - ctx.text_extents(lab).x_advance / 2, bar_y0 - 24)
            ctx.show_text(lab)
            font(ctx, "dbold", 72)
            s = f"{frac * 100:.1f}%" if n else "—"
            ctx.set_source_rgba(*rgb(col), alpha)
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, bar_y0 + 100)
            ctx.show_text(s)
            font(ctx, "bold", 18)
            ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
            s = f"WON {wins:,} OF {n:,}"
            show_tracked(ctx, s, cx - tracked_width(ctx, s, 0.08 * 18) / 2, bar_y0 + 136, 0.08 * 18)
        # HUD counter
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
        show_tracked(ctx, "GAMES", 70, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgba(*rgb(WHITE), alpha)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22})
        show_cells(ctx, f"{n:,}", 70, 545, big)
        # rain: the last 600 games fall into their column as dots (emissive)
        lo = max(0, n - 600)
        for g in range(lo, n):
            age = (n - g) / 600.0
            for ci, won in ((0, self.stay_win[g]), (1, not self.stay_win[g])):
                cx = cols[ci][0]
                x = cx + (self.rain[g, 0] - 0.5) * (bar_w - 30)
                y = bar_y0 - 160 + age * 900 * (0.3 + 0.7 * self.rain[g, 1])
                if y > bar_y1 - 6:
                    continue
                col = self.col_win if won else self.col_lose
                xi, yi = int(x), int(y)
                if 0 <= xi < W - 1 and 0 <= yi < H - 1:
                    emis[yi:yi + 2, xi:xi + 2] += col * 0.5 * alpha * (1 - age)

    def _hundred(self, ctx, t, alpha):
        w, h, gx, gy = 72, 96, 86, 112
        x0 = CX + 70 - (10 * gx - (gx - w)) / 2
        y0 = 700
        n_open = int(98 * ease_out_cubic(seg(t, T_H0, T_H1)) ** 1.0)
        opened = set(int(i) for i in self.h_order[:n_open])
        last_t = {int(self.h_order[k]): k for k in range(min(n_open, 98))}
        for i in range(100):
            r, c = divmod(i, 10)
            x, y = x0 + c * gx, y0 + r * gy
            is_pick = i == self.h_pick and t >= T_H_PICK
            if i in opened:
                k = last_t[i]
                t_open = T_H0 + (T_H1 - T_H0) * ((k + 1) / 98) ** 1.0
                ou = 1.0 if t > t_open + 0.25 else ease_out_cubic(seg(t, t_open - 0.25, t_open + 0.05))
                ou = max(ou, 0.6)
                self._door(ctx, x, y, w, h, i + 1, "goat", ou, pick=False, alpha=alpha, number_size=18)
            else:
                self._door(ctx, x, y, w, h, i + 1, None, 0.0, pick=is_pick, alpha=alpha, number_size=18)
        if n_open >= 98 and t >= T_H1 + 0.2:
            # the skipped door pulses
            r, c = divmod(self.h_car, 10)
            x, y = x0 + c * gx, y0 + r * gy
            pulse = 0.5 + 0.5 * math.sin((t - T_H1) * 6)
            ctx.set_source_rgba(*rgb(CYAN), (0.5 + 0.5 * pulse) * alpha)
            ctx.set_line_width(4)
            ctx.rectangle(x - 4, y - 4, w + 8, h + 8)
            ctx.stroke()
            a = alpha * ease_out_cubic(seg(t, T_H1 + 0.2, T_H1 + 0.5))
            self._tag(ctx, x + w / 2, y - 26, "HE SKIPPED THIS", CYAN, a, 16)
        if t >= T_H_PICK:
            r, c = divmod(self.h_pick, 10)
            x, y = x0 + c * gx, y0 + r * gy
            a = alpha * ease_out_cubic(seg(t, T_H_PICK, T_H_PICK + 0.3))
            self._tag(ctx, x + w / 2, y - 26, "YOUR PICK", AMBER, a, 16)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
        show_tracked(ctx, "GOATS OPENED", 70, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgba(*rgb(WHITE), alpha)
        big = {str(c): 50 for c in range(10)}
        show_cells(ctx, f"{min(n_open, 98)}", 70, 545, big)
        w2 = tracked_width(ctx, "DOORS LEFT", 0.14 * 24)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
        show_tracked(ctx, "DOORS LEFT", 910 - w2, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        s = f"{100 - min(n_open, 98)}"
        ctx.set_source_rgba(*rgb(AMBER if n_open < 98 else CYAN), alpha)
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)

    # ------------------------------------------------------------ frame

    def _section_alpha(self, t, a, b):
        """1 inside [a, b] with XFADE ramps at both ends."""
        if t < a - XFADE or t > b + XFADE:
            return 0.0
        return min(1.0, seg(t, a - XFADE, a), 1.0 - seg(t, b, b + XFADE))

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        surf, ctx = layer()
        emis = np.zeros((H, W, 3), np.float32)
        if fade < 1.0:
            # dissolve to the hook's frame-0 state
            a_hook0 = 1 - fade
            self._hook(ctx, 0.0, a_hook0)
        a = self._section_alpha(t, *HOOK) * fade
        if a > 0:
            self._hook(ctx, t, a)
        a = self._section_alpha(t, *WORLDS) * fade
        if a > 0:
            self._worlds(ctx, t, a)
        a = self._section_alpha(t, *RACE) * fade
        if a > 0:
            self._race(ctx, emis, t, a)
        a = self._section_alpha(t, *HUNDRED) * fade
        if a > 0:
            self._hundred(ctx, t, a)
        # compose: emissive rain with bloom over the background, then the cairo layer
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (emis + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        _ruin.Scene._headlines(self, img, f, t)
        return dither(img, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s stay {sc.p_stay:.4f} switch {sc.p_switch:.4f}")
    frames = [0, 80, 150, 230, 300, 380, 460, 540, 640, 720, 800, 900, 1020, 1120, 1200, 1290, 1420, 1499]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (150, 540, 800, 1290):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
