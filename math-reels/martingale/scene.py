"""The Martingale: bet $1, double after every loss, reset after a win. One gambler first (so the
bets 1, 2, 4 ... 512 and the cliff are visible), then 1,000 Martingale players as a waterfall,
then the expected value per round. Roulette red/black, p = 18/38. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, linear_to_srgb,  # noqa: E402
                     over, rgb, show_cells, show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1440                     # 24 s
SEED = 3
SOLO_IDX = 9                        # which of the 1,000 is the "one gambler" (busts at spin 651)
N, BANK = 1000, 1023                # $1,023 covers $1 + 2 + 4 + ... + 512: nine doublings
P_WIN = 18 / 38
SPINS = 5000
DS = 5                              # draw every 5th spin for the crowd (1,001 samples per walk)
STREAK = 10
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
CX = 470
CHART = (90, 720, 880, 1440)
SOLO_T0, STREAK_T0, STREAK_DT = 5.0, 8.2, 0.19
SOLO_YMAX = 1500.0
CROWD_T0, CROWD_T1 = 12.2, 15.6
CROWD_YMAX = 3600.0
LOOP_FADE = (23.4, 24.0)
MAX_TEXT_W = 820
P10 = (1 - P_WIN) ** STREAK
EV_ROUND = (1 - P10) * 1 - BANK * P10


@njit(cache=True)
def splat_paths(acc, paths, start, end, cols, x0, y0, xw, yh, s_total, ymax, w):
    """Draw segments (s-1, s) for s in [start[i], end[i]) of each path; x by sample index, y by value."""
    n_walks = paths.shape[0]
    for i in range(n_walks):
        for s in range(start[i], end[i]):
            if s < 1 or s >= paths.shape[1]:
                continue
            xa = x0 + xw * (s - 1) / (s_total - 1)
            xb = x0 + xw * s / (s_total - 1)
            ya = y0 + yh * (1 - paths[i, s - 1] / ymax)
            yb = y0 + yh * (1 - paths[i, s] / ymax)
            dx = xb - xa
            dy = yb - ya
            length = math.sqrt(dx * dx + dy * dy)
            n = int(length / 0.8) + 1
            for k in range(n):
                u = (k + 0.5) / n
                x = xa + dx * u
                y = ya + dy * u
                xi = int(x)
                yi = int(y)
                if xi < 0 or yi < 0 or xi >= acc.shape[1] - 1 or yi >= acc.shape[0] - 1:
                    continue
                fx = x - xi
                fy = y - yi
                for c in range(3):
                    v = cols[i, c] * w * 0.8
                    acc[yi, xi, c] += v * (1 - fx) * (1 - fy)
                    acc[yi, xi + 1, c] += v * fx * (1 - fy)
                    acc[yi + 1, xi, c] += v * (1 - fx) * fy
                    acc[yi + 1, xi + 1, c] += v * fx * fy


@njit(cache=True)
def splat_dots(acc, xs, ys, col, r, gain):
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for yy in range(int(y) - r, int(y) + r + 2):
            for xx in range(int(x) - r, int(x) + r + 2):
                if 0 <= xx < acc.shape[1] and 0 <= yy < acc.shape[0]:
                    d2 = (xx - x) ** 2 + (yy - y) ** 2
                    g = math.exp(-d2 / (r * r * 0.5)) * gain
                    for c in range(3):
                        acc[yy, xx, c] += col[c] * g


def simulate(rng):
    """Martingale for N players over SPINS spins. A player is out after STREAK losses in a row: the
    $1,023 stake is gone (what's left is the $1 wins banked before). Returns bankroll paths (N, SPINS+1), bust spin (SPINS+1 if never), win mask."""
    win = rng.random((N, SPINS)) < P_WIN
    bank = np.full(N, float(BANK))
    bet = np.ones(N)
    streak = np.zeros(N, np.int64)
    alive = np.ones(N, bool)
    paths = np.zeros((N, SPINS + 1))
    paths[:, 0] = BANK
    bust = np.full(N, SPINS + 1)
    for s in range(SPINS):
        w = win[:, s]
        bank = np.where(alive, np.where(w, bank + bet, bank - bet), bank)
        bet = np.where(w, 1.0, bet * 2)
        streak = np.where(w, 0, streak + 1)
        out = alive & (streak >= STREAK)           # ten losses in a row: the $1,023 stake is gone, you're out
        bust[out] = s + 1
        alive &= ~out
        paths[:, s + 1] = bank
    return paths, bust, win


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.paths_full, self.bust, self.win = simulate(rng)
        self.busted = self.bust <= SPINS
        self.n_bust = int(self.busted.sum())
        self.n_alive = N - self.n_bust
        self.net_final = int(round(self.paths_full[:, -1].sum() - N * BANK))
        self.avg_alive = float(self.paths_full[~self.busted, -1].mean()) if self.n_alive else 0.0
        # solo gambler
        self.solo = self.paths_full[SOLO_IDX]
        self.solo_bust = int(self.bust[SOLO_IDX])
        assert 500 <= self.solo_bust <= 850, self.solo_bust
        assert not self.win[SOLO_IDX, self.solo_bust - STREAK:self.solo_bust].any()
        self.solo_x = int(math.ceil((self.solo_bust + 40) / 100.0) * 100)
        self.solo_left = int(self.solo[self.solo_bust])
        self.solo_peak = int(self.solo[:self.solo_bust].max())
        # crowd (downsampled)
        self.paths = np.ascontiguousarray(self.paths_full[:, ::DS])
        S = self.paths.shape[1]
        self.alive_until = np.minimum(np.ceil(self.bust / DS).astype(np.int64), S - 1)
        amber = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.cols_crowd = np.tile(amber * 0.9, (N, 1)).astype(np.float32)
        self.col_amber = amber
        self.col_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.col_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    def _cues(self):
        hook = [("Lose? Double your bet.", "black", 66, WHITE, 300, 1), ("Win? You're $1 up. Start again.", "dbold", 44, WHITE, 392, 0.9)]
        p10 = f"{P10 * 100:.2f}%"
        return [
            (0.0, 2.5, hook, 0, 6, "none"),
            (2.7, 4.8, [("The Martingale. $1,023 in your pocket.", "black", 50, AMBER, 304, 1), ("Enough to double nine times: $1 to $512.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (5.2, 7.8, [("One gambler. Every round ends $1 up.", "black", 50, AMBER, 350, 1)], 8, 6, "pop"),
            (8.0, 10.3, [("Then ten losses in a row.", "black", 64, MAGENTA, 350, 1)], 8, 6, "pop"),
            (10.5, 12.0, [("$1,023 gone in ten spins.", "black", 58, MAGENTA, 312, 1), (f"Next bet: $1,024. You have ${self.solo_left}.", "dbold", 42, WHITE, 396, 0.9)], 6, 6, "chaos"),
            (CROWD_T0, 14.6, [("1,000 Martingale players.", "black", 60, AMBER, 312, 1), ("5,000 spins each.", "dbold", 46, WHITE, 396, 0.9)], 8, 6, "pop"),
            (15.8, 17.6, [(f"Busted: {self.n_bust} of 1,000.", "black", 68, MAGENTA, 312, 1), (f"All 1,000 together: −${-self.net_final:,}.", "dbold", 44, WHITE, 396, 0.9)], 6, 6, "chaos"),
            (17.8, 20.0, [(f"Each round: {100 - P10 * 100:.2f}% chance of +$1.", "black", 46, GREEN, 312, 1), (f"{p10} chance of −$1,023.", "black", 52, MAGENTA, 404, 1)], 8, 6, "pop"),
            (20.2, 22.2, [(f"Average: −{abs(EV_ROUND) * 100:.0f}¢ a round.", "black", 64, WHITE, 312, 1), ("The house's 5.3% of everything you bet.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (22.4, 23.5, [("It doesn't change if you lose.", "black", 54, WHITE, 312, 1), ("Only when.", "black", 80, MAGENTA, 404, 1)], 8, 6, "pop"),
            (23.6, 24.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ schedules

    def solo_spins(self, t):
        """How many of the solo gambler's spins are shown at time t."""
        if t < SOLO_T0:
            return 0
        pre = self.solo_bust - STREAK
        if t < STREAK_T0:
            u = (t - SOLO_T0) / (STREAK_T0 - 0.3 - SOLO_T0)
            return int(pre * min(1.0, u) ** 0.9)
        return pre + min(STREAK, 1 + int((t - STREAK_T0) / STREAK_DT))

    def crowd_spins(self, t):
        if t < CROWD_T0:
            return 0
        if t >= CROWD_T1:
            return SPINS
        u = (t - CROWD_T0) / (CROWD_T1 - CROWD_T0)
        return int(SPINS * u ** 1.3)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        mode = "hook" if t < SOLO_T0 or fade < 0.5 else "solo" if t < CROWD_T0 else "crowd"
        x0, y0, x1, y1 = CHART
        acc = np.zeros((H, W, 3), np.float32)
        if mode == "solo":
            n = self.solo_spins(t)
            pre = self.solo_bust - STREAK
            p = self.solo[None, :]
            splat_paths(acc, p, np.array([1]), np.array([min(n, pre) + 1]), self.col_amber[None, :] * 0.9,
                        float(x0), float(y0), float(x1 - x0), float(y1 - y0), self.solo_x + 1, SOLO_YMAX, 0.9)
            if n > pre:
                splat_paths(acc, p, np.array([pre + 1]), np.array([n + 1]), self.col_mag[None, :],
                            float(x0), float(y0), float(x1 - x0), float(y1 - y0), self.solo_x + 1, SOLO_YMAX, 1.2)
            # head dot
            xh = x0 + (x1 - x0) * n / self.solo_x
            yh = y0 + (y1 - y0) * (1 - self.solo[n] / SOLO_YMAX)
            col = self.col_mag if n >= self.solo_bust else self.col_amber
            splat_dots(acc, np.array([xh]), np.array([yh]), col, 6, 1.2 if n >= self.solo_bust else 0.7)
        elif mode == "crowd":
            n = self.crowd_spins(t)
            n_show = min(self.paths.shape[1] - 1, n // DS)
            end = np.minimum(n_show, self.alive_until) + 1
            splat_paths(acc, self.paths, np.ones(N, np.int64), end.astype(np.int64), self.cols_crowd,
                        float(x0), float(y0), float(x1 - x0), float(y1 - y0), self.paths.shape[1], CROWD_YMAX, 0.035)
            done = self.busted & (self.alive_until <= n_show)
            if done.any():
                s = self.alive_until[done]
                xs = x0 + (x1 - x0) * s / (self.paths.shape[1] - 1)
                ys = y0 + (y1 - y0) * (1 - self.paths[done, s] / CROWD_YMAX)
                splat_dots(acc, xs.astype(np.float64), ys.astype(np.float64), self.col_mag, 4, 0.9)
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, mode)
        return dither(img, f)

    # ------------------------------------------------------------ HUD

    def _solo_state(self, n):
        """(bankroll, next bet, trailing losses, busted) after n shown spins."""
        bank = int(self.solo[n])
        if n >= self.solo_bust:
            return bank, 2 ** STREAK, STREAK, True
        k = 0
        while k < n and not self.win[SOLO_IDX, n - 1 - k]:
            k += 1
        return bank, 2 ** k, k, False

    def _hud(self, img, f, t, mode):
        surf, ctx = layer()
        x0, y0, x1, y1 = CHART
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, "$": 44})
        if mode == "crowd":
            n = self.crowd_spins(t)
            n_show = min(self.paths.shape[1] - 1, n // DS)
            n_bust = int((self.busted & (self.bust <= n)).sum())
            net = int(round(self.paths_full[:, n].sum() - N * BANK))
            left_l, left_v, left_c = "SPINS EACH", f"{n:,}", WHITE
            right_l, right_v, right_c = "STILL PLAYING", f"{N - n_bust:,}", AMBER if N - n_bust else MAGENTA
            pills = ((f"BUSTED  {n_bust:,}", MAGENTA, 70, "l"), (f"PLAYERS' NET  {'+' if net >= 0 else '−'}${abs(net):,}", GREEN if net >= 0 else MAGENTA, 910, "r"))
            ymax, lines = CROWD_YMAX, ((BANK, "$1,023  START", 0.12), (0.0, "$0", 0.22))
        else:
            n = self.solo_spins(t) if mode == "solo" else 0
            bank, bet, k, busted = self._solo_state(n)
            left_l, left_v, left_c = "NEXT BET", f"${bet:,}", MAGENTA if k >= 3 else WHITE
            right_l, right_v, right_c = "BANKROLL", f"${bank:,}", MAGENTA if busted else AMBER
            pills = ((f"SPIN  {n:,}", WHITE, 70, "l"), ((f"LOSSES IN A ROW  {k}" if k else "ROUNDS WON  " + f"{int(self.win[SOLO_IDX, :n].sum()):,}"), MAGENTA if k else GREEN, 910, "r"))
            ymax, lines = SOLO_YMAX, ((BANK, "$1,023  START", 0.12), (0.0, "$0", 0.22))
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, left_l, 70, 470, 0.14 * 24)
        w = tracked_width(ctx, right_l, 0.14 * 24)
        show_tracked(ctx, right_l, 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(left_c))
        # pop the bet when it changes during the streak
        sc = 1.0
        if mode == "solo" and t >= STREAK_T0:
            k_since = ((t - STREAK_T0) % STREAK_DT) * FPS if n < self.solo_bust else (t - (STREAK_T0 + (STREAK - 1) * STREAK_DT)) * FPS
            sc = 1.0 + 0.25 * max(0.0, 1 - k_since / 7.0)
        ctx.save()
        ctx.translate(70, 545)
        ctx.scale(sc, sc)
        show_cells(ctx, left_v, 0, 0, big)
        ctx.restore()
        ctx.set_source_rgb(*rgb(right_c))
        tw = cells_width(ctx, right_v, big)
        show_cells(ctx, right_v, 910 - tw, 545, big)
        font(ctx, "bold", 26)
        for (lab, col, xx, align) in pills:
            lw = tracked_width(ctx, lab, 0.08 * 26)
            px = xx if align == "l" else xx - lw - 36
            ctx.set_source_rgba(*rgb(col), 0.14)
            ctx.rectangle(px, 586, lw + 36, 44)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), 0.95)
            show_tracked(ctx, lab, px + 18, 617, 0.08 * 26)
        # outcome strip (solo): last 16 spins as cells
        if mode == "solo" and n > 0:
            k0 = max(0, n - 16)
            cw = 44
            sx = 910 - (n - k0) * cw
            for j in range(k0, n):
                won = self.win[SOLO_IDX, j]
                col = GREEN if won else MAGENTA
                a = 0.35 + 0.65 * (j - k0 + 1) / (n - k0)
                ctx.set_source_rgba(*rgb(col), 0.18 * a)
                ctx.rectangle(sx + (j - k0) * cw + 4, 652, cw - 8, 34)
                ctx.fill()
                ctx.set_source_rgba(*rgb(col), 0.95 * a)
                font(ctx, "bold", 22)
                s = "W" if won else "L"
                ctx.move_to(sx + (j - k0) * cw + cw / 2 - ctx.text_extents(s).x_advance / 2, 677)
                ctx.show_text(s)
        # chart axes
        if mode != "hook":
            ctx.set_line_width(1)
            for (val, lab, a) in lines:
                y = y0 + (y1 - y0) * (1 - val / ymax)
                ctx.set_source_rgba(1, 1, 1, a)
                ctx.set_dash([4, 6] if val else [])
                ctx.move_to(x0, y + 0.5)
                ctx.line_to(x1, y + 0.5)
                ctx.stroke()
                ctx.set_dash([])
                font(ctx, "bold", 20)
                ctx.set_source_rgba(*(rgb(MAGENTA) if val == 0 else (1, 1, 1)), 0.75)
                show_tracked(ctx, lab, x0, y - 8 if val else y + 26, 0.08 * 20)
            for val in ((1000, 2000, 3000) if mode == "crowd" else (500,)):
                y = y0 + (y1 - y0) * (1 - val / ymax)
                if val == BANK:
                    continue
                ctx.set_source_rgba(1, 1, 1, 0.07)
                ctx.move_to(x0, y + 0.5)
                ctx.line_to(x1, y + 0.5)
                ctx.stroke()
                font(ctx, "bold", 18)
                ctx.set_source_rgba(1, 1, 1, 0.4)
                s = f"${val:,}"
                ctx.move_to(x1 - ctx.text_extents(s).x_advance, y - 6)
                ctx.show_text(s)
            font(ctx, "bold", 18)
            ctx.set_source_rgba(1, 1, 1, 0.45)
            xmax, step = (SPINS, 1000) if mode == "crowd" else (self.solo_x, 100)
            for k in range(0, xmax + 1, step):
                xx = x0 + (x1 - x0) * k / xmax
                s = (f"{k // 1000}k" if k else "0") if mode == "crowd" else f"{k}"
                ctx.move_to(xx - ctx.text_extents(s).x_advance / 2, y1 + 50)
                ctx.show_text(s)
            s = "SPINS"
            ctx.move_to(x1 - ctx.text_extents(s).x_advance, y1 + 72)
            ctx.show_text(s)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        _ruin.Scene._headlines(self, img, f, t)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s busted {sc.n_bust} alive {sc.n_alive} net {sc.net_final} "
          f"solo bust {sc.solo_bust} left {sc.solo_left} peak {sc.solo_peak} EV {EV_ROUND:.4f} p10 {P10:.6f}")
    frames = [0, 170, 320, 420, 480, 500, 540, 580, 610, 660, 740, 800, 870, 960, 1080, 1200, 1340, 1439]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
