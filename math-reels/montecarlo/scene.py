"""Monte Carlo pi. Darts rain into a square with a circle inside; pi ~ 4 x inside / total, the
estimate's correct digits turning green, up to one million darts. Then 300 independent dart-throwers
drawn side by side form a funnel that narrows like 1/sqrt(N): every extra digit costs 100x more darts.
frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, font, layer, linear_to_srgb, over, rgb, show_cells,  # noqa: E402
                     show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ruin = _load("ruin_scene", ("ruin", "scene.py"))
_mart = _load("martingale_scene", ("martingale", "scene.py"))
splat_paths = _mart.splat_paths

FPS = 60
N_FRAMES = 1380                     # 23 s
SEED = 1
N_MAX = 1_000_000
N_RUNS, N_CK = 300, 400             # funnel: independent runs, log-spaced checkpoints from 10 to 1e6
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
CX = 470
SQ = (110, 680, 970, 1540)          # the dartboard square
CHART = (90, 760, 880, 1440)        # the funnel chart
RAIN_T0, RAIN_T1 = 2.6, 11.0
FUNNEL_T0, FUNNEL_T1 = 13.2, 16.6
Y_LO, Y_HI = 2.4, 3.9
LOOP_FADE = (22.3, 22.85)
MAX_TEXT_W = 820
PI_DIGITS = "31415926535897932384626433832795"
SIGMA = 4 * math.sqrt((math.pi / 4) * (1 - math.pi / 4))      # 1.642: std of the estimate is SIGMA / sqrt(N)


@njit(cache=True)
def splat_darts(acc, xs, ys, inside, n, col_in, col_out, x0, y0, side, r, gain):
    """Draw the first n darts. r > 0: soft dots of that radius; r == 0: single bilinear splats."""
    for i in range(n):
        x = x0 + (xs[i] + 1) * 0.5 * side
        y = y0 + (1 - ys[i]) * 0.5 * side
        col = col_in if inside[i] else col_out
        if r > 0:
            for yy in range(int(y) - r, int(y) + r + 2):
                for xx in range(int(x) - r, int(x) + r + 2):
                    if 0 <= xx < acc.shape[1] and 0 <= yy < acc.shape[0]:
                        d2 = (xx - x) ** 2 + (yy - y) ** 2
                        g = math.exp(-d2 / (r * r * 0.45)) * gain
                        for c in range(3):
                            acc[yy, xx, c] += col[c] * g
        else:
            xi = int(x)
            yi = int(y)
            if xi < 0 or yi < 0 or xi >= acc.shape[1] - 1 or yi >= acc.shape[0] - 1:
                continue
            fx = x - xi
            fy = y - yi
            for c in range(3):
                v = col[c] * gain
                acc[yi, xi, c] += v * (1 - fx) * (1 - fy)
                acc[yi, xi + 1, c] += v * fx * (1 - fy)
                acc[yi + 1, xi, c] += v * (1 - fx) * fy
                acc[yi + 1, xi + 1, c] += v * fx * fy


@njit(cache=True)
def funnel_runs(n_runs, checkpoints, seed):
    """Running pi estimates of n_runs independent dart-throwers at the given dart counts."""
    np.random.seed(seed)
    out = np.zeros((n_runs, len(checkpoints)))
    n_max = checkpoints[-1]
    for r in range(n_runs):
        inside = 0
        k = 0
        for i in range(1, n_max + 1):
            x = np.random.random() * 2 - 1
            y = np.random.random() * 2 - 1
            if x * x + y * y <= 1.0:
                inside += 1
            if i == checkpoints[k]:
                out[r, k] = 4.0 * inside / i
                k += 1
    return out


def digits_correct(est):
    s = f"{est:.10f}".replace(".", "")
    k = 0
    for a, b in zip(s, PI_DIGITS):
        if a != b:
            break
        k += 1
    return k


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.xs = rng.random(N_MAX) * 2 - 1
        self.ys = rng.random(N_MAX) * 2 - 1
        self.inside = (self.xs ** 2 + self.ys ** 2 <= 1.0)
        self.cum_in = np.cumsum(self.inside).astype(np.int64)
        self.est_final = 4 * self.cum_in[-1] / N_MAX
        self.digits_final = digits_correct(self.est_final)
        ck = np.unique(np.round(np.logspace(1, 6, N_CK)).astype(np.int64))
        self.ck = ck
        self.runs = funnel_runs(N_RUNS, ck, SEED + 100)
        self.runs_scaled = np.ascontiguousarray(np.clip(self.runs, Y_LO, Y_HI) - Y_LO)
        self.spread_final = float(np.std(self.runs[:, -1]))
        self.within_final = int((np.abs(self.runs[:, -1] - math.pi) < 1.96 * SIGMA / math.sqrt(ck[-1])).sum())
        self.col_in = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.col_out = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.cols_runs = np.tile(np.array(srgb_to_linear(rgb(AMBER)), np.float32), (N_RUNS, 1)).astype(np.float32)
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    def _cues(self):
        hook = [("Throw darts at random.", "black", 64, WHITE, 300, 1), ("Count the ones inside the circle. That's π.", "dbold", 42, WHITE, 392, 0.9)]
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.6, 5.2, [("Circle ÷ square = π/4.", "black", 56, WHITE, 312, 1), ("So π ≈ 4 × inside ÷ total.", "dbold", 44, WHITE, 396, 0.9)], 8, 6, "pop"),
            (6.0, 8.4, [("Watch the digits lock in.", "black", 56, AMBER, 350, 1)], 8, 6, "pop"),
            (9.0, 10.8, [("Slowly.", "black", 72, MAGENTA, 350, 1)], 8, 6, "pop"),
            (11.2, 13.0, [(f"1,000,000 darts: π ≈ {self.est_final:.4f}", "black", 56, WHITE, 312, 1), (f"Correct digits: {self.digits_final}.", "black", 60, GREEN, 404, 1)], 6, 6, "chaos"),
            (13.4, 15.6, [("300 dart-throwers, side by side.", "black", 54, AMBER, 312, 1), ("Each one's estimate as the darts pile up.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (16.0, 18.4, [("The spread shrinks like 1/√N.", "black", 54, WHITE, 312, 1), ("Each extra digit costs 100× more darts.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (18.8, 20.8, [("10 digits of π: 10¹⁹ darts.", "black", 56, MAGENTA, 312, 1), ("A billion a second, for a thousand years.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (21.0, 22.4, [("Slow for π.", "black", 60, WHITE, 312, 1), ("Unbeatable in 1,000 dimensions.", "black", 50, CYAN, 404, 1)], 8, 6, "pop"),
            (22.6, 23.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ schedules

    def darts_shown(self, t):
        if t < RAIN_T0:
            return 0
        if t >= RAIN_T1:
            return N_MAX
        u = (t - RAIN_T0) / (RAIN_T1 - RAIN_T0)
        return max(1, min(N_MAX, int(round(10 ** (6 * u)))))

    def funnel_shown(self, t):
        if t < FUNNEL_T0:
            return 0
        if t >= FUNNEL_T1:
            return len(self.ck)
        u = (t - FUNNEL_T0) / (FUNNEL_T1 - FUNNEL_T0)
        return max(1, int(len(self.ck) * u))

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        mix = 0.0 if t < FUNNEL_T0 - 0.3 else smooth((t - (FUNNEL_T0 - 0.3)) / 0.4)
        if fade < 1.0:
            mix *= fade
        out = None
        if mix < 1.0:
            out = self._darts_frame(t if fade >= 1.0 else 0.0)
        if mix > 0.0:
            fimg = self._funnel_frame(t)
            out = fimg if out is None else out * (1 - mix) + fimg * mix
        _ruin.Scene._headlines(self, out, f, t)
        return dither(out, f)

    def _compose(self, acc, k=1.5):
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-k * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        return linear_to_srgb(lin).astype(np.float32)

    # ------------------------------------------------------------ darts

    def _darts_frame(self, t):
        n = self.darts_shown(t)
        acc = np.zeros((H, W, 3), np.float32)
        x0, y0, x1, y1 = SQ
        side = float(x1 - x0)
        if n > 0:
            # per-dart light falls off with the count so the board fills instead of blowing out
            energy = 20.0 * min(1.0, (300.0 / n) ** 0.8)
            r = 4 if n <= 300 else 3 if n <= 3000 else 2
            gain = float(min(3.0, energy / (0.45 * math.pi * r * r)))
            splat_darts(acc, self.xs, self.ys, self.inside, n, self.col_in, self.col_out,
                        float(x0), float(y0), side, r, gain)
        img = self._compose(acc)
        self._darts_hud(img, t, n)
        return img

    def _darts_hud(self, img, t, n):
        surf, ctx = layer()
        x0, y0, x1, y1 = SQ
        # square and circle
        ctx.set_line_width(2)
        ctx.set_source_rgba(1, 1, 1, 0.35)
        ctx.rectangle(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(CYAN), 0.6)
        ctx.arc((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, 0, 2 * math.pi)
        ctx.stroke()
        inside = int(self.cum_in[n - 1]) if n else 0
        est = 4 * inside / n if n else 0.0
        k = digits_correct(est) if n else 0
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "DARTS", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "INSIDE", 0.14 * 24)
        show_tracked(ctx, "INSIDE", 910 - w, 470, 0.14 * 24)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22})
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{n:,}", 70, 545, big)
        s = f"{inside:,}"
        ctx.set_source_rgb(*rgb(CYAN))
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
        # the estimate line: pi ~ 4 x inside / darts = 3.14xxxx with correct digits green
        font(ctx, "bold", 26)
        lab = "π ≈ 4 × INSIDE ÷ DARTS  ="
        ctx.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(ctx, lab, 70, 626, 0.06 * 26)
        lw = tracked_width(ctx, lab, 0.06 * 26)
        font(ctx, "dbold", 44)
        cw = 30
        sx = 70 + lw + 26
        if n:
            s = f"{est:.6f}"
            di = 0
            for ch in s:
                if ch == ".":
                    ctx.set_source_rgba(1, 1, 1, 0.9)
                    ctx.move_to(sx, 632)
                    ctx.show_text(ch)
                    sx += 14
                    continue
                ok = di < k
                ctx.set_source_rgba(*(rgb(GREEN) if ok else (1, 1, 1)), 0.95 if ok else 0.45)
                ctx.move_to(sx + cw / 2 - ctx.text_extents(ch).x_advance / 2, 632)
                ctx.show_text(ch)
                sx += cw
                di += 1
            font(ctx, "bold", 22)
            ctx.set_source_rgba(*rgb(GREEN), 0.9)
            lab2 = f"{k} DIGIT{'S' if k != 1 else ''}"
            show_tracked(ctx, lab2, 910 - tracked_width(ctx, lab2, 0.08 * 22), 632, 0.08 * 22)
        # true pi, small, under the board
        font(ctx, "bold", 20)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        show_tracked(ctx, "π = 3.14159265…", x0, y1 + 44, 0.08 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ funnel

    def _funnel_frame(self, t):
        n_show = self.funnel_shown(t)
        acc = np.zeros((H, W, 3), np.float32)
        x0, y0, x1, y1 = CHART
        if n_show > 1:
            start = np.ones(N_RUNS, np.int64)
            end = np.full(N_RUNS, n_show, np.int64)
            splat_paths(acc, self.runs_scaled, start, end, self.cols_runs, float(x0), float(y0),
                        float(x1 - x0), float(y1 - y0), len(self.ck), Y_HI - Y_LO, 0.05)
        img = self._compose(acc)
        self._funnel_hud(img, t, n_show)
        return img

    def _funnel_hud(self, img, t, n_show):
        surf, ctx = layer()
        x0, y0, x1, y1 = CHART

        def px(nd):
            return x0 + (x1 - x0) * (math.log10(nd) - 1) / 5

        def py(v):
            return y1 - (v - Y_LO) / (Y_HI - Y_LO) * (y1 - y0)

        n_d = int(self.ck[max(0, n_show - 1)]) if n_show else 10
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "DARTS EACH", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "SPREAD (±2σ)", 0.14 * 24)
        show_tracked(ctx, "SPREAD (±2σ)", 910 - w, 470, 0.14 * 24)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22})
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{n_d:,}", 70, 545, big)
        sp = 1.96 * SIGMA / math.sqrt(n_d)
        s = f"±{sp:.4f}" if sp < 0.1 else f"±{sp:.3f}"
        ctx.set_source_rgb(*rgb(AMBER))
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, f"{N_RUNS} INDEPENDENT RUNS", 70, 626, 0.08 * 24)
        # pi line and the 95% envelope
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(*rgb(GREEN), 0.8)
        ctx.move_to(x0, py(math.pi) + 0.5)
        ctx.line_to(x1, py(math.pi) + 0.5)
        ctx.stroke()
        font(ctx, "bold", 22)
        show_tracked(ctx, "π", x1 + 12, py(math.pi) + 8, 0)
        ctx.set_dash([5, 7])
        ctx.set_source_rgba(1, 1, 1, 0.55)
        for sgn in (1, -1):
            first = True
            for nd in np.logspace(1, 6, 200):
                v = math.pi + sgn * 1.96 * SIGMA / math.sqrt(nd)
                v = min(max(v, Y_LO), Y_HI)
                if first:
                    ctx.move_to(px(nd), py(v))
                    first = False
                else:
                    ctx.line_to(px(nd), py(v))
            ctx.stroke()
        ctx.set_dash([])
        font(ctx, "bold", 20)
        ctx.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(ctx, "95% BAND  ±1.96 × 1.64 / √N", x0, y0 - 14, 0.08 * 20)
        # axes
        ctx.set_line_width(1)
        ctx.set_source_rgba(1, 1, 1, 0.2)
        ctx.move_to(x0, y1 + 0.5)
        ctx.line_to(x1, y1 + 0.5)
        ctx.stroke()
        font(ctx, "bold", 18)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        for nd, lab in ((10, "10"), (100, "100"), (1000, "1k"), (10000, "10k"), (100000, "100k"), (1000000, "1M")):
            ctx.move_to(px(nd) - ctx.text_extents(lab).x_advance / 2, y1 + 32)
            ctx.show_text(lab)
        s = "DARTS"
        ctx.move_to(x1 - ctx.text_extents(s).x_advance, y1 + 56)
        ctx.show_text(s)
        for v in (2.5, 3.0, 3.5):
            ctx.set_source_rgba(1, 1, 1, 0.08)
            ctx.move_to(x0, py(v) + 0.5)
            ctx.line_to(x1, py(v) + 0.5)
            ctx.stroke()
            ctx.set_source_rgba(1, 1, 1, 0.45)
            s = f"{v:.1f}"
            ctx.move_to(x0 - ctx.text_extents(s).x_advance - 10, py(v) + 6)
            ctx.show_text(s)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s est {sc.est_final} digits {sc.digits_final} "
          f"funnel std@1M {sc.spread_final:.4f} within95 {sc.within_final}/{N_RUNS}")
    frames = [0, 170, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760, 820, 880, 960, 1060, 1200, 1379]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (400, 700, 960):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
