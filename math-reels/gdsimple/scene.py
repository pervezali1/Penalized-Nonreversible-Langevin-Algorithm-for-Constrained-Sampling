"""Gradient descent in two dimensions, defined as simply as possible, on a white background. A bowl
drawn as a wireframe grid surface in perspective, one ball, and at every step one arrow: the slope
under the ball, pointing downhill, as long as the slope is steep. The ball takes the step, the arrow
is redrawn, the steps shrink as the floor flattens, and it settles. Slow pacing, 30 s.
frame(f) -> uint8 1080x1920 RGB."""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402

FPS = 60
N_FRAMES = 1800                     # 30 s
# light theme
INK, INK2, GRID_C = "#111827", "#6B7280", "#9CA3AF"
BLUE, AMBER, MAGENTA, GREEN = "#2563EB", "#D97706", "#DB2777", "#059669"
SURF_LO, SURF_HI = "#BFDBFE", "#FFFFFF"       # surface fill: low ground tinted blue, high ground white
CX = 540
# the bowl: an elongated bowl rotated so the path curves
ROT = math.radians(32)
A_, B_ = 0.5, 1.8
LR = 0.19
START = (2.4, 0.2)
N_STEPS = 24
# timeline
T_SURF = 2.6                        # surface draws in
T_BALL = 6.0
T_FIRST = 9.6                       # first arrow
STEP_DT = 0.62                      # per step: arrow 0.26 s, move 0.26 s, rest
LOOP_FADE = (29.3, 29.85)
MAX_TEXT_W = 880
# projection
DOM = 2.6
S_XY = 118.0
P_CX, P_CY = 540, 1190
Z_PX = 430.0                          # pixel height of the bowl's rim


def f_xy(x, y):
    u = math.cos(ROT) * x + math.sin(ROT) * y
    v = -math.sin(ROT) * x + math.cos(ROT) * y
    return A_ * u * u + B_ * v * v


def grad_xy(x, y):
    u = math.cos(ROT) * x + math.sin(ROT) * y
    v = -math.sin(ROT) * x + math.cos(ROT) * y
    gu, gv = 2 * A_ * u, 2 * B_ * v
    return (math.cos(ROT) * gu - math.sin(ROT) * gv, math.sin(ROT) * gu + math.cos(ROT) * gv)


def z_max():
    return max(f_xy(sx * DOM, sy * DOM) for sx in (-1, 1) for sy in (-1, 1))


S_Z = Z_PX / z_max()


def project(x, y, z):
    """A gentle isometric view: x to the lower right, y to the lower left, z up."""
    sx = P_CX + (x - y) * math.cos(math.radians(30)) * S_XY
    sy = P_CY + (x + y) * math.sin(math.radians(30)) * S_XY - z * S_Z
    return sx, sy


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


class Scene:
    def __init__(self):
        # the path of the ball
        pts = [START]
        for _ in range(N_STEPS):
            x, y = pts[-1]
            gx, gy = grad_xy(x, y)
            pts.append((x - LR * gx, y - LR * gy))
        self.path = pts
        self.heights = [f_xy(*p) for p in pts]
        self.slopes = [math.hypot(*grad_xy(*p)) for p in pts]
        self.cues = self._cues()
        self._surf_cache = {}

    # ------------------------------------------------------------ cues

    def _cues(self):
        h0, hN = self.heights[0], self.heights[-1]
        hook = [("Gradient descent,", "black", 64, INK, 300, 1), ("in the simplest way I can.", "black", 50, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 5.8, [("A hill.", "black", 64, INK, 300, 1), ("Height = how wrong you are.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.2, 9.2, [("A ball, dropped anywhere.", "black", 54, INK, 300, 1), ("It wants the lowest point. It can't see the hill.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.4, [("One rule.", "black", 64, AMBER, 300, 1), ("Feel the slope under your feet. Step downhill.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.8, 17.4, [("The arrow is the slope.", "black", 50, INK, 300, 1), ("Steep hill, long arrow, big step.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.8, 21.4, [("Near the bottom the slope fades.", "black", 46, INK, 300, 1), ("Smaller arrows, smaller steps. It settles.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.8, 25.2, [("That's all gradient descent is.", "black", 50, INK, 300, 1), ("Gradient = the slope. Descent = walk down it.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (25.6, 29.0, [("In a neural network the two axes are two weights.", "black", 40, INK, 300, 1), ("Real networks: millions of axes. Same rule.", "dbold", 40, AMBER, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ schedule

    def state(self, t):
        """(step index k of the ball's current position, phase, u): phase 'rest' | 'arrow' | 'move'."""
        if t < T_FIRST:
            return 0, "rest", 0.0
        k = int((t - T_FIRST) / STEP_DT)
        if k >= N_STEPS:
            return N_STEPS, "rest", 0.0
        u = (t - T_FIRST - k * STEP_DT) / STEP_DT
        if u < 0.42:
            return k, "arrow", u / 0.42
        if u < 0.84:
            return k, "move", (u - 0.42) / 0.42
        return k + 1, "rest", 0.0

    # ------------------------------------------------------------ drawing

    def _surface(self, ctx, reveal, alpha):
        """Filled, stroked grid quads back to front; reveal in [0,1] grows the drawn region from the far corner."""
        n = 22
        xs = np.linspace(-DOM, DOM, n + 1)
        lo, hi = rgb(SURF_LO), rgb(SURF_HI)
        zmax = z_max()
        # draw order: far (x+y small) to near
        cells = [(i, j) for i in range(n) for j in range(n)]
        cells.sort(key=lambda ij: xs[ij[0]] + xs[ij[1]])
        n_show = int(len(cells) * ease_out_cubic(reveal))
        ctx.set_line_width(1.2)
        for (i, j) in cells[:n_show]:
            quad = [(xs[i], xs[j]), (xs[i + 1], xs[j]), (xs[i + 1], xs[j + 1]), (xs[i], xs[j + 1])]
            zs = [f_xy(x, y) for x, y in quad]
            zm = sum(zs) / 4
            k = min(1.0, zm / zmax) ** 0.55
            col = tuple(lo[c] + (hi[c] - lo[c]) * k for c in range(3))
            pts = [project(x, y, z) for (x, y), z in zip(quad, zs)]
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source_rgba(*col, 0.92 * alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*rgb(GRID_C), 0.55 * alpha)
            ctx.stroke()
        # axis labels
        if reveal >= 1.0:
            font(ctx, "bold", 20)
            ctx.set_source_rgba(*rgb(INK2), 0.9 * alpha)
            x_, y_ = project(DOM + 0.3, -DOM - 0.3, 0)
            show_tracked(ctx, "WEIGHT 1", x_ - 40, y_ + 54, 0.1 * 20)
            x_, y_ = project(-DOM - 0.3, DOM + 0.3, 0)
            show_tracked(ctx, "WEIGHT 2", x_ - 60, y_ + 54, 0.1 * 20)
            x_, y_ = project(-DOM, -DOM, zmax)
            show_tracked(ctx, "\u2191 ERROR", x_ + 16, y_ + 8, 0.1 * 20)

    def _arrow(self, ctx, p0, p1, col, alpha, width=7):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 2:
            return
        ux, uy = dx / L, dy / L
        head = min(26, L * 0.5)
        ctx.set_source_rgba(*rgb(col), alpha)
        ctx.set_line_width(width)
        ctx.move_to(x0, y0)
        ctx.line_to(x1 - ux * head * 0.8, y1 - uy * head * 0.8)
        ctx.stroke()
        ctx.move_to(x1, y1)
        ctx.line_to(x1 - ux * head - uy * head * 0.55, y1 - uy * head + ux * head * 0.55)
        ctx.line_to(x1 - ux * head + uy * head * 0.55, y1 - uy * head - ux * head * 0.55)
        ctx.close_path()
        ctx.fill()

    def _ball_and_steps(self, ctx, t, alpha):
        k, phase, u = self.state(t)
        if t < T_BALL:
            return
        a_in = ease_out_cubic(seg(t, T_BALL, T_BALL + 0.6))
        # trail of past positions
        for i in range(1, min(k, N_STEPS) + 1):
            xa, ya = self.path[i - 1]
            xb, yb = self.path[i]
            pa = project(xa, ya, f_xy(xa, ya))
            pb = project(xb, yb, f_xy(xb, yb))
            ctx.set_source_rgba(*rgb(BLUE), 0.55 * alpha)
            ctx.set_line_width(3)
            ctx.move_to(*pa)
            ctx.line_to(*pb)
            ctx.stroke()
            ctx.set_source_rgba(*rgb(BLUE), 0.7 * alpha)
            ctx.arc(pa[0], pa[1], 5, 0, 2 * math.pi)
            ctx.fill()
        # current position (interpolated during the move)
        x, y = self.path[min(k, N_STEPS)]
        if phase == "move" and k < N_STEPS:
            xn, yn = self.path[k + 1]
            m = ease_out_cubic(u)
            x, y = x + (xn - x) * m, y + (yn - y) * m
        z = f_xy(x, y)
        bx, by = project(x, y, z)
        # drop shadow on the surface, then the ball slightly above
        ctx.set_source_rgba(0, 0, 0, 0.18 * alpha * a_in)
        ctx.save()
        ctx.translate(bx, by + 4)
        ctx.scale(1.0, 0.5)
        ctx.arc(0, 0, 16, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()
        # the arrow: the slope under the ball, pointing to where the step lands; it grows, then the
        # ball slides along it while it stays
        if k < N_STEPS and t >= T_FIRST - 0.01:
            xs_, ys_ = self.path[k]
            xn, yn = self.path[k + 1]
            pa = project(xs_, ys_, f_xy(xs_, ys_))
            pb = project(xn, yn, f_xy(xn, yn))
            g = ease_out_cubic(u) if phase == "arrow" else 1.0
            self._arrow(ctx, pa, (pa[0] + (pb[0] - pa[0]) * g, pa[1] + (pb[1] - pa[1]) * g), AMBER, 0.95 * alpha)
        ctx.set_source_rgba(*rgb(INK), alpha * a_in)
        ctx.arc(bx, by - 16, 18, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.8 * alpha * a_in)
        ctx.arc(bx - 6, by - 22, 6, 0, 2 * math.pi)
        ctx.fill()
        # readouts
        kk = min(k, N_STEPS)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(*rgb(INK2), 0.9 * alpha * a_in)
        show_tracked(ctx, "STEP", 70, 1560, 0.14 * 22)
        w = tracked_width(ctx, "SLOPE UNDER THE BALL", 0.14 * 22)
        show_tracked(ctx, "SLOPE UNDER THE BALL", 1010 - w, 1560, 0.14 * 22)
        font(ctx, "dbold", 60)
        ctx.set_source_rgba(*rgb(INK), alpha * a_in)
        ctx.move_to(70, 1628)
        ctx.show_text(f"{kk}")
        s = f"{self.slopes[kk]:.2f}"
        ctx.set_source_rgba(*rgb(AMBER), alpha * a_in)
        ctx.move_to(1010 - ctx.text_extents(s).x_advance, 1628)
        ctx.show_text(s)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(*rgb(INK2), 0.9 * alpha * a_in)
        s = "HEIGHT (ERROR)"
        show_tracked(ctx, s, CX - tracked_width(ctx, s, 0.14 * 22) / 2, 1560, 0.14 * 22)
        font(ctx, "dbold", 60)
        ctx.set_source_rgba(*rgb(BLUE), alpha * a_in)
        s = f"{z:.2f}"
        ctx.move_to(CX - ctx.text_extents(s).x_advance / 2, 1628)
        ctx.show_text(s)

    def _headlines(self, ctx, t):
        for (t_in, t_out, lines, n_in, n_out, style) in self.cues:
            if not (t_in - 1e-6 <= t < t_out + n_out / FPS):
                continue
            k_in = (t - t_in) * FPS
            k_out = (t - t_out) * FPS
            alpha = 1.0
            if n_in and k_in < n_in:
                alpha = ease_out_cubic(k_in / n_in)
            if k_out >= 0 and n_out:
                alpha = min(alpha, 1 - k_out / n_out)
            if alpha <= 0:
                continue
            dy = 14 * (1 - ease_out_cubic(k_in / n_in)) if (style == "pop" and n_in and k_in < n_in) else 0.0
            for (s, kind, size, colr, base, la) in lines:
                font(ctx, kind, size)
                w = ctx.text_extents(s).x_advance
                fs = min(1.0, MAX_TEXT_W / w)
                ctx.save()
                ctx.translate(CX, base + dy)
                ctx.scale(fs, fs)
                ctx.translate(-CX, -base)
                ctx.set_source_rgba(*rgb(colr), la * alpha)
                ctx.move_to(CX - w / 2, base)
                ctx.show_text(s)
                ctx.restore()

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = np.ones((H, W, 3), np.float32)
        surf, ctx = layer()
        reveal = seg(t, T_SURF, T_SURF + 2.6) if fade >= 1.0 else seg(0.0, T_SURF, T_SURF + 2.6)
        self._surface(ctx, reveal if fade >= 1.0 else 1.0, 1.0)
        if fade >= 1.0:
            self._ball_and_steps(ctx, t, 1.0)
        else:
            # dissolve: current state fades, frame-0 state (bare surface) remains
            self._ball_and_steps(ctx, t, fade)
            # at frame 0 the surface has not been drawn yet; fade it out too
        trgb, ta = surf_rgba(surf)
        if fade < 1.0:
            # the drawing fades toward the empty white frame 0; the headlines live on their own layer
            trgb = trgb * fade
            ta = ta * fade
        over(img, trgb, ta)
        surf2, ctx2 = layer()
        self._headlines(ctx2, t)
        trgb, ta = surf_rgba(surf2)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s heights {sc.heights[0]:.2f} -> {sc.heights[-1]:.4f} slopes {sc.slopes[0]:.2f} -> {sc.slopes[-1]:.4f}")
    frames = [0, 120, 220, 300, 400, 500, 580, 620, 660, 720, 800, 900, 1000, 1140, 1300, 1450, 1620, 1799]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (620, 900):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
