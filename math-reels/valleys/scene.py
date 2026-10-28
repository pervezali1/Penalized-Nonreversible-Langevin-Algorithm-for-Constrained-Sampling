"""Two valleys, in the white style of reels 23-27, on a new hill with two bottoms: a double well along the
long axis (one valley shallow, one deep) and the usual bowl across it. Two balls dropped a hair apart on
the ridge roll into different valleys and stay there: plain descent only sees the slope under its feet.
Then the ball in the shallow valley has noise added to the slope it feels (like a small batch, reel 27),
for 16 steps: it rattles, climbs the ridge and drops into the deep valley; with the noise off it settles.
The hill is rendered by hillgen.py, a general-surface ray-marcher with the reel-23 look. 40 steps of
0.38 s. Background: a faint warm tint with a light grid, by request. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import numba as nb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import textkit  # noqa: E402
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
import hillgen  # noqa: E402

# ---------------------------------------------------------------- the hill: double well along u, bowl across v
K_, A2_, T_, BB_ = 2.2, 2.0, 0.6, 1.8


def _well(u):
    return K_ * ((u / A2_) ** 2 - 1) ** 2 + T_ * u


_us = np.linspace(-4, 4, 800001)
_w = _well(_us)
_i0 = int(np.argmax(np.where(np.abs(_us) < 1.5, _w, -1e9)))
U_RIDGE = float(_us[_i0])
U_DEEP = float(_us[int(np.argmin(np.where(_us < U_RIDGE, _w, 1e9)))])
U_SHALLOW = float(_us[int(np.argmin(np.where(_us > U_RIDGE, _w, 1e9)))])
C0_ = float(-min(_well(U_DEEP), _well(U_SHALLOW)))          # lifts the deep bottom to height 0


@nb.njit(cache=False)
def F(u, v):
    q = (u / A2_) ** 2 - 1.0
    return K_ * q * q + T_ * u + C0_ + BB_ * v * v


@nb.njit(cache=False)
def FU(u, v):
    q = (u / A2_) ** 2 - 1.0
    return K_ * 2.0 * q * (2.0 * u / (A2_ * A2_)) + T_


@nb.njit(cache=False)
def FV(u, v):
    return 2.0 * BB_ * v


gds.V_CUT = 0.85
N_STEPS, STEP_DT = 18, 0.34
T_FIRST = gds.T_FIRST
for mod in (gds, lr3):
    mod.N_STEPS, mod.STEP_DT = N_STEPS, STEP_DT
_mix = hillgen.install(gds, F, FU, FV, textkit)

FPS, N_FRAMES, T_SURF, T_BALL, LOOP_FADE = gds.FPS, gds.N_FRAMES, gds.T_SURF, gds.T_BALL, gds.LOOP_FADE
Z_RIM, CX = gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
seg, lerp, poly_prefix = gds.seg, gds.lerp, gds.poly_prefix
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi

LR = 0.1
DELTA = 0.1                          # the hair: the balls start 2 * DELTA apart along the ridge
V0 = -0.9
SIGMA = 3.5                          # noise added to each slope component of a shaken copy
N_COPIES, N_SHAKE, N_SETTLE = 12, 80, 12
DT2 = 0.11                           # seconds per shaken / settling step
T_SHAKE = T_FIRST + (N_STEPS - 1) * STEP_DT + 0.84 * STEP_DT + 0.45    # the copies start shaking here
T_OFF = T_SHAKE + N_SHAKE * DT2
T_STILL = T_OFF + N_SETTLE * DT2
COPY_SEEDS = [1, 11, 39, 42, 53, 60, 70, 78, 94, 104, 107, 179]     # see pick_copies(): first runs that stay on the drawn hill
N_POP = 1000
VERM, BLUE = gds.AMBER, gds.BLUE
BALLS = [
    dict(key="a", name="BALL A", pill="A", side=-1, col=VERM, word="deep valley"),
    dict(key="b", name="BALL B", pill="B", side=+1, col=BLUE, word="shallow valley"),
]
COL_X = (330, 750)
GREY = "#9CA3AF"
BG = "#F4F5F9"                       # the faint background tint
GRID_A = 0.05                        # grid line alpha (very light)


def run_ball(side):
    u, v = U_RIDGE + side * DELTA, V0
    pts = [(u, v)]
    for _ in range(N_STEPS):
        u, v = u - LR * FU(u, v), v - LR * FV(u, v)
        pts.append((u, v))
    return pts


def run_copy(start, seed):
    """One copy of B: N_SHAKE steps with Gaussian noise (sd SIGMA) on each slope component, then N_SETTLE plain steps."""
    rng = np.random.default_rng(seed)
    u, v = start
    pts = [(u, v)]
    for _ in range(N_SHAKE):
        u = u - LR * (FU(u, v) + SIGMA * rng.standard_normal())
        v = v - LR * (FV(u, v) + SIGMA * rng.standard_normal())
        pts.append((u, v))
    for _ in range(N_SETTLE):
        u, v = u - LR * FU(u, v), v - LR * FV(u, v)
        pts.append((u, v))
    return pts


def population(start, n=N_POP):
    """Of n random copies, the fraction that end in the deep valley, and the fraction that ever leave the drawn hill."""
    esc = on = 0
    for seed in range(n):
        p = run_copy(start, seed)
        esc += p[-1][0] < U_RIDGE
        on += max(v for _, v in p) < gds.V_CUT - 0.03 and max(F(u, v) for u, v in p) < 5.4
    return esc / n, on / n


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, grad_xy, surf_pt = gds.f_xy, gds.grad_xy, gds.surf_pt
        self.balls = []
        for b in BALLS:
            uv = run_ball(b["side"])
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9),
                     heights=[float(f_xy(*p)) for p in path], slopes=[float(math.hypot(*grad_xy(*p))) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
            self.balls.append(d)
        a = self.balls[0]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            a["path"], a["heights"], a["slopes"], a["spt"], a["seg_px"], a["step_poly"])
        # the copies of B, shaken from B's settled position
        start = self.balls[1]["uv"][-1]
        self.copies = []
        for seed in COPY_SEEDS:
            uv = run_copy(start, seed)
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            over = next((k for k, (u, v) in enumerate(uv) if u < U_RIDGE), None)
            self.copies.append(dict(seed=seed, uv=uv, path=path, spt=spt, heights=[float(f_xy(*p)) for p in path], over=over,
                                    escaped=uv[-1][0] < U_RIDGE))
            assert max(v for _, v in uv) < gds.V_CUT - 0.02 and max(F(u, v) for u, v in uv) < Z_RIM, seed
        self.pop_escape, self.pop_on = population(start)
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()
        self._build_grid()

    def copy_state(self, t):
        """(landing index j, slide fraction) for the copies at time t; j counts shaken+settling steps."""
        if t < T_SHAKE:
            return 0, 0.0
        x = (t - T_SHAKE) / DT2
        j = int(math.floor(x))
        if j >= N_SHAKE + N_SETTLE:
            return N_SHAKE + N_SETTLE, 0.0
        return j, smooth(min(1.0, (x - j) / 0.8))

    def n_over(self, j):
        return sum(1 for c in self.copies if c["uv"][j][0] < U_RIDGE)

    # ------------------------------------------------------------ cues (every number from the run)

    @staticmethod
    def t_land(k):
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def _cues(self):
        a, b = self.balls
        ha, hb = a["heights"], b["heights"]
        firsts = sorted(c["over"] for c in self.copies if c["over"] is not None)
        k_first = firsts[0]
        n_end = sum(1 for c in self.copies if c["escaped"])
        n_mid = self.n_over(N_SHAKE)
        self.facts = dict(ridge=float(F(U_RIDGE, 0.0)), shallow=float(F(U_SHALLOW, 0.0)), deep=float(F(U_DEEP, 0.0)),
                          u_ridge=U_RIDGE, u_shallow=U_SHALLOW, u_deep=U_DEEP, h_settled=(ha[-1], hb[-1]),
                          k_first=k_first, firsts=firsts, n_over_at_off=n_mid, n_end=n_end, pop_escape=self.pop_escape, pop_on=self.pop_on,
                          h_max_copies=max(max(c["heights"]) for c in self.copies))
        t_first = T_SHAKE + k_first * DT2
        hook = [("Two valleys.", "black", 64, INK, 300, 1), ("Where you start decides where you end.", "black", 44, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [("A hill with two bottoms.", "black", 48, INK, 300, 1), (f"One valley is shallow (height {self.facts['shallow']:.1f}), the other deep (0).", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [(f"Two balls, dropped {2 * DELTA:.1f} apart on the ridge.", "black", 44, INK, 300, 1), ("Same rule, same step size.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.1, [("Plain descent only sees the slope under its feet.", "black", 40, INK, 300, 1), ("Whichever way the ground tilts, that's the way it goes.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.4, T_SHAKE - 0.3, [(f"Settled: {ha[-1]:.2f} and {hb[-1]:.2f}.", "black", 46, INK, 300, 1), ("B would sit in the shallow valley forever.", "dbold", 38, BLUE, 390, 0.95)], 10, 8, "pop"),
            (T_SHAKE, T_SHAKE + 3.2, [(f"Now shake it: {N_COPIES} copies of B, noise on the slope each one feels.", "black", 36, BLUE, 300, 1), (f"Like a small batch (last reel). {N_SHAKE} shaken steps.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_SHAKE + 3.5, T_SHAKE + 6.3, [(f"The first came over the ridge at step {k_first}.", "black", 42, BLUE, 300, 1), ("A noisy step can go uphill. Plain descent never can.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_SHAKE + 6.6, T_OFF - 0.2, [("Most rattle in the shallow valley. Some climb out.", "black", 40, INK, 300, 1), ("Watch the count.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_OFF + 0.1, T_STILL + 0.8, [(f"Shaking off: {n_mid} of {N_COPIES} are over the ridge, and settle at 0.", "black", 36, INK, 300, 1), ("Places they could never have walked to.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_STILL + 1.1, 29.0, [("Where you start decides where you end. Unless you shake.", "black", 38, INK, 300, 1), (f"Over {N_POP:,} random copies, {self.pop_escape:.0%} climb out in {N_SHAKE} steps.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the copies of B

    def copy_alpha(self, t):
        return ease_out_cubic(seg(t, T_SHAKE - 0.4, T_SHAKE))

    def draw_copies(self, ctx, t, fade):
        a_in = self.copy_alpha(t) * fade
        if a_in <= 0:
            return
        j, frac = self.copy_state(t)
        ink = rgb(INK)
        col = rgb(BLUE)
        ctx.save()
        self._path(ctx, self.surf_clip, close=True)
        ctx.clip()
        # trails: the last 16 landings of each copy, as a hairline and small dots fading with age
        for c in self.copies:
            j0 = max(0, j - 16)
            pts = c["spt"][j0:j + 1]
            if len(pts) >= 2:
                for i in range(1, len(pts)):
                    age = (j - (j0 + i)) / 16.0
                    ctx.set_source_rgba(*col, 0.55 * (1 - age) * a_in)
                    ctx.set_line_width(1.6)
                    ctx.move_to(*pts[i - 1])
                    ctx.line_to(*pts[i])
                    ctx.stroke()
            for i, p in enumerate(pts):
                age = (j - (j0 + i)) / 16.0
                ctx.set_source_rgba(*ink, 0.5 * (1 - age) * a_in)
                ctx.arc(p[0], p[1], 2.2, 0, TAU)
                ctx.fill()
        ctx.restore()
        # the copies themselves: small spheres at the interpolated position, farther ones first
        items = []
        for c in self.copies:
            (x0, y0), (x1, y1) = c["path"][j], c["path"][min(j + 1, N_SHAKE + N_SETTLE)]
            x, y = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
            P, n, rb_px, rb_w, (cx, cy), (bx, by) = self.ball_geom(x, y)
            sc_ = 0.58
            cxs, cys, _ = gds.project_w(P + n * rb_w * sc_)
            items.append((rb_px, cxs, cys, rb_px * sc_, c))
        items.sort(key=lambda it: it[0])
        for rb_px, cx, cy, r, c in items:
            self.sphere(ctx, cx, cy, r, a_in, contact=None, squash=0.0)
            # a thin blue ring marks a copy, so they read as B's
            ctx.set_source_rgba(*col, 0.9 * a_in)
            ctx.set_line_width(1.5)
            ctx.arc(cx, cy, r + 2.0, 0, TAU)
            ctx.stroke()

    # ------------------------------------------------------------ the true slope beside the shaken arrow

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        a_ball = a_ball * self.ball_vis(b, t)
        if b["key"] == "b":
            a_ball *= 1 - self.copy_alpha(t)            # B becomes its copies
        if a_ball <= 0.002:
            return
        cx, cy = cxy
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=bxy, squash=squash)
        s = b["pill"]
        font(ctx, "dbold", 24)
        tw = ctx.text_extents(s).x_advance
        if b["key"] == "a":
            px, py = cx - rb_px - tw / 2 - 14, cy - 80 * drop - rb_px - 4
        else:
            px, py = cx + rb_px + tw / 2 + 14, cy - 80 * drop - rb_px - 4
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)

    def _draw(self, ctx, t, fade):
        super()._draw(ctx, t, fade)
        self.draw_copies(ctx, t, fade)

    # ------------------------------------------------------------ readouts: two columns

    def readouts3(self, ctx, t, fade):
        a = fade * ease_out_cubic(seg(t, 6.4, 6.9))
        if a <= 0:
            return
        k, phase, u = self.state(t)
        Y_LAB, Y_VAL, Y_BAR, Y_WORD = 1404, 1466, 1489, 1522
        font(ctx, "dbold", 46)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        ca = self.copy_alpha(t)
        j, _ = self.copy_state(t)
        for b, xc in zip(self.balls, COL_X):
            kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
            z = max(0.0, b["heights"][kz])
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            if b["key"] == "b" and ca > 0:
                # the copies: how many are over the ridge
                n = self.n_over(j)
                lab = f"{N_COPIES} COPIES OF B   \u00b7   OVER THE RIDGE" + ("   \u00b7   SHAKING" if t < T_OFF else "")
                w = tracked_width(ctx, lab, tr)
                ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a * ca)
                show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
                font(ctx, "dbold", 46)
                s = f"{n} / {N_COPIES}"
                w = ctx.text_extents(s).x_advance
                ctx.set_source_rgba(*rgb(INK), a * ca)
                ctx.move_to(xc - w / 2, Y_VAL)
                ctx.show_text(s)
                ctx.set_line_width(6)
                ctx.set_source_rgba(*rgb("#E5E7EB"), a * ca)
                ctx.move_to(xc - 140, Y_BAR)
                ctx.line_to(xc + 140, Y_BAR)
                ctx.stroke()
                L = 280 * n / N_COPIES
                if L > 0.5:
                    ctx.set_source_rgba(*rgb(b["col"]), a * ca)
                    ctx.move_to(xc - 140, Y_BAR)
                    ctx.line_to(xc - 140 + L, Y_BAR)
                    ctx.stroke()
                if ca >= 1:
                    continue
                a_b = a * (1 - ca)
            else:
                a_b = a
            lab = b["name"]
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a_b)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            s = gds._fmt(z, 2 if z >= 1.0 else 3)
            w = cells_width(ctx, s, widths)
            ctx.set_source_rgba(*rgb(INK), a_b)
            show_cells(ctx, s, xc - w / 2, Y_VAL, widths)
            ctx.set_line_width(6)
            ctx.set_source_rgba(*rgb("#E5E7EB"), a_b)
            ctx.move_to(xc - 140, Y_BAR)
            ctx.line_to(xc + 140, Y_BAR)
            ctx.stroke()
            L = 280 * min(1.0, z / b["heights"][0])
            if L > 0.5:
                ctx.set_source_rgba(*rgb(b["col"]), a_b)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
            a_w = a_b * ease_out_cubic(seg(t, T_STILL + 1.7, T_STILL + 2.3))
            if a_w > 0 and b["key"] == "a":
                font(ctx, "regular", 24)
                s = b["word"]
                ctx.set_source_rgba(*rgb(INK3), a_w)
                ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
                ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = f"HEIGHT (ERROR) AFTER EACH STEP   ·   STEP SIZE {LR:g}"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2 + 40, 1372, tr)
        s = f"STEP {min(k, N_STEPS)}" if ca <= 0 else f"SHAKE {min(j, N_SHAKE)}" + (f" + {max(0, j - N_SHAKE)}" if j > N_SHAKE else "")
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, 9.6, 10.0)))
        show_tracked(ctx, s, 70, 1372, tr)

    # ------------------------------------------------------------ background: faint tint and grid

    def _build_grid(self):
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(INK2), GRID_A)
        ctx.set_line_width(1.0)
        step = 54
        for x in range(step, W, step):
            ctx.move_to(x + 0.5, 0)
            ctx.line_to(x + 0.5, H)
        for y in range(step, H, step):
            ctx.move_to(0, y + 0.5)
            ctx.line_to(W, y + 0.5)
        ctx.stroke()
        grgb, ga = surf_rgba(surf)
        base = np.ones((H, W, 3), np.float32) * np.array(rgb(BG), np.float32)
        base = base * (1 - ga)[..., None] + grgb
        self.bg = np.ascontiguousarray(base)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        x0, x1, y0, y1 = gds.CROP
        sub = img[y0:y1, x0:x1]
        over(sub, self.b_rgb * gds.GHOST_A, self.b_a * gds.GHOST_A)
        reveal = seg(t, T_SURF, T_SURF + 2.6)
        if reveal > 0 and fade > 0:
            sh = self.sh_a * (smooth(seg(reveal, 0.45, 1.0)) * fade)
            sub *= (1 - sh)[..., None]
            sub += sh[..., None] * self.sh_rgb
            if reveal < 1:
                zf = 0.4 + (Z_RIM - 0.38) * ease_out_cubic(reveal) ** 2
                m = np.clip((zf - self.b_z) / self.b_gpx + 0.5, 0, 1)
                m = np.where(self.b_stroke & (self.b_z > zf + 0.05), 0.0, m).astype(np.float32)
                front = np.clip(1.2 - np.abs(self.b_z - zf) / self.b_gpx, 0, 1) * (1 - reveal) ** 0.5
                m *= fade * min(1.0, reveal / 0.08)
                over(sub, self.b_rgb * (m * (1 - 0.35 * front))[..., None], self.b_a * m)
            elif fade < 1:
                over(sub, self.b_rgb * fade, self.b_a * fade)
            else:
                over(sub, self.b_rgb, self.b_a)
        if fade > 0 and t >= T_BALL:
            surf, ctx = layer()
            self._draw(ctx, t, fade)
            trgb, ta = surf_rgba(surf)
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
    print(f"init {time.time() - t0:.1f}s  exterior px {sc.n_exterior_px}")
    print("hill: ridge", sc.facts["u_ridge"], sc.facts["ridge"], "shallow", sc.facts["u_shallow"], sc.facts["shallow"], "deep", sc.facts["u_deep"], sc.facts["deep"])
    for b in sc.balls:
        print(f"  {b['name']} height {b['heights'][0]:.2f} -> end {b['heights'][-1]:.4f}")
    for c in sc.copies:
        print(f"  copy seed {c['seed']:4d}: over at {c['over']}, escaped {c['escaped']}, max h {max(c['heights']):.2f}, end {c['heights'][-1]:.3f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 300, 500, 640, 800, 940, 1000, 1050, 1120, 1200, 1300, 1400, 1500, 1580, 1660, 1700, 1760]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    cv2.imwrite(f"{out}/f300.png", sc.frame(300)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
