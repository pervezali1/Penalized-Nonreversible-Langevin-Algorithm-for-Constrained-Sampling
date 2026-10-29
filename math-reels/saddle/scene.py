"""The saddle, in the white style of reels 23-29, with the maths on screen: one ball on the two-valley hill
with no tilt, whose pass is a true saddle point (slope zero in every direction, not a bottom). The ball is
dropped 0.01 off the line through the pass. It rolls down to the pass, looks parked there while its tiny
offset multiplies by 1.22 every step (1 - 0.1 x the pass's curvature of -2.2), tips off at step 20 and
rolls to the bottom of a valley. A card at the bottom of the frame carries the maths of each stage: the
hill and the saddle test, the update rule, the per-step multiplication with the live offset, the fall, and
the n-dimensional argument. 40 s. Rendered by ../valleys/hillgen.py on the faint tinted grid."""

import importlib.util
import math
import os
import sys

import numba as nb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "valleys"))
sys.path.insert(0, HERE)
import textkit  # noqa: E402
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
import hillgen  # noqa: E402

# ---------------------------------------------------------------- the hill: the symmetric double well, saddle at the centre
K_, A2_, BB_ = 2.2, 2.0, 1.8


@nb.njit(cache=False)
def F(u, v):
    q = (u / A2_) ** 2 - 1.0
    return K_ * q * q + BB_ * v * v


@nb.njit(cache=False)
def FU(u, v):
    q = (u / A2_) ** 2 - 1.0
    return K_ * 2.0 * q * (2.0 * u / (A2_ * A2_))


@nb.njit(cache=False)
def FV(u, v):
    return 2.0 * BB_ * v


gds.V_CUT = 0.75
# a 40 s reel: 40 steps of 0.5 s, then room for the closing maths
N_STEPS, STEP_DT = 40, 0.5
N_FRAMES = 2400
LOOP_FADE = (39.3, 39.85)
T_FIRST = gds.T_FIRST
for mod in (gds, lr3):
    mod.N_STEPS, mod.STEP_DT, mod.N_FRAMES, mod.LOOP_FADE = N_STEPS, STEP_DT, N_FRAMES, LOOP_FADE
_mix = hillgen.install(gds, F, FU, FV, textkit)

FPS, T_SURF, T_BALL = gds.FPS, gds.T_SURF, gds.T_BALL
Z_RIM, CX = gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
seg, lerp = gds.seg, gds.lerp
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi

LR = 0.1
V0 = -0.9
OFF = 0.01                                   # the offset from the saddle line
SADDLE_H = float(F(0.0, 0.0))
CURV_U = -4 * K_ / (A2_ * A2_)               # f_uu at the pass: -2.2 (down along the pass)
CURV_V = 2 * BB_                             # f_vv at the pass: +3.6 (up across it)
GROWTH = 1 - LR * CURV_U                     # the offset's factor per step near the pass: 1.22
SHRINK = 1 - LR * CURV_V                     # the across-pass factor per step: 0.64
BLUE = gds.BLUE
BALLS = [dict(key="b", name="THE BALL", pill=f"{OFF:g} off", u0=OFF, col=BLUE)]
COL_X = (330, 750)
BG = "#F4F5F9"
GRID_A = 0.05
CARD = (70, 1575, 940, 300)                  # the maths card: x, y, w, h
CARD_C = "#FFFFFF"


def run_ball(u0):
    u, v = u0, V0
    pts = [(u, v)]
    for _ in range(N_STEPS):
        u, v = u - LR * FU(u, v), v - LR * FV(u, v)
        pts.append((u, v))
    return pts


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, grad_xy, surf_pt = gds.f_xy, gds.grad_xy, gds.surf_pt
        self.balls = []
        for b in BALLS:
            uv = run_ball(b["u0"])
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9),
                     heights=[float(f_xy(*p)) for p in path], slopes=[float(math.hypot(*grad_xy(*p))) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])],
                     off=[abs(u) for u, _ in uv])
            self.balls.append(d)
        b = self.balls[0]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            b["path"], b["heights"], b["slopes"], b["spt"], b["seg_px"], b["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()
        self._build_grid()

    # ------------------------------------------------------------ cues and cards (every number from the run)

    @staticmethod
    def t_land(k):
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def _cues(self):
        b = self.balls[0]
        hs, off = b["heights"], b["off"]
        k_pass = next(k for k, h in enumerate(hs) if h < SADDLE_H + 0.01)          # on the pass
        k_off = next(k for k, o in enumerate(off) if o > 0.5)                       # off the pass
        k_bottom = next(k for k, h in enumerate(hs) if h < 0.01)                    # at the bottom
        double = math.log(2) / math.log(GROWTH)
        n_half = math.log(0.5 / OFF) / math.log(GROWTH)
        self.facts = dict(saddle_h=SADDLE_H, curv_u=CURV_U, curv_v=CURV_V, growth=GROWTH, shrink=SHRINK, double_every=double,
                          n_to_half=n_half, k_pass=k_pass, k_off=k_off, k_bottom=k_bottom, h_end=hs[-1],
                          off_at=[(k, off[k]) for k in (0, 5, 10, 15, 20)], u_end=b["uv"][-1][0])
        self.k_pass, self.k_off, self.k_bottom = k_pass, k_off, k_bottom
        tp, to, tb, tend = self.t_land(k_pass), self.t_land(k_off), self.t_land(k_bottom), self.t_land(N_STEPS)
        self.t_pass, self.t_off, self.t_bottom, self.t_end = tp, to, tb, tend
        hook = [("The saddle.", "black", 64, INK, 300, 1), ("Slope zero, and not a bottom.", "black", 44, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [("Two valleys, and a pass between them.", "black", 46, INK, 300, 1), (f"The top of the pass is flat in every direction. Height {SADDLE_H:.1f}.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [(f"One ball, dropped {OFF:g} off the line through the pass.", "black", 40, INK, 300, 1), (f"The usual rule: step = slope × {LR:g}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, tp + 0.3, [("It rolls down towards the pass.", "black", 46, INK, 300, 1), (f"Across the pass the ground is steep: each step keeps only {SHRINK:.0%} of the distance.", "dbold", 32, INK2, 390, 0.95)], 10, 8, "pop"),
            (tp + 0.6, tp + 2.9, [(f"On the pass. Height {SADDLE_H:.2f}, slope almost zero.", "black", 42, INK, 300, 1), ("It looks parked. It isn't: its offset is multiplying.", "dbold", 36, BLUE, 390, 0.95)], 10, 8, "pop"),
            (tp + 3.2, to - 0.3, [(f"{OFF:g} off becomes {off[1]:.3f}, {off[2]:.3f}, {off[3]:.3f} …  × {GROWTH:.2f} a step.", "black", 36, BLUE, 300, 1), (f"Doubling every {double:.1f} steps. Flat ground is slow, not a trap.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (to, to + 3.6, [(f"Step {k_off}: off the pass and falling.", "black", 44, BLUE, 300, 1), ("The slope grows with the offset, and the offset is big now.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (to + 3.9, tend + 0.3, [("Only an offset of exactly zero would have stayed.", "black", 40, INK, 300, 1), ("In millions of dimensions, exactly never happens.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (tend + 0.6, 34.4, [(f"{k_pass} steps down, {k_off - k_pass} parked, {k_bottom - k_off} to the bottom.", "black", 40, INK, 300, 1), (f"Height after {N_STEPS} steps: {hs[-1]:.3f}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (34.7, 39.0, [("Saddles slow training down. They almost never stop it.", "black", 38, INK, 300, 1), ("The plateau in a loss curve, and the drop that follows it.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (39.4, 40.1, hook, 12, 0, "fade"),
        ]

    def card(self, t):
        """The maths card for time t: (title, lines, alpha) with lines as (text, kind, size, colour)."""
        b = self.balls[0]
        tp, to, tend = self.t_pass, self.t_off, self.t_end
        k, phase, u = self.state(t)
        kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
        off = b["off"][kz]
        R, D = "regular", "dbold"
        cards = [
            (3.0, 9.3, "THE HILL, AND WHY ITS PASS IS A SADDLE", [
                (f"f(u, v) = {K_:g}\u00b7((u/{A2_:g})\u00b2 \u2212 1)\u00b2 + {BB_:g}\u00b7v\u00b2", R, 30, INK),
                (f"At the pass (0, 0):   \u2202f/\u2202u = 0,  \u2202f/\u2202v = 0.   Slope zero.", R, 27, INK),
                (f"Along the pass:  \u2202\u00b2f/\u2202u\u00b2 = {CURV_U:+.1f},  curving down.", R, 27, INK),
                (f"Across the pass:  \u2202\u00b2f/\u2202v\u00b2 = {CURV_V:+.1f},  curving up.", R, 27, INK),
                ("A bottom needs both positive. One is negative: a saddle.", D, 27, INK2)]),
            (9.6, tp + 0.3, "THE RULE", [
                (f"u → u − {LR:g}·∂f/∂u        v → v − {LR:g}·∂f/∂v", R, 30, INK),
                (f"Across the pass  ∂f/∂v = {CURV_V:g}·v,  so  v → v − {LR:g}·{CURV_V:g}·v = {SHRINK:.2f}·v.", R, 27, INK),
                (f"The distance across shrinks to {SHRINK:.0%} every step: {abs(V0):g} → {abs(V0) * SHRINK:.2f} → {abs(V0) * SHRINK ** 2:.2f} → {abs(V0) * SHRINK ** 3:.2f} …", R, 26, INK),
                ("That is the quick part.", D, 27, INK2)]),
            (tp + 0.6, to + 0.3, "ON THE PASS: THE OFFSET MULTIPLIES", [
                (f"Near the pass  ∂f/∂u = {CURV_U:g}·u,  so  u → u − {LR:g}·({CURV_U:g}·u) = {GROWTH:.2f}·u.", R, 27, INK),
                (f"step {kz}:   u = {off:.4f}      next:  {GROWTH:.2f} × {off:.4f} = {min(off * GROWTH, 9.9):.4f}", D, 28, BLUE),
                (f"{OFF:g} × {GROWTH:.2f} × {GROWTH:.2f} × … reaches 0.5 after {self.facts['n_to_half']:.0f} steps.", R, 27, INK),
                (f"If u were exactly 0:  {GROWTH:.2f} × 0 = 0, forever. Anything else leaves.", D, 27, INK2)]),
            (to + 0.6, tend + 0.3, "OFF THE PASS", [
                (f"Away from the pass the slope is the full  ∂f/∂u = {K_:g}·((u/{A2_:g})² − 1)·u,", R, 27, INK),
                (f"no longer small: the steps grow, then shrink again as the bottom nears.", R, 27, INK),
                (f"step {kz}:   u = {b['uv'][kz][0]:+.3f}     height f = {max(0.0, b['heights'][kz]):.3f}", D, 28, BLUE),
                (f"The bottom is at u = ±{A2_:g}, f = 0. Reached (below 0.01) at step {self.k_bottom}.", R, 27, INK)]),
            (tend + 0.6, 39.0, "IN n DIMENSIONS", [
                ("A flat point: all n slopes are zero.   A bottom: all n curvatures positive.", R, 27, INK),
                ("If each curvature's sign were a coin flip, a flat point would be a bottom", R, 27, INK),
                ("one time in 2ⁿ.   n = 20: one in a million.   Almost every flat point is a saddle.", R, 27, INK),
                ("And a saddle holds only an offset of exactly zero.", D, 27, INK2)]),
        ]
        for t0, t1, title, lines in cards:
            if t0 - 0.3 <= t < t1 + 0.3:
                a = ease_out_cubic(seg(t, t0 - 0.3, t0)) * (1 - smooth(seg(t, t1, t1 + 0.3)))
                return title, lines, a
        return None, None, 0.0

    def draw_card(self, ctx, t, fade):
        title, lines, a = self.card(t)
        a *= fade
        if a <= 0 or lines is None:
            return
        x, y, w, h = CARD
        ctx.set_source_rgba(*rgb(CARD_C), 0.88 * a)
        self._pill(ctx, x, y, w, h, 18)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(INK3), 0.45 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        show_tracked(ctx, title, x + 34, y + 44, tr)
        yy = y + 92
        gap = 52 if len(lines) <= 4 else 44
        for text, kind, size, col in lines:
            font(ctx, kind, size)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x + 34, yy)
            ctx.show_text(text)
            yy += gap

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        a_ball = a_ball * self.ball_vis(b, t)
        if a_ball <= 0.002:
            return
        cx, cy = cxy
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=bxy, squash=squash)
        s = b["pill"]
        font(ctx, "dbold", 24)
        tw = ctx.text_extents(s).x_advance
        px, py = cx, cy - 80 * drop - rb_px - 16
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)

    # ------------------------------------------------------------ readouts: height and offset

    def readouts3(self, ctx, t, fade):
        a = fade * ease_out_cubic(seg(t, 6.4, 6.9))
        if a <= 0:
            return
        k, phase, u = self.state(t)
        b = self.balls[0]
        kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
        Y_LAB, Y_VAL, Y_BAR = 1404, 1466, 1489
        font(ctx, "dbold", 46)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        z = max(0.0, b["heights"][kz])
        off = b["off"][kz]
        for (lab, val, frac, col), xc in zip(
                (("HEIGHT (ERROR)", gds._fmt(z, 2 if z >= 1.0 else 3), min(1.0, z / b["heights"][0]), INK),
                 ("OFF THE SADDLE LINE", f"{off:.4f}" if off < 1 else f"{off:.3f}", min(1.0, off / A2_), BLUE)), COL_X):
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(INK2 if col == INK else col), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            w = cells_width(ctx, val, widths)
            ctx.set_source_rgba(*rgb(INK), a)
            show_cells(ctx, val, xc - w / 2, Y_VAL, widths)
            ctx.set_line_width(6)
            ctx.set_source_rgba(*rgb("#E5E7EB"), a)
            ctx.move_to(xc - 140, Y_BAR)
            ctx.line_to(xc + 140, Y_BAR)
            ctx.stroke()
            L = 280 * frac
            if L > 0.5:
                ctx.set_source_rgba(*rgb(BLUE), a)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
        font(ctx, "bold", 22)
        tr = 0.12 * 22
        lab = f"AFTER EACH STEP   ·   STEP SIZE {LR:g}"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2 + 90, 1372, tr)
        s = f"STEP {min(k, N_STEPS)}"
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, 9.6, 10.0)))
        show_tracked(ctx, s, 70, 1372, tr)

    # ------------------------------------------------------------ background and frame

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
        if fade > 0:
            surf3, ctx3 = layer()
            self.draw_card(ctx3, t, fade)
            trgb, ta = surf_rgba(surf3)
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
    print(f"init {time.time() - t0:.1f}s")
    b = sc.balls[0]
    print(f"  start u {b['u0']:g} h {b['heights'][0]:.2f} -> end {b['heights'][-1]:.4f}, u end {b['uv'][-1][0]:+.3f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 200, 320, 500, 620, 720, 820, 920, 1020, 1140, 1240, 1340, 1460, 1600, 1760, 1900, 2100, 2300]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (320, 1000, 1240):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
