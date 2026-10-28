"""Two valleys, in the white style of reels 23-27, on a hill with two bottoms: a double well along the long
axis (one valley shallow, one deep) and the usual bowl across it. Four balls, plain gradient descent, step
size 0.1, four starting points: two well inside each side, two a hair either side of the ridge. Plain
descent only sees the slope under its feet, so each ball ends in whichever valley its own side of the
ridge tilts towards. The hill is rendered by hillgen.py, a general-surface ray-marcher with the reel-23
look. 24 steps of 0.62 s. Background: a faint tint with a light grid. frame(f) -> uint8 1080x1920 RGB."""

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


gds.V_CUT = 0.75
N_STEPS, STEP_DT = 24, 0.62
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
V0 = -0.9                            # all four start up the back wall
DELTA = 0.1                          # B and C start DELTA either side of the ridge
VERM, BLUE, GREEN, MAGENTA = gds.AMBER, gds.BLUE, gds.GREEN, gds.MAGENTA
BALLS = [
    dict(key="a", name="A", pill="A", u0=-1.1, col=VERM),
    dict(key="b", name="B", pill="B", u0=U_RIDGE - DELTA, col=BLUE),
    dict(key="c", name="C", pill="C", u0=U_RIDGE + DELTA, col=GREEN),
    dict(key="d", name="D", pill="D", u0=1.3, col=MAGENTA),
]
COL_X = (165, 415, 665, 915)
BG = "#F4F5F9"                       # the faint background tint
GRID_A = 0.05                        # grid line alpha (very light)


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
                     deep=uv[-1][0] < U_RIDGE)
            d["word"] = "deep valley" if d["deep"] else "shallow valley"
            self.balls.append(d)
        a = self.balls[0]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            a["path"], a["heights"], a["slopes"], a["spt"], a["seg_px"], a["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()
        self._build_grid()

    # ------------------------------------------------------------ cues (every number from the run)

    @staticmethod
    def t_land(k):
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def _cues(self):
        hs = [b["heights"] for b in self.balls]
        deep = [b["name"] for b in self.balls if b["deep"]]
        shallow = [b["name"] for b in self.balls if not b["deep"]]
        # when B and C are first on opposite sides of the ridge by a visible margin (0.3 apart)
        bu, cu = [p[0] for p in self.balls[1]["uv"]], [p[0] for p in self.balls[2]["uv"]]
        k_part = next((k for k in range(N_STEPS + 1) if cu[k] - bu[k] > 0.3), None)
        self.facts = dict(ridge=float(F(U_RIDGE, 0.0)), shallow=float(F(U_SHALLOW, 0.0)), deep=float(F(U_DEEP, 0.0)),
                          u_ridge=U_RIDGE, starts=[b["u0"] for b in self.balls], h_start=[h[0] for h in hs], h_end=[h[-1] for h in hs],
                          ends=[b["word"] for b in self.balls], k_part=k_part)
        t_part = self.t_land(k_part) if k_part else 15.0
        hook = [("Two valleys.", "black", 64, INK, 300, 1), ("Where you start decides where you end.", "black", 44, INK2, 390, 1)]
        joined = lambda names: " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1]  # noqa: E731
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [("A hill with two bottoms.", "black", 48, INK, 300, 1), (f"One valley is shallow (height {self.facts['shallow']:.1f}), the other deep (0).", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [(f"Four balls. Same rule: step = slope \u00d7 {LR:g}.", "black", 44, INK, 300, 1), (f"Four starting points. B and C are {2 * DELTA:.1f} apart, either side of the ridge.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.4, [("Plain descent only sees the slope under its feet.", "black", 40, INK, 300, 1), ("Whichever way the ground tilts, that's the way it goes.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.8, 17.4, [(f"B and C had parted by step {k_part}.", "black", 44, INK, 300, 1), ("Same rule, same step size, opposite valleys.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.8, 21.4, [(f"{joined(deep)} settle in the deep valley.", "black", 42, INK, 300, 1), (f"{joined(shallow)} settle in the shallow one, and will never leave it.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.8, 25.2, [(f"After 24 steps: " + ", ".join(f"{h[-1]:.2f}" for h in hs) + ".", "black", 44, INK, 300, 1), ("Two found the bottom. Two never will.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (25.6, 29.0, [("No ball can see the other valley. It only feels the ground under it.", "black", 34, INK, 300, 1), ("That's a local minimum, and why the starting point matters.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        a_ball = a_ball * self.ball_vis(b, t)
        if a_ball <= 0.002:
            return
        cx, cy = cxy
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=bxy, squash=squash)
        s = b["pill"]
        font(ctx, "dbold", 24)
        tw = ctx.text_extents(s).x_advance
        if b["key"] == "b":
            px, py = cx - rb_px - tw / 2 - 14, cy - 80 * drop - rb_px - 4
        elif b["key"] == "c":
            px, py = cx + rb_px + tw / 2 + 14, cy - 80 * drop - rb_px - 4
        else:
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
        for b, xc in zip(self.balls, COL_X):
            kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
            z = max(0.0, b["heights"][kz])
            font(ctx, "bold", 22)
            lab = f"BALL {b['name']}"
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            s = gds._fmt(z, 2 if z >= 1.0 else 3)
            w = cells_width(ctx, s, widths)
            ctx.set_source_rgba(*rgb(INK), a)
            show_cells(ctx, s, xc - w / 2, Y_VAL, widths)
            ctx.set_line_width(6)
            ctx.set_source_rgba(*rgb("#E5E7EB"), a)
            ctx.move_to(xc - 100, Y_BAR)
            ctx.line_to(xc + 100, Y_BAR)
            ctx.stroke()
            L = 200 * min(1.0, z / max(h[0] for h in (bb["heights"] for bb in self.balls)))
            if L > 0.5:
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
            a_w = a * ease_out_cubic(seg(t, 21.8, 22.4))
            if a_w > 0:
                font(ctx, "regular", 22)
                s = b["word"]
                ctx.set_source_rgba(*rgb(INK3), a_w)
                ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
                ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = f"HEIGHT (ERROR) AFTER EACH STEP   \u00b7   STEP SIZE {LR:g}"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2 + 40, 1372, tr)
        s = f"STEP {min(k, N_STEPS)}"
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
    print("hill: ridge", sc.facts["u_ridge"], sc.facts["ridge"], "shallow", sc.facts["shallow"], "deep", sc.facts["deep"])
    for b in sc.balls:
        print(f"  ball {b['name']} start u {b['u0']:.3f} h {b['heights'][0]:.2f} -> end {b['heights'][-1]:.4f} ({b['word']})  max v {max(v for _, v in b['uv']):.2f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 300, 500, 600, 660, 760, 860, 960, 1060, 1160, 1260, 1360, 1460, 1500, 1600, 1700, 1760]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
