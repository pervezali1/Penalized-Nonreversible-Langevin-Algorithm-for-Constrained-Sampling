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


gds.V_CUT = 0.75
N_STEPS, STEP_DT = 40, 0.38
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
SIGMA = 5.0                          # noise added to each slope component while shaking
SHAKE = (18, 34)                     # steps [18, 34) are shaken
SEED = 59
VERM, BLUE = gds.AMBER, gds.BLUE
BALLS = [
    dict(key="a", name="BALL A", pill="A", side=-1, col=VERM, word="deep valley"),
    dict(key="b", name="BALL B", pill="B", side=+1, col=BLUE, word="climbed out"),
]
COL_X = (330, 750)
GREY = "#9CA3AF"
BG = "#F4F5F9"                       # the faint background tint
GRID_A = 0.05                        # grid line alpha (very light)


def run_ball(side, shake, sigma, seed):
    rng = np.random.default_rng(seed)
    u, v = U_RIDGE + side * DELTA, V0
    pts, noisy, true_steps = [(u, v)], [], []
    for s in range(N_STEPS):
        gu, gv = FU(u, v), FV(u, v)
        true_steps.append((-LR * gu, -LR * gv))
        shaken = shake is not None and shake[0] <= s < shake[1]
        if shaken:
            gu += sigma * rng.standard_normal()
            gv += sigma * rng.standard_normal()
        noisy.append(shaken)
        u, v = u - LR * gu, v - LR * gv
        pts.append((u, v))
    return pts, noisy, true_steps


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, grad_xy, surf_pt = gds.f_xy, gds.grad_xy, gds.surf_pt
        self.balls = []
        for b in BALLS:
            uv, noisy, true_steps = run_ball(b["side"], SHAKE if b["key"] == "b" else None, SIGMA, SEED)
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9), noisy=noisy,
                     heights=[float(f_xy(*p)) for p in path], slopes=[float(math.hypot(*grad_xy(*p))) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
            assert max(d["heights"]) < Z_RIM and max(v for _, v in uv) < gds.V_CUT - 0.02, (b["key"], max(d["heights"]), max(v for _, v in uv))
            true_poly, true_px = [], []
            for k in range(N_STEPS):
                q = uv_to_xy(uv[k][0] + true_steps[k][0], uv[k][1] + true_steps[k][1])
                true_poly.append(self.surf_poly(path[k], q, n=16))
                a, c = spt[k], surf_pt(*q)
                true_px.append(math.hypot(c[0] - a[0], c[1] - a[1]))
            d.update(true_poly=true_poly, true_px=true_px)
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
        a, b = self.balls
        ha, hb = a["heights"], b["heights"]
        k_esc = next((k for k, (u, v) in enumerate(b["uv"]) if u < U_RIDGE), None)
        self.facts = dict(ridge=float(F(U_RIDGE, 0.0)), shallow=float(F(U_SHALLOW, 0.0)), deep=float(F(U_DEEP, 0.0)),
                          u_ridge=U_RIDGE, u_shallow=U_SHALLOW, u_deep=U_DEEP, k_esc=k_esc,
                          h_settled=(ha[SHAKE[0]], hb[SHAKE[0]]), h_end=(ha[-1], hb[-1]), h_max_shake=max(hb[SHAKE[0]:SHAKE[1] + 1]))
        t_shake = T_FIRST + SHAKE[0] * STEP_DT            # the first shaken slide begins
        t_esc = self.t_land(k_esc) if k_esc else t_shake + 3
        t_off = T_FIRST + SHAKE[1] * STEP_DT
        self.t_shake, self.t_off = t_shake, t_off
        hook = [("Two valleys.", "black", 64, INK, 300, 1), ("Where you start decides where you end.", "black", 44, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [("A hill with two bottoms.", "black", 48, INK, 300, 1), (f"One valley is shallow (height {self.facts['shallow']:.1f}), the other deep (0).", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [(f"Two balls, dropped {2 * DELTA:.1f} apart on the ridge.", "black", 44, INK, 300, 1), ("Same rule, same step size.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.4, [("Plain descent only sees the slope under its feet.", "black", 40, INK, 300, 1), ("Whichever way the ground tilts, that's the way it goes.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.8, t_shake - 0.3, [(f"Settled: {ha[SHAKE[0]]:.2f} and {hb[SHAKE[0]]:.2f}.", "black", 46, INK, 300, 1), ("The blue ball would sit in the shallow valley forever.", "dbold", 36, BLUE, 390, 0.95)], 10, 8, "pop"),
            (t_shake, t_esc + 0.3, [("Now shake it: noise on the slope it feels, like a small batch.", "black", 36, BLUE, 300, 1), (f"Grey arrow: the true slope.  Blue: what it felt.  {SHAKE[1] - SHAKE[0]} shaken steps.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_esc + 0.6, t_off - 0.2, [(f"Step {k_esc}: over the ridge, into the deep valley.", "black", 42, BLUE, 300, 1), ("Noise isn't only a cost.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_off + 0.1, 25.6, [("Shaking off. It settles at the bottom.", "black", 44, INK, 300, 1), (f"Height {hb[-1]:.3f}: a place it could never have walked to.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (26.0, 29.0, [("Where you start decides where you end. Unless you shake.", "black", 38, INK, 300, 1), (f"With this noise, {ESCAPE_PCT}% of runs climb out within {SHAKE[1] - SHAKE[0]} steps. This is one.", "dbold", 32, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the true slope beside the shaken arrow

    def draw_ball_surface(self, ctx, b, t, fade, a_ball, drop):
        out = super().draw_ball_surface(ctx, b, t, fade, a_ball, drop)
        k, phase, u = self.state(t)
        if not (k < N_STEPS and b["noisy"][k] and phase in ("arrow", "move")):
            return out
        if phase == "arrow":
            g, a = smooth(u), 1.0
        else:
            g, a = 1.0, 1 - smooth(seg(u, 0.0, 0.5))
        if b["true_px"][k] >= gds.ARROW_MIN_PX and a > 0:
            gds.ACC = GREY
            self.arrow(ctx, poly_prefix(b["true_poly"][k], g), 0.55 * a * fade, b["true_px"][k], halo=0.0)
            gds.ACC = b["col"]
        return out

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        a_ball = a_ball * self.ball_vis(b, t)
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
            lab = b["name"]
            if k < N_STEPS and b["noisy"][min(k, N_STEPS - 1)]:
                lab += "   ·   SHAKING"
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
            ctx.move_to(xc - 140, Y_BAR)
            ctx.line_to(xc + 140, Y_BAR)
            ctx.stroke()
            L = 280 * min(1.0, z / b["heights"][0])
            if L > 0.5:
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
            a_w = a * ease_out_cubic(seg(t, self.t_land(N_STEPS) + 0.3, self.t_land(N_STEPS) + 0.9))
            if a_w > 0:
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


def escape_fraction(n_runs=100):
    """Of n_runs random shakes of ball B (same noise, same steps), how many end in the deep valley."""
    n = 0
    for seed in range(n_runs):
        pts, _, _ = run_ball(+1, SHAKE, SIGMA, seed)
        u, v = pts[-1]
        if u < U_RIDGE and F(u, v) < 0.4:
            n += 1
    return n


ESCAPE_PCT = round(100 * escape_fraction() / 100)


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
        print(f"  {b['name']} height {b['heights'][0]:.2f} -> step 18 {b['heights'][18]:.3f} -> end {b['heights'][-1]:.4f}  max v {max(v for _, v in b['uv']):.2f} max h {max(b['heights']):.2f}")
    print("facts", sc.facts, "escape %", ESCAPE_PCT)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 220, 300, 400, 560, 640, 760, 880, 960, 1000, 1060, 1120, 1200, 1300, 1400, 1560, 1700]
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
