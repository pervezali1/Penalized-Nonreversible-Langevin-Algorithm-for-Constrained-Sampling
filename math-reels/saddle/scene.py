"""The saddle, in the white style of reels 23-29: the two-valley hill with no tilt, so the pass between the
two equal valleys is a true saddle point, slope zero in every direction and not a bottom. Three balls, same
rule, step size 0.1. A starts exactly on the saddle line: it rolls down to the pass and stops there for
good. B starts 0.01 off the line: its offset grows 22% a step, so it creeps along the flat top for twenty
steps before sliding into a valley. C starts 0.01 off too but measures its slope on a batch of 4 of the
1,000 examples (reel 29's tilted data): the noise throws it off the pass within a few steps. Rendered by
../valleys/hillgen.py in the reel-23 look on the faint tinted grid. frame(f) -> uint8 1080x1920 RGB."""

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
sys.path.insert(0, os.path.join(ROOT, "valleys2"))
sys.path.insert(0, HERE)
import textkit  # noqa: E402
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)
import tiltsim as TS  # noqa: E402

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
import hillgen  # noqa: E402

# ---------------------------------------------------------------- the hill: the symmetric double well (no tilt), saddle at the centre
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


gds.V_CUT = 0.85
N_STEPS, STEP_DT = 40, 0.42
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
V0 = -0.9
OFF = 0.01                            # B's and C's offset from the saddle line
BATCH_C, SEED_C = 4, 2                # C measures its slope on 4 of the 1,000 examples
SADDLE_H = float(F(0.0, 0.0))
GROWTH = 1 + LR * 4 * K_ / (A2_ * A2_)  # per-step growth of a small offset at the saddle: 1 - lr * f_uu(0)
VERM, BLUE, MAGENTA = gds.AMBER, gds.BLUE, gds.MAGENTA
BALLS = [
    dict(key="a", name="A", pill="A", u0=0.0, batch=None, col=VERM),
    dict(key="b", name="B", pill="B", u0=OFF, batch=None, col=BLUE),
    dict(key="c", name="C", pill="C", u0=OFF, batch=BATCH_C, col=MAGENTA),
]
COL_X = (190, 540, 890)
GREY = "#9CA3AF"
BG = "#F4F5F9"
GRID_A = 0.05


def run_ball(u0, batch, seed, tau):
    """Plain descent; with a batch, the slope is the true slope plus the mean tilt of a fresh random batch."""
    rng = np.random.default_rng(seed)
    u, v = u0, V0
    pts, noisy, true_steps = [(u, v)], [], []
    for _ in range(N_STEPS):
        gu, gv = FU(u, v), FV(u, v)
        true_steps.append((-LR * gu, -LR * gv))
        if batch:
            n = TS.batch_noise(tau, rng, batch)
            gu, gv = gu + n[0], gv + n[1]
        noisy.append(bool(batch))
        u, v = u - LR * gu, v - LR * gv
        pts.append((u, v))
    return pts, noisy, true_steps


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, grad_xy, surf_pt = gds.f_xy, gds.grad_xy, gds.surf_pt
        self.tau = TS.make_tilts()
        self.balls = []
        for b in BALLS:
            uv, noisy, true_steps = run_ball(b["u0"], b["batch"], SEED_C, self.tau)
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9), noisy=noisy,
                     heights=[float(f_xy(*p)) for p in path], slopes=[float(math.hypot(*grad_xy(*p))) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])],
                     off=[abs(u) for u, _ in uv])
            d["k_off"] = next((k for k, (u, _) in enumerate(uv) if abs(u) > 0.5), None)     # off the pass
            d["side"] = "left" if uv[-1][0] < 0 else "right"
            assert max(d["heights"]) < Z_RIM, b["key"]
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
        A, B, C = self.balls
        kb, kc = B["k_off"], C["k_off"]
        k_pass = next(k for k, h in enumerate(A["heights"]) if h < SADDLE_H + 0.01)     # A has reached the pass
        double = math.log(2) / math.log(GROWTH)
        self.facts = dict(saddle_h=SADDLE_H, growth=GROWTH, double_every=double, k_pass=k_pass, kb=kb, kc=kc,
                          side_b=B["side"], side_c=C["side"], h_end=[b["heights"][-1] for b in self.balls],
                          off_b=[B["off"][k] for k in (0, 5, 10, 15, 20)], max_h_c=max(C["heights"]), max_v_c=max(v for _, v in C["uv"]))
        t_pass, t_b, t_c = self.t_land(k_pass), self.t_land(kb), self.t_land(kc)
        hook = [("The saddle.", "black", 64, INK, 300, 1), ("Slope zero, and not a bottom.", "black", 44, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [("Two valleys, and a pass between them.", "black", 46, INK, 300, 1), (f"On top of the pass the ground is flat in every direction. Height {SADDLE_H:.1f}.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [(f"Three balls. Same rule, step = slope × {LR:g}.", "black", 42, INK, 300, 1), (f"A: exactly on the line.  B: {OFF:g} off.  C: {OFF:g} off, slope from {BATCH_C} examples.", "dbold", 32, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, t_pass + 0.4, [("All three roll down towards the pass.", "black", 44, INK, 300, 1), (f"Grey arrow: C's true slope. Magenta: what it measured from {BATCH_C} examples.", "dbold", 33, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_pass + 0.7, t_b - 0.3, [(f"A is on the pass: slope zero, stuck at {SADDLE_H:.1f}. C was thrown off at step {kc}.", "black", 33, INK, 300, 1), (f"B is creeping: its offset grows {GROWTH - 1:.0%} a step, doubling every {double:.1f}.", "dbold", 33, BLUE, 390, 0.95)], 10, 8, "pop"),
            (t_b, t_b + 3.4, [(f"Step {kb}: B is off the pass and falling.", "black", 42, BLUE, 300, 1), ("Flat ground is slow. It isn't a trap.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_b + 3.7, self.t_land(N_STEPS) + 0.3, [("Only the ball placed exactly on the line stays.", "black", 40, INK, 300, 1), ("In millions of dimensions, exactly never happens.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (self.t_land(N_STEPS) + 0.6, 29.0, [(f"After {N_STEPS} steps: {A['heights'][-1]:.2f}, {B['heights'][-1]:.3f}, {C['heights'][-1]:.3f}.", "black", 42, INK, 300, 1), ("Saddles slow training down. They almost never stop it.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the true slope beside C's measured arrow

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
        # A and B share a spot for twenty steps: A's pill to the upper left, B's to the upper right, C's above
        if b["key"] == "a":
            px, py = cx - rb_px - tw / 2 - 14, cy - 80 * drop - rb_px - 4
        elif b["key"] == "b":
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

    # ------------------------------------------------------------ readouts: three columns, height and offset

    def readouts3(self, ctx, t, fade):
        a = fade * ease_out_cubic(seg(t, 6.4, 6.9))
        if a <= 0:
            return
        k, phase, u = self.state(t)
        Y_LAB, Y_VAL, Y_BAR, Y_WORD = 1404, 1466, 1489, 1522
        font(ctx, "dbold", 46)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        hmax = max(b["heights"][0] for b in self.balls)
        for b, xc in zip(self.balls, COL_X):
            kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
            z = max(0.0, b["heights"][kz])
            font(ctx, "bold", 22)
            lab = {"a": "A  ·  ON THE LINE", "b": f"B  ·  {OFF:g} OFF", "c": f"C  ·  {OFF:g} OFF, {BATCH_C} EXAMPLES"}[b["key"]]
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
            ctx.move_to(xc - 110, Y_BAR)
            ctx.line_to(xc + 110, Y_BAR)
            ctx.stroke()
            L = 220 * min(1.0, z / hmax)
            if L > 0.5:
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
            # the offset from the saddle line, small, under the bar (the creeping is invisible in the height)
            off = b["off"][kz]
            font(ctx, "regular", 22)
            s = f"off the line: {off:.3f}" if off < 1 else f"off the line: {off:.2f}"
            ctx.set_source_rgba(*rgb(INK3), a * ease_out_cubic(seg(t, 9.6, 10.1)))
            ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
            ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = f"HEIGHT (ERROR) AFTER EACH STEP   ·   STEP SIZE {LR:g}"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, max(300, CX - w / 2 + 60), 1372, tr)
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
    for b in sc.balls:
        print(f"  ball {b['name']} start u {b['u0']:g} h {b['heights'][0]:.2f} -> end {b['heights'][-1]:.4f} ({b['side']})  off the pass at step {b['k_off']}  max v {max(v for _, v in b['uv']):.2f}  max h {max(b['heights']):.2f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 300, 500, 600, 680, 760, 860, 960, 1060, 1160, 1260, 1360, 1460, 1540, 1620, 1700, 1760]
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
