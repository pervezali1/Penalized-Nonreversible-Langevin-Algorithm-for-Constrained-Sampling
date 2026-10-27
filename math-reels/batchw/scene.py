"""The noisy slope (mini-batches), in the white style of reels 23-26. The hill is the average squared error
of a two-weight least-squares fit over 1,000 data points, built so that average is EXACTLY the bowl of
reels 23-26 plus a floor of 0.10 from the labels' own noise (batchsim.py). Three balls, same rule
(step = slope x 0.2), same start; the only difference is how many of the 1,000 points each ball looks at
to measure the slope before every step: 4, 32, or all 1,000. The full-batch ball walks the reel-24 path;
the others feel the true slope plus noise and are shaken by it. During each arrow phase the noisy balls
also show the true slope as a faint grey arrow next to the one they measured. frame(f) -> uint8 RGB."""

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)
import batchsim as BS  # noqa: E402

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
gds.V_CUT = 0.7

FPS, N_FRAMES = gds.FPS, gds.N_FRAMES
T_SURF, T_BALL, T_FIRST, STEP_DT, N_STEPS, LOOP_FADE = gds.T_SURF, gds.T_BALL, gds.T_FIRST, gds.STEP_DT, gds.N_STEPS, gds.LOOP_FADE
A_, B_, Z_RIM, CX = gds.A_, gds.B_, gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
f_xy, grad_xy, surf_pt, seg, lerp, poly_prefix = gds.f_xy, gds.grad_xy, gds.surf_pt, gds.seg, gds.lerp, gds.poly_prefix
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi
assert abs(BS.A_ - A_) < 1e-12 and abs(BS.B_ - B_) < 1e-12

LR = 0.2
START_UV = (2.85, -0.64)
VERM, BLUE, MAGENTA = gds.AMBER, gds.BLUE, gds.MAGENTA
BALLS = [
    dict(key="few", name="4 EXAMPLES A STEP", pill="4", batch=4, seed=27, col=MAGENTA, word="bouncing"),
    dict(key="mid", name="32 EXAMPLES A STEP", pill="32", batch=32, seed=2, col=BLUE, word="a wobble"),
    dict(key="all", name="ALL 1,000", pill="1,000", batch=BS.N, seed=1, col=VERM, word="still"),
]
COL_X = (190, 540, 890)
GREY = "#9CA3AF"


class Scene(lr3.Scene):
    def __init__(self):
        self.X, self.e = BS.make_data()
        self.balls = []
        for b in BALLS:
            pts, grads = BS.run(self.X, self.e, START_UV, LR, b["batch"], N_STEPS, b["seed"])
            uv = [(float(p[0]), float(p[1])) for p in pts]
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            base_h = [f_xy(*p) for p in path]
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9),
                     heights=[float(BS.loss(self.X, self.e, np.array(p))) for p in uv],          # the true average error, floor included
                     slopes=[float(np.linalg.norm(g)) for g in grads] + [float(np.linalg.norm(BS.true_grad(np.array(uv[-1]))))],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
            assert max(base_h) < Z_RIM and max(v for _, v in uv) < gds.V_CUT - 0.02, b["key"]
            # the true slope's step, for the grey ghost arrow next to the measured one
            true_poly, true_px = [], []
            for k in range(N_STEPS):
                g = BS.true_grad(np.array(uv[k]))
                q = uv_to_xy(uv[k][0] - LR * g[0], uv[k][1] - LR * g[1])
                true_poly.append(self.surf_poly(path[k], q, n=16))
                a, c = spt[k], surf_pt(*q)
                true_px.append(math.hypot(c[0] - a[0], c[1] - a[1]))
            d.update(true_poly=true_poly, true_px=true_px)
            self.balls.append(d)
        full = self.balls[2]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            full["path"], full["heights"], full["slopes"], full["spt"], full["seg_px"], full["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()

    # ------------------------------------------------------------ cues

    @staticmethod
    def t_land(k):
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def _cues(self):
        few, mid, full = self.balls
        hs = {b["key"]: b["heights"] for b in self.balls}
        tail = {b["key"]: (float(np.mean(b["heights"][16:])), float(np.std(b["heights"][16:]))) for b in self.balls}
        looked = {b["key"]: b["batch"] * N_STEPS for b in self.balls}
        # how far the 4-example ball's measured slope strayed from the true one, step by step (angle), for the facts
        angs = []
        for k in range(N_STEPS):
            g_true = BS.true_grad(np.array(few["uv"][k]))
            step = np.array(few["uv"][k + 1]) - np.array(few["uv"][k])
            g_meas = -step / LR
            c = float(g_true @ g_meas / (np.linalg.norm(g_true) * np.linalg.norm(g_meas) + 1e-12))
            angs.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        self.facts = dict(floor=BS.FLOOR, tail=tail, looked=looked, h24={k: v[-1] for k, v in hs.items()}, max_angle_few=max(angs), mean_angle_few=float(np.mean(angs)))

        def fmt(h):
            return f"{h:.3f}" if h >= 0.001 else "<0.001"

        hook = [("The noisy slope.", "black", 64, INK, 300, 1), ("A network never feels the whole hill. Only a few examples of it.", "black", 38, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 5.8, [(f"This hill is the average error over {BS.N:,} examples.", "black", 42, INK, 300, 1), ("A step can look at all of them, or at a handful.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.2, 9.2, [("Three balls. Same rule, same step size, same start.", "black", 40, INK, 300, 1), (f"{few['batch']} examples a step.   {mid['batch']}.   All {BS.N:,}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.2, [("Grey arrow: the true slope.  Coloured: the slope it measured.", "black", 36, INK, 300, 1), ("Fewer examples, more noise.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.6, 17.2, [("On average the noise cancels: the expected path is the same.", "black", 36, INK, 300, 1), ("It doesn't slow them down. It shakes them.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.6, 21.2, [(f"The floor is {BS.FLOOR:.2f}: the labels themselves are noisy.", "black", 40, INK, 300, 1), ("No weights can do better than that, so no ball can.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.6, 25.0, [(f"After 24 steps: {hs['few'][-1]:.3f}, {hs['mid'][-1]:.3f}, {hs['all'][-1]:.3f}.", "black", 42, INK, 300, 1), (f"{few['batch']}: still bouncing.  {mid['batch']}: a wobble.  {BS.N:,}: still.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (25.4, 29.0, [(f"Examples looked at: {looked['few']:,} vs {looked['mid']:,} vs {looked['all']:,}.", "black", 40, INK, 300, 1), ("Small batches: cheap, noisy steps. That's the trade.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the true-slope ghost next to the measured arrow

    def draw_ball_surface(self, ctx, b, t, fade, a_ball, drop):
        out = super().draw_ball_surface(ctx, b, t, fade, a_ball, drop)
        if b["key"] == "all":
            return out
        k, phase, u = self.state(t)
        if not (k < N_STEPS and phase in ("arrow", "move")):
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
        # three balls share a start: pills above (all), lower right (32) and lower left (4)
        if b["key"] == "all":
            px, py = cx, cy - 80 * drop - rb_px - 16
        elif b["key"] == "mid":
            px, py = cx + rb_px + tw / 2 + 16, cy - 80 * drop + rb_px + 4
        else:
            px, py = cx - rb_px - tw / 2 - 16, cy - 80 * drop + rb_px + 4
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)

    # ------------------------------------------------------------ readouts: three columns

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
            z = b["heights"][kz]
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            w = tracked_width(ctx, b["name"], tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, b["name"], xc - w / 2, Y_LAB, tr)
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
            L = 220 * min(1.0, z / b["heights"][0])
            if L > 0.5:
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - L / 2, Y_BAR)
                ctx.line_to(xc + L / 2, Y_BAR)
                ctx.stroke()
            # the floor, as a small tick on the bar
            Lf = 220 * BS.FLOOR / b["heights"][0]
            ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
            ctx.set_line_width(2)
            for sgn in (-1, 1):
                ctx.move_to(xc + sgn * Lf / 2, Y_BAR - 7)
                ctx.line_to(xc + sgn * Lf / 2, Y_BAR + 7)
            ctx.stroke()
            a_w = a * ease_out_cubic(seg(t, 21.6, 22.2))
            if a_w > 0:
                font(ctx, "regular", 24)
                s = b["word"]
                ctx.set_source_rgba(*rgb(INK3), a_w)
                ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
                ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = f"ERROR AFTER EACH STEP   ·   STEP SIZE {LR:g}   ·   FLOOR {BS.FLOOR:.2f}"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2, 1372, tr)
        s = f"STEP {min(k, N_STEPS)}"
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, 9.6, 10.0)))
        show_tracked(ctx, s, 70, 1372, tr)


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s")
    for b in sc.balls:
        print(f"  {b['name']:20s} error {b['heights'][0]:.3f} -> {b['heights'][-1]:.4f}  tail mean/std {np.mean(b['heights'][16:]):.3f}/{np.std(b['heights'][16:]):.4f}  max v {max(v for _, v in b['uv']):.2f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 220, 300, 400, 560, 600, 640, 700, 780, 900, 960, 1100, 1250, 1400, 1520, 1650, 1799]
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
