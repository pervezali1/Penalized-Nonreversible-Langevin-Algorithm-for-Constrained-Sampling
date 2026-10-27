"""Momentum, in the white style of reels 23 and 24: the same cut hill, two balls, the same step size and
the same start. Plain descent steps by step size x slope and starts every step from nothing. Momentum
keeps 80% of its last step and adds the slope, so speed builds up along the valley: it arrives far
sooner, overshoots the bottom a little, and comes back. During each arrow phase the momentum ball shows
its two ingredients, a blue "memory" arrow (0.8 x the last step) and a vermilion slope arrow, and then
slides along their sum. Reuses the three-ball scene of lr3/scene.py (bowl, sphere, arrows, readouts).
frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
gds.V_CUT = 0.45                    # both paths stay on the far side of the valley; a modest cut keeps the reel-23 look

FPS, N_FRAMES = gds.FPS, gds.N_FRAMES
T_SURF, T_BALL, T_FIRST, STEP_DT, N_STEPS, LOOP_FADE = gds.T_SURF, gds.T_BALL, gds.T_FIRST, gds.STEP_DT, gds.N_STEPS, gds.LOOP_FADE
A_, B_, Z_RIM, CX = gds.A_, gds.B_, gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
f_xy, grad_xy, surf_pt, seg, lerp, poly_prefix = gds.f_xy, gds.grad_xy, gds.surf_pt, gds.seg, gds.lerp, gds.poly_prefix
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi

LR = 0.03                           # the one step size, for both balls
BETA = 0.8                          # momentum: fraction of the last step that is kept
START_UV = (2.85, -0.64)            # the reel-23 start, for both balls
BLUE, VERM = gds.BLUE, gds.AMBER
BALLS = [
    dict(key="plain", name="PLAIN DESCENT", pill="plain", col=VERM, word="crawling"),
    dict(key="mom", name=f"MOMENTUM {BETA:g}", pill="momentum", col=BLUE, word="settled"),
]
COL_X = (330, 750)


def descend_plain(u, v, lr, n):
    pts = [(u, v)]
    for _ in range(n):
        u, v = u * (1 - 2 * A_ * lr), v * (1 - 2 * B_ * lr)
        pts.append((u, v))
    return pts, [None] * (n + 1)


def descend_momentum(u, v, lr, beta, n):
    """Heavy-ball momentum on f = A u^2 + B v^2 in the bowl's own axes: step = beta * last step - lr * grad.
    Returns the landings and, per step, the remembered part of that step (beta * last step)."""
    pts, mem = [(u, v)], [None]
    du = dv = 0.0
    for _ in range(n):
        mu, mv = beta * du, beta * dv
        du, dv = mu - lr * 2 * A_ * u, mv - lr * 2 * B_ * v
        u, v = u + du, v + dv
        pts.append((u, v))
        mem.append((mu, mv))
    return pts, mem


class Scene(lr3.Scene):
    def __init__(self):
        self.balls = []
        for b in BALLS:
            if b["key"] == "plain":
                uv, mem = descend_plain(*START_UV, LR, N_STEPS)
            else:
                uv, mem = descend_momentum(*START_UV, LR, BETA, N_STEPS)
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9),
                     heights=[f_xy(*p) for p in path], slopes=[math.hypot(*grad_xy(*p)) for p in path],
                     step_len=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(uv[:-1], uv[1:])],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
            if b["key"] == "mom":
                # the two ingredients of step k (k >= 1): memory from path[k] to path[k] + beta*last, slope from there to path[k+1]
                mem_poly, slope_poly, mem_px, slope_px = [None], [None], [0.0], [0.0]
                mem_vec, slope_vec = [None], [None]
                for k in range(1, N_STEPS):
                    mx, my = uv_to_xy(*mem[k + 1])
                    tip = (path[k][0] + mx, path[k][1] + my)
                    mem_poly.append(self.surf_poly(path[k], tip, n=16))
                    slope_poly.append(self.surf_poly(tip, path[k + 1], n=16))
                    a, c = surf_pt(*path[k]), surf_pt(*tip)
                    e = spt[k + 1]
                    mem_px.append(math.hypot(c[0] - a[0], c[1] - a[1]))
                    slope_px.append(math.hypot(e[0] - c[0], e[1] - c[1]))
                    mem_vec.append((c[0] - a[0], c[1] - a[1]))
                    slope_vec.append((e[0] - c[0], e[1] - c[1]))
                d.update(mem_poly=mem_poly, slope_poly=slope_poly, mem_px=mem_px, slope_px=slope_px, mem_vec=mem_vec, slope_vec=slope_vec)
            self.balls.append(d)
        mom = self.balls[1]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            mom["path"], mom["heights"], mom["slopes"], mom["spt"], mom["seg_px"], mom["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()

    # ------------------------------------------------------------ cues (every number from the two runs)

    @staticmethod
    def t_land(k):
        """Time at which landing k (1-based step count) is reached."""
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def _cues(self):
        plain, mom = self.balls
        hp, hm = plain["heights"], mom["heights"]
        # where momentum's step is longest, and how it compares with plain's step at that moment
        k_fast = int(np.argmax(mom["step_len"])) + 1
        ratio = mom["step_len"][k_fast - 1] / plain["step_len"][k_fast - 1]
        # the overshoot: the first landing past the bottom along the valley (u < 0), then the height's local rise
        k_over = next((k for k, (u, v) in enumerate(mom["uv"]) if u < 0), None)
        k_min = int(np.argmin(hm[:k_over + 1])) if k_over else int(np.argmin(hm))
        k_peak = (k_min + int(np.argmax(hm[k_min:]))) if k_over else k_min
        self.facts = dict(k_fast=k_fast, ratio=ratio, k_over=k_over, k_min=k_min, k_peak=k_peak)
        t_over = self.t_land(k_over) if k_over else 20.0
        t_mid = 15.9
        k_mid = int((t_mid - T_FIRST) / STEP_DT)           # the step in progress at t_mid; quote the last landing
        def fmt(h):
            return f"{h:.3f}" if h >= 0.01 else f"{h:.4f}"

        hook = [("Momentum.", "black", 64, INK, 300, 1), ("The ball that remembers which way it was going.", "black", 42, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 5.8, [("Plain descent: feel the slope, step downhill.", "black", 44, VERM, 300, 1), ("Every step starts from nothing.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.2, 9.2, [(f"Momentum: keep {BETA:.0%} of your last step, add the slope.", "black", 40, BLUE, 300, 1), ("Same hill. Same step size. Same start.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 12.8, [("Blue arrow: what it remembers.  Red arrow: the slope.", "black", 38, INK, 300, 1), ("Its step is the two added together.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.1, t_mid - 0.3, [(f"Step {k_fast}: its step is {ratio:.1f}× longer than plain's.", "black", 42, BLUE, 300, 1), ("The slope kept pointing the same way, so speed built up.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_mid, t_over - 0.4, [(f"Step {k_mid}: plain {hp[k_mid]:.2f}, momentum {hm[k_mid]:.3f}.", "black", 42, INK, 300, 1), ("Plain descent is still crawling.", "dbold", 40, VERM, 390, 0.95)], 10, 8, "pop"),
            (t_over - 0.1, t_over + 3.0, [("It overshoots the bottom, and comes back.", "black", 42, BLUE, 300, 1), (f"Height {fmt(hm[k_min])} at step {k_min}, {fmt(hm[k_peak])} at step {k_peak}, then down again.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_over + 3.4, self.t_land(N_STEPS) + 0.2, [("Faster wherever the slope agrees with you.", "black", 42, BLUE, 300, 1), ("Slower to stop wherever it flips.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (self.t_land(N_STEPS) + 0.5, 29.0, [(f"After 24 steps: plain {hp[-1]:.2f}, momentum {hm[-1]:.3f}.", "black", 42, INK, 300, 1), ("Nearly every network is trained this way. Adam has it built in.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the momentum ball's two ingredient arrows

    def draw_ball_surface(self, ctx, b, t, fade, a_ball, drop):
        out = super().draw_ball_surface(ctx, b, t, fade, a_ball, drop)
        if b["key"] != "mom":
            return out
        k, phase, u = self.state(t)
        if not (1 <= k < N_STEPS and phase in ("arrow", "move")):
            return out
        if phase == "arrow":
            a_mem, a_slope = smooth(seg(u, 0.0, 0.5)), smooth(seg(u, 0.3, 0.85))
            g_mem, g_slope = smooth(seg(u, 0.0, 0.55)), smooth(seg(u, 0.3, 0.9))
        else:
            w = 1 - smooth(seg(u, 0.0, 0.5))
            a_mem = a_slope = w
            g_mem = g_slope = 1.0
        if b["mem_px"][k] >= gds.ARROW_MIN_PX and a_mem > 0:
            gds.ACC = BLUE
            self.arrow(ctx, poly_prefix(b["mem_poly"][k], g_mem), 0.75 * a_mem * fade, b["mem_px"][k], halo=0.0)
        if b["slope_px"][k] >= gds.ARROW_MIN_PX and a_slope > 0:
            gds.ACC = VERM
            self.arrow(ctx, poly_prefix(b["slope_poly"][k], g_slope), 0.9 * a_slope * fade, b["slope_px"][k], halo=0.0)
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
        px, py = cx, cy - 80 * drop - rb_px - 16
        if b["key"] == "mom":
            # the two balls share a start, so the momentum pill rides to the lower right while plain's sits above
            px, py = cx + rb_px + tw / 2 + 16, cy - 80 * drop + rb_px + 4
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)

    # ------------------------------------------------------------ inset: momentum's current step, magnified

    INSET_C, INSET_R, INSET_M = (874.0, 1158.0), 122.0, 4.0

    def inset(self, ctx, t, fade):
        """The momentum ball's current step as a vector triangle, the main view's own screen vectors magnified INSET_M
        times: blue memory arrow (BETA x last step), vermilion slope arrow from its tip, and their sum, the step."""
        b = self.balls[1]
        t_in = T_FIRST + STEP_DT - 0.3
        t_out = self.t_land(N_STEPS) + 0.3
        a = fade * ease_out_cubic(seg(t, t_in, t_in + 0.45)) * (1 - smooth(seg(t, t_out, t_out + 0.5)))
        if a <= 0:
            return
        k, phase, u = self.state(t)
        k = max(1, min(k, N_STEPS - 1))
        Cx, Cy, R, M = self.INSET_C[0], self.INSET_C[1], self.INSET_R, self.INSET_M
        # disc
        ctx.set_source_rgba(1, 1, 1, 0.96 * a)
        ctx.arc(Cx, Cy, R, 0, TAU)
        ctx.fill()
        ctx.set_source_rgba(*rgb(INK3), 0.55 * a)
        ctx.set_line_width(1.5)
        ctx.arc(Cx, Cy, R, 0, TAU)
        ctx.stroke()
        mv, sv = b["mem_vec"][k], b["slope_vec"][k]
        rv = (mv[0] + sv[0], mv[1] + sv[1])
        ox, oy = Cx - 0.5 * M * rv[0], Cy - 0.5 * M * rv[1]        # the triangle centred in the disc
        tip = (ox + M * mv[0], oy + M * mv[1])
        end = (ox + M * rv[0], oy + M * rv[1])
        if phase == "arrow":
            g_mem, g_slope, a_res = smooth(seg(u, 0.0, 0.55)), smooth(seg(u, 0.3, 0.9)), smooth(seg(u, 0.7, 1.0))
            a_cmp = 1.0
        else:
            g_mem = g_slope = a_res = 1.0
            a_cmp = 0.6
        ctx.save()
        ctx.arc(Cx, Cy, R - 1, 0, TAU)
        ctx.clip()
        Lm, Ls, Lr = M * math.hypot(*mv), M * math.hypot(*sv), M * math.hypot(*rv)
        if Lm > 4 and g_mem > 0:
            gds.ACC = BLUE
            self.arrow(ctx, [(ox, oy), (ox + g_mem * M * mv[0], oy + g_mem * M * mv[1])], 0.75 * a_cmp * a, Lm, halo=0.0)
        if Ls > 4 and g_slope > 0:
            gds.ACC = VERM
            self.arrow(ctx, [tip, (tip[0] + g_slope * M * sv[0], tip[1] + g_slope * M * sv[1])], 0.9 * a_cmp * a, Ls, halo=0.0)
        if Lr > 4 and a_res > 0:
            # where the step lands: a dashed ring, as in the main view
            ctx.set_source_rgba(*rgb(INK), 0.55 * a_res * a)
            ctx.set_line_width(1.6)
            ctx.set_dash([4.0, 4.0])
            ctx.arc(end[0], end[1], 11.0, 0, TAU)
            ctx.stroke()
            ctx.set_dash([])
        gds.ACC = VERM
        # the ball, as a fixed marker at the start of the step
        self.sphere(ctx, ox, oy, 11.0, a, contact=None, squash=0.0)
        ctx.restore()
        # caption under the disc
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        lab = f"MOMENTUM'S NEXT STEP   \u00b7   \u00d7{M:g}"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        show_tracked(ctx, lab, Cx - w / 2, Cy + R + 30, tr)
        font(ctx, "regular", 21)
        legend = [(f"{BETA:g} \u00d7 last step", BLUE), ("+  slope \u00d7 " + f"{LR:g}", VERM)]
        y = Cy + R + 60
        wtot = sum(ctx.text_extents(s_)[4] for s_, _ in legend) + 2 * 22
        x = Cx - wtot / 2
        for s_, col in legend:
            ctx.set_source_rgba(*rgb(col), 0.95 * a)
            ctx.move_to(x, y)
            ctx.show_text(s_)
            x += ctx.text_extents(s_)[4] + 22

    # ------------------------------------------------------------ readouts: two columns

    def readouts3(self, ctx, t, fade):
        self.inset(ctx, t, fade)
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
            lab = b["name"]
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            s = gds._fmt(z, 2 if z >= 0.1 else 3) if z >= 0.001 else "<0.001"
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
        show_tracked(ctx, lab, CX - w / 2, 1372, tr)
        font(ctx, "bold", 22)
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
        print(f"  {b['name']:14s} height {b['heights'][0]:.2f} -> {b['heights'][-1]:.4f}  uv end ({b['uv'][-1][0]:.3f}, {b['uv'][-1][1]:.3f})  max step {max(b['step_len']):.3f}")
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
    for fr in (660, 900, 1200, 1480):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
