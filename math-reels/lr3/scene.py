"""The learning rate, in the white style of reel 23: the same cut hill, three balls, three step sizes.
Too small (0.02 x slope) crawls, just right (0.09 x slope) settles, too big (2 x slope) jumps clean
over the bottom and lands at the same height on the far side every time. Reuses the shaded bowl,
sphere and arrow renderer of gdsimple/scene.py. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
import os as _os
gds.V_CUT = float(_os.environ.get('LR3_VCUT', '1.25'))   # the cut plane sits further back than in reel 23 so most of the big ball's widening crossings stay on the drawn surface

FPS, N_FRAMES = gds.FPS, gds.N_FRAMES
T_SURF, T_BALL, T_FIRST, STEP_DT, N_STEPS, LOOP_FADE = gds.T_SURF, gds.T_BALL, gds.T_FIRST, gds.STEP_DT, gds.N_STEPS, gds.LOOP_FADE
A_, B_, KZ, Z_RIM, CX = gds.A_, gds.B_, gds.KZ, gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
f_xy, grad_xy, surf_pt, seg, lerp, poly_prefix = gds.f_xy, gds.grad_xy, gds.surf_pt, gds.seg, gds.lerp, gds.poly_prefix
_cr, _sr = gds._cr, gds._sr
TAU = 2 * math.pi

# the three balls: step size (learning rate), start in the bowl's own (u, v) axes, colour
BALLS = [
    dict(key="small", name="TOO SMALL", lr=0.03, start_uv=(2.6, -0.95), col="#059669", word="crawling"),
    dict(key="right", name="JUST RIGHT", lr=0.2, start_uv=(2.85, -0.64), col="#E4572E", word="settled"),
    dict(key="big", name="TOO BIG", lr=0.6, start_uv=(2.9, -0.42), col="#DB2777", word="gone"),
]
COL_X = (190, 540, 890)             # readout columns


def uv_to_xy(u, v):
    return _cr * u - _sr * v, _sr * u + _cr * v


def descend_uv(u, v, lr, n):
    """Gradient descent on f = A u^2 + B v^2, done in the bowl's own axes (exact: the gradient is diagonal there)."""
    pts = [(u, v)]
    for _ in range(n):
        u, v = u * (1 - 2 * A_ * lr), v * (1 - 2 * B_ * lr)
        pts.append((u, v))
    return pts


class Scene(gds.Scene):
    def __init__(self):
        self.balls = []
        for b in BALLS:
            uv = descend_uv(b["start_uv"][0], b["start_uv"][1], b["lr"], N_STEPS)
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b)
            # gone: the first landing whose true height is above the rim; the readout says "left the hill" from there.
            # vis_gone: the first landing that is off the drawn hill (above the rim, or in the cut-away region in
            # front of the cut plane, where the surface is not drawn); the ball fades out on the slide to it.
            gone = next((i for i, (uu, vv) in enumerate(uv) if f_xy(*uv_to_xy(uu, vv)) > Z_RIM), None)
            vis_gone = next((i for i, (uu, vv) in enumerate(uv) if f_xy(*uv_to_xy(uu, vv)) > Z_RIM or vv > gds.V_CUT - 0.02), None)
            d.update(path=path, uv=uv, spt=spt, gone=gone, vis_gone=vis_gone,
                     heights=[f_xy(*p) for p in path], slopes=[math.hypot(*grad_xy(*p)) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
            self.balls.append(d)
        right = self.balls[1]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            right["path"], right["heights"], right["slopes"], right["spt"], right["seg_px"], right["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hs = {b["key"]: b["heights"] for b in self.balls}
        big = self.balls[2]
        g = big["gone"]
        t_gone = T_FIRST + (g - 1) * STEP_DT + 0.84 * STEP_DT if g is not None else 13.8
        t_jr = max(17.8, t_gone + 3.0)      # the just-right cue waits for the gone cue to have its say
        # across the valley each crossing is |1 - 2*B*lr| as wide as the last; the cue quotes the growth per crossing
        widen = round((abs(1 - 2 * B_ * big["lr"]) - 1) * 100)
        settle = next((k for k in range(1, N_STEPS + 1) if hs["right"][k] < 0.01), None)

        def fmt(h):
            return f"{h:.3f}" if h >= 0.001 else "<0.001"

        hook = [("The step size.", "black", 64, INK, 300, 1), ("Same hill. Three balls. Three step sizes.", "black", 46, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 5.8, [("The rule: feel the slope, step downhill.", "black", 46, INK, 300, 1), ("The step is the slope times a number you choose.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.2, 9.2, [("Three balls. Three numbers.", "black", 54, INK, 300, 1), ("0.03 \u00d7 slope.   0.2 \u00d7 slope.   0.6 \u00d7 slope.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, t_gone + 0.1, [("Too small: safe, and slow.", "black", 50, BALLS[0]["col"], 300, 1), ("Too big: it crosses the floor every step, wider each time.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_gone + 0.3, t_jr - 0.3, [(f"Too big left the hill in {g} steps.", "black", 50, BALLS[2]["col"], 300, 1), (f"Each crossing {widen}% wider than the last.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_jr, 21.4, [("Just right: long steps first, short steps last.", "black", 42, BALLS[1]["col"], 300, 1), (f"It settles: below 0.01 by step {settle}.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.8, 25.2, [(f"After 24 steps: {hs['small'][-1]:.2f}, {fmt(hs['right'][-1])}, gone.", "black", 46, INK, 300, 1), ("Crawling. Settled. Gone.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (25.6, 29.0, [("The step size is the one number you choose.", "black", 42, INK, 300, 1), ("Too small wastes time. Too big never arrives.", "dbold", 40, BALLS[1]["col"], 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ one ball

    def ball_vis(self, b, t):
        """1 while the ball is on the hill; fades to 0 during the slide that takes it off; 0 after."""
        g = b["vis_gone"]
        if g is None:
            return 1.0
        k, phase, u = self.state(t)
        if k < g - 1 or (k == g - 1 and phase != "move"):
            return 1.0
        if k == g - 1 and phase == "move":
            return 1.0 - smooth(seg(u, 0.35, 0.9))
        return 0.0

    def ball_pos(self, b, t):
        k, phase, u = self.state(t)
        x, y = b["path"][min(k, N_STEPS)]
        mm = 0.0
        if phase == "move" and k < N_STEPS:
            xn, yn = b["path"][k + 1]
            mm = smooth(u)
            x, y = x + (xn - x) * mm, y + (yn - y) * mm
        return x, y, mm

    def draw_ball_surface(self, ctx, b, t, fade, a_ball, drop):
        """Everything of this ball that lies on the surface: trail, ghost arrows, live arrow, landing ring, shadow, dots."""
        k, phase, u = self.state(t)
        kk = min(k, N_STEPS)
        vis = self.ball_vis(b, t)
        g = b["vis_gone"]
        if g is not None:
            kk = min(kk, g - 1)                      # nothing of the ball exists beyond the step that took it off
            if k > g - 1:
                phase, u = "rest", 0.0
        x, y, mm = self.ball_pos(b, t)
        P, n, rb_px, rb_w, (cx, cy), (bx, by) = self.ball_geom(x, y)
        ink = rgb(INK)
        gds.ACC = b["col"]
        # trail hairline
        ctx.set_source_rgba(*ink, 0.35 * fade)
        ctx.set_line_width(2.0)
        for i in range(1, kk + 1):
            self._path(ctx, b["step_poly"][i - 1])
            ctx.stroke()
        if phase == "move" and k < N_STEPS and k <= kk:
            self._path(ctx, self.surf_poly(b["path"][k], (x, y), n=8))
            ctx.stroke()
        # walked arrows as ghosts, the current one settling into its ghost
        for i in range(0, kk):
            L = b["seg_px"][i]
            if L >= gds.ARROW_MIN_PX:
                self.arrow(ctx, b["step_poly"][i], gds.ARROW_GHOST_A * fade, L, halo=0.0)
        if k < N_STEPS and phase in ("arrow", "move") and k <= kk:
            L = b["seg_px"][k]
            if L >= gds.ARROW_MIN_PX:
                if phase == "arrow":
                    if L >= rb_px + 8:
                        self.arrow(ctx, poly_prefix(b["step_poly"][k], smooth(u)), fade, L)
                else:
                    w = smooth(seg(u, 0.15, 0.85))
                    if L >= rb_px + 8:
                        self.arrow(ctx, b["step_poly"][k], lerp(1.0, gds.ARROW_GHOST_A, w) * fade, L, halo=1.0 - w)
                    else:
                        self.arrow(ctx, b["step_poly"][k], gds.ARROW_GHOST_A * w * fade, L, halo=0.0)
        # landing ring
        if k < N_STEPS and phase in ("arrow", "move") and k <= kk and (g is None or k + 1 < g) and b["seg_px"][k] >= 3.0 * rb_px:
            ga = min(1.0, max(0.0, (smooth(u) - 0.45) / 0.4)) if phase == "arrow" else 1 - smooth(seg(u, 0.5, 0.92))
            if ga > 0:
                self.ghost(ctx, b["path"][k + 1], rb_w, ga * fade)
        # contact shadow and dots
        if vis > 0:
            self.contact_shadow(ctx, P, n, rb_w, vis * a_ball * (0.35 + 0.65 * (1 - drop)), scale=0.6 + 0.4 * (1 - drop))
        for i in range(0, kk + 1):
            r = 5.0 if i == 0 else min(5.0, max(2.2, 0.36 * b["seg_px"][i - 1]))
            ctx.set_source_rgba(*ink, 0.65 * fade * (a_ball if i == 0 else 1.0))
            pa = b["spt"][i]
            ctx.arc(pa[0], pa[1], r, 0, TAU)
            ctx.fill()
        return (cx, cy), (bx, by), rb_px

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        a_ball = a_ball * self.ball_vis(b, t)
        if a_ball <= 0.002:
            return
        cx, cy = cxy
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=bxy, squash=squash)
        # a small pill with the step size, in the ball's colour, riding above the ball
        s = f"{b['lr']:g}"
        font(ctx, "dbold", 24)
        tw = ctx.text_extents(s).x_advance
        px, py = cx, cy - 80 * drop - rb_px - 16
        if b["key"] == "big":
            # the zigzag ball ends up beside the just-right ball; its pill rides to the upper right so the two never overlap
            px, py = cx + rb_px + tw / 2 + 22, cy - 80 * drop - rb_px + 2
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)

    @staticmethod
    def _pill(ctx, x, y, w, h, r):
        ctx.new_sub_path()
        ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()

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
            # the readout holds each landing's height (the live height would dip as the big ball flies over the bottom)
            kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
            gone_now = b["gone"] is not None and kz >= b["gone"]
            z = b["heights"][min(kz, (b["gone"] - 1) if b["gone"] is not None else kz)]
            font(ctx, "bold", 22)
            lab = f"{b['name']}  ·  {b['lr']:g}"
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            if gone_now:
                s = "left the hill"
                font(ctx, "regular", 30)
                w = ctx.text_extents(s).x_advance
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - w / 2, Y_VAL - 4)
                ctx.show_text(s)
                z = 0.0
            else:
                s = gds._fmt(z, 2 if z >= 0.1 else 3) if z >= 0.001 else "<0.001"
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
            a_w = a * ease_out_cubic(seg(t, 21.8, 22.4))
            if a_w > 0:
                font(ctx, "regular", 24)
                s = b["word"]
                ctx.set_source_rgba(*rgb(INK3), a_w)
                ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
                ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = "HEIGHT (ERROR) AFTER EACH STEP"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2, 1372, tr)
        # step counter, small, top left of the readouts
        font(ctx, "bold", 22)
        s = f"STEP {min(k, N_STEPS)}"
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, 9.6, 10.0)))
        show_tracked(ctx, s, 70, 1372, tr)

    # ------------------------------------------------------------ frame

    def _draw(self, ctx, t, fade):
        if t < T_BALL:
            return
        k, phase, u = self.state(t)
        drop_u = seg(t, T_BALL, T_BALL + 0.5)
        a_ball = min(1.0, drop_u / 0.4) * fade
        drop = (1 - drop_u) ** 2
        squash = 0.0
        tl = (t - T_FIRST - 0.84 * STEP_DT) / STEP_DT
        il = int(math.floor(tl))
        age = (tl - il) * STEP_DT
        if 0 <= il < N_STEPS and age < 0.16:
            squash = 1 - ease_out_cubic(age / 0.16)
        if T_BALL + 0.5 <= t < T_BALL + 0.66:
            squash = 1 - ease_out_cubic((t - T_BALL - 0.5) / 0.16)
        tops = []
        ctx.save()
        self._path(ctx, self.surf_clip, close=True)
        ctx.clip()
        for b in self.balls:
            tops.append(self.draw_ball_surface(ctx, b, t, fade, a_ball, drop))
        ctx.restore()
        gds.ACC = gds.AMBER
        # spheres back to front (farther = smaller on screen)
        order = sorted(range(len(self.balls)), key=lambda i: -tops[i][2])
        for i in order:
            cxy, bxy, rb_px = tops[i]
            self.draw_ball_top(ctx, self.balls[i], t, fade, a_ball, drop, cxy, bxy, rb_px, squash)
        self.readouts3(ctx, t, fade)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = np.ones((H, W, 3), np.float32)
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
        print(f"  {b['name']:10s} lr {b['lr']:<5g} height {b['heights'][0]:.2f} -> {b['heights'][-1]:.4f}  uv start {b['uv'][0]}  uv end ({b['uv'][-1][0]:.3f}, {b['uv'][-1][1]:.3f})  max |u| {max(abs(p[0]) for p in b['uv']):.2f}")
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
    for fr in (620, 760, 1000, 1480):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
