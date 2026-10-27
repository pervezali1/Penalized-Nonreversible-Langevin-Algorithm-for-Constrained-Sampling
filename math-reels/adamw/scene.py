"""Adam, in the white style of reels 23-25: the step size that does not care how steep the hill is.
Three acts on the same cut hill, two balls each, the same number (0.25) for both. Plain descent steps by
slope x 0.25; Adam divides the slope by a running measure of its own size, then steps by 0.25 x that.
Act 1: the hill as before, plain descent wins. Act 2: the same hill 100 times steeper, plain descent
leaves in one step, Adam walks the identical path. Act 3: the same hill 100 times flatter, plain
descent freezes, Adam walks the identical path again. Reuses the three-ball machinery of lr3/scene.py.
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
gds.V_CUT = 0.45

# three short acts instead of one long run: 12 steps of 0.45 s each, the balls dropping 1 s before the first step
N_STEPS, STEP_DT = 12, 0.45
T_FIRST = gds.T_FIRST                                   # act-local time of the first step (9.6), as in reels 23-25
T_BALL = T_FIRST - 1.0
for mod in (gds, lr3):
    mod.N_STEPS, mod.STEP_DT, mod.T_BALL = N_STEPS, STEP_DT, T_BALL
FPS, N_FRAMES, T_SURF, LOOP_FADE = gds.FPS, gds.N_FRAMES, gds.T_SURF, gds.LOOP_FADE
A_, B_, Z_RIM, CX = gds.A_, gds.B_, gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
f_xy, grad_xy, surf_pt, seg, lerp, poly_prefix = gds.f_xy, gds.grad_xy, gds.surf_pt, gds.seg, gds.lerp, gds.poly_prefix
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi

LR = 0.25                               # the one number, for both balls
B1, B2, EPS = 0.9, 0.999, 1e-8          # Adam's defaults
START_UV = (2.85, -0.64)
VERM, GREEN = gds.AMBER, gds.GREEN
ACTS = [
    dict(scale=1.0, label="× 1", t_first=6.4),
    dict(scale=100.0, label="× 100", t_first=13.0),
    dict(scale=0.01, label="÷ 100", t_first=19.6),
]
BALL_DEFS = [
    dict(key="plain", name="PLAIN DESCENT", pill="plain", col=VERM),
    dict(key="adam", name="ADAM", pill="adam", col=GREEN),
]
COL_X = (330, 750)


def run_plain(u, v, lr, scale, n):
    """Plain descent on scale * (A u^2 + B v^2): exact in the bowl's own axes."""
    pts = [(u, v)]
    for _ in range(n):
        u, v = u * (1 - 2 * A_ * lr * scale), v * (1 - 2 * B_ * lr * scale)
        pts.append((u, v))
    return pts


def run_adam(u, v, lr, scale, n):
    """Adam (Kingma & Ba defaults, bias-corrected) on scale * (A u^2 + B v^2)."""
    pts = [(u, v)]
    mu = mv = su = sv = 0.0
    for t in range(1, n + 1):
        gu, gv = scale * 2 * A_ * u, scale * 2 * B_ * v
        mu, mv = B1 * mu + (1 - B1) * gu, B1 * mv + (1 - B1) * gv
        su, sv = B2 * su + (1 - B2) * gu * gu, B2 * sv + (1 - B2) * gv * gv
        c1, c2 = 1 - B1 ** t, 1 - B2 ** t
        u -= lr * (mu / c1) / (math.sqrt(su / c2) + EPS)
        v -= lr * (mv / c1) / (math.sqrt(sv / c2) + EPS)
        pts.append((u, v))
    return pts


class Scene(lr3.Scene):
    def __init__(self):
        self.acts = []
        for act in ACTS:
            balls = []
            for b in BALL_DEFS:
                uv = (run_plain if b["key"] == "plain" else run_adam)(*START_UV, LR, act["scale"], N_STEPS)
                path = [uv_to_xy(*p) for p in uv]
                spt = [surf_pt(*p) for p in path]
                d = dict(b, lr=LR)
                base_h = [f_xy(*p) for p in path]
                gone = next((i for i, h in enumerate(base_h) if h > Z_RIM), None)
                vis_gone = next((i for i, (h, (uu, vv)) in enumerate(zip(base_h, uv)) if h > Z_RIM or vv > gds.V_CUT - 0.02), None)
                d.update(path=path, uv=uv, spt=spt, gone=gone, vis_gone=vis_gone,
                         heights=[act["scale"] * h for h in base_h], slopes=[act["scale"] * math.hypot(*grad_xy(*p)) for p in path],
                         step_len=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(uv[:-1], uv[1:])],
                         seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                         step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])])
                d["exit_fade"] = (0.05, 0.4) if vis_gone is not None and base_h[vis_gone] > 2 * Z_RIM else (0.35, 0.9)
                if vis_gone is not None and base_h[vis_gone] > 4 * Z_RIM:
                    # the slide that takes it off lands absurdly far away (the x100 hill: a step of about 14 bowl-lengths); draw
                    # that one arrow and slide towards a point 6 units out in the same direction, so the line leaves the
                    # picture cleanly instead of being folded across the bowl. The readout keeps the true heights.
                    i = vis_gone - 1
                    (x0, y0), (x1, y1) = path[i], path[i + 1]
                    L = math.hypot(x1 - x0, y1 - y0)
                    far = (x0 + (x1 - x0) * 6.0 / L, y0 + (y1 - y0) * 6.0 / L)
                    path[i + 1] = far
                    spt[i + 1] = surf_pt(*far)
                    d["seg_px"][i] = math.hypot(spt[i + 1][0] - spt[i][0], spt[i + 1][1] - spt[i][1])
                    d["step_poly"][i] = self.surf_poly(path[i], far, n=24)
                balls.append(d)
            self.acts.append(dict(act, balls=balls, t_drop=act["t_first"] - 1.0, t_end=act["t_first"] + (N_STEPS - 1) * STEP_DT + 0.84 * STEP_DT))
        self.balls = self.acts[0]["balls"]
        adam = self.balls[1]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            adam["path"], adam["heights"], adam["slopes"], adam["spt"], adam["seg_px"], adam["step_poly"])
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()

    # ------------------------------------------------------------ cues

    def _cues(self):
        a1, a2, a3 = self.acts
        hp1, ha1 = a1["balls"][0]["heights"], a1["balls"][1]["heights"]
        g2 = a2["balls"][0]["gone"]
        hp3 = a3["balls"][0]["heights"]
        moved = (1 - hp3[-1] / hp3[0]) * 100
        self.facts = dict(plain1=hp1[-1], adam1=ha1[-1], gone2=g2, plain3_pct=moved, adam_end_uv=a1["balls"][1]["uv"][-1])
        hook = [("Adam.", "black", 64, INK, 300, 1), ("The step size that doesn't care how steep the hill is.", "black", 40, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [(f"Plain descent: step = slope × {LR:g}.", "black", 42, VERM, 300, 1), (f"Adam: slope ÷ its own typical size, × {LR:g}.", "dbold", 38, GREEN, 390, 0.95)], 10, 8, "pop"),
            (a1["t_first"], a2["t_drop"] - 0.3, [(f"The hill as before. The same number, {LR:g}, for both.", "black", 40, INK, 300, 1), ("On this hill, plain descent wins.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (a2["t_drop"], a3["t_drop"] - 0.3, [("The same hill, 100 times steeper.", "black", 44, INK, 300, 1), (f"Plain descent: gone in {g2} step.  Adam: the same path.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (a3["t_drop"], a3["t_end"] + 0.5, [("The same hill, 100 times flatter.", "black", 44, INK, 300, 1), (f"Plain descent: down {moved:.0f}% in 12 steps.  Adam: the same path, again.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (a3["t_end"] + 0.8, 29.0, [("Adam divides the slope by its own size.", "black", 42, INK, 300, 1), ("One step size fits every steepness. That's why 0.001 works almost everywhere.", "dbold", 32, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ acts: local time

    def act_at(self, t):
        """(act, local time, act fade) for global t: the latest act whose drop has begun; its picture fades out in
        the half second before the next act's drop."""
        cur = None
        for i, a in enumerate(self.acts):
            if t >= a["t_drop"]:
                cur = i
        if cur is None:
            return None, 0.0, 0.0
        a = self.acts[cur]
        afade = 1.0
        if cur + 1 < len(self.acts):
            afade = 1 - smooth(seg(t, self.acts[cur + 1]["t_drop"] - 0.5, self.acts[cur + 1]["t_drop"]))
        return a, t - a["t_first"] + T_FIRST, afade

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
        if b["key"] == "adam":
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

    @staticmethod
    def fmt_h(z):
        if z >= 100:
            return f"{z:.0f}"
        if z >= 10:
            return f"{z:.1f}"
        if z >= 0.1:
            return f"{z:.2f}"
        if z >= 0.001:
            return f"{z:.3f}"
        return f"{z:.4f}"

    def readouts3(self, ctx, t, fade):
        """Two columns for the current act; t is act-local."""
        a = fade * ease_out_cubic(seg(t, T_BALL - 0.2, T_BALL + 0.3))
        if a <= 0:
            return
        act = self.cur_act
        k, phase, u = self.state(t)
        Y_LAB, Y_VAL, Y_BAR = 1404, 1466, 1489
        font(ctx, "dbold", 46)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        for b, xc in zip(self.balls, COL_X):
            kz = min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)
            gone_now = b["gone"] is not None and kz >= b["gone"]
            z = b["heights"][min(kz, (b["gone"] - 1) if b["gone"] is not None else kz)]
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            w = tracked_width(ctx, b["name"], tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, b["name"], xc - w / 2, Y_LAB, tr)
            if gone_now:
                s = "left the hill"
                font(ctx, "regular", 30)
                w = ctx.text_extents(s).x_advance
                ctx.set_source_rgba(*rgb(b["col"]), a)
                ctx.move_to(xc - w / 2, Y_VAL - 4)
                ctx.show_text(s)
                z = 0.0
            else:
                font(ctx, "dbold", 46)
                s = self.fmt_h(z)
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
        font(ctx, "bold", 22)
        lab = f"HEIGHT (ERROR) AFTER EACH STEP   ·   HILL {act['label']}"
        tr = 0.12 * 22
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, CX - w / 2, 1372, tr)
        s = f"STEP {min(k, N_STEPS)}"
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, T_FIRST, T_FIRST + 0.4)))
        show_tracked(ctx, s, 70, 1372, tr)

    # ------------------------------------------------------------ frame

    def _draw(self, ctx, t, fade):
        act, tl, afade = self.act_at(t)
        if act is None or afade <= 0:
            return
        self.cur_act = act
        self.balls = act["balls"]
        super()._draw(ctx, tl, fade * afade)

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
        if fade > 0 and t >= self.acts[0]["t_drop"]:
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
    for a in sc.acts:
        for b in a["balls"]:
            print(f"  hill {a['label']:6s} {b['name']:14s} gone {b['gone']}  heights {[round(h, 4) for h in b['heights'][:4]]} ... {b['heights'][-1]:.4g}  uv end ({b['uv'][-1][0]:.4f}, {b['uv'][-1][1]:.4f})")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 260, 340, 400, 470, 560, 640, 720, 800, 870, 930, 1000, 1100, 1200, 1320, 1450, 1620]
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
