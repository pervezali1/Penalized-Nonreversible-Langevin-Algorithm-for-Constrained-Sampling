"""The wrong valley, shaken: what batch size does to a ball stuck in the shallow valley of the two-valley
hill (reel 28). The hill is the average error over 1,000 examples; each example's own error is the hill
plus a tilt (tiltsim.py), so a mini-batch's slope is the true slope plus real sampling noise that shrinks
like 1/sqrt(batch). Ball C looks at all 1,000 examples: its slope at the bottom is exactly zero and it
never moves. Three groups of eight copies measure the slope on 4, 8 or 32 examples a step, from the
same spot, with their own random batches; a live count says how many are over the ridge. Rendered by
../valleys/hillgen.py in the reel-23 look, on the faint tinted grid. frame(f) -> uint8 1080x1920 RGB."""

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
sys.path.insert(0, os.path.join(ROOT, "valleys"))
import textkit  # noqa: E402
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked,  # noqa: E402
                     smooth, surf_rgba, tracked_width)
import tiltsim as TS  # noqa: E402

_spec = importlib.util.spec_from_file_location("lr3_scene", os.path.join(ROOT, "lr3", "scene.py"))
lr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr3)
gds = lr3.gds
import hillgen  # noqa: E402

# ---------------------------------------------------------------- the hill of reel 28: double well along u, bowl across v
K_, A2_, T_, BB_ = 2.2, 2.0, 0.6, 1.8


def _well(u):
    return K_ * ((u / A2_) ** 2 - 1) ** 2 + T_ * u


_us = np.linspace(-4, 4, 800001)
_w = _well(_us)
_i0 = int(np.argmax(np.where(np.abs(_us) < 1.5, _w, -1e9)))
U_RIDGE = float(_us[_i0])
U_DEEP = float(_us[int(np.argmin(np.where(_us < U_RIDGE, _w, 1e9)))])
U_SHALLOW = float(_us[int(np.argmin(np.where(_us > U_RIDGE, _w, 1e9)))])
C0_ = float(-min(_well(U_DEEP), _well(U_SHALLOW)))


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


gds.V_CUT = 0.9
_mix = hillgen.install(gds, F, FU, FV, textkit)

FPS, N_FRAMES, T_SURF, LOOP_FADE = gds.FPS, gds.N_FRAMES, gds.T_SURF, gds.LOOP_FADE
Z_RIM, CX = gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
seg, lerp = gds.seg, gds.lerp
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi

LR = 0.1
T_BALL = 6.0                         # the balls appear at the shallow bottom
T_SHAKE = 9.6                        # the copies start measuring their batches
N_SHAKE, N_SETTLE, DT2 = 140, 12, 0.1
T_OFF = T_SHAKE + N_SHAKE * DT2
T_STILL = T_OFF + N_SETTLE * DT2
N_COPIES = 8
N_POP = 500
VERM, BLUE, GREEN, MAGENTA = gds.AMBER, gds.BLUE, gds.GREEN, gds.MAGENTA
GROUPS = [
    dict(key="b4", batch=4, name="4 A STEP", col=MAGENTA, seed0=100),
    dict(key="b8", batch=8, name="8 A STEP", col=BLUE, seed0=200),
    dict(key="b32", batch=32, name="32 A STEP", col=GREEN, seed0=300),
]
FULL = dict(key="full", batch=TS.N, name="ALL 1,000", pill="1,000", col=VERM)
COL_X = (165, 415, 665, 915)
BG = "#F4F5F9"
GRID_A = 0.05
START = (U_SHALLOW, 0.0)             # the bottom of the shallow valley: slope exactly zero


def run_copy(tau, batch, seed):
    """From the shallow bottom: N_SHAKE steps on the batch's slope (true slope + mean tilt of a fresh random
    batch), then N_SETTLE steps on the true slope."""
    rng = np.random.default_rng(seed)
    u, v = START
    pts = [(u, v)]
    for _ in range(N_SHAKE):
        n = TS.batch_noise(tau, rng, batch)
        u, v = u - LR * (FU(u, v) + n[0]), v - LR * (FV(u, v) + n[1])
        pts.append((u, v))
    for _ in range(N_SETTLE):
        u, v = u - LR * FU(u, v), v - LR * FV(u, v)
        pts.append((u, v))
    return pts


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, surf_pt = gds.f_xy, gds.surf_pt
        self.tau = TS.make_tilts()
        self.groups = []
        for g in GROUPS:
            copies, skipped, i = [], 0, 0
            while len(copies) < N_COPIES:
                # the eight shown are the first runs (by seed) that never go above the rim of the drawn hill
                uv = run_copy(self.tau, g["batch"], g["seed0"] + i)
                i += 1
                heights = [float(f_xy(*uv_to_xy(*p))) for p in uv]
                if max(heights) > Z_RIM:
                    skipped += 1
                    continue
                path = [uv_to_xy(*p) for p in uv]
                spt = [surf_pt(*p) for p in path]
                over = next((k for k, (u, v) in enumerate(uv) if u < U_RIDGE), None)
                copies.append(dict(uv=uv, path=path, spt=spt, heights=heights, over=over, escaped=uv[-1][0] < U_RIDGE, seed=g["seed0"] + i - 1))
            kick = LR * TS.TILT / math.sqrt(g["batch"]) * math.sqrt((TS.N - g["batch"]) / (TS.N - 1))
            # population: of N_POP random copies with this batch size, the fraction that end in the deep valley
            n_esc = sum(run_copy(self.tau, g["batch"], 10_000 + s)[-1][0] < U_RIDGE for s in range(N_POP))
            self.groups.append(dict(g, copies=copies, kick=kick, pop=n_esc / N_POP, skipped=skipped))
        self.full = dict(FULL)
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()
        self._build_grid()

    # ------------------------------------------------------------ time

    @staticmethod
    def copy_state(t):
        if t < T_SHAKE:
            return 0, 0.0
        x = (t - T_SHAKE) / DT2
        j = int(math.floor(x))
        if j >= N_SHAKE + N_SETTLE:
            return N_SHAKE + N_SETTLE, 0.0
        return j, smooth(min(1.0, (x - j) / 0.8))

    @staticmethod
    def n_over(g, j):
        return sum(1 for c in g["copies"] if c["uv"][j][0] < U_RIDGE)

    # ------------------------------------------------------------ cues (every number from the runs)

    def _cues(self):
        firsts = [(c["over"], g["batch"]) for g in self.groups for c in g["copies"] if c["over"] is not None]
        k_first, b_first = min(firsts) if firsts else (None, None)
        n_off = [self.n_over(g, N_SHAKE) for g in self.groups]
        n_end = [sum(1 for c in g["copies"] if c["escaped"]) for g in self.groups]
        barrier = float(F(U_RIDGE, 0.0) - F(U_SHALLOW, 0.0))
        self.facts = dict(ridge=float(F(U_RIDGE, 0.0)), shallow=float(F(U_SHALLOW, 0.0)), barrier=barrier,
                          k_first=k_first, b_first=b_first, n_off=n_off, n_end=n_end,
                          kicks=[g["kick"] for g in self.groups], pop=[g["pop"] for g in self.groups],
                          firsts={g["batch"]: sorted(c["over"] for c in g["copies"] if c["over"] is not None) for g in self.groups},
                          skipped=[g["skipped"] for g in self.groups], seeds={g["batch"]: [c["seed"] for c in g["copies"]] for g in self.groups})
        t_first = T_SHAKE + k_first * DT2 if k_first else T_SHAKE + 4
        kicks = ", ".join(f"{g['kick']:.2f}" for g in self.groups)
        hook = [("The wrong valley, shaken.", "black", 58, INK, 300, 1), ("What batch size does to a ball stuck in the shallow valley.", "black", 36, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 6.0, [(f"This hill is the average error over {TS.N:,} examples.", "black", 40, INK, 300, 1), ("Each example's own error is the hill plus a tilt.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.4, 9.4, [("A ball at the bottom of the shallow valley. Slope: zero.", "black", 38, INK, 300, 1), ("Plain descent, all 1,000 examples, sits here forever.", "dbold", 36, VERM, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.2, [("Now 24 copies measure the slope on a batch: 4, 8 or 32 examples.", "black", 34, INK, 300, 1), ("Smaller batch, noisier slope, bigger kicks.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.5, 17.0, [(f"Kicks per step: {kicks}.  The ridge is {barrier:.1f} high.", "black", 36, INK, 300, 1), (f"The first copy over was a {b_first}-example one, at step {k_first}.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.3, T_OFF - 0.2, [("A noisy step can go uphill. An exact one never can.", "black", 40, INK, 300, 1), ("Watch the counts.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_OFF + 0.1, T_STILL + 1.2, [(f"Shaking off: {n_off[0]}, {n_off[1]} and {n_off[2]} of 8 are over the ridge.", "black", 38, INK, 300, 1), ("They settle at 0. The rest, and ball C, stay at 2.39.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_STILL + 1.5, 29.0, [("Small batches are noisy. Sometimes that is the point.", "black", 38, INK, 300, 1), (f"Over {N_POP} runs each: {self.groups[0]['pop']:.0%}, {self.groups[1]['pop']:.0%} and {self.groups[2]['pop']:.0%} climb out in {N_SHAKE} steps.", "dbold", 33, INK2, 390, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ drawing

    def _draw(self, ctx, t, fade):
        if t < T_BALL:
            return
        a_in = ease_out_cubic(seg(t, T_BALL, T_BALL + 0.5)) * fade
        drop = (1 - seg(t, T_BALL, T_BALL + 0.5)) ** 2
        j, frac = self.copy_state(t)
        ink = rgb(INK)
        # trails, clipped to the drawn surface
        ctx.save()
        self._path(ctx, self.surf_clip, close=True)
        ctx.clip()
        if t >= T_SHAKE:
            for g in self.groups:
                col = rgb(g["col"])
                for c in g["copies"]:
                    j0 = max(0, j - 12)
                    pts = c["spt"][j0:j + 1]
                    for i in range(1, len(pts)):
                        age = (j - (j0 + i)) / 12.0
                        ctx.set_source_rgba(*col, 0.5 * (1 - age) * a_in)
                        ctx.set_line_width(1.5)
                        ctx.move_to(*pts[i - 1])
                        ctx.line_to(*pts[i])
                        ctx.stroke()
                    for i, p in enumerate(pts):
                        age = (j - (j0 + i)) / 12.0
                        ctx.set_source_rgba(*ink, 0.45 * (1 - age) * a_in)
                        ctx.arc(p[0], p[1], 2.0, 0, TAU)
                        ctx.fill()
        # ball C's contact shadow
        x, y = uv_to_xy(*START)
        P, n, rb_px, rb_w, (cx, cy), (bx, by) = self.ball_geom(x, y)
        self.contact_shadow(ctx, P, n, rb_w, a_in * (0.35 + 0.65 * (1 - drop)), scale=0.6 + 0.4 * (1 - drop))
        ctx.restore()
        # the copies: small ringed spheres, drawn farther-first; the group with the biggest kicks on top
        items = []
        if t >= T_SHAKE - 0.4:
            a_c = ease_out_cubic(seg(t, T_SHAKE - 0.4, T_SHAKE)) * fade
            for gi, g in enumerate(self.groups):
                for c in g["copies"]:
                    (x0, y0), (x1, y1) = c["path"][j], c["path"][min(j + 1, N_SHAKE + N_SETTLE)]
                    xx, yy = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
                    Pc, nc, rpx, rw, _, _ = self.ball_geom(xx, yy)
                    cxs, cys, _ = gds.project_w(Pc + nc * rw * 0.55)
                    items.append((rpx, gi, cxs, cys, rpx * 0.55, g["col"], a_c))
            items.sort(key=lambda it: (it[0], it[1]))
            for rpx, gi, cxs, cys, r, col, a_c in items:
                self.sphere(ctx, cxs, cys, r, a_c, contact=None, squash=0.0)
                ctx.set_source_rgba(*rgb(col), 0.9 * a_c)
                ctx.set_line_width(1.6)
                ctx.arc(cxs, cys, r + 2.0, 0, TAU)
                ctx.stroke()
        # ball C, full batch, with its pill
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_in, contact=(bx, by), squash=0.0)
        s = self.full["pill"]
        font(ctx, "dbold", 24)
        tw = ctx.text_extents(s).x_advance
        px, py = cx, cy - 80 * drop - rb_px - 16
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_in)
        self._pill(ctx, px - tw / 2 - 10, py - 20, tw + 20, 30, 15)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(VERM), 0.9 * a_in)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(VERM), a_in)
        ctx.move_to(px - tw / 2, py + 2)
        ctx.show_text(s)
        self.readouts3(ctx, t, fade)

    # ------------------------------------------------------------ readouts: four columns of counts

    def readouts3(self, ctx, t, fade):
        a = fade * ease_out_cubic(seg(t, 6.4, 6.9))
        if a <= 0:
            return
        j, _ = self.copy_state(t)
        Y_LAB, Y_VAL, Y_BAR, Y_WORD = 1404, 1466, 1489, 1522
        tr = 0.12 * 22
        cols = [(g["name"], g["col"], self.n_over(g, j), N_COPIES) for g in self.groups] + [(self.full["name"], VERM, 0, 1)]
        for (name, col, n, tot), xc in zip(cols, COL_X):
            font(ctx, "bold", 22)
            w = tracked_width(ctx, name, tr)
            ctx.set_source_rgba(*rgb(col), 0.95 * a)
            show_tracked(ctx, name, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            s = f"{n} / {tot}" if tot > 1 else "still"
            if tot == 1:
                font(ctx, "regular", 30)
            w = ctx.text_extents(s).x_advance
            ctx.set_source_rgba(*rgb(INK if tot > 1 else col), a)
            ctx.move_to(xc - w / 2, Y_VAL - (4 if tot == 1 else 0))
            ctx.show_text(s)
            ctx.set_line_width(6)
            ctx.set_source_rgba(*rgb("#E5E7EB"), a)
            ctx.move_to(xc - 100, Y_BAR)
            ctx.line_to(xc + 100, Y_BAR)
            ctx.stroke()
            L = 200 * n / tot
            if L > 0.5:
                ctx.set_source_rgba(*rgb(col), a)
                ctx.move_to(xc - 100, Y_BAR)
                ctx.line_to(xc - 100 + L, Y_BAR)
                ctx.stroke()
            a_w = a * ease_out_cubic(seg(t, T_STILL + 1.5, T_STILL + 2.1))
            if a_w > 0:
                font(ctx, "regular", 22)
                s = "over the ridge" if tot > 1 else "never moved"
                ctx.set_source_rgba(*rgb(INK3), a_w)
                ctx.move_to(xc - ctx.text_extents(s).x_advance / 2, Y_WORD)
                ctx.show_text(s)
        font(ctx, "bold", 22)
        lab = f"OVER THE RIDGE   \u00b7   EXAMPLES PER STEP   \u00b7   STEP SIZE {LR:g}"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, max(300, CX - w / 2 + 60), 1372, tr)
        s = f"SHAKE {min(j, N_SHAKE)}" + (f" + {j - N_SHAKE}" if j > N_SHAKE else "")
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, T_SHAKE, T_SHAKE + 0.4)))
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
    for g in sc.groups:
        print(f"  batch {g['batch']:4d}: kick {g['kick']:.3f}  over at off {sc.n_over(g, N_SHAKE)}/8  escaped {sum(c['escaped'] for c in g['copies'])}/8  firsts {sorted(c['over'] for c in g['copies'] if c['over'] is not None)}  pop {g['pop']:.1%}  max v {max(v for c in g['copies'] for _, v in c['uv']):.2f}  max h {max(max(c['heights']) for c in g['copies']):.2f}")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 160, 300, 400, 560, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1450, 1520, 1620, 1740]
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
