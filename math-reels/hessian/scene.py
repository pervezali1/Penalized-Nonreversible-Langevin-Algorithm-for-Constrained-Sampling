"""Gradient descent vs the saddle: why second derivatives and the Hessian matter. White style of reels 23-30,
the maths on screen. The same two-valley hill as reel 30, f(u, v) = 2.2((u/2)^2 - 1)^2 + 1.8 v^2, whose pass
is a saddle. First the second derivative in 1D (curving up or down), then the Hessian: the curvature in
every direction, drawn on the hill at the pass (down along it, up across it) and at a valley bottom (up
both ways), with the two matrices and the verdict from their signs. Then three balls from the same start
(0.01 off the line through the pass): gradient descent (step 0.1), Newton's method (step = H^-1 x slope)
and saddle-free Newton (step = |H|^-1 x slope, the negative curvature flipped). Along the pass the offset
multiplies by 1.22, 0 and 2 per step: gradient descent escapes slowly, Newton walks onto the saddle and
stays, saddle-free Newton escapes fastest. 48 s. Rendered by ../valleys/hillgen.py."""

import importlib.util
import math
import os
import sys

import cairo
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

# ---------------------------------------------------------------- the hill of reel 30: two valleys, a saddle at the centre
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


def FUU(u, v):
    return K_ * (2.0 * (2.0 * u / A2_ ** 2) ** 2 + 2.0 * ((u / A2_) ** 2 - 1.0) * (2.0 / A2_ ** 2))


FVV, FUV = 2.0 * BB_, 0.0

gds.V_CUT = 0.75
N_STEPS, STEP_DT = 36, 0.5
N_FRAMES = 3000                              # 50 s
LOOP_FADE = (49.3, 49.85)
T_BALL, T_FIRST = 18.2, 21.0                 # the balls drop after the curvature story; the race starts at 21 s
for mod in (gds, lr3):
    mod.N_STEPS, mod.STEP_DT, mod.N_FRAMES, mod.LOOP_FADE = N_STEPS, STEP_DT, N_FRAMES, LOOP_FADE
    mod.T_BALL, mod.T_FIRST = T_BALL, T_FIRST
_mix = hillgen.install(gds, F, FU, FV, textkit)

FPS, T_SURF = gds.FPS, gds.T_SURF
Z_RIM, CX = gds.Z_RIM, gds.CX
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
seg, lerp = gds.seg, gds.lerp
uv_to_xy = lr3.uv_to_xy
TAU = 2 * math.pi
BLUE, AMBER, GREEN = gds.BLUE, gds.AMBER, gds.GREEN
DOWN_C, UP_C = AMBER, GREEN                  # curving down / curving up

LR = 0.1
U0, V0 = -0.01, -0.9                         # the common start: 0.01 off the line through the pass (towards the open valley
                                             # at u = -2, the bottom whose Hessian is shown), up the back wall
SADDLE_H = float(F(0.0, 0.0))
H_PASS = (FUU(0.0, 0.0), FVV)                # (-2.2, 3.6)
H_BOT = (FUU(-A2_, 0.0), FVV)                # (4.4, 3.6)
BALLS = [dict(key="gd", name="GRADIENT DESCENT", pill="GRADIENT", col=BLUE, pill_dx=0, pill_dy=0),
         dict(key="newton", name="NEWTON", pill="NEWTON", col=AMBER, pill_dx=-118, pill_dy=34),
         dict(key="sfn", name="SADDLE-FREE NEWTON", pill="SADDLE-FREE", col=GREEN, pill_dx=128, pill_dy=34)]
COL_X = (190, 540, 890)
BG = "#F4F5F9"
GRID_A = 0.05
CARD = (70, 1575, 940, 300)
CARD_C = "#FFFFFF"
T_CURV = (9.6, 17.0)                         # the curvature strokes on the hill
T_CLOSE = 39.4


def run_method(key):
    u, v = U0, V0
    pts = [(u, v)]
    for _ in range(N_STEPS):
        gu, gv = FU(u, v), FV(u, v)
        if key == "gd":
            u, v = u - LR * gu, v - LR * gv
        elif key == "newton":
            u, v = u - gu / FUU(u, v), v - gv / FVV
        else:
            u, v = u - gu / max(abs(FUU(u, v)), 1e-3), v - gv / FVV
        pts.append((u, v))
    return pts


# ---------------------------------------------------------------- a tiny maths typesetter (as in reel 34)

def mfont(ctx, size, italic=False, bold=False):
    ctx.select_font_face("Inter", cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def T(s, c=None, bold=False):
    return dict(k="t", s=s, c=c, it=False, bold=bold)


def V(s, sub=None, c=None):
    out = [dict(k="t", s=s, c=c, it=True, bold=False)]
    if sub is not None:
        out.append(dict(k="sub", e=[dict(k="t", s=sub, c=c, it=False, bold=False)]))
    return out


def FR(num, den, c=None):
    return dict(k="frac", n=num, d=den, c=c)


def SUP(elems):
    return dict(k="sup", e=elems)


def flat(*items):
    out = []
    for it in items:
        out.extend(it if isinstance(it, list) else [it])
    return out


def D2(a, b=None, c=None):
    """The second derivative of f: d2f/da2, or d2f/dadb."""
    top = [T("∂", c), SUP([T("2", c)]), dict(k="t", s="f", c=c, it=True, bold=False)]
    bot = [T("∂", c)] + V(a, c=c) + ([SUP([T("2", c)])] if b is None else [T("∂", c)] + V(b, c=c))
    return FR(top, bot, c=c)


def typeset(ctx, elems, size, col):
    ops, bars = [], []
    x, asc, desc, prev_it = 0.0, 0.0, 0.0, False
    for e in elems:
        k = e["k"]
        if k == "t":
            mfont(ctx, size, e["it"], e["bold"])
            ext = ctx.text_extents(e["s"])
            if prev_it and not e["it"]:
                x += 0.05 * size
            ops.append((x, 0.0, e["s"], size, e["it"], e["bold"], e["c"] or col))
            x += ext.x_advance
            asc = max(asc, -ext.y_bearing)
            desc = max(desc, ext.height + ext.y_bearing)
            prev_it = e["it"]
        elif k in ("sub", "sup"):
            fs = size * 0.64
            sops, sbars, w, a, d = typeset(ctx, e["e"], fs, col)
            dy = 0.22 * size if k == "sub" else -0.42 * size
            x0 = x + (0.01 * size if k == "sub" else 0.04 * size)
            ops += [(x0 + ox, dy + oy) + tuple(rest) for (ox, oy, *rest) in sops]
            bars += [(x0 + a0, x0 + a1, dy + by) + tuple(rest) for (a0, a1, by, *rest) in sbars]
            x = x0 + w + 0.03 * size
            asc, desc = max(asc, a - dy), max(desc, d + dy)
            prev_it = False
        elif k == "frac":
            fs = size * 0.80
            fc = e.get("c") or col
            nops, nbars, nw, na, nd = typeset(ctx, e["n"], fs, fc)
            dops, dbars, dw, da, dd = typeset(ctx, e["d"], fs, fc)
            pad = 0.10 * size
            wf = max(nw, dw) + 2 * pad
            x += 0.06 * size
            axis, gap = -0.29 * size, 0.17 * size
            ny, dy_ = axis - gap - nd, axis + gap + da
            nx, dx = x + (wf - nw) / 2, x + (wf - dw) / 2
            ops += [(nx + ox, ny + oy) + tuple(rest) for (ox, oy, *rest) in nops]
            ops += [(dx + ox, dy_ + oy) + tuple(rest) for (ox, oy, *rest) in dops]
            bars += [(nx + a0, nx + a1, ny + by) + tuple(rest) for (a0, a1, by, *rest) in nbars]
            bars += [(dx + a0, dx + a1, dy_ + by) + tuple(rest) for (a0, a1, by, *rest) in dbars]
            bars.append((x, x + wf, axis, max(1.6, 0.05 * size), fc))
            x += wf + 0.06 * size
            asc, desc = max(asc, -(ny - na)), max(desc, dy_ + dd)
            prev_it = False
    return ops, bars, x, asc, desc


def draw_typeset(ctx, ts, x0, base, a):
    ops, bars = ts[0], ts[1]
    for (ox, oy, s, fs, it, bold, c) in ops:
        mfont(ctx, fs, it, bold)
        ctx.set_source_rgba(*rgb(c), a)
        ctx.move_to(x0 + ox, base + oy)
        ctx.show_text(s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for (b0, b1, by, lw, c) in bars:
        ctx.set_source_rgba(*rgb(c), a)
        ctx.set_line_width(lw)
        ctx.move_to(x0 + b0, base + by)
        ctx.line_to(x0 + b1, base + by)
        ctx.stroke()


def draw_matrix(ctx, cells, x, mid, size, col, a, row_gap=10, col_gap=34, measure=False):
    """A 2x2 matrix of typeset cells, centred vertically on mid (a screen y); returns its width."""
    ts = [[typeset(ctx, c, size, col) for c in row] for row in cells]
    cw = [max(ts[0][j][2], ts[1][j][2]) for j in range(2)]
    ra = [max(ts[i][0][3], ts[i][1][3]) for i in range(2)]
    rd = [max(ts[i][0][4], ts[i][1][4]) for i in range(2)]
    total = ra[0] + rd[0] + row_gap + ra[1] + rd[1]
    pad = 16
    width = pad * 2 + cw[0] + col_gap + cw[1]
    if measure:
        return width
    top = mid - total / 2
    bases = [top + ra[0], top + ra[0] + rd[0] + row_gap + ra[1]]
    for i in range(2):
        cx = x + pad
        for j in range(2):
            t_ = ts[i][j]
            draw_typeset(ctx, t_, cx + (cw[j] - t_[2]) / 2, bases[i], a)
            cx += cw[j] + col_gap
    ctx.set_source_rgba(*rgb(col), a)
    ctx.set_line_width(2.0)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    y0, y1 = top - 6, top + total + 6
    for xb, s in ((x + 3, 1), (x + width - 3, -1)):
        ctx.move_to(xb + 9 * s, y0)
        ctx.line_to(xb, y0)
        ctx.line_to(xb, y1)
        ctx.line_to(xb + 9 * s, y1)
        ctx.stroke()
    return width


def num(v, c=None, bold=False):
    s = "0" if v == 0 else f"{v:.1f}".replace("-", "−")
    return [T(s, c, bold)]


class Scene(lr3.Scene):
    _raycast = _mix["_raycast"]
    _build_bowl = _mix["_build_bowl"]
    _build_shadow = _mix["_build_shadow"]

    def __init__(self):
        f_xy, grad_xy, surf_pt = gds.f_xy, gds.grad_xy, gds.surf_pt
        self.balls = []
        for b in BALLS:
            uv = run_method(b["key"])
            path = [uv_to_xy(*p) for p in uv]
            spt = [surf_pt(*p) for p in path]
            d = dict(b, lr=LR)
            d.update(path=path, uv=uv, spt=spt, gone=None, vis_gone=None, exit_fade=(0.35, 0.9),
                     heights=[float(f_xy(*p)) for p in path], slopes=[float(math.hypot(*grad_xy(*p))) for p in path],
                     seg_px=[math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(spt[:-1], spt[1:])],
                     step_poly=[self.surf_poly(p, q, n=24) for p, q in zip(path[:-1], path[1:])],
                     off=[abs(u) for u, _ in uv])
            hs, off = d["heights"], d["off"]
            d["k_off"] = next((k for k, o in enumerate(off) if o > 0.5), None)
            d["k_bottom"] = next((k for k, (u, v) in enumerate(uv) if abs(abs(u) - A2_) < 0.01 and abs(v) < 0.01), None)
            self.balls.append(d)
        b = self.balls[0]
        self.path, self.heights, self.slopes, self.spt, self.seg_px, self.step_poly = (
            b["path"], b["heights"], b["slopes"], b["spt"], b["seg_px"], b["step_poly"])
        S = {b["key"]: b for b in self.balls}
        self.S = S
        nw = S["newton"]
        self.facts = dict(h_pass=H_PASS, h_bottom=H_BOT, saddle_h=SADDLE_H,
                          gd_factor=1 - LR * H_PASS[0], newton_factor=0.0, sfn_factor=1 + 1.0,
                          gd_off=S["gd"]["k_off"], gd_bottom=S["gd"]["k_bottom"], sfn_off=S["sfn"]["k_off"], sfn_bottom=S["sfn"]["k_bottom"],
                          newton_u=[float(u) for u, _ in nw["uv"][:4]], newton_h_end=nw["heights"][-1], newton_slope_end=nw["slopes"][-1],
                          sfn_u=[round(float(u), 4) for u, _ in S["sfn"]["uv"][:12]], gd_u=[round(float(u), 4) for u, _ in S["gd"]["uv"][:6]],
                          end=dict((b["key"], (round(b["uv"][-1][0], 4), round(b["heights"][-1], 4))) for b in self.balls))
        assert self.facts["gd_off"] is not None and self.facts["sfn_off"] is not None and S["newton"]["k_off"] is None
        self.cues = self._cues()
        self._build_bowl()
        self._build_shadow()
        self._build_grid()
        self._curv_polys()

    # ------------------------------------------------------------ time

    @staticmethod
    def t_land(k):
        return T_FIRST + (k - 1) * STEP_DT + 0.84 * STEP_DT

    def kz_at(self, t):
        k, phase, u = self.state(t)
        return min(N_STEPS, k + 1) if (phase == "move" and u >= 0.9) else min(N_STEPS, k)

    # ------------------------------------------------------------ cues

    def _cues(self):
        F_ = self.facts
        S = self.S
        t1 = self.t_land(1)
        t_sfn_bot = self.t_land(F_["sfn_bottom"])
        t_gd_off = self.t_land(F_["gd_off"])
        hook = [("Slope zero. Valley, or saddle?", "black", 58, INK, 300, 1), ("The slope can't tell. The second derivative can.", "black", 40, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 9.3, [("The slope says which way is down.", "black", 46, INK, 300, 1), ("The second derivative says how the ground curves.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 12.9, [("On a hill, curvature is a matrix: the Hessian.", "black", 42, INK, 300, 1), ("The curvature in every direction at once.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.2, 16.9, [("At the pass: down one way, up the other. A saddle.", "black", 38, INK, 300, 1), ("At the bottom: up both ways. A minimum.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.2, t1 - 0.3, [("Three balls, one start, three rules.", "black", 46, INK, 300, 1), ("Plain slope. Newton. Newton with the curvature flipped.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (t1, t1 + 3.0, [("Newton jumps straight onto the saddle.", "black", 44, AMBER, 300, 1), ("The slope there is zero, so it stops. \"Converged.\"", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (t1 + 3.3, max(t_sfn_bot + 0.6, t1 + 7.1), [("Saddle-free Newton: the offset doubles every step.", "black", 38, GREEN, 300, 1), (f"Off the pass at step {F_['sfn_off']}, at the bottom by step {F_['sfn_bottom']}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (max(t_sfn_bot + 0.9, t1 + 7.4), t_gd_off + 0.6, [(f"Gradient descent: × {F_['gd_factor']:.2f} a step. Slow.", "black", 44, BLUE, 300, 1), (f"Off the pass at step {F_['gd_off']}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (t_gd_off + 0.9, T_CLOSE - 0.3, [(f"Newton never moves again: height {S['newton']['heights'][-1]:.1f}, slope 0.000.", "black", 38, AMBER, 300, 1), (f"Gradient descent gets to the bottom too, at step {F_['gd_bottom']}.", "dbold", 37, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 44.0, [("Same curvature, read two ways.", "black", 48, INK, 300, 1), ("Trust it: you walk into the saddle. Flip it: you escape.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (44.3, 49.0, [("So why not always use the Hessian?", "black", 46, INK, 300, 1), ("A million weights: a trillion second derivatives.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (49.4, 50.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ the card

    def card_draw(self, ctx, t, fade):
        x, y, w, h = CARD
        cards = [(3.0, 9.3, "HOW THE GROUND CURVES", self.card_1d), (9.6, 12.9, "THE HESSIAN: CURVATURE IN EVERY DIRECTION", self.card_hessian),
                 (13.2, 16.9, "READ THE SIGNS", self.card_signs), (17.2, 20.9, "THREE WAYS TO STEP", self.card_rules),
                 (21.0, T_CLOSE - 0.2, "ALONG THE PASS, ONE STEP", self.card_race), (T_CLOSE, 49.0, "WHY THE HESSIAN MATTERS", self.card_close)]
        for t0, t1, title, fn in cards:
            if t0 - 0.3 <= t < t1 + 0.3:
                a = ease_out_cubic(seg(t, t0 - 0.3, t0)) * (1 - smooth(seg(t, t1, t1 + 0.3))) * fade
                if a <= 0:
                    return
                ctx.set_source_rgba(*rgb(CARD_C), 0.9 * a)
                self._pill(ctx, x, y, w, h, 18)
                ctx.fill_preserve()
                ctx.set_source_rgba(*rgb(INK3), 0.45 * a)
                ctx.set_line_width(1.2)
                ctx.stroke()
                font(ctx, "bold", 20)
                ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
                show_tracked(ctx, title, x + 34, y + 44, 0.12 * 20)
                fn(ctx, t, a, x, y, w, h)
                return

    def mini_curve(self, ctx, cx, cy, up, a):
        """A small 1D curve: a valley (up) or a hilltop (down), with its flat tangent at the turning point."""
        ws, hs = 120, 46
        s = 1 if up else -1
        ctx.set_source_rgba(*rgb(UP_C if up else DOWN_C), a)
        ctx.set_line_width(4.0)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for i in range(41):
            u = -1 + 2 * i / 40
            px, py = cx + u * ws / 2, cy - s * (u * u - 0.5) * hs
            (ctx.move_to if i == 0 else ctx.line_to)(px, py)
        ctx.stroke()
        ty = cy + s * 0.5 * hs
        ctx.set_source_rgba(*rgb(INK3), 0.8 * a)
        ctx.set_line_width(1.6)
        ctx.set_dash([6.0, 5.0])
        ctx.move_to(cx - 44, ty)
        ctx.line_to(cx + 44, ty)
        ctx.stroke()
        ctx.set_dash([])
        ctx.set_source_rgba(*rgb(INK), a)
        ctx.arc(cx, ty, 5.0, 0, TAU)
        ctx.fill()

    def card_1d(self, ctx, t, a, x, y, w, h):
        for k, (up, cx) in enumerate(((True, x + 120), (False, x + 590))):
            ak = a * ease_out_cubic(seg(t, 3.4 + 1.2 * k, 3.9 + 1.2 * k))
            if ak <= 0:
                continue
            self.mini_curve(ctx, cx, y + 140, up, ak)
            row = flat(dict(k="t", s="f", c=None, it=True, bold=False), T("″ "), T("> 0" if up else "< 0"))
            ts = typeset(ctx, row, 34, UP_C if up else DOWN_C)
            draw_typeset(ctx, ts, cx + 90, y + 130, ak)
            font(ctx, "regular", 24)
            ctx.set_source_rgba(*rgb(INK), ak)
            ctx.move_to(cx + 90, y + 166)
            ctx.show_text("curves up: a valley" if up else "curves down: a hilltop")
        al = a * ease_out_cubic(seg(t, 6.2, 6.7))
        if al > 0:
            font(ctx, "dbold", 25)
            ctx.set_source_rgba(*rgb(INK2), al)
            ctx.move_to(x + 34, y + 252)
            ctx.show_text("At both dots the slope f′ is zero. Only f″ tells them apart.")

    def card_hessian(self, ctx, t, a, x, y, w, h):
        cells = [[[D2("u")], [D2("u", "v")]], [[D2("v", "u")], [D2("v")]]]
        ts = typeset(ctx, flat(V("H"), T("  =")), 36, INK)
        draw_typeset(ctx, ts, x + 34, y + 156, a)
        mw = draw_matrix(ctx, cells, x + 34 + ts[2] + 16, y + 144, 32, INK, a, row_gap=14)
        font(ctx, "regular", 23)
        notes = ["curvature along u", "the twist between them", "curvature along v"]
        xn = x + 34 + ts[2] + 16 + mw + 26
        for k, s in enumerate(notes):
            ctx.set_source_rgba(*rgb(INK2), a * ease_out_cubic(seg(t, 10.2 + 0.4 * k, 10.6 + 0.4 * k)))
            ctx.move_to(xn, y + 108 + 38 * k)
            ctx.show_text(s)
        al = a * ease_out_cubic(seg(t, 11.4, 11.9))
        font(ctx, "dbold", 24)
        ctx.set_source_rgba(*rgb(INK2), al)
        ctx.move_to(x + 34, y + 268)
        ctx.show_text("1D had one second derivative. A hill has a matrix of them.")

    def card_signs(self, ctx, t, a, x, y, w, h):
        rows = [(f"At the pass (0, 0)", H_PASS, "one down, one up: a SADDLE", AMBER, y + 118, 13.4),
                (f"At a bottom (−{A2_:g}, 0)", H_BOT, "both up: a MINIMUM", GREEN, y + 228, 14.6)]
        for lab, (huu, hvv), verdict, col, mid, t0 in rows:
            ak = a * ease_out_cubic(seg(t, t0, t0 + 0.45))
            if ak <= 0:
                continue
            font(ctx, "regular", 24)
            ctx.set_source_rgba(*rgb(INK2), ak)
            ctx.move_to(x + 34, mid + 8)
            ctx.show_text(lab)
            ts = typeset(ctx, flat(V("H"), T("  =")), 30, INK)
            xh = x + 262
            draw_typeset(ctx, ts, xh, mid + 10, ak)
            cells = [[num(huu, DOWN_C if huu < 0 else UP_C, True), num(0.0)], [num(0.0), num(hvv, DOWN_C if hvv < 0 else UP_C, True)]]
            mw = draw_matrix(ctx, cells, xh + ts[2] + 12, mid, 28, INK, ak, row_gap=6, col_gap=26)
            font(ctx, "dbold", 25)
            ctx.set_source_rgba(*rgb(col), ak * ease_out_cubic(seg(t, t0 + 0.5, t0 + 0.9)))
            ctx.move_to(xh + ts[2] + 12 + mw + 22, mid + 9)
            ctx.show_text("→  " + verdict)

    def card_rules(self, ctx, t, a, x, y, w, h):
        rows = [("Gradient descent", "step = −0.1 × slope", "the same small fraction of the slope", BLUE),
                ("Newton", "step = −H⁻¹ × slope", "jump to where the curvature says the slope is 0", AMBER),
                ("Saddle-free Newton", "step = −|H|⁻¹ × slope", "the same, with negative curvature flipped to +", GREEN)]
        for k, (name, rule, note, col) in enumerate(rows):
            ak = a * ease_out_cubic(seg(t, 17.4 + 0.6 * k, 17.85 + 0.6 * k))
            if ak <= 0:
                continue
            yy = y + 104 + 68 * k
            font(ctx, "dbold", 25)
            ctx.set_source_rgba(*rgb(col), ak)
            ctx.move_to(x + 34, yy)
            ctx.show_text(name)
            font(ctx, "dbold", 27)
            ctx.set_source_rgba(*rgb(INK), ak)
            ctx.move_to(x + 300, yy)
            ctx.show_text(rule)
            font(ctx, "regular", 20)
            ctx.set_source_rgba(*rgb(INK2), ak)
            ctx.move_to(x + 300, yy + 28)
            ctx.show_text(note)

    def card_race(self, ctx, t, a, x, y, w, h):
        kz = self.kz_at(t)
        c = H_PASS[0]
        rows = [("gd", BLUE, f"u − 0.1·({c:.1f}·u) = {1 - LR * c:.2f}·u".replace("-", "−")),
                ("newton", AMBER, f"u − ({c:.1f}·u) / ({c:.1f}) = 0·u".replace("-", "−")),
                ("sfn", GREEN, f"u − ({c:.1f}·u) / |{c:.1f}| = 2·u".replace("-", "−"))]
        font(ctx, "regular", 21)
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(x + 34, y + 84)
        ctx.show_text(f"Near the pass the slope along it is {c:.1f}·u, so one step sends u to".replace("-", "−"))
        for k, (key, col, rule) in enumerate(rows):
            b = self.S[key]
            u = float(b["uv"][kz][0])
            yy = y + 140 + 56 * k
            font(ctx, "dbold", 24)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x + 34, yy)
            ctx.show_text({"gd": "Gradient", "newton": "Newton", "sfn": "Saddle-free"}[key])
            font(ctx, "regular", 25)
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(x + 200, yy)
            ctx.show_text(rule)
            if abs(u) < 0.5:
                s = f"u = {abs(u):.4f}"
            elif b["k_bottom"] is not None and kz >= b["k_bottom"]:
                s = "at the bottom"
            else:
                s = f"u = {abs(u):.2f}, off the pass"
            font(ctx, "dbold", 25)
            tw = ctx.text_extents(s).x_advance
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x + w - 34 - tw, yy)
            ctx.show_text(s)

    def card_close(self, ctx, t, a, x, y, w, h):
        F_ = self.facts
        lines = [("The Hessian's signs tell a bottom from a saddle. The slope alone can't.", "regular", 24, INK, T_CLOSE + 0.2),
                 ("Newton trusts the curvature and steps onto the saddle: one step, then stuck.", "regular", 24, AMBER, T_CLOSE + 1.0),
                 (f"Flip the negative curvature (|H|): off the pass in {F_['sfn_off']} steps, not {F_['gd_off']}.", "regular", 24, GREEN, T_CLOSE + 1.8),
                 ("The catch: n weights have n² second derivatives. 10⁶ weights → 10¹².", "dbold", 24, INK, 44.5)]
        for k, (s, kind, size, col, t0) in enumerate(lines):
            ak = a * ease_out_cubic(seg(t, t0, t0 + 0.45))
            if ak <= 0:
                continue
            font(ctx, kind, size)
            assert ctx.text_extents(s).x_advance < w - 60, s
            ctx.set_source_rgba(*rgb(col), ak)
            ctx.move_to(x + 34, y + 96 + 52 * k)
            ctx.show_text(s)

    # ------------------------------------------------------------ the curvature strokes on the hill

    def _curv_polys(self):
        L = 0.75
        self.curv = [
            dict(poly=self.surf_poly(uv_to_xy(-L, 0.0), uv_to_xy(L, 0.0), n=40), col=DOWN_C, lab=f"{H_PASS[0]:+.1f}".replace("-", "−"), at=uv_to_xy(L, 0.0), t0=10.0),
            dict(poly=self.surf_poly(uv_to_xy(0.0, -0.62), uv_to_xy(0.0, 0.62), n=40), col=UP_C, lab=f"{H_PASS[1]:+.1f}", at=uv_to_xy(0.0, -0.62), t0=10.6),
            dict(poly=self.surf_poly(uv_to_xy(-A2_ - 0.55, 0.0), uv_to_xy(-A2_ + 0.55, 0.0), n=40), col=UP_C, lab=f"{H_BOT[0]:+.1f}", at=uv_to_xy(-A2_ - 0.55, 0.0), t0=14.6),
            dict(poly=self.surf_poly(uv_to_xy(-A2_, -0.55), uv_to_xy(-A2_, 0.55), n=40), col=UP_C, lab=f"{H_BOT[1]:+.1f}", at=uv_to_xy(-A2_, -0.55), t0=14.9),
        ]

    def draw_curvature(self, ctx, t, fade):
        a_all = (1 - smooth(seg(t, T_CURV[1] - 0.4, T_CURV[1] + 0.2))) * fade
        if a_all <= 0 or t < T_CURV[0]:
            return
        for c in self.curv:
            k = ease_out_cubic(seg(t, c["t0"], c["t0"] + 0.8))
            if k <= 0:
                continue
            pts = c["poly"]
            m = max(2, int(len(pts) * k))
            for lw, al, col in ((11.0, 0.9, "#FFFFFF"), (6.0, 1.0, c["col"])):
                ctx.set_source_rgba(*rgb(col), al * a_all)
                ctx.set_line_width(lw)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self._path(ctx, pts[:m])
                ctx.stroke()
            if k > 0.9:
                px, py = gds.surf_pt(*c["at"])
                s = c["lab"]
                font(ctx, "dbold", 26)
                tw = ctx.text_extents(s).x_advance
                la = a_all * seg(k, 0.9, 1.0)
                ctx.set_source_rgba(1, 1, 1, 0.94 * la)
                self._pill(ctx, px - tw / 2 - 12, py - 46, tw + 24, 36, 18)
                ctx.fill_preserve()
                ctx.set_source_rgba(*rgb(c["col"]), 0.9 * la)
                ctx.set_line_width(1.8)
                ctx.stroke()
                ctx.set_source_rgba(*rgb(c["col"]), la)
                ctx.move_to(px - tw / 2, py - 19)
                ctx.show_text(s)

    # ------------------------------------------------------------ the bottom the balls are heading for

    def draw_bottom_marker(self, ctx, t, fade):
        a = ease_out_cubic(seg(t, T_BALL, T_BALL + 0.6)) * fade
        if a <= 0:
            return
        p = uv_to_xy(-A2_, 0.0)
        ctx.save()
        self._path(ctx, self.surf_clip, close=True)
        ctx.clip()
        self.ghost(ctx, p, 0.34, 0.9 * a)
        ctx.restore()
        px, py = gds.surf_pt(*p)
        s = "THE BOTTOM  ·  HEIGHT 0"
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        tw = tracked_width(ctx, s, tr)
        bx, by = px - tw / 2, py + 74
        ctx.set_source_rgba(1, 1, 1, 0.92 * a)
        self._pill(ctx, bx - 12, by - 22, tw + 24, 32, 16)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(INK2), 0.6 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(INK), a)
        show_tracked(ctx, s, bx, by, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.7 * a)
        ctx.set_line_width(1.4)
        ctx.move_to(px, py + 14)
        ctx.line_to(px, by - 22)
        ctx.stroke()

    # ------------------------------------------------------------ the balls

    def draw_ball_top(self, ctx, b, t, fade, a_ball, drop, cxy, bxy, rb_px, squash):
        if a_ball <= 0.002:
            return
        cx, cy = cxy
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=bxy, squash=squash)
        ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a_ball)
        ctx.set_line_width(3.0)
        ctx.arc(cx, cy - 80 * drop, rb_px + (7.0 if b["key"] == "sfn" else 2.5), 0, TAU)   # concentric when two share a spot
        ctx.stroke()
        s = b["pill"]
        font(ctx, "dbold", 21)
        tw = ctx.text_extents(s).x_advance
        px, py = cx + b["pill_dx"], cy - 80 * drop - rb_px - 16 - b["pill_dy"]
        ctx.set_source_rgba(1, 1, 1, 0.92 * a_ball)
        self._pill(ctx, px - tw / 2 - 10, py - 19, tw + 20, 28, 14)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(b["col"]), 0.9 * a_ball)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(b["col"]), a_ball)
        ctx.move_to(px - tw / 2, py + 1)
        ctx.show_text(s)

    def readouts3(self, ctx, t, fade):
        a = fade * ease_out_cubic(seg(t, T_BALL + 0.2, T_BALL + 0.7))
        if a <= 0:
            return
        k, phase, u = self.state(t)
        kz = self.kz_at(t)
        Y_LAB, Y_VAL, Y_SUB = 1404, 1458, 1494
        font(ctx, "dbold", 40)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        for b, xc in zip(self.balls, COL_X):
            font(ctx, "bold", 20)
            tr = 0.12 * 20
            lab = b["name"]
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(b["col"]), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            z = max(0.0, b["heights"][kz])
            val = gds._fmt(z, 2 if z >= 1.0 else 3)
            font(ctx, "dbold", 40)
            w = cells_width(ctx, val, widths)
            ctx.set_source_rgba(*rgb(INK), a)
            show_cells(ctx, val, xc - w / 2, Y_VAL, widths)
            off = b["off"][kz]
            s = f"height  ·  off the pass {off:.4f}" if off < 1 else f"height  ·  off the pass {off:.3f}"
            font(ctx, "regular", 19)
            w = ctx.text_extents(s).x_advance
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(xc - w / 2, Y_SUB)
            ctx.show_text(s)
        font(ctx, "bold", 22)
        tr = 0.12 * 22
        s = f"STEP {min(k, N_STEPS)}"
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a * ease_out_cubic(seg(t, T_FIRST, T_FIRST + 0.4)))
        show_tracked(ctx, s, 70, 1372, tr)
        lab = "SAME START: 0.01 OFF THE LINE THROUGH THE PASS"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 1010 - w, 1372, tr)

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
        if fade > 0:
            surf, ctx = layer()
            self.draw_curvature(ctx, t, fade)
            self.draw_bottom_marker(ctx, t, fade)
            if t >= T_BALL:
                self._draw(ctx, t, fade)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
            surf3, ctx3 = layer()
            self.card_draw(ctx3, t, fade)
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
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 330, 480, 620, 760, 840, 960, 1080, 1180, 1300, 1400, 1560, 1720, 1900, 2150, 2400, 2700, 2990]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (480, 700, 960, 1180, 1300, 1560, 1900, 2400, 2700):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
