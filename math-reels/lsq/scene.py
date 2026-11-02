"""The line of best fit is a shadow: least squares as a projection. The 3D stage of reels 38-43 (stage3d.py),
narrated by Kokoro (voicekit.py).

Three points, (-1, 1), (0, 3), (1, 2), and a straight line y = a + b x. Their three heights are one arrow in 3D,
y = (1, 3, 2). Every line gives three heights too, a (1, 1, 1) + b (-1, 0, 1): together the lines form a plane through
the origin. A line is a point on that plane; how badly it misses the data is the distance from that point to the
tip of y (squared: the sum of the squared misses). The best line is the foot of the perpendicular from y, its
shadow on the plane; the leftover miss is at a right angle to the plane, X^T (y - X b) = 0, the normal equations.
A 2D panel shows the same line through the three points, with the misses. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import CYAN, GOLD, GREEN, INK, INK2, INK3, MAGENTA, TAU, VERM, paste  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg

FPS = 60

# ---------------------------------------------------------------- the data and every number the reel states
XS = np.array([-1.0, 0.0, 1.0])
Y = np.array([1.0, 3.0, 2.0])
X = np.column_stack([np.ones(3), XS])
XTX = X.T @ X
XTY = X.T @ Y
BETA = np.linalg.solve(XTX, XTY)                  # (a, b) = (2, 0.5)
YHAT = X @ BETA                                   # (1.5, 2, 2.5): the shadow
RES = Y - YHAT                                    # (-0.5, 1, -0.5)
SSE = float(RES @ RES)                            # 1.5
assert np.allclose(X.T @ RES, 0) and np.allclose(BETA, [2.0, 0.5])
# the lines we try before the best one, (a, b)
TRIES = [(2.0, 0.0), (1.2, 1.0), (2.6, -0.5), (1.6, 0.9)]


def sse(a, b):
    r = Y - (a + b * XS)
    return float(r @ r)


SCALE = 0.56
TARGET = SCALE * np.array([0.9, 1.4, 1.3])
CARD = (70, 1430, 940, 250)
PANEL = (110, 1158, 400, 236)                     # the 2D plot: x0, y0, w, h
P_X, P_Y = (-1.5, 1.5), (0.0, 4.0)

SCRIPT = [
    "The line of best fit is just a shadow. Here's what I mean.",
    "Three points. We want the straight line, a plus b x, that misses them the least.",
    "Stack the three heights into one arrow in 3D. Y equals one, three, two.",
    "Any line gives three heights too: a times one, one, one, plus b times minus one, zero, one. All the lines together make a plane.",
    "Pick a line, and it's a point on that plane. How much it misses is the distance from that point to the tip of y.",
    "The best line is the point right under y: its shadow. The leftover miss stands at a right angle to the plane.",
    "A right angle means X transpose times the miss is zero. Solve those two equations: a equals two, and b equals one half.",
    "Least squares is a projection. The line of best fit is the shadow of the data.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# ---------------------------------------------------------------- the timeline, sized to the narration
# 0 hook, 1 three points, 2 y as an arrow, 3 the plane of lines, 4 tries, 5 the shadow, 6 normal equations, 7 close
SEG_MIN = [3.5, 6.0, 6.0, 8.5, 10.0, 7.5, 8.5, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[8]
N_FRAMES = int(round((T_END + 0.7) * FPS))
HOOK_OUT = (SEG[1], SEG[1] + 0.7)                 # the opening picture clears, then is built again
TRY = (SEG[4] + 1.0, SEG[5] - 0.2)                # the tried lines
SETTLE = (SEG[5] + 0.3, SEG[5] + 2.6)             # the glide to the shadow
SWING_AZ = 20.0


def ab_at(t):
    """The line (a, b) being shown: the best one in the opening, the tries, then the best one again."""
    if t < TRY[0]:
        return BETA.copy()
    keys = [np.array(k) for k in TRIES]
    if t < TRY[1]:
        u = (t - TRY[0]) / (TRY[1] - TRY[0]) * (len(keys) - 0.4)
        i = min(int(u), len(keys) - 1)
        f = smooth(min(1.0, (u - i) / 0.55))
        k0 = keys[i - 1] if i > 0 else keys[0]
        return k0 + (keys[i] - k0) * f
    k = smooth(seg(t, *SETTLE))
    return keys[-1] + (BETA - keys[-1]) * k


def vis(t, t_in, dur=0.9):
    """Shown in the opening picture, cleared at HOOK_OUT, built again from t_in; stays to the end (the loop)."""
    hook = 1 - smooth(seg(t, *HOOK_OUT))
    return max(hook, ease_out_cubic(seg(t, t_in, t_in + dur)))


def qa_frames():
    fr = []
    for k in range(8):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(points=np.column_stack([XS, Y]).tolist(), XtX=XTX.tolist(), Xty=XTY.tolist(), a=float(BETA[0]), b=float(BETA[1]), yhat=YHAT.tolist(),
                          residual=RES.tolist(), sse=SSE, tries=[[a, b, sse(a, b)] for a, b in TRIES], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG

    def _cues(self):
        hook = [("The line of best fit is a shadow.", "black", 48, INK, 300, 1), ("Least squares, in 3D.", "dbold", 40, INK2, 390, 0.95)]
        S_ = SEG
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Three points. One straight line.", "black", 48, INK, 300, 1), ("y = a + b·x, missing them the least.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Three heights: one arrow.", "black", 50, INK, 300, 1), ("y = (1, 3, 2)", "dbold", 40, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Every line is a point on a plane.", "black", 46, INK, 300, 1), ("a·(1, 1, 1) + b·(−1, 0, 1)", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Its miss: the distance to y.", "black", 50, INK, 300, 1), ("error² = the sum of the squared misses", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The best line: the shadow of y.", "black", 48, INK, 300, 1), ("The miss meets the plane at 90°.", "dbold", 38, MAGENTA, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("90° gives two equations.", "black", 50, INK, 300, 1), (f"a = {BETA[0]:g},  b = {BETA[1]:g}", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("Least squares is a projection.", "black", 48, INK, 300, 1), ("The best fit is the data's shadow.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END - 0.3, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ camera

    def camera(self, t):
        L = N_FRAMES / FPS
        w = smooth(seg(t, SEG[5] + 0.2, SEG[5] + 2.2)) * (1 - smooth(seg(t, SEG[7], SEG[7] + 2.5)))
        az = -46 + 8 * math.sin(TAU * t / L) + SWING_AZ * w
        el = 24 + 3 * math.sin(2 * TAU * t / L) - 8 * w
        tgt = TARGET + (SCALE * YHAT - TARGET) * 0.55 * w            # push in towards the shadow during the swing
        return S3.Camera(tgt, az, el, dist=10.8, focal=2100.0 + 900.0 * w, centre=(540, 800))

    def W(self, u):
        return np.atleast_2d(u) * SCALE

    # ------------------------------------------------------------ the 3D stage

    def draw_world(self, img, t, a):
        cam = self.camera(t)
        A, B = ab_at(t)
        p = A * X[:, 0] + B * X[:, 1]
        S3.floor_light(img, cam, -0.9, a, radius=3.4)
        surf, ctx = layer()
        ctx.rectangle(0, 430, 1080, 712)                 # the 3D stage stays above the 2D panel
        ctx.clip()
        o = cam.project(self.W(np.zeros(3)))[0]
        # the three axes: the heights of point 1, 2 and 3
        ax_a = vis(t, SEG[2] + 0.1) * a
        self.axis_labels = []
        ctx.set_dash([5.0, 7.0])
        for i in range(3):
            e = np.eye(3)[i]
            pa, pb = cam.project(self.W(np.array([-0.4 * e, 3.6 * e])))
            ctx.set_source_rgba(*rgb(INK3), 0.85 * ax_a)
            ctx.set_line_width(1.3)
            ctx.move_to(*pa)
            ctx.line_to(*pb)
            ctx.stroke()
            self.axis_labels.append((i, cam.project(self.W(3.9 * e))[0], ax_a))
        ctx.set_dash([])
        # the plane of all lines: a (1,1,1) + b (-1,0,1)
        pl = vis(t, SEG[3] + 4.2, 1.2) * a
        if pl > 0.01:
            a0, a1, b0, b1 = -0.2, 3.0, -0.8, 1.3

            def P(aa, bb):
                return cam.project(self.W(aa * X[:, 0] + bb * X[:, 1]))[0]
            cs = [P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)]
            ctx.move_to(*cs[0])
            for c in cs[1:]:
                ctx.line_to(*c)
            ctx.close_path()
            ctx.set_source_rgba(*rgb(CYAN), 0.08 * pl)
            ctx.fill_preserve()
            ctx.set_source_rgba(*rgb(CYAN), 0.40 * pl)
            ctx.set_line_width(1.3)
            ctx.stroke()
            ctx.set_line_width(0.9)
            ctx.set_source_rgba(*rgb(CYAN), 0.16 * pl)
            for aa in np.arange(0.0, a1, 0.5):
                ctx.move_to(*P(aa, b0))
                ctx.line_to(*P(aa, b1))
            for bb in np.arange(-0.5, b1, 0.5):
                ctx.move_to(*P(a0, bb))
                ctx.line_to(*P(a1, bb))
            ctx.stroke()
        # the two arrows that span it, and y
        self.vec_labels = []
        one_a = vis(t, SEG[3] + 1.0) * a
        x_a = vis(t, SEG[3] + 2.6) * a
        y_a = vis(t, SEG[2] + 0.9) * a
        for v, col, al, lab in ((X[:, 0], GOLD, one_a, r"$(1,\,1,\,1)$"), (X[:, 1], GREEN, x_a, r"$x = (-1,\,0,\,1)$"), (Y, VERM, y_a, r"$y = (1,\,3,\,2)$")):
            if al > 0.01:
                S3.arrow3(ctx, cam, self.W(np.zeros(3))[0], self.W(v)[0], col, al, width=4.6, head=24)
                self.vec_labels.append((lab, cam.project(self.W(v))[0], o, col, al))
        # the line being tried: a point on the plane, its miss to the tip of y
        cand_a = max(1 - smooth(seg(t, *HOOK_OUT)), ease_out_cubic(seg(t, TRY[0] - 0.6, TRY[0]))) * a
        self.cand = None
        if cand_a > 0.01:
            pp, yy = cam.project(self.W(np.array([p, Y])))
            best = 1 - min(1.0, math.hypot(A - BETA[0], B - BETA[1]) / 0.05)
            ctx.set_dash([] if best > 0.5 else [7.0, 6.0])
            ctx.set_source_rgba(*rgb(MAGENTA), 0.95 * cand_a)
            ctx.set_line_width(3.0)
            ctx.move_to(*pp)
            ctx.line_to(*yy)
            ctx.stroke()
            ctx.set_dash([])
            # the right angle at the shadow
            if best > 0.01:
                r = (Y - p) / np.linalg.norm(Y - p)
                # the in-plane direction that shows the corner best: the plane's own version of screen-right
                # the in-plane direction whose picture is most nearly at a right angle to the miss's picture
                e1 = X[:, 0] / np.linalg.norm(X[:, 0])
                e2 = X[:, 1] / np.linalg.norm(X[:, 1])
                o2, r2 = cam.project(self.W(np.array([p, p + r])))
                rs = (r2 - o2) / (np.linalg.norm(r2 - o2) + 1e-9)
                best_d, best_s = e1, -1.0
                for th in np.linspace(0, math.pi, 37):
                    u = math.cos(th) * e1 + math.sin(th) * e2
                    for sg in (1, -1):
                        q2 = cam.project(self.W(p + sg * u))[0]
                        us = (q2 - o2) / (np.linalg.norm(q2 - o2) + 1e-9)
                        sc = abs(us[0] * rs[1] - us[1] * rs[0]) - 0.001 * th
                        if sc > best_s:
                            best_s, best_d = sc, sg * u
                d = best_d
                eps = 0.34
                q = [p + eps * r, p + eps * r + eps * d, p + eps * d]
                qq = cam.project(self.W(np.array(q)))
                ctx.set_source_rgba(*rgb(INK), 0.9 * best * cand_a)
                ctx.set_line_width(1.8)
                ctx.move_to(*qq[0])
                ctx.line_to(*qq[1])
                ctx.line_to(*qq[2])
                ctx.stroke()
            ctx.set_source_rgba(1, 1, 1, cand_a)
            ctx.arc(*pp, 7.0, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(MAGENTA), cand_a)
            ctx.set_line_width(2.0)
            ctx.arc(*pp, 7.0, 0, TAU)
            ctx.stroke()
            self.cand = (pp, yy, math.sqrt(sse(A, B)), best, cand_a)
        ctx.set_source_rgba(1, 1, 1, 0.9 * a)
        ctx.arc(*o, 4.0, 0, TAU)
        ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return A, B

    def labels(self, img, t, a):
        for i, pnt, al in self.axis_labels:
            paste(img, rf"$\mathrm{{point}}\ {i + 1}$", 22, min(max(float(pnt[0]), 80.0), 1000.0), min(float(pnt[1]), 1110.0) + 8, INK3, al, align="c", halo=4)
        placed = []
        for lab, tip, o, col, al in self.vec_labels:
            d = np.asarray(tip, float) - np.asarray(o, float)
            n = float(np.hypot(*d))
            d = d / n if n > 1e-6 else np.array([0.0, -1.0])
            m, bl, wd = S3.tex(lab, 24)
            hh = m.shape[0]
            cx, cy = np.asarray(tip, float) + d * (14 + abs(d[0]) * wd / 2 + abs(d[1]) * hh / 2)
            cx = min(max(cx, 24 + wd / 2), 1056 - wd / 2)
            for _ in range(20):
                box = (cx - wd / 2 - 4, cy - hh / 2 - 3, cx + wd / 2 + 4, cy + hh / 2 + 3)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                    break
                cx, cy = cx + 5 * d[0], cy + 5 * d[1]
            placed.append((cx - wd / 2 - 4, cy - hh / 2 - 3, cx + wd / 2 + 4, cy + hh / 2 + 3))
            paste(img, lab, 24, cx, cy - hh / 2 + bl, col, al, align="c", halo=4)
        if self.cand is not None:
            pp, yy, miss, best, al = self.cand
            mid = (np.asarray(pp, float) + np.asarray(yy, float)) / 2
            lab = rf"$\mathrm{{miss}} = {miss:.2f}$"
            m, bl, wd = S3.tex(lab, 24)
            hh = m.shape[0]
            seg_d = np.asarray(yy, float) - np.asarray(pp, float)
            nd = np.array([-seg_d[1], seg_d[0]]) / (np.hypot(*seg_d) + 1e-9)
            if nd[1] < 0:
                nd = -nd                                    # to the lower side of the miss
            cx, cy = mid + nd * (16 + abs(nd[0]) * wd / 2 + abs(nd[1]) * hh / 2)
            for _ in range(30):
                box = (cx - wd / 2 - 4, cy - hh / 2 - 3, cx + wd / 2 + 4, cy + hh / 2 + 3)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                    break
                cx, cy = cx + 5 * nd[0], cy + 5 * nd[1]
            paste(img, lab, 24, cx, cy - hh / 2 + bl, MAGENTA, al, align="c", halo=4)
            sh = best * al * (1 if t >= SEG[5] or t < SEG[1] else 0)
            if sh > 0.01:
                paste(img, r"$\mathrm{the\ shadow}\ \hat{y}$", 24, pp[0] - 16, pp[1] + 34, INK, sh, align="r", halo=4)

    # ------------------------------------------------------------ the 2D panel and the readouts

    def panel(self, img, t, a, A, B):
        x0, y0, w, h = PANEL

        def px(x):
            return x0 + (x - P_X[0]) / (P_X[1] - P_X[0]) * w

        def py(y):
            return y0 + h - (y - P_Y[0]) / (P_Y[1] - P_Y[0]) * h
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(INK3), 0.7 * a)
        ctx.set_line_width(1.2)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0, y0)
        ctx.stroke()
        for xv in (-1, 0, 1):
            ctx.move_to(px(xv), y0 + h)
            ctx.line_to(px(xv), y0 + h + 6)
        for yv in (1, 2, 3, 4):
            ctx.move_to(x0 - 6, py(yv))
            ctx.line_to(x0, py(yv))
        ctx.stroke()
        line_a = max(1 - smooth(seg(t, *HOOK_OUT)), ease_out_cubic(seg(t, TRY[0] - 0.6, TRY[0]))) * a
        if line_a > 0.01:
            ctx.save()
            ctx.rectangle(x0, y0 - 10, w, h + 10)
            ctx.clip()
            ctx.set_source_rgba(*rgb(CYAN), 0.95 * line_a)
            ctx.set_line_width(3.0)
            ctx.move_to(px(P_X[0]), py(A + B * P_X[0]))
            ctx.line_to(px(P_X[1]), py(A + B * P_X[1]))
            ctx.stroke()
            ctx.restore()
            ctx.set_dash([4.0, 4.0])
            ctx.set_source_rgba(*rgb(MAGENTA), 0.95 * line_a)
            ctx.set_line_width(2.2)
            for xv, yv in zip(XS, Y):
                ctx.move_to(px(xv), py(yv))
                ctx.line_to(px(xv), py(A + B * xv))
            ctx.stroke()
            ctx.set_dash([])
        pts_a = max(1 - smooth(seg(t, *HOOK_OUT)), ease_out_cubic(seg(t, SEG[1] + 0.8, SEG[1] + 1.6))) * a
        for i, (xv, yv) in enumerate(zip(XS, Y)):
            ctx.set_source_rgba(*rgb(VERM), pts_a)
            ctx.arc(px(xv), py(yv), 7.5, 0, TAU)
            ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        for xv in (-1, 0, 1):
            paste(img, rf"${xv}$", 20, px(xv), y0 + h + 28, INK3, a, align="c")
        for yv in (1, 2, 3, 4):
            paste(img, rf"${yv}$", 20, x0 - 14, py(yv) + 7, INK3, a, align="r")
        hi = ease_out_cubic(seg(t, SEG[2] + 0.3, SEG[2] + 1.2)) * (1 - smooth(seg(t, SEG[3] + 0.5, SEG[3] + 1.2)))
        for i, (xv, yv) in enumerate(zip(XS, Y)):
            paste(img, rf"$y_{i + 1} = {yv:g}$", 20, px(xv) + 12, py(yv) - 10, VERM, pts_a * max(hi, 0.0), align="l", halo=3)
        # the readouts
        surf, ctx = layer()
        ro_a = max(1 - smooth(seg(t, *HOOK_OUT)), ease_out_cubic(seg(t, TRY[0] - 0.6, TRY[0]))) * a
        rows = (("THE LINE", f"y = {A:.2f} + {B:.2f}x" if B >= 0 else f"y = {A:.2f} − {-B:.2f}x", INK),
                ("ERROR² (SUM OF SQUARED MISSES)", f"{sse(A, B):.2f}", MAGENTA))
        for k, (lab, val, col) in enumerate(rows):
            yy = 1196 + k * 104
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * ro_a)
            show_tracked(ctx, lab, 578, yy, 0.12 * 17)
            font(ctx, "dbold", 38)
            ctx.set_source_rgba(*rgb(col), ro_a)
            ctx.move_to(578, yy + 46)
            ctx.show_text(val)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a, A, B):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if t < SEG[1] or t >= SEG[7]:
            paste(img, r"$X^{\top} X\,\hat{\beta} = X^{\top} y$", 44, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{least\ squares:\ drop\ a\ perpendicular\ from}\ y\ \mathrm{onto\ the\ plane}$", 21, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[2]:
            paste(img, r"$y = a + b\,x$", 44, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{points}\ (-1,\,1),\ (0,\,3),\ (1,\,2)$", 26, 540, y + 168, VERM, a, align="c")
            paste(img, r"$\mathrm{the\ best\ line\ misses\ them\ the\ least}$", 21, 540, y + h - 20, INK2, a, align="c")
        elif t < SEG[3]:
            name = r"$y\ =$"
            lw = S3.tex(name, 32)[2]
            mw = S3.matrix_block(img, Y[:, None], 0, 0, size=26, row_h=38, measure=True)
            x0 = 540 - (lw + 16 + mw) / 2
            paste(img, name, 32, x0, y + 112, VERM, a)
            S3.matrix_block(img, Y[:, None], x0 + lw + 16, y + 104, size=26, row_h=38, a=a)
            paste(img, r"$\mathrm{the\ three\ heights,\ as\ one\ vector\ in\ 3D}$", 21, 540, y + h - 18, INK2, a, align="c")
        elif t < SEG[4]:
            name = r"$X\beta\ =$"
            lw = S3.tex(name, 30)[2]
            mw = S3.matrix_block(img, X, 0, 0, size=26, row_h=38, measure=True)
            bw = S3.matrix_block(img, np.array([[0.0], [0.0]]), 0, 0, size=26, row_h=38, measure=True)
            tot = lw + 14 + mw + 10 + bw + 18 + S3.tex(r"$= a\,(1,1,1) + b\,(-1,0,1)$", 26)[2]
            x0 = 540 - tot / 2
            paste(img, name, 30, x0, y + 112, INK, a)
            S3.matrix_block(img, X, x0 + lw + 14, y + 104, size=26, row_h=38, a=a, cols=(GOLD, GREEN))
            xb = x0 + lw + 14 + mw + 10
            surf, ctx = layer()
            S3.brackets(ctx, xb, xb + bw, y + 104 - 38 + 6, y + 104 + 38 - 2, a)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
            paste(img, r"$a$", 26, xb + bw / 2, y + 104 - 19 + 10, INK, a, align="c")
            paste(img, r"$b$", 26, xb + bw / 2, y + 104 + 19 + 10, INK, a, align="c")
            paste(img, r"$= a\,(1,1,1) + b\,(-1,0,1)$", 26, xb + bw + 18, y + 112, INK, a)
            paste(img, r"$\mathrm{the\ heights\ of\ the\ line}\ a + b\,x\ \mathrm{at}\ x = -1,\,0,\,1$", 21, 540, y + h - 18, INK2, a, align="c")
        elif t < SEG[5]:
            paste(img, r"$\mathrm{error}^2 = \sum_i\,(y_i - a - b\,x_i)^2 = \|\,y - X\beta\,\|^2$", 32, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{the\ squared\ distance\ from\ the\ point\ on\ the\ plane\ to\ the\ tip\ of}\ y$", 21, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[6]:
            paste(img, rf"$\hat{{y}} = ({YHAT[0]:g},\ {YHAT[1]:g},\ {YHAT[2]:g})\qquad y - \hat{{y}} = ({{{RES[0]:g}}},\ {RES[1]:g},\ {{{RES[2]:g}}})$", 28, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\mathrm{{error}}^2 = {RES[0] ** 2:g} + {RES[1] ** 2:g} + {RES[2] ** 2:g} = {SSE:g}\ \ \mathrm{{(the\ smallest)}}$", 28, 540, y + 160, MAGENTA,
                  a * ease_out_cubic(seg(t, SETTLE[1] - 0.3, SETTLE[1] + 0.3)), align="c")
            paste(img, r"$\mathrm{the\ miss}\ y - \hat{y}\ \mathrm{is\ perpendicular\ to\ the\ plane}$", 21, 540, y + h - 20, INK2, a, align="c")
        else:
            paste(img, r"$X^{\top}(y - X\hat{\beta}) = 0\ \ \Rightarrow\ \ X^{\top} X\,\hat{\beta} = X^{\top} y$", 30, 540, y + 74, INK, a, align="c")
            k = ease_out_cubic(seg(t, SEG[6] + 2.2, SEG[6] + 3.0))
            # [[3, 0], [0, 2]] (a, b) = (6, 1)
            parts_w = []
            mw = S3.matrix_block(img, XTX, 0, 0, size=26, row_h=36, measure=True)
            vw = S3.matrix_block(img, np.array([[0.0], [0.0]]), 0, 0, size=26, row_h=36, measure=True)
            rw = S3.matrix_block(img, XTY[:, None], 0, 0, size=26, row_h=36, measure=True)
            sol = rf"$\Rightarrow\ \ a = {BETA[0]:g},\ \ b = {BETA[1]:g}$"
            sw = S3.tex(sol, 30)[2]
            eq_w = S3.tex(r"$=$", 28)[2]
            tot = mw + 8 + vw + 12 + eq_w + 12 + rw + 26 + sw
            xx = 540 - tot / 2
            cy = y + 158
            S3.matrix_block(img, XTX, xx, cy, size=26, row_h=36, a=a * k)
            xx += mw + 8
            surf, ctx = layer()
            S3.brackets(ctx, xx, xx + vw, cy - 36 + 6, cy + 36 - 2, a * k)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
            paste(img, r"$a$", 26, xx + vw / 2, cy - 18 + 10, INK, a * k, align="c")
            paste(img, r"$b$", 26, xx + vw / 2, cy + 18 + 10, INK, a * k, align="c")
            xx += vw + 12
            paste(img, r"$=$", 28, xx, cy + 10, INK, a * k)
            xx += eq_w + 12
            S3.matrix_block(img, XTY[:, None], xx, cy, size=26, row_h=36, a=a * k)
            xx += rw + 26
            paste(img, sol, 30, xx, cy + 10, CYAN, a * ease_out_cubic(seg(t, SEG[6] + 3.6, SEG[6] + 4.4)))
            parts_w.append(tot)
            paste(img, r"$\mathrm{the\ normal\ equations:\ two\ equations,\ two\ unknowns}$", 21, 540, y + h - 14, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        A, B = self.draw_world(img, t, a)
        self.labels(img, t, a)
        self.panel(img, t, a, A, B)
        self.card(img, t, a, A, B)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "3 POINTS   ·   1 STRAIGHT LINE   ·   COMPUTED LIVE"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sc = Scene()
    print("facts", sc.facts)
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        ims.append(cv2.resize(sc.frame(fr)[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
