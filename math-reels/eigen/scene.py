"""Eigenvectors: the directions a matrix only stretches. The 3D stage of reel 38 (stage3d.py), narrated by Kokoro
(voicekit.py). A ball of 80 unit arrows; a symmetric matrix A (eigenvalues 2, 1.2, 0.5 on tilted axes) is applied
and the ball becomes an egg: every arrow turns (one is followed with its angle measured live), except three
directions that only stretch, the eigenvectors, drawn in vermilion, green and gold with their stretch factors.
Their product is the determinant (reel 38). Then A applied again and again, continuously (A^k with real k,
fine for positive eigenvalues), each arrow rescaled to length 1: the arrows swing onto the strongest
eigenvector, the mean angle and the count within a degree measured live. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import CYAN, GOLD, GREEN, INK, INK2, INK3, MAGENTA, TAU, VERM, fmt, paste  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg

FPS = 60


def rot(ax, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- the matrix and every number the reel states
R = rot("z", 35) @ rot("y", -30) @ rot("x", 20)
LAM = np.array([2.0, 1.2, 0.5])
A = R @ np.diag(LAM) @ R.T
EV = [R[:, i] * (1 if R[2, i] >= 0 else -1) for i in range(3)]       # eigenvectors, pointing up
EV_C = (VERM, GREEN, GOLD)
ARROWS = S3.fib_sphere(80)
_AU = ARROWS @ A.T
TURN = np.degrees(np.arccos(np.clip(np.sum(ARROWS * _AU, 1) / np.linalg.norm(_AU, axis=1), -1, 1)))
FOLLOW = int(np.argmin(np.abs(TURN - 30)))                           # the arrow we follow: turned about 30 degrees
K_MAX = 12                                                           # applications in the power-iteration segment


def power(k):
    """A^k for real k (A is symmetric with positive eigenvalues)."""
    return R @ np.diag(LAM ** k) @ R.T


def power_stats(k):
    P = ARROWS @ power(k).T
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.abs(P @ EV[0]), 0, 1)))
    return P, ang


_, ANG_END = power_stats(K_MAX)
_, ANG_0 = power_stats(0)
DET = float(np.linalg.det(A))
NUM_WORDS = {0.5: "one half", 1.2: "one point two", 2.0: "two"}

SCRIPT = [
    "A matrix knocks almost every arrow off its line. Almost.",
    "Here are eighty arrows, all of length one, pointing every way.",
    f"Apply a matrix, and the ball becomes an egg. Every arrow turns. This one, by about {round(TURN[FOLLOW]):d} degrees.",
    "But three directions never turn. They only stretch. These are the eigenvectors.",
    "Their stretch factors are the eigenvalues: two, one point two, and one half. A times v equals lambda times v.",
    "Multiply them, and you get one point two. That's the determinant: the volume from the last video.",
    "Now apply the matrix again, and again. Rescale every arrow back to length one, and watch them swing.",
    f"Each step, the strongest direction wins by a factor of {LAM[0] / LAM[1]:.2g}. After {K_MAX} steps, {int(np.sum(ANG_END < 1))} of the 80 arrows lie within a degree of it.",
    "Eigenvectors: the directions a matrix only stretches. Their stretch factors, multiplied, give the determinant.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# ---------------------------------------------------------------- the timeline, sized to the narration
# 0 hook, 1 ball, 2 apply, 3 eigenvectors, 4 eigenvalues, 5 determinant, 6 power, 7 power result, 8 close
SEG_MIN = [2.8, 6.0, 7.5, 7.0, 7.0, 6.5, 8.0, 7.0, 6.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[9]
LOOP_FADE = (T_END, T_END + 0.55)
N_FRAMES = int(round((T_END + 0.7) * FPS))
APPLY = (SEG[2] + 0.3, SEG[2] + 3.3)                                 # the ball becomes the egg
UNAPPLY = (SEG[6] - 0.2, SEG[6] + 1.2)                               # back to the unit ball for the power steps
POWER = (SEG[6] + 1.6, SEG[8] - 0.4)                                 # k: 0 -> K_MAX
BACK = (SEG[8] + 0.2, SEG[8] + 2.6)                                  # the arrows return to the ball for the loop
Z0 = np.array([0.0, 0.0, 0.25])                                      # the origin of everything, above the floor
Z_FLOOR = -2.1
CARD = (70, 1430, 940, 250)


def apply_k(t):
    return smooth(seg(t, *APPLY)) * (1 - smooth(seg(t, *UNAPPLY)))


def power_k(t):
    return K_MAX * smooth(seg(t, *POWER)) * (1 - smooth(seg(t, *BACK)))


def qa_frames():
    fr = []
    for k in range(9):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.8 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.verts, self.quads = S3.sphere_mesh(14, 28)
        self.facts = dict(A=np.round(A, 4).tolist(), eigenvalues=LAM.tolist(), det=DET, turn_min=float(TURN.min()), turn_max=float(TURN.max()),
                          turn_follow=float(TURN[FOLLOW]), follow=FOLLOW, power_mean_end=float(ANG_END.mean()), power_within1_end=int(np.sum(ANG_END < 1)),
                          power_mean_start=float(ANG_0.mean()), k_max=K_MAX, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG

    def _cues(self):
        hook = [("A matrix turns almost every arrow.", "black", 50, INK, 300, 1), ("Almost.", "black", 44, INK2, 390, 1)]
        S_ = SEG
        n1 = int(np.sum(ANG_END < 1))
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("80 arrows, all of length 1.", "black", 50, INK, 300, 1), ("Pointing every way.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Apply a matrix: every arrow turns.", "black", 46, INK, 300, 1), (f"This one by {TURN[FOLLOW]:.0f}°. The least: {TURN.min():.0f}°.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Except three directions.", "black", 50, INK, 300, 1), ("They only stretch: the eigenvectors.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("The stretch factors: the eigenvalues.", "black", 46, INK, 300, 1), ("2, 1.2 and 0.5.   A v = λ v.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [(f"2 × 1.2 × 0.5 = {DET:.1f} = det A.", "black", 48, INK, 300, 1), ("The volume from the last video.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("Apply it again, and again.", "black", 50, INK, 300, 1), ("Rescale each arrow to length 1 every time.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], S_[8] - 0.3, [("The strongest direction wins.", "black", 48, VERM, 300, 1), (f"After {K_MAX} steps: {n1} of 80 within 1°.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[8], T_END - 0.3, [("Eigenvectors: the directions a matrix only stretches.", "black", 38, INK, 300, 1), ("Their stretch factors, multiplied: the determinant.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END + 0.1, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ camera

    def camera(self, t):
        L = N_FRAMES / FPS
        az = -62 + 16 * math.sin(TAU * t / L)
        el = 20 + 4 * math.sin(2 * TAU * t / L)
        return S3.Camera(Z0 + np.array([0, 0, -0.15]), az, el, dist=10.8, focal=2650.0, centre=(540, 960))

    # ------------------------------------------------------------ drawing

    def draw_world(self, img, t, a):
        cam = self.camera(t)
        k = apply_k(t)
        M = np.eye(3) + k * (A - np.eye(3))
        kp = power_k(t)
        P_pow, ang_pow = power_stats(kp)
        in_power = kp > 1e-6
        # the solid: the egg M·ball (the unit ball during the power steps)
        V = self.verts @ M.T + Z0
        S3.floor_light(img, cam, Z_FLOOR, a)
        S3.floor_shadow(img, cam, V, Z_FLOOR, a, strength=0.55)
        surf, ctx = layer()
        S3.draw_mesh(ctx, cam, V, self.quads, a, alpha=0.30, grid=True)
        # the arrows: from the origin to M·u (or to the rescaled A^k u during the power steps)
        tips = (ARROWS @ M.T) if not in_power else P_pow
        o = cam.project(Z0)[0]
        ev_show = ease_out_cubic(seg(t, SEG[3] + 0.2, SEG[3] + 1.4)) * (1 - smooth(seg(t, BACK[0], BACK[1])))
        follow_show = ease_out_cubic(seg(t, SEG[2] + 0.6, SEG[2] + 1.2)) * (1 - smooth(seg(t, SEG[3], SEG[3] + 0.8)))
        TP = cam.project(tips + Z0)
        dep = cam.depth(tips + Z0)
        order = np.argsort(-dep)
        for i in order:
            if i == FOLLOW and follow_show > 0:
                continue
            near = 1.0
            if in_power:
                near = 1 - min(1.0, ang_pow[i] / 30.0)
            col = np.array(rgb(INK)) * (1 - 0.6 * near) + np.array(rgb(VERM)) * 0.6 * near if in_power else np.array(rgb(INK))
            al = (0.32 + 0.25 * near if in_power else 0.36) * a * (1 - 0.45 * ev_show * (not in_power))
            ctx.set_source_rgba(*col, al)
            ctx.set_line_width(1.6)
            ctx.move_to(*o)
            ctx.line_to(*TP[i])
            ctx.stroke()
            ctx.set_source_rgba(*np.clip(np.array(rgb(CYAN)) * (1 - near) + np.array(rgb(VERM)) * near, 0, 1) if in_power else rgb(CYAN), 0.95 * a)
            ctx.arc(*TP[i], 3.6, 0, TAU)
            ctx.fill()
        # the followed arrow: its original line dashed, the turned arrow in cyan, the angle live
        if follow_show > 0:
            u = ARROWS[FOLLOW]
            Mu = M @ u
            ctx.set_dash([7.0, 6.0])
            pa, pb = cam.project(np.array([Z0 - 2.4 * u, Z0 + 2.4 * u]))
            ctx.set_source_rgba(*rgb(INK2), 0.7 * follow_show * a)
            ctx.set_line_width(1.6)
            ctx.move_to(*pa)
            ctx.line_to(*pb)
            ctx.stroke()
            ctx.set_dash([])
            S3.arrow3(ctx, cam, Z0, Z0 + Mu, CYAN, follow_show * a, width=4.0)
            self.follow_label = (cam.project(Z0 + Mu * 1.12)[0], float(np.degrees(np.arccos(np.clip(Mu @ u / np.linalg.norm(Mu), -1, 1)))), follow_show * a)
        else:
            self.follow_label = None
        # the eigenvectors: their lines dashed through the origin, the arrows M·v in colour
        self.ev_labels = []
        if ev_show > 0:
            for i in range(3):
                if in_power and i > 0:
                    continue
                v = EV[i]
                Mv = (M @ v) if not in_power else v
                ctx.set_dash([4.0, 7.0])
                pa, pb = cam.project(np.array([Z0 - 2.6 * v, Z0 + 2.6 * v]))
                ctx.set_source_rgba(*rgb(EV_C[i]), 0.55 * ev_show * a)
                ctx.set_line_width(1.6)
                ctx.move_to(*pa)
                ctx.line_to(*pb)
                ctx.stroke()
                ctx.set_dash([])
                S3.arrow3(ctx, cam, Z0, Z0 + Mv, EV_C[i], ev_show * a, width=5.0, head=26)
                S3.arrow3(ctx, cam, Z0, Z0 - Mv, EV_C[i], 0.55 * ev_show * a, width=3.0, head=18)
                self.ev_labels.append((i, cam.project(Z0 + Mv * 1.13 + np.array([0, 0, 0.05]))[0], float(np.linalg.norm(Mv)), ev_show * a))
        ctx.set_source_rgba(1, 1, 1, 0.95 * a)
        ctx.arc(*o, 4.5, 0, TAU)
        ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return k, kp, ang_pow

    def labels(self, img, t, a):
        if self.follow_label:
            p, ang, al = self.follow_label
            paste(img, rf"${ang:.1f}^\circ$", 24, p[0] + 10, p[1] + 8, CYAN, al)
        for i, p, s, al in self.ev_labels:
            p = (min(float(p[0]), 940.0), float(p[1]))      # keep the labels inside the frame
            if power_k(t) > 1e-6:
                paste(img, r"$v_1$", 24, p[0] + 8, p[1] + 8, EV_C[i], al)
            else:
                paste(img, rf"$v_{i + 1}\ \times {fmt(s, 2)}$", 22, p[0] + 8, p[1] + 8, EV_C[i], al)

    def card(self, img, t, a, k, kp, ang_pow):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if t < SEG[3] or t >= SEG[8] + 0.0 and kp < 1e-6:
            # the matrix itself
            lw = S3.tex(r"$A\ =$", 30)[2]
            mw = S3.matrix_block(img, A, 0, 0, measure=True)
            x0 = 540 - (lw + 16 + mw) / 2
            paste(img, r"$A\ =$", 30, x0, y + 114, INK, a)
            S3.matrix_block(img, A, x0 + lw + 16, y + 104, a=a)
            line = r"$\mathrm{symmetric}:\ A^{\top} = A$" if (t < SEG[2] or t >= T_END - 0.3) else rf"$\mathrm{{this\ arrow\ turned}}\ {TURN[FOLLOW]:.1f}^\circ;\ \mathrm{{all\ 80\ turned\ at\ least}}\ {TURN.min():.1f}^\circ$"
            if SEG[8] <= t < T_END - 0.3:
                line = r"$A v = \lambda v \qquad \det A = \lambda_1 \lambda_2 \lambda_3$"
            paste(img, line, 22, 540, y + h - 24, INK2, a, align="c")
        elif t < SEG[5]:
            paste(img, r"$A\,v = \lambda\,v$", 44, 540, y + 92, INK, a, align="c")
            xs = (230, 540, 850)
            for i in range(3):
                al = a * ease_out_cubic(seg(t, SEG[4] + 0.3 + 0.5 * i, SEG[4] + 0.8 + 0.5 * i))
                paste(img, rf"$\lambda_{i + 1} = {fmt(LAM[i])}$", 32, xs[i], y + 170, EV_C[i], al, align="c")
            paste(img, r"$\mathrm{the\ eigenvectors\ only\ stretch;\ the\ eigenvalues\ say\ by\ how\ much}$", 21, 540, y + h - 22, INK2,
                  a * ease_out_cubic(seg(t, SEG[3] + 0.4, SEG[3] + 0.9)), align="c")
        elif t < SEG[6]:
            paste(img, rf"$\lambda_1 \lambda_2 \lambda_3 = 2 \times 1.2 \times 0.5 = {fmt(LAM.prod())}$", 34, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\det A = {fmt(DET, 4)}$", 34, 540, y + 170, CYAN, a * ease_out_cubic(seg(t, SEG[5] + 0.8, SEG[5] + 1.3)), align="c")
            paste(img, r"$\mathrm{the\ egg\ holds}\ 1.2\times\ \mathrm{the\ ball's\ volume}$", 21, 540, y + h - 22, INK2, a, align="c")
        else:
            paste(img, r"$\frac{A^k u}{\|A^k u\|}$", 36, 330, y + 118, INK, a, align="c")
            paste(img, rf"$k = {kp:.1f}$", 34, 640, y + 96, INK, a, align="l")
            paste(img, rf"$\mathrm{{mean\ angle\ to}}\ v_1:\ {ang_pow.mean():.1f}^\circ$", 26, 560, y + 150, VERM, a, align="l")
            paste(img, rf"$\mathrm{{within}}\ 1^\circ:\ {int(np.sum(ang_pow < 1))}\ \mathrm{{of}}\ 80$", 26, 560, y + 192, INK2, a, align="l")
            paste(img, rf"$\mathrm{{each\ step\ the\ rest\ shrinks\ by}}\ \lambda_2/\lambda_1 = {LAM[1] / LAM[0]:.1f}$", 21, 540, y + h - 18, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        img = self.bg.copy()
        k, kp, ang_pow = self.draw_world(img, t, a)
        self.labels(img, t, a)
        self.card(img, t, a, k, kp, ang_pow)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "80 UNIT ARROWS   ·   A SYMMETRIC 3 × 3 MATRIX   ·   COMPUTED LIVE"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
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
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
