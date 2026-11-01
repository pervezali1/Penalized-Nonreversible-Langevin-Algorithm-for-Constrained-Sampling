"""Every matrix is a turn, a stretch and a turn: the singular value decomposition, on the 3D stage of reels 38-39
(stage3d.py), narrated by Kokoro (voicekit.py). A ball with three painted arrows (the right singular vectors
v1, v2, v3) is sent through M all at once (it becomes a tilted egg), then in slow motion: the turn V^T (slerp:
the arrows line up with x, y, z), the stretch Sigma along the axes (2.4, 1.25, 0.5), the turn U, ending exactly
on the first egg (its outline stays as a dashed ghost). Then M = U Sigma V^T on the card, the product of the
singular values is det M (reel 38), and the low-rank versions: drop the smallest stretch (a flat disc, 97% of
the total squared stretch kept) and keep only the largest (a needle, 76%). frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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

# ---------------------------------------------------------------- the matrix, built from its decomposition
SIG = np.array([2.4, 1.25, 0.5])
U = Rotation.from_euler("zyx", [40, -25, 30], degrees=True).as_matrix()
V = Rotation.from_euler("zyx", [-55, 35, -20], degrees=True).as_matrix()
M = U @ np.diag(SIG) @ V.T
_u, _s, _vt = np.linalg.svd(M)
assert np.allclose(_s, SIG) and abs(np.linalg.det(U) - 1) < 1e-9 and abs(np.linalg.det(V) - 1) < 1e-9
DET = float(np.linalg.det(M))
ENERGY = SIG ** 2 / np.sum(SIG ** 2)
KEEP2, KEEP1 = float(ENERGY[:2].sum()), float(ENERGY[0])
COLS = (VERM, GREEN, GOLD)


def words(x):
    return {2.4: "two point four", 1.25: "one point two five", 0.5: "one half", 1.5: "one point five"}[round(x, 2)]


SCRIPT = [
    "Every matrix does just three things: turn, stretch, turn.",
    "Here's a matrix acting on a ball, all at once. The ball becomes a tilted egg.",
    "Now in slow motion. First, a turn. The ball looks the same, but its three painted arrows line up with the axes.",
    f"Then a stretch along the axes: by {words(SIG[0])}, {words(SIG[1])}, and {words(SIG[2])}.",
    "Then one more turn. And it lands exactly on the same egg.",
    "Turn, stretch, turn. M equals U, sigma, V transpose. Every matrix can be written this way. It's called the singular value decomposition.",
    f"The stretches are the singular values. Multiply them, and you get {words(SIG.prod())}: the determinant, again.",
    f"Drop the smallest stretch, and the egg goes flat. But it still keeps {round(100 * KEEP2)} percent of the matrix.",
    f"Keep only the biggest, and it's a single line, still {round(100 * KEEP1)} percent. That's the idea behind PCA, and image compression.",
    "Any matrix: a turn, a stretch, and a turn. The stretches tell you what it really does.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# 0 hook, 1 direct, 2 turn V^T, 3 stretch, 4 turn U, 5 M = U S V^T, 6 det, 7 rank 2, 8 rank 1, 9 close
SEG_MIN = [2.8, 6.5, 7.0, 6.5, 6.0, 7.5, 6.5, 6.5, 7.5, 6.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
NSEG = len(SEG_MIN)
T_END = SEG[NSEG]
LOOP_FADE = (T_END, T_END + 0.55)
N_FRAMES = int(round((T_END + 0.7) * FPS))
DIRECT = (SEG[1] + 0.4, SEG[1] + 3.0)              # the ball becomes the egg in one go
DIRECT_BACK = (SEG[2] - 0.6, SEG[2] + 0.3)         # and back to the ball
TURN1 = (SEG[2] + 0.8, SEG[2] + 3.6)
STRETCH = (SEG[3] + 0.4, SEG[3] + 3.0)
TURN2 = (SEG[4] + 0.4, SEG[4] + 3.0)
RANK2 = (SEG[7] + 0.3, SEG[7] + 2.3)
RANK1 = (SEG[8] + 0.3, SEG[8] + 2.3)
RESTORE = (SEG[9] + 0.2, SEG[9] + 2.0)
TO_BALL = (T_END - 2.2, T_END - 0.4)                # the egg turns back into the ball for the loop
Z0 = np.array([0.0, 0.0, 0.2])
Z_FLOOR = -2.5
CARD = (70, 1430, 940, 250)
AX_TOP, AX_BOTTOM = 470, 1420                               # the axes stay between the headlines and the card
SWING_AZ = 36.0
_SL_V = Slerp([0, 1], Rotation.from_matrix([np.eye(3), V.T]))
_SL_U = Slerp([0, 1], Rotation.from_matrix([np.eye(3), U]))


def transform(t):
    """The matrix applied to the ball at time t, and a label for what it is."""
    if t < SEG[2]:
        k = smooth(seg(t, *DIRECT)) * (1 - smooth(seg(t, *DIRECT_BACK)))
        return np.eye(3) + k * (M - np.eye(3)), "direct"
    r1 = _SL_V([smooth(seg(t, *TURN1))]).as_matrix()[0]
    s = 1 + smooth(seg(t, *STRETCH)) * (SIG - 1)
    # the low-rank steps: sigma3, then sigma2, squashed (to a sliver, so the mesh stays drawable), then restored
    rest = smooth(seg(t, *RESTORE))
    s3 = s[2] + smooth(seg(t, *RANK2)) * (1 - rest) * (0.02 - s[2])
    s2 = s[1] + smooth(seg(t, *RANK1)) * (1 - rest) * (0.02 - s[1])
    s = np.array([s[0], s2, s3])
    r2 = _SL_U([smooth(seg(t, *TURN2))]).as_matrix()[0]
    T = r2 @ np.diag(s) @ r1
    back = smooth(seg(t, *TO_BALL))
    return T + back * (np.eye(3) - T), "steps"


def qa_frames():
    fr = []
    for k in range(NSEG):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.verts, self.quads = S3.sphere_mesh(16, 32)
        self.facts = dict(M=np.round(M, 4).tolist(), U=np.round(U, 4).tolist(), V=np.round(V, 4).tolist(), sigma=SIG.tolist(), det=DET,
                          keep2=KEEP2, keep1=KEEP1, svd_check=float(np.abs(U @ np.diag(SIG) @ V.T - M).max()), segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG

    def _cues(self):
        S_ = SEG
        hook = [("Every matrix: turn, stretch, turn.", "black", 50, INK, 300, 1), ("The singular value decomposition.", "black", 40, INK2, 390, 1)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("A matrix, all at once.", "black", 52, INK, 300, 1), ("The ball becomes a tilted egg.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Slow motion. 1: a turn.", "black", 52, INK, 300, 1), ("The arrows line up with the axes.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("2: a stretch along the axes.", "black", 50, INK, 300, 1), (f"× {fmt(SIG[0])},  × {fmt(SIG[1])},  × {fmt(SIG[2])}.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("3: one more turn.", "black", 52, INK, 300, 1), ("Exactly the same egg.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("M = U Σ Vᵗ, for every matrix.", "black", 48, INK, 300, 1), ("The singular value decomposition.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [(f"{fmt(SIG[0])} × {fmt(SIG[1])} × {fmt(SIG[2])} = {fmt(DET)} = det M.", "black", 46, INK, 300, 1), ("The stretches: the singular values.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], S_[8] - 0.3, [("Drop the smallest stretch.", "black", 50, INK, 300, 1), (f"Flat, and still {KEEP2:.0%} of the matrix.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[8], S_[9] - 0.3, [("Keep only the biggest.", "black", 52, VERM, 300, 1), (f"A line, still {KEEP1:.0%}. The idea behind PCA.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[9], T_END - 0.3, [("Any matrix: a turn, a stretch, a turn.", "black", 44, INK, 300, 1), ("The stretches say what it really does.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END + 0.1, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    def camera(self, t):
        L = N_FRAMES / FPS
        # for the low-rank steps the camera swings round to look along the flat disc's edge, so its flatness shows
        w = smooth(seg(t, SEG[7] - 0.6, SEG[7] + 1.2)) * (1 - smooth(seg(t, SEG[9], SEG[9] + 1.8)))
        return S3.Camera(Z0 + np.array([0, 0, -0.1]), -60 + 16 * math.sin(TAU * t / L) + SWING_AZ * w, 20 + 4 * math.sin(2 * TAU * t / L),
                         dist=10.8, focal=1950.0, centre=(540, 960))

    def draw_world(self, img, t, a):
        cam = self.camera(t)
        T, mode = transform(t)
        Vt = self.verts @ T.T + Z0
        S3.floor_light(img, cam, Z_FLOOR, a)
        S3.floor_shadow(img, cam, Vt, Z_FLOOR, a, strength=0.55)
        surf, ctx = layer()
        # the axes, faint, dashed: the stretch happens along them
        ax_a = a * (0.25 + 0.5 * smooth(seg(t, SEG[2] + 2.5, SEG[3])) * (1 - smooth(seg(t, SEG[4] + 1.0, SEG[5]))))
        # (clipped to the stage, so an axis never runs up into the headlines or down behind the card)
        ctx.save()
        ctx.rectangle(0, AX_TOP, 1080, AX_BOTTOM - AX_TOP)
        ctx.clip()
        ctx.set_dash([5.0, 7.0])
        for i in range(3):
            e = np.eye(3)[i]
            pa, pb = cam.project(np.array([Z0 - 2.9 * e, Z0 + 2.9 * e]))
            ctx.set_source_rgba(*rgb(INK3), ax_a)
            ctx.set_line_width(1.3)
            ctx.move_to(*pa)
            ctx.line_to(*pb)
            ctx.stroke()
        ctx.restore()
        o2 = cam.project(Z0)[0]
        self.axis_ends = []
        for i in range(3):
            p = cam.project(Z0 + 3.05 * np.eye(3)[i])[0]
            if p[1] < AX_TOP - 18 and o2[1] - p[1] > 1e-6:
                # the end is off the stage: put the letter just above where the clipped line stops
                k = (o2[1] - (AX_TOP - 18)) / (o2[1] - p[1])
                p = o2 + k * (p - o2)
            self.axis_ends.append((p, ax_a))
        # the ghost of the direct result, from the moment the slow motion starts until it is matched
        ghost = smooth(seg(t, SEG[2] + 0.4, SEG[2] + 1.2)) * (1 - smooth(seg(t, SEG[5] + 0.5, SEG[5] + 1.5)))
        if ghost > 0:
            G = self.verts @ M.T + Z0
            P = cam.project(G)
            ctx.set_dash([3.0, 5.0])
            ctx.set_source_rgba(*rgb(INK2), 0.30 * ghost * a)
            ctx.set_line_width(1.0)
            for q in self.quads:
                if q[4] % 4 == 0:
                    ctx.move_to(*P[q[0]])
                    ctx.line_to(*P[q[1]])
                if q[5] % 4 == 0:
                    ctx.move_to(*P[q[0]])
                    ctx.line_to(*P[q[3]])
            ctx.stroke()
            ctx.set_dash([])
        S3.draw_mesh(ctx, cam, Vt, self.quads, a, alpha=0.42, grid=True, grid_every=(2, 4))
        # the three painted arrows: T v_i
        self.arrow_labels = []
        for i in range(3):
            tip = T @ V[:, i]
            S3.arrow3(ctx, cam, Z0, Z0 + tip, COLS[i], a, width=4.6, head=24)
            self.arrow_labels.append((i, cam.project(Z0 + tip)[0], cam.project(Z0)[0], float(np.linalg.norm(tip)), a))
        o = cam.project(Z0)[0]
        ctx.set_source_rgba(1, 1, 1, 0.95 * a)
        ctx.arc(*o, 4.5, 0, TAU)
        ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return T

    def labels(self, img, t, a):
        for i, (p, al) in enumerate(self.axis_ends):
            paste(img, f"${'xyz'[i]}$", 24, p[0], p[1] + 8, INK3, al, align="c")
        stretch_phase = SEG[3] <= t < SEG[7]
        placed = []
        for i, tip, o, length, al in self.arrow_labels:
            lab = rf"$v_{i + 1}$" if not stretch_phase else rf"$\sigma_{i + 1} = {fmt(length)}$"
            # centre the label just beyond the arrowhead, along the arrow, far enough that its box clears the head
            d = np.asarray(tip, float) - np.asarray(o, float)
            n = float(np.hypot(*d))
            d = d / n if n > 1e-6 else np.array([0.0, -1.0])
            m, bl, w = S3.tex(lab, 22)
            hh = m.shape[0]
            cx, cy = np.asarray(tip, float) + d * (14 + abs(d[0]) * w / 2 + abs(d[1]) * hh / 2)
            cx = min(max(cx, 24 + w / 2), 1056 - w / 2)
            # two short arrows can put their labels on top of each other: walk this one further out until it is clear
            for _ in range(20):
                box = (cx - w / 2 - 4, cy - hh / 2 - 3, cx + w / 2 + 4, cy + hh / 2 + 3)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                    break
                cx, cy = cx + 5 * d[0], cy + 5 * d[1] - (5 if abs(d[1]) < 0.3 else 0)
                cx = min(max(cx, 24 + w / 2), 1056 - w / 2)
            placed.append((cx - w / 2 - 4, cy - hh / 2 - 3, cx + w / 2 + 4, cy + hh / 2 + 3))
            paste(img, lab, 22, cx, cy - hh / 2 + bl, COLS[i], al, align="c", halo=4)

    def card(self, img, t, a):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

        def one(name, Mx, note, cols=None, kz=True):
            lw = S3.tex(name, 30)[2]
            mw = S3.matrix_block(img, Mx, 0, 0, measure=True, keep_zeros=kz)
            x0 = 540 - (lw + 16 + mw) / 2
            paste(img, name, 30, x0, y + 114, INK, a)
            S3.matrix_block(img, Mx, x0 + lw + 16, y + 104, a=a, cols=cols, keep_zeros=kz)
            paste(img, note, 21, 540, y + h - 22, INK2, a, align="c")

        if t < SEG[2] or t >= T_END - 0.3:
            one(r"$M\ =$", M, r"$\mathrm{any\ matrix\ (here\ a\ random-looking\ one)}$")
        elif t < SEG[3]:
            one(r"$V^{\top} =$", V.T, r"$\mathrm{a\ rotation:\ lengths\ and\ angles\ stay\ the\ same}$")
        elif t < SEG[4]:
            one(r"$\Sigma\ =$", np.diag(SIG), r"$\mathrm{a\ stretch\ along\ the\ axes:\ the\ singular\ values}$", cols=COLS, kz=False)
        elif t < SEG[5]:
            one(r"$U\ =$", U, r"$\mathrm{another\ rotation}$")
        elif t < SEG[6]:
            sz, rh = 17, 30
            parts = [(r"$M$", None), (r"$=$", None), (None, U), (None, np.diag(SIG)), (None, V.T)]
            kzs = [True, True, True, False, True]
            widths = [S3.tex(p, 26)[2] if p else S3.matrix_block(img, Mx, 0, 0, size=sz, row_h=rh, measure=True, keep_zeros=kz)
                      for (p, Mx), kz in zip(parts, kzs)]
            labels = [None, None, r"$U$", r"$\Sigma$", r"$V^{\top}$"]
            total = sum(widths) + 10 * (len(parts) - 1)
            xx = 540 - total / 2
            for (p, Mx), wd, lab, kz in zip(parts, widths, labels, kzs):
                if p:
                    paste(img, p, 26, xx, y + 106, INK, a)
                else:
                    S3.matrix_block(img, Mx, xx, y + 98, size=sz, row_h=rh, a=a, cols=COLS if lab == r"$\Sigma$" else None, keep_zeros=kz)
                    paste(img, lab, 22, xx + wd / 2, y + 182, INK2, a, align="c")
                xx += wd + 10
            paste(img, r"$\mathrm{turn,\ stretch,\ turn:\ true\ for\ every\ matrix}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[7]:
            paste(img, rf"$\sigma_1 \sigma_2 \sigma_3 = {fmt(SIG[0])} \times {fmt(SIG[1])} \times {fmt(SIG[2])} = {fmt(SIG.prod())}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\det M = {fmt(DET, 4)}$", 32, 540, y + 168, CYAN, a * ease_out_cubic(seg(t, SEG[6] + 1.0, SEG[6] + 1.5)), align="c")
            paste(img, r"$\mathrm{the\ volume\ of\ the\ egg,\ in\ balls}$", 21, 540, y + h - 22, INK2, a, align="c")
        elif t < SEG[9]:
            r1 = t >= SEG[8]
            num = r"\sigma_1^2" if r1 else r"\sigma_1^2 + \sigma_2^2"
            val = KEEP1 if r1 else KEEP2
            paste(img, rf"$\mathrm{{kept}} = \frac{{{num}}}{{\sigma_1^2 + \sigma_2^2 + \sigma_3^2}} = {100 * val:.0f}\%$", 32, 540, y + 118, VERM if r1 else INK, a, align="c")
            note = r"$\mathrm{rank\ 1:\ one\ direction\ in,\ one\ direction\ out}$" if r1 else r"$\mathrm{rank\ 2:\ everything\ lands\ in\ a\ plane}$"
            paste(img, note, 21, 540, y + h - 22, INK2, a, align="c")
        else:
            paste(img, r"$M = U\,\Sigma\,V^{\top}$", 46, 540, y + 120, INK, a, align="c")
            paste(img, r"$\mathrm{turn,\ stretch,\ turn}$", 22, 540, y + h - 26, INK2, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        img = self.bg.copy()
        self.draw_world(img, t, a)
        self.labels(img, t, a)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "A 3 × 3 MATRIX ON A UNIT BALL   ·   M = U Σ Vᵗ   ·   COMPUTED LIVE"
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
