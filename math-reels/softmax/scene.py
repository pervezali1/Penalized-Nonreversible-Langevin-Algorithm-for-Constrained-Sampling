"""Three classes and softmax. Dark theme of reels 31-35 and 41, maths in Computer Modern (stage3d.tex), narrated by
Kokoro (voicekit). A three-arm spiral and a network with three outputs (smsim.py): every point gets three scores,
softmax turns them into three probabilities (worked on the card for one ringed point, live), the map shows the
probabilities mixed (the stronger the colour, the surer the network) with the decision boundaries, and training
runs live from the snapshots. For the ringed point the blame at the output, p - y, is three numbers, drawn as
three bars that shrink as it learns. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import paste  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg
FPS = 60
TAU = 2 * math.pi
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA, GOLD = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6", "#F6C56E"
CLS = (VERM, BLUE, GREEN)
CLS_NAME = ("vermilion", "blue", "green")
BG_TOP, BG_BOT = "#0E1226", "#05060C"
PANEL_BASE = "#161C33"
NPZ = os.path.join(HERE, "softmax_sim.npz")
if not os.path.exists(NPZ):
    import smsim
    smsim.main(NPZ)
D = np.load(NPZ)
X, Y = D["X"], D["y"]
FIELDS = D["field"].astype(np.float32)
ACC, PZ, PP = D["acc"], D["pz"], D["pp"]
PROBE, RIGHT_FROM = int(D["probe"]), int(D["right_from"])
N_SNAP = len(ACC)
SNAP_STEPS = 15
YP = np.eye(3)[Y[PROBE]]
FINAL_RIGHT = int(round(ACC[-1] * len(Y)))

SCRIPT = [
    "Two classes need one probability. Three classes need softmax.",
    "Three spiral arms, three colors. For every point, the network gives three scores, one per color.",
    "Softmax turns the scores into probabilities: raise e to each score, then divide by the total. They always add up to one.",
    "Now train it. The map shows the probabilities, mixed: the stronger the color, the surer the network.",
    "Watch the ringed point. Its blame at the output is guess minus answer. Now that's three numbers, one per color.",
    f"Early on, it leans towards the wrong colors. From step {RIGHT_FROM * SNAP_STEPS}, blue stays on top, and the blame keeps shrinking.",
    f"After fifteen hundred steps, {round(100 * ACC[-1]):d} percent of the points are right.",
    "Softmax plus cross-entropy: three probabilities, and the blame is still just guess minus answer.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 three scores, 2 softmax, 3 train, 4 blame, 5 blame shrinks, 6 result, 7 close
SEG_MIN = [2.8, 6.0, 7.5, 6.5, 7.0, 7.5, 5.5, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
NSEG = len(SEG_MIN)
T_END = SEG[NSEG]
LOOP_FADE = (T_END, T_END + 0.55)
N_FRAMES = int(round((T_END + 0.7) * FPS))
TRAIN = (SEG[3] + 0.6, SEG[6] - 0.4)
UNTRAIN = (SEG[7] + 2.5, T_END - 0.2)             # the map relaxes back to the untrained one for the loop
PANEL = (190, 480, 700)
CARD = (70, 1430, 940, 250)


def snap_at(t):
    u = seg(t, *TRAIN)
    u = 1 - (1 - u) ** 1.8
    return u * (N_SNAP - 1) * (1 - smooth(seg(t, *UNTRAIN)))


def lerp_snap(arr, sn):
    i0 = int(math.floor(sn))
    i1 = min(i0 + 1, N_SNAP - 1)
    k = sn - i0
    return arr[i0] * (1 - k) + arr[i1] * k


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def qa_frames():
    fr = []
    for k in range(NSEG):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(seed=int(D["seed"]), finals=D["finals"].tolist(), final_acc=float(ACC[-1]), final_right=FINAL_RIGHT, probe=PROBE,
                          probe_x=X[PROBE].tolist(), right_from_step=RIGHT_FROM * SNAP_STEPS, p_start=PP[0].tolist(), p_end=PP[-1].tolist(),
                          segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        n = PANEL[2]
        surf, ctx = layer(n, n)
        S3.pill(ctx, 0, 0, n, n, 18)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
        _, a = surf_rgba(surf)
        self.mask = a.astype(np.float32)

    def _cues(self):
        S_ = SEG
        hook = [("Two classes: one probability.", "black", 52, INK, 300, 1), ("Three classes: softmax.", "black", 48, GOLD, 390, 1)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Three arms, three colours.", "black", 50, INK, 300, 1), ("Every point gets three scores.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Softmax: e to each score, ÷ the total.", "black", 44, INK, 300, 1), ("Three probabilities that add up to 1.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Now train it.", "black", 54, INK, 300, 1), ("Stronger colour: a surer network.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("The blame: guess − answer.", "black", 50, INK, 300, 1), ("Now three numbers, one per colour.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [(f"From step {RIGHT_FROM * SNAP_STEPS}, blue stays on top.", "black", 46, BLUE, 300, 1), ("And the blame keeps shrinking.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [(f"{FINAL_RIGHT} of {len(Y)} points right.", "black", 52, INK, 300, 1), (f"{ACC[-1]:.1%} after 1,500 steps.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("Softmax + cross-entropy:", "black", 52, INK, 300, 1), ("the blame is still guess − answer.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END + 0.1, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    def _build_bg(self):
        surf, ctx = layer()
        ctx.set_source_rgba(1, 1, 1, 0.045)
        ctx.set_line_width(1.0)
        for x in range(54, W, 54):
            ctx.move_to(x + 0.5, 0)
            ctx.line_to(x + 0.5, H)
        for y in range(54, H, 54):
            ctx.move_to(0, y + 0.5)
            ctx.line_to(W, y + 0.5)
        ctx.stroke()
        grgb, ga = surf_rgba(surf)
        c0 = np.array(rgb(BG_TOP), np.float32) ** 2.2
        c1 = np.array(rgb(BG_BOT), np.float32) ** 2.2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = np.clip(np.sqrt((xx - W / 2) ** 2 + (yy - 1000) ** 2) / 1100, 0, 1)[..., None]
        base = (c0 * (1 - u) + c1 * u) ** (1 / 2.2)
        base = base * (1 - ga)[..., None] + grgb
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        self.bg = np.ascontiguousarray(base * vig[..., None])

    # ------------------------------------------------------------ the map

    def to_panel(self, x, y):
        px, py, n = PANEL
        lim = 1.05
        return px + (x + lim) / (2 * lim) * n, py + (lim - y) / (2 * lim) * n

    def paste_field(self, img, P, a):
        px, py, n = PANEL
        up = cv2.resize(P[::-1], (n, n), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)
        up /= up.sum(2, keepdims=True) + 1e-9
        base = np.array(rgb(PANEL_BASE), np.float32)
        cols = np.array([rgb(c) for c in CLS], np.float32)
        mixc = up @ cols
        conf = np.clip((up.max(2) - 1 / 3) * 1.5, 0, 1)[..., None] * 0.82
        rgbf = base * (1 - conf) + mixc * conf
        r = np.sqrt((np.mgrid[0:n, 0:n][1] - n / 2) ** 2 + (np.mgrid[0:n, 0:n][0] - n / 2) ** 2) / (n / 2)
        rgbf *= (1 - 0.18 * np.clip(r, 0, 1.2) ** 2)[..., None]
        sub = img[py:py + n, px:px + n]
        m = (self.mask * a)[..., None]
        sub[:] = sub * (1 - m) + rgbf * m
        return up

    def draw_overlay(self, ctx, up, t, a):
        px, py, n = PANEL
        ctx.save()
        S3.pill(ctx, px, py, n, n, 18)
        ctx.clip()
        lab = up.argmax(2).astype(np.uint8)
        if float(up.max(2).mean()) > 0.4:
            for c in range(3):
                cs, _ = cv2.findContours((lab == c).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for lw, al in ((5.0, 0.12), (1.6, 0.7)):
                    ctx.set_source_rgba(*rgb(INK), al * a)
                    ctx.set_line_width(lw)
                    for cc in cs:
                        if len(cc) < 8:
                            continue
                        pts = cc[:, 0, :]
                        ctx.move_to(px + pts[0][0], py + pts[0][1])
                        for x, y in pts[1:]:
                            ctx.line_to(px + x, py + y)
                        ctx.close_path()
                        ctx.stroke()
        for (x, y), c in zip(X, Y):
            sx, sy = self.to_panel(x, y)
            ctx.set_source_rgba(*rgb(CLS[c]), 0.95 * a)
            ctx.arc(sx, sy, 3.0, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(0.02, 0.02, 0.05, 0.8 * a)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, 3.0, 0, TAU)
            ctx.stroke()
        ctx.restore()
        # the ringed point
        sx, sy = self.to_panel(*X[PROBE])
        br = 0.5 + 0.5 * math.sin(TAU * t / 1.6)
        ctx.set_source_rgba(*rgb(GOLD), 0.25 * a)
        ctx.arc(sx, sy, 20 + 4 * br, 0, TAU)
        ctx.fill()
        ctx.set_source_rgba(*rgb(GOLD), (0.75 + 0.25 * br) * a)
        ctx.set_line_width(2.6)
        ctx.arc(sx, sy, 10 + 2 * br, 0, TAU)
        ctx.stroke()
        for lw, al in ((5.0, 0.08), (1.4, 0.55)):
            ctx.set_source_rgba(*rgb(INK2), al * a)
            ctx.set_line_width(lw)
            S3.pill(ctx, px + 0.5, py + 0.5, n - 1, n - 1, 18)
            ctx.stroke()

    # ------------------------------------------------------------ the ringed point's numbers

    def draw_probe(self, ctx, img, t, a, p, z):
        """Three bars: the probabilities (or, in the blame segments, p - y), with values."""
        blame = SEG[4] <= t < SEG[6] + 0.0
        x0, y0 = 150, 1245
        bw, bh, gap = (165 if blame else 220), 18, 62
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        title = "THE RINGED POINT: BLAME  p − y" if blame else "THE RINGED POINT: PROBABILITIES"
        ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
        show_tracked(ctx, title, x0 - 50, y0 - 30, tr)
        vals = (p - YP) if blame else p
        for c in range(3):
            y = y0 + c * gap
            font(ctx, "bold", 18)
            ctx.set_source_rgba(*rgb(CLS[c]), a)
            ctx.move_to(x0 - 50, y + 14)
            ctx.show_text(CLS_NAME[c])
            bx = x0 + (255 if blame else 120)
            ctx.set_source_rgba(*rgb("#2A3148"), a)
            S3.pill(ctx, bx - (bw if blame else 0), y, bw * (2 if blame else 1), bh, 9)
            ctx.fill()
            v = float(vals[c])
            L = bw * abs(v)
            if L > 0.5:
                ctx.set_source_rgba(*rgb(CLS[c]), 0.9 * a)
                S3.pill(ctx, bx if v >= 0 else bx - L, y, max(L, bh), bh, 9)
                ctx.fill()
            if blame:
                ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
                ctx.set_line_width(1.6)
                ctx.move_to(bx, y - 6)
                ctx.line_to(bx, y + bh + 6)
                ctx.stroke()
            s = f"{v:+.2f}".replace("-", "−") if blame else f"{v:.2f}"
            font(ctx, "dbold", 28)
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(bx + bw + 24, y + 18)
            ctx.show_text(s)
        # step and accuracy on the right
        sn = snap_at(t)
        i = int(round(sn))
        for k, (lab, val) in enumerate((("TRAINING STEP", f"{i * SNAP_STEPS:,}"), ("RIGHT", f"{int(round(ACC[i] * len(Y)))} / {len(Y)}"))):
            yy = y0 + k * 92
            font(ctx, "bold", 18)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, lab, 790, yy - 6, tr)
            font(ctx, "dbold", 38)
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(790, yy + 40)
            ctx.show_text(val)

    def card(self, img, t, a, z, p):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        f2 = S3.fmt
        if SEG[2] <= t < SEG[4]:
            paste(img, r"$p_i = \frac{e^{z_i}}{e^{z_1} + e^{z_2} + e^{z_3}}$", 36, 300, y + 110, INK, a, align="c")
            e = np.exp(z)
            rows = ((r"z", z), (r"e^{z}", e), (r"p", p))
            for k, (name, v) in enumerate(rows):
                yy = y + 64 + k * 52
                paste(img, rf"${name}:$", 26, 600, yy, INK2, a, align="r")
                for c in range(3):
                    paste(img, rf"${f2(v[c])}$", 26, 680 + c * 105, yy, CLS[c], a, align="c")
            paste(img, rf"$\mathrm{{sum\ of}}\ p = {f2(p.sum())}$", 21, 540, y + h - 22, INK3, a, align="c")
        elif SEG[4] <= t < SEG[6]:
            d = p - YP
            paste(img, rf"$p - y = ({f2(p[0])},\ {f2(p[1])},\ {f2(p[2])}) - (0,\ 1,\ 0)$", 30, 540, y + 78, INK, a, align="c")
            vec = r",\ ".join(f2(v) for v in d)
            paste(img, rf"$= ({vec})$", 34, 540, y + 142, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{right\ colour\ negative:\ push\ up.\ \ Others\ positive:\ push\ down}$", 20, 540, y + 192, INK2, a, align="c")
            paste(img, r"$L = -\log p_{\mathrm{right}}\ \ \Rightarrow\ \ \partial L / \partial z = p - y$", 22, 540, y + h - 18, INK3, a, align="c")
        else:
            paste(img, r"$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$", 36, 540, y + 110, INK, a, align="c")
            note = r"$\mathrm{2\ classes:\ one\ probability\ (sigmoid).\ \ 3\ classes:\ three\ (softmax)}$" if (t < SEG[2] or t >= T_END - 0.3) else \
                r"$\mathrm{the\ blame\ at\ the\ output\ is\ still}\ p - y$"
            paste(img, note, 21, 540, y + h - 24, INK2, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        img = self.bg.copy()
        sn = snap_at(t)
        P = lerp_snap(FIELDS, sn)
        z = lerp_snap(PZ, sn)
        p = softmax(z)
        up = self.paste_field(img, P, a)
        surf, ctx = layer()
        self.draw_overlay(ctx, up, t, a)
        self.draw_probe(ctx, img, t, a, p, z)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "450 POINTS, 3 CLASSES   ·   2 → 16 → 16 → 3, ReLU, SOFTMAX   ·   ADAM, MEDIAN OF 6 SEEDS"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.card(img, t, a, z, p)
        surf, ctx = layer()
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

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
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims) - len(ims) % 6, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
