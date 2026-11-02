"""Why gradient descent zig-zags: the condition number. The 3D stage of reels 38-44 (stage3d.py), narrated by Kokoro
(voicekit.py).

A bowl f(x, y) = (mu x^2 + y^2) / 2, stretched along x (mu = 1 / kappa; its Hessian has eigenvalues mu and 1, so
kappa is its condition number). Gradient descent from the same start, each time with its best step size (zzsim.py):
1 step on the round bowl, 29 when stretched 10 times, 284 when stretched 100 times, bouncing across the valley and
crawling along it. Then momentum (23 steps) and Adam (29 steps, the same on all three bowls) on the 100x bowl. Every
path and count is the real run. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
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

# ---------------------------------------------------------------- the runs (zzsim.py) and every number the reel states
with open(os.path.join(HERE, "zz_results.json")) as fh:
    RUNS = json.load(fh)
PATH = {k: np.array(v["path"]) for k, v in RUNS.items()}
STEPS = {k: v["steps"] for k, v in RUNS.items()}
LR = {k: v["lr"] for k, v in RUNS.items()}
BETA_M = RUNS["momentum_100"]["beta"]
METHOD_C = {"gd": GOLD, "momentum": MAGENTA, "adam": GREEN}
METHOD_NAME = {"gd": "GRADIENT DESCENT", "momentum": "MOMENTUM", "adam": "ADAM"}
X0 = PATH["gd_1"][0]

XR, YR = 3.2, 2.0                                  # the patch of the bowl drawn
ZS = 0.26                                          # height scale
Z_BASE = -0.6
CARD = (70, 1430, 940, 250)


def f_bowl(x, y, kappa):
    return 0.5 * (x * x / kappa + y * y)


SCRIPT = [
    f"Same bowl, just stretched. Gradient descent goes from one step to {STEPS['gd_100']}.",
    "A round bowl. With the right step size, gradient descent lands on the bottom in one step.",
    f"Stretch it ten times along one direction. Even with its best step size, it now takes {STEPS['gd_10']} steps, and it starts to zig-zag.",
    f"Stretch it a hundred times: {STEPS['gd_100']} steps. It bounces between the steep walls, and crawls along the flat floor.",
    "The curvatures across and along the valley are the Hessian's eigenvalues: one, and one hundredth. Their ratio is the condition number. "
    "The step has to suit the steepest direction, so along the floor it barely moves.",
    f"Momentum remembers its last step, and builds up speed along the floor: {STEPS['momentum_100']} steps.",
    f"Adam rescales each direction separately: {STEPS['adam_100']} steps, on every one of these bowls, round or stretched.",
    "Steepest over flattest curvature: the condition number. The bigger it is, the more plain gradient descent zig-zags.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# ---------------------------------------------------------------- the timeline, sized to the narration
# 0 hook, 1 round, 2 x10, 3 x100, 4 why, 5 momentum, 6 Adam, 7 close
SEG_MIN = [4.0, 6.5, 8.5, 10.0, 11.0, 7.5, 8.0, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[8]
N_FRAMES = int(round((T_END + 0.7) * FPS))
# the bowl's stretch over time: 100 in the opening, 1, 10, 100, and 100 to the end
MORPH = [(SEG[1] - 0.1, SEG[1] + 1.4, 100, 1), (SEG[2] + 0.1, SEG[2] + 1.6, 1, 10), (SEG[3] + 0.1, SEG[3] + 1.6, 10, 100)]
# the runs shown: (key, start, end) of the ball's journey
RUN_T = {"gd_1": (SEG[1] + 3.4, SEG[1] + 4.5), "gd_10": (SEG[2] + 2.2, SEG[2] + 6.6), "gd_100": (SEG[3] + 2.0, SEG[3] + 8.8),
         "momentum_100": (SEG[5] + 0.8, SEG[5] + 4.6), "adam_100": (SEG[6] + 0.8, SEG[6] + 5.0)}


def kappa_at(t):
    k = 100.0
    for t0, t1, a, b in MORPH:
        if t < t0:
            break
        u = smooth(seg(t, t0, t1))
        k = math.exp(math.log(a) + (math.log(b) - math.log(a)) * u)
    return k


def progress(key, t):
    """Steps taken (float) of a run at time t."""
    t0, t1 = RUN_T[key]
    n = STEPS[key]
    u = seg(t, t0, t1)
    if key == "gd_100":
        u = u ** 1.5                                  # slower at first, so the first bounces are seen, then faster
    return n * smooth(u) if key != "gd_1" else n * smooth(u)


def path_vis(key, t):
    """How visible a run's path is: built while it runs, kept, cleared when the bowl changes again."""
    t0, _ = RUN_T[key]
    show = 1.0 if t >= t0 - 0.2 else 0.0
    if key == "gd_1":
        show *= 1 - smooth(seg(t, MORPH[1][0] - 0.3, MORPH[1][0] + 0.3))
    elif key == "gd_10":
        show *= 1 - smooth(seg(t, MORPH[2][0] - 0.3, MORPH[2][0] + 0.3))
    elif key in ("momentum_100", "adam_100"):
        show *= 1 - smooth(seg(t, SEG[7] + 1.5, SEG[7] + 3.5))
    return show


def gd100_vis(t):
    """The 284-step path: drawn in full in the opening (the loop), cleared, run again, faint under the others."""
    if t < SEG[1]:
        return 1.0, STEPS["gd_100"]
    if t < MORPH[0][0] + 0.6:
        return 1 - smooth(seg(t, SEG[1] - 0.1, MORPH[0][0] + 0.6)), STEPS["gd_100"]
    t0, _ = RUN_T["gd_100"]
    if t < t0 - 0.2:
        return 0.0, 0
    faint = 1 - 0.65 * smooth(seg(t, SEG[5] + 0.2, SEG[5] + 0.8)) * (1 - smooth(seg(t, SEG[7] + 1.5, SEG[7] + 3.0)))
    return faint, progress("gd_100", t)


def qa_frames():
    fr = []
    for k in range(8):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(runs={k: dict(lr=RUNS[k]["lr"], beta=RUNS[k]["beta"], steps=STEPS[k]) for k in RUNS}, start=X0.tolist(), tol=0.01,
                          segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG
        nx, ny = 44, 26
        self.gx = np.linspace(-XR, XR, nx + 1)
        self.gy = np.linspace(-YR, YR, ny + 1)

    def _cues(self):
        n = STEPS
        hook = [("Same bowl. Stretched 100×.", "black", 50, INK, 300, 1), (f"Gradient descent: 1 step → {n['gd_100']}.", "dbold", 40, GOLD, 390, 0.95)]
        S_ = SEG
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [(f"A round bowl: {n['gd_1']} step.", "black", 50, INK, 300, 1), ("With the right step size.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [(f"Stretched 10×: {n['gd_10']} steps.", "black", 50, INK, 300, 1), ("Its best step size. It starts to zig-zag.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"Stretched 100×: {n['gd_100']} steps.", "black", 50, GOLD, 300, 1), ("Bouncing across, crawling along.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Curvatures: 1 and 0.01.", "black", 50, INK, 300, 1), ("Their ratio, 100: the condition number.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [(f"Momentum: {n['momentum_100']} steps.", "black", 50, MAGENTA, 300, 1), ("Speed builds up along the floor.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [(f"Adam: {n['adam_100']} steps.", "black", 50, GREEN, 300, 1), (f"The same {n['adam_1']} on every one of these bowls.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("The condition number.", "black", 50, INK, 300, 1), ("The bigger it is, the more it zig-zags.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END - 0.3, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ camera and the bowl

    def camera(self, t):
        L = N_FRAMES / FPS
        az = -90 + 9 * math.sin(TAU * t / L)
        el = 38 + 3 * math.sin(2 * TAU * t / L)
        return S3.Camera(np.array([0.0, 0.0, Z_BASE + 0.45]), az, el, dist=10.8, focal=1330.0, centre=(540, 840))

    def P3(self, x, y, kappa, lift=0.0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.stack([x, y, Z_BASE + ZS * f_bowl(x, y, kappa) + lift], -1)

    def draw_bowl(self, ctx, cam, kappa, a):
        GX, GY = np.meshgrid(self.gx, self.gy)
        V = self.P3(GX, GY, kappa)
        P = cam.project(V.reshape(-1, 3)).reshape(V.shape[0], V.shape[1], 2)
        D = cam.depth(V.reshape(-1, 3)).reshape(V.shape[0], V.shape[1])
        items = []
        ny, nx = V.shape[0] - 1, V.shape[1] - 1
        for j in range(ny):
            for i in range(nx):
                items.append((D[j:j + 2, i:i + 2].mean(), j, i))
        items.sort(key=lambda q: -q[0])
        base = S3.BASE
        for _, j, i in items:
            q = [V[j, i], V[j, i + 1], V[j + 1, i + 1], V[j + 1, i]]
            n = np.cross(q[2] - q[0], q[3] - q[1])
            n /= np.linalg.norm(n) + 1e-12
            if np.dot(n, cam.eye - q[0]) < 0:
                n = -n
            lam = max(0.0, float(np.dot(n, S3.LIGHT)))
            col = np.clip(base * (0.32 + 0.6 * lam) + 0.05, 0, 1)
            pts = [P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]]
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source_rgba(*col, 0.82 * a)
            ctx.fill_preserve()
            ctx.set_line_width(0.7)
            ctx.stroke()
            hi = np.clip(base * 0.5 + 0.55, 0, 1)
            ctx.set_source_rgba(*hi, 0.20 * a)
            ctx.set_line_width(1.0)
            if i % 4 == 0:
                ctx.move_to(*pts[0])
                ctx.line_to(*pts[3])
                ctx.stroke()
            if j % 2 == 0:
                ctx.move_to(*pts[0])
                ctx.line_to(*pts[1])
                ctx.stroke()

    def draw_run(self, ctx, cam, key, kappa, upto, vis, a, ball=True):
        if vis <= 0.01:
            return None
        path = PATH[key]
        n = int(math.floor(upto))
        fr = upto - n
        pts = path[:n + 1]
        if n < len(path) - 1 and fr > 0:
            pts = np.vstack([pts, path[n] + (path[n + 1] - path[n]) * fr])
        # the path as straight hops between the steps, drawn over the surface
        hops = []
        for k in range(len(pts) - 1):
            seg_pts = np.linspace(pts[k], pts[k + 1], 6)
            hops.append(seg_pts)
        col = METHOD_C[key.split("_")[0]]
        ctx.set_source_rgba(*rgb(col), 0.9 * vis * a)
        ctx.set_line_width(2.4)
        for hp in hops:
            pp = cam.project(self.P3(hp[:, 0], hp[:, 1], kappa, 0.03))
            ctx.move_to(*pp[0])
            for p in pp[1:]:
                ctx.line_to(*p)
            ctx.stroke()
        steps_pp = cam.project(self.P3(path[:n + 1, 0], path[:n + 1, 1], kappa, 0.03))
        ctx.set_source_rgba(*rgb(col), vis * a)
        for p in steps_pp:
            ctx.arc(*p, 3.0, 0, TAU)
            ctx.fill()
        if ball:
            bp = cam.project(self.P3(pts[-1][0], pts[-1][1], kappa, 0.06))[0]
            ctx.set_source_rgba(0.04, 0.05, 0.1, 0.5 * vis * a)
            ctx.arc(bp[0] + 3, bp[1] + 4, 12, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(1, 1, 1, vis * a)
            ctx.arc(*bp, 11, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), vis * a)
            ctx.set_line_width(3.5)
            ctx.arc(*bp, 11, 0, TAU)
            ctx.stroke()
        return n

    def draw_world(self, img, t, a):
        cam = self.camera(t)
        kappa = kappa_at(t)
        S3.floor_light(img, cam, Z_BASE - 0.15, a, radius=4.2)
        surf, ctx = layer()
        self.draw_bowl(ctx, cam, kappa, a)
        # the bottom
        bt = cam.project(self.P3(0.0, 0.0, kappa, 0.02))[0]
        ctx.set_source_rgba(1, 1, 1, 0.85 * a)
        ctx.set_line_width(1.6)
        ctx.arc(*bt, 7, 0, TAU)
        ctx.stroke()
        # the start
        st = cam.project(self.P3(X0[0], X0[1], kappa, 0.03))[0]
        ctx.set_source_rgba(*rgb(INK), 0.8 * a)
        ctx.arc(*st, 5, 0, TAU)
        ctx.fill()
        counts = {}
        v100, up100 = gd100_vis(t)
        for key in ("gd_1", "gd_10"):
            if t >= RUN_T[key][0] - 0.2 and path_vis(key, t) > 0.01:
                counts[key] = self.draw_run(ctx, cam, key, kappa, progress(key, t), path_vis(key, t), a)
        if v100 > 0.01:
            counts["gd_100"] = self.draw_run(ctx, cam, "gd_100", kappa, up100, v100, a, ball=t < SEG[5] + 0.5 or t < SEG[1])
        for key in ("momentum_100", "adam_100"):
            if t >= RUN_T[key][0] - 0.2 and path_vis(key, t) > 0.01:
                counts[key] = self.draw_run(ctx, cam, key, kappa, progress(key, t), path_vis(key, t), a)
        # the two curvatures, in the explanation: across (steep) and along (flat)
        self.curv = None
        cv = ease_out_cubic(seg(t, SEG[4] + 0.6, SEG[4] + 1.6)) * (1 - smooth(seg(t, SEG[5] - 0.6, SEG[5])))
        if cv > 0.01:
            for xs, ys, col in ((np.zeros(30), np.linspace(-YR, YR, 30), VERM), (np.linspace(-XR, XR, 30), np.zeros(30), CYAN)):
                pp = cam.project(self.P3(xs, ys, kappa, 0.025))
                ctx.set_source_rgba(*rgb(col), 0.95 * cv * a)
                ctx.set_line_width(3.4)
                ctx.move_to(*pp[0])
                for p in pp[1:]:
                    ctx.line_to(*p)
                ctx.stroke()
            self.curv = (cam.project(self.P3(0.0, YR, kappa, 0.05))[0], cam.project(self.P3(XR, 0.0, kappa, 0.05))[0], cv * a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return kappa, counts

    def labels(self, img, t, a):
        if self.curv is not None:
            p_across, p_along, al = self.curv
            paste(img, r"$\mathrm{across:\ curvature}\ 1$", 24, p_across[0], p_across[1] - 18, VERM, al, align="c", halo=4)
            paste(img, r"$\mathrm{along:\ curvature}\ 0.01$", 24, min(p_along[0], 900), p_along[1] + 40, CYAN, al, align="c", halo=4)

    def readouts(self, img, t, a, kappa, counts):
        surf, ctx = layer()
        font(ctx, "bold", 17)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "STRETCH  κ", 110, 1222, 0.12 * 17)
        font(ctx, "dbold", 46)
        ctx.set_source_rgba(*rgb(INK), a)
        ctx.move_to(110, 1278)
        ctx.show_text(f"{kappa:.0f}" if abs(kappa - round(kappa)) < 0.05 else f"{kappa:.1f}")
        rows = []
        if "gd_1" in counts and t < MORPH[1][0]:
            rows.append(("gd", counts["gd_1"]))
        elif "gd_10" in counts and t < MORPH[2][0]:
            rows.append(("gd", counts["gd_10"]))
        elif "gd_100" in counts:
            rows.append(("gd", counts["gd_100"]))
        for key in ("momentum_100", "adam_100"):
            if key in counts and path_vis(key, t) > 0.3:
                rows.append((key.split("_")[0], counts[key]))
        for k, (m, n) in enumerate(rows):
            yy = 1214 + k * 62
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*rgb(METHOD_C[m]), 0.95 * a)
            show_tracked(ctx, METHOD_NAME[m], 430, yy, 0.12 * 17)
            font(ctx, "dbold", 34)
            ctx.set_source_rgba(*rgb(INK), a)
            txt = f"{n} step" + ("" if n == 1 else "s")
            ctx.move_to(1000 - ctx.text_extents(txt).x_advance, yy + 4)
            ctx.show_text(txt)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a, kappa):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        g = STEPS
        if t < SEG[1] or t >= SEG[7]:
            paste(img, r"$f(x, y) = \frac{1}{2}\left(\frac{x^2}{\kappa} + y^2\right)$", 36, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\kappa = 1:\ {g['gd_1']}\ \mathrm{{step}}\qquad \kappa = 10:\ {g['gd_10']}\qquad \kappa = 100:\ {g['gd_100']}$", 28, 540, y + 170, GOLD, a, align="c")
            paste(img, r"$\mathrm{best\ step\ size\ per\ bowl;\ steps\ until\ within\ 0.01\ of\ the\ bottom}$", 20, 540, y + h - 16, INK3, a, align="c")
        elif t < SEG[4]:
            k = 1 if t < SEG[2] else (10 if t < SEG[3] else 100)
            key = f"gd_{k}"
            paste(img, rf"$x_{{k+1}} = x_k - \eta\,\nabla f(x_k)$", 36, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\kappa = {k}:\ \ \mathrm{{best\ step}}\ \eta = {LR[key]:.2f}\ \ \rightarrow\ \ {g[key]}\ \mathrm{{step{'s' if g[key] != 1 else ''}}}$", 28, 540, y + 168, GOLD, a, align="c")
            paste(img, r"$\mathrm{steps\ until\ within\ 0.01\ of\ the\ bottom;\ step\ sizes\ tried:\ 0.01\ to\ 1.99}$", 20, 540, y + h - 16, INK3, a, align="c")
        elif t < SEG[5]:
            paste(img, r"$\lambda_{\max} = 1,\ \ \lambda_{\min} = 0.01\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = 100$", 32, 540, y + 92, INK, a, align="c")
            paste(img, rf"$\eta < 2 / \lambda_{{\max}}\ \Rightarrow\ \mathrm{{along\ the\ floor,\ each\ step\ closes}}\ \eta\,\lambda_{{\min}} \approx {100 * LR['gd_100'] * 0.01:.0f}\%$", 24, 540, y + 162, INK2,
                  a * ease_out_cubic(seg(t, SEG[4] + 8.6, SEG[4] + 9.4)), align="c")
            paste(img, r"$\mathrm{the\ Hessian's\ eigenvalues:\ the\ curvatures\ across\ and\ along\ the\ valley}$", 20, 540, y + h - 16, INK3, a, align="c")
        elif t < SEG[6]:
            paste(img, r"$v_{k+1} = \beta\,v_k - \eta\,\nabla f(x_k),\qquad x_{k+1} = x_k + v_{k+1}$", 29, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\mathrm{{best}}\ \eta = {LR['momentum_100']:.2f},\ \beta = {BETA_M:.2f}\ \ \rightarrow\ \ {g['momentum_100']}\ \mathrm{{steps}}\ \ (\kappa = 100)$", 28, 540, y + 166, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{the\ velocity\ adds\ up\ along\ the\ floor\ and\ cancels\ out\ across\ it}$", 20, 540, y + h - 16, INK3, a, align="c")
        else:
            paste(img, r"$x_{k+1} = x_k - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}$", 36, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\mathrm{{best}}\ \eta = {LR['adam_100']:.2f}:\ \ \kappa = 1,\ 10,\ 100\ \rightarrow\ {g['adam_1']},\ {g['adam_10']},\ {g['adam_100']}\ \mathrm{{steps}}$", 28, 540, y + 166, GREEN, a, align="c")
            paste(img, r"$\mathrm{each\ direction\ scaled\ on\ its\ own\ (the\ valley\ runs\ along\ an\ axis)}$", 20, 540, y + h - 16, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        kappa, counts = self.draw_world(img, t, a)
        self.labels(img, t, a)
        self.readouts(img, t, a, kappa, counts)
        self.card(img, t, a, kappa)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "SAME START   ·   BEST SETTINGS FOR EACH   ·   REAL RUNS"
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
