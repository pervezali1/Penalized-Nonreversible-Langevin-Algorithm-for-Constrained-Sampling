"""A network is a bunch of hinges, on the dark theme of reels 1-22 and 31-35. One hidden layer of ReLU
neurons fitted to one curve (hgsim.py) with 1, 2, 4, 8, 16 and 32 neurons, each trained live from its
snapshots: the target dashed, the network's output in gold with a dot at every hinge that lands inside
the plot; below it the pieces, one bent line per neuron (its output weight times max(0, w x + b),
vermilion for a positive output weight, blue for negative), whose sum plus a constant is the gold curve;
readouts for neurons, hinges inside and the error. No formulas on screen. 50 s. frame(f) -> uint8 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba,  # noqa: E402
                     tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 3000                       # 50 s
LOOP_FADE = (49.3, 49.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6"
GOLD = "#F6C56E"
seg = gds.seg
TAU = 2 * math.pi
BG_TOP, BG_BOT = "#0E1226", "#05060C"
GRID_C, GRID_A = "#FFFFFF", 0.045
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"
NPZ = os.path.join(HERE, "hinges_sim.npz")

MAIN = (60, 480, 960, 600)            # x, y, w, h of the main plot
PIECES = (60, 1120, 960, 330)         # the pieces
Y_RANGE = (-1.05, 0.85)
WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "Thirty-two"}
STAGES = [dict(N=1, t0=2.6, train=(1.4, 4.6)), dict(N=2, t0=10.6, train=(0.9, 3.6)), dict(N=4, t0=16.4, train=(0.9, 3.6)),
          dict(N=8, t0=22.2, train=(0.9, 3.6)), dict(N=16, t0=28.0, train=(0.9, 3.6)), dict(N=32, t0=33.8, train=(0.9, 4.0))]
T_CLOSE = 40.6


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


class Scene(gds.Scene):
    def __init__(self):
        if not os.path.exists(NPZ):
            import hgsim
            hgsim.main(NPZ)
        d = np.load(NPZ)
        self.x, self.yt = d["x"], d["y"]
        self.xs = np.linspace(-1, 1, 721)                # drawing resolution
        self.yts = np.sin(2 * np.pi * self.xs) * (0.55 + 0.45 * self.xs)
        self.stages = []
        for i, st in enumerate(STAGES):
            N = st["N"]
            s = dict(st, snaps=d[f"n{N}_snaps"], seed=int(d[f"n{N}_seed"]), finals=d[f"n{N}_finals"], rms=float(d[f"n{N}_rms"]))
            s["t1"] = STAGES[i + 1]["t0"] if i + 1 < len(STAGES) else T_CLOSE
            s["n_snap"] = len(s["snaps"])
            p = self.params(s, s["n_snap"] - 1)
            hx = -p[1] / p[0]
            s["inside"] = int(np.sum((hx > -1) & (hx < 1)))
            # the pieces' scale for this stage: the largest piece anywhere in the final fit
            h = np.maximum(self.xs[:, None] * p[0] + p[1], 0) * p[2]
            s["pscale"] = max(0.5, float(np.abs(h).max()))
            self.stages.append(s)
        self.facts = {s["N"]: dict(rms=round(s["rms"], 4), inside=s["inside"], seed=s["seed"], finals=[round(float(v), 4) for v in s["finals"]]) for s in self.stages}
        self.cues = self._cues()
        self._build_grid()

    # -------------------------------------------------------------- parameters

    def params(self, s, sn):
        N = s["N"]
        i0 = int(math.floor(sn))
        i1 = min(i0 + 1, s["n_snap"] - 1)
        k = sn - i0
        v = s["snaps"][i0] * (1 - k) + s["snaps"][i1] * k
        return v[:N], v[N:2 * N], v[2 * N:3 * N], v[3 * N]

    def stage_at(self, t):
        if t >= T_CLOSE - 0.3 or t < STAGES[0]["t0"] - 0.3:
            return self.stages[-1], self.stages[-1]["n_snap"] - 1.0, 99.0
        cur = None
        for s in self.stages:
            if t >= s["t0"] - 0.3:
                cur = s
        tl = t - cur["t0"]
        a, b = cur["train"]
        u = seg(tl, a, b)
        u = 1 - (1 - u) ** 2.2
        return cur, u * (cur["n_snap"] - 1), tl

    # -------------------------------------------------------------- cues (every number from hgsim.py)

    def _cues(self):
        S = {s["N"]: s for s in self.stages}
        hook = [("Every ReLU neuron is a hinge.", "black", 48, INK, 300, 1), ("Add enough hinges and you can draw any curve.", "black", 38, INK2, 390, 1)]

        def tr(N):
            return S[N]["t0"] + S[N]["train"][1] + 0.3

        def ins(N):
            k = S[N]["inside"]
            return f"{k} hinge{'s' if k != 1 else ''} inside"

        cues = [(0.0, 2.4, hook, 0, 8, "none"),
                (S[1]["t0"], tr(1) - 0.3, [("One neuron: flat, then a straight ramp.", "black", 42, INK, 300, 1), ("One bend: a hinge. Dashed is the target, gold is the network.", "dbold", 33, INK2, 390, 0.95)], 10, 8, "pop"),
                (tr(1), S[2]["t0"] - 0.3, [(f"Trained: error {S[1]['rms']:.3f}.", "black", 46, VERM, 300, 1), ("One bend can't follow a wave.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop")]
        for a, b in ((2, 4), (4, 8), (8, 16), (16, 32)):
            s = S[a]
            first = f"{WORDS[a]} neurons: {WORDS[a].lower()} hinges, added." if s["inside"] == a else f"{WORDS[a]} neurons, but only {ins(a)}."
            second = {2: self.dead_note(S[2]), 4: "The pieces below add up to the gold curve.",
                      8: "Now it can follow the wave.", 16: "Each hinge bends the curve a little."}[a]
            cues.append((s["t0"], tr(a) - 0.3, [(first, "black", 40, INK, 300, 1), (second, "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"))
            prev = S[{2: 1, 4: 2, 8: 4, 16: 8}[a]]["rms"]
            note = {2: "No better than one neuron." if abs(s["rms"] - prev) < 5e-4 else "Barely better.", 4: "Training found a poor fit.", 8: "A big jump.", 16: "Closer."}[a]
            cues.append((tr(a), S[b]["t0"] - 0.3, [(f"Error {s['rms']:.3f}.", "black", 46, VERM, 300, 1), (note, "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"))
        s = S[32]
        cues.append((s["t0"], tr(32) - 0.3, [("Thirty-two hinges.", "black", 48, INK, 300, 1), (f"{ins(32).capitalize()}: many short straight pieces.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"))
        cues.append((tr(32), T_CLOSE - 0.3, [(f"Error {s['rms']:.3f}.", "black", 46, VERM, 300, 1), ("You can barely tell the gold from the dashed.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"))
        cues.append((T_CLOSE, 49.0, [("One layer of hinges can draw any curve,", "black", 42, INK, 300, 1), ("if it is wide enough. That is the universal approximation theorem.", "dbold", 32, INK2, 390, 0.95)], 10, 8, "pop"))
        cues.append((49.4, 50.1, hook, 12, 0, "fade"))
        return cues

    def dead_note(self, s):
        """What the neurons whose hinge lands outside the plot do there: always on (a straight tilt) or always off (nothing)."""
        W1, b1, W2, b2 = self.params(s, s["n_snap"] - 1)
        on = np.mean(np.maximum(self.x[:, None] * W1 + b1, 0) > 0, axis=0)
        hx = -b1 / W1
        out = (hx <= -1) | (hx >= 1)
        dead = int(np.sum(out & (on == 0)))
        tilt = int(np.sum(out & (on == 1)))
        s["dead"], s["tilt"] = dead, tilt
        if dead and not tilt:
            return "The other is off everywhere here: it adds nothing." if dead == 1 else f"The other {dead} are off everywhere: they add nothing."
        if tilt and not dead:
            return "The other lands outside: it only tilts the line."
        return f"{dead} off everywhere, {tilt} only tilting the line."

    # -------------------------------------------------------------- drawing

    def to_main(self, x, y):
        mx, my, mw, mh = MAIN
        return mx + 30 + (x + 1) / 2 * (mw - 60), my + 20 + (Y_RANGE[1] - y) / (Y_RANGE[1] - Y_RANGE[0]) * (mh - 40)

    def draw_main(self, ctx, s, sn, a):
        mx, my, mw, mh = MAIN
        W1, b1, W2, b2 = self.params(s, sn)
        xs = self.xs
        ys = np.maximum(xs[:, None] * W1 + b1, 0) @ W2 + b2
        # axes: y = 0 and the frame of x in [-1, 1]
        ctx.set_source_rgba(*rgb(INK3), 0.45 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(*self.to_main(-1, 0))
        ctx.line_to(*self.to_main(1, 0))
        ctx.stroke()
        for xv in (-1, 1):
            ctx.set_source_rgba(*rgb(INK3), 0.25 * a)
            ctx.move_to(*self.to_main(xv, Y_RANGE[0]))
            ctx.line_to(*self.to_main(xv, Y_RANGE[1]))
            ctx.stroke()
        # the target, dashed
        ctx.set_source_rgba(*rgb(INK), 0.65 * a)
        ctx.set_line_width(2.2)
        ctx.set_dash([9.0, 7.0])
        for i, (x, y) in enumerate(zip(xs, self.yts)):
            (ctx.move_to if i == 0 else ctx.line_to)(*self.to_main(x, y))
        ctx.stroke()
        ctx.set_dash([])
        # the network: gold, with a glow; clipped to the plot
        ctx.save()
        pill(ctx, mx, my, mw, mh, 18)
        ctx.clip()
        ys_c = np.clip(ys, Y_RANGE[0] - 0.5, Y_RANGE[1] + 0.5)
        for lw, al in ((14.0, 0.10), (7.0, 0.22), (3.4, 1.0)):
            ctx.set_source_rgba(*rgb(GOLD), al * a)
            ctx.set_line_width(lw)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            for i, (x, y) in enumerate(zip(xs, ys_c)):
                (ctx.move_to if i == 0 else ctx.line_to)(*self.to_main(x, y))
            ctx.stroke()
        # the hinges: a dot on the curve, and a tick on the axis
        hx = -b1 / np.where(np.abs(W1) < 1e-9, 1e-9, W1)
        for j in range(len(W1)):
            if not (-1 < hx[j] < 1):
                continue
            yv = float(np.maximum(hx[j] * W1 + b1, 0) @ W2 + b2)
            px, py = self.to_main(hx[j], yv)
            col = VERM if W2[j] >= 0 else BLUE
            ctx.set_source_rgba(*rgb(col), 0.30 * a)
            ctx.arc(px, py, 11, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb("#05060C"), a)
            ctx.arc(px, py, 5.5, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.set_line_width(2.2)
            ctx.arc(px, py, 5.5, 0, TAU)
            ctx.stroke()
            tx, ty = self.to_main(hx[j], 0)
            ctx.set_source_rgba(*rgb(col), 0.8 * a)
            ctx.set_line_width(2.0)
            ctx.move_to(tx, ty - 7)
            ctx.line_to(tx, ty + 7)
            ctx.stroke()
        ctx.restore()
        # frame and labels
        for lw, al in ((5.0, 0.08), (1.3, 0.5)):
            ctx.set_source_rgba(*rgb(INK2), al * a)
            ctx.set_line_width(lw)
            pill(ctx, mx + 0.5, my + 0.5, mw - 1, mh - 1, 18)
            ctx.stroke()
        font(ctx, "bold", 19)
        tr = 0.12 * 19
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        show_tracked(ctx, "THE TARGET (DASHED) AND THE NETWORK (GOLD)", mx + 26, my - 16, tr)
        lw_ = tracked_width(ctx, "● HINGE", tr)
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        show_tracked(ctx, "● HINGE", mx + mw - 26 - lw_, my - 16, tr)
        return ys

    def draw_pieces(self, ctx, s, sn, a):
        px0, py0, pw, ph = PIECES
        W1, b1, W2, b2 = self.params(s, sn)
        xs = self.xs
        sc = s["pscale"]
        ctx.set_source_rgba(*rgb(CARD_FILL), 0.6 * a)
        pill(ctx, px0, py0, pw, ph, 18)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(CARD_EDGE), 0.8 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()

        def P(x, y):
            return px0 + 30 + (x + 1) / 2 * (pw - 60), py0 + ph / 2 - y / sc * (ph / 2 - 26)

        ctx.set_source_rgba(*rgb(INK3), 0.45 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(*P(-1, 0))
        ctx.line_to(*P(1, 0))
        ctx.stroke()
        ctx.save()
        pill(ctx, px0, py0, pw, ph, 18)
        ctx.clip()
        N = len(W1)
        lw = 3.0 if N <= 4 else (2.2 if N <= 16 else 1.6)
        for j in range(N):
            h = np.maximum(xs * W1[j] + b1[j], 0) * W2[j]
            col = VERM if W2[j] >= 0 else BLUE
            for w_, al in ((lw * 3, 0.12), (lw, 0.9 if N <= 8 else 0.75)):
                ctx.set_source_rgba(*rgb(col), al * a)
                ctx.set_line_width(w_)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                for i, (x, y) in enumerate(zip(xs, h)):
                    (ctx.move_to if i == 0 else ctx.line_to)(*P(x, y))
                ctx.stroke()
        ctx.restore()
        font(ctx, "bold", 19)
        tr = 0.12 * 19
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        lab = f"THE PIECES: ONE BENT LINE PER NEURON, {N} IN ALL"
        show_tracked(ctx, lab, px0 + 26, py0 - 16, tr)
        font(ctx, "regular", 18)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        s_ = f"scale ±{sc:.1f}"
        ctx.move_to(px0 + pw - 24 - ctx.text_extents(s_).x_advance, py0 + 30)
        ctx.show_text(s_)

    def readouts(self, ctx, s, sn, a, ys_draw):
        W1, b1, W2, b2 = self.params(s, sn)
        y = np.maximum(self.x[:, None] * W1 + b1, 0) @ W2 + b2
        rms = float(np.sqrt(np.mean((y - self.yt) ** 2)))
        hx = -b1 / np.where(np.abs(W1) < 1e-9, 1e-9, W1)
        inside = int(np.sum((hx > -1) & (hx < 1)))
        items = (("NEURONS", f"{s['N']}"), ("HINGES INSIDE", f"{inside}"), ("ERROR", f"{rms:.3f}"))
        Y_LAB, Y_VAL = 1520, 1584
        for (lab, val), xc in zip(items, (190, 540, 890)):
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 50)
            w = ctx.text_extents(val).x_advance
            if lab == "ERROR":
                for dx, dy in ((-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)):
                    ctx.set_source_rgba(*rgb(VERM), 0.18 * a)
                    ctx.move_to(xc - w / 2 + dx, Y_VAL + dy)
                    ctx.show_text(val)
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(xc - w / 2, Y_VAL)
            ctx.show_text(val)
        # the error bar: log scale from 1 down to 0.001
        fr = (math.log10(max(rms, 1e-3)) - 0) / (-3)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(*rgb("#2A3148"), a)
        ctx.set_line_width(6)
        ctx.move_to(780, 1610)
        ctx.line_to(1000, 1610)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(VERM), a)
        ctx.move_to(780, 1610)
        ctx.line_to(780 + 220 * max(0.01, min(1.0, 1 - fr)), 1610)
        ctx.stroke()

    def close_table(self, ctx, t, a):
        """At the end: the error for every width, as a staircase."""
        if a <= 0:
            return
        x0, y0, w, h = 120, 1120, 840, 330
        ctx.set_source_rgba(*rgb(CARD_FILL), 0.85 * a)
        pill(ctx, x0 - 60, y0, w + 120, h, 18)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(CARD_EDGE), 0.9 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        font(ctx, "bold", 19)
        tr = 0.12 * 19
        ctx.set_source_rgba(*rgb(INK2), 0.9 * a)
        show_tracked(ctx, "ERROR FOR EACH WIDTH (MEDIAN OF 6 SEEDS)", x0 - 34, y0 - 16, tr)
        n = len(self.stages)
        bw = w / n
        for k, s in enumerate(self.stages):
            ka = ease_out_cubic(seg(t, T_CLOSE + 0.2 + 0.15 * k, T_CLOSE + 0.6 + 0.15 * k)) * a
            if ka <= 0:
                continue
            cx = x0 + bw * (k + 0.5)
            frac = s["rms"] / self.stages[0]["rms"]
            top = y0 + 70 + (1 - frac) * (h - 150)
            ctx.set_source_rgba(*rgb(VERM), 0.25 * ka)
            pill(ctx, cx - 30, top, 60, y0 + h - 80 - top + 1, 8)
            ctx.fill()
            ctx.set_source_rgba(*rgb(VERM), 0.95 * ka)
            pill(ctx, cx - 30, top, 60, max(6, y0 + h - 80 - top + 1), 8)
            ctx.set_line_width(2)
            ctx.stroke()
            font(ctx, "dbold", 26)
            v = f"{s['rms']:.3f}"
            ctx.set_source_rgba(*rgb(INK), ka)
            ctx.move_to(cx - ctx.text_extents(v).x_advance / 2, top - 14)
            ctx.show_text(v)
            font(ctx, "bold", 22)
            lab = f"{s['N']}"
            ctx.set_source_rgba(*rgb(INK2), ka)
            ctx.move_to(cx - ctx.text_extents(lab).x_advance / 2, y0 + h - 42)
            ctx.show_text(lab)
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        lab = "NEURONS IN THE LAYER"
        lw = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - lw / 2, y0 + h - 14, tr)

    def footer(self, ctx, a):
        if a <= 0:
            return
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "ONE HIDDEN LAYER OF ReLU NEURONS   ·   ADAM, 6,000 STEPS   ·   MEDIAN OF 6 SEEDS"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        lab = "ERROR: ROOT-MEAN-SQUARE GAP TO THE TARGET OVER 256 POINTS"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1788, tr)

    # -------------------------------------------------------------- background and frame

    def _build_grid(self):
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(GRID_C), GRID_A)
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
        c0 = np.array(rgb(BG_TOP), np.float32) ** 2.2
        c1 = np.array(rgb(BG_BOT), np.float32) ** 2.2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = np.clip(np.sqrt((xx - W / 2) ** 2 + (yy - 1000) ** 2) / 1100, 0, 1)[..., None]
        base = (c0 * (1 - u) + c1 * u) ** (1 / 2.2)
        base = base * (1 - ga)[..., None] + grgb
        for (cx, cy, rad, col, amp) in ((540, 780, 620.0, GOLD, 0.05), (540, 1290, 560.0, BLUE, 0.05)):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
            g = np.clip(1 - d, 0, 1) ** 2 * amp
            base = base * (1 - g)[..., None] + np.array(rgb(col), np.float32) * g[..., None]
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        # the main plot's backing card
        surf2, ctx2 = layer()
        mx, my, mw, mh = MAIN
        ctx2.set_source_rgba(*rgb(CARD_FILL), 0.55)
        pill(ctx2, mx, my, mw, mh, 18)
        ctx2.fill()
        crgb, ca = surf_rgba(surf2)
        base = base * vig[..., None]
        self.bg = np.ascontiguousarray(base)
        self.card = (crgb, ca)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        a_all = max(fade, ease_out_cubic(seg(t, 49.45, 49.95)))
        over(img, self.card[0] * a_all, self.card[1] * a_all)
        s, sn, tl = self.stage_at(t)
        # a short dip between stages, so each width starts visibly fresh
        dip = 1.0
        for st in self.stages[1:]:
            dip = min(dip, 1 - 0.85 * (1 - smooth(abs(t - (st["t0"] - 0.05)) / 0.3)))
        dip = min(dip, 1 - 0.85 * (1 - smooth(abs(t - (T_CLOSE - 0.3)) / 0.3)), 1 - 0.85 * (1 - smooth(abs(t - 2.4) / 0.3)))
        a = a_all * dip
        surf, ctx = layer()
        ys = self.draw_main(ctx, s, sn, a)
        a_close = ease_out_cubic(seg(t, T_CLOSE, T_CLOSE + 0.4)) * fade
        if a_close < 1:
            self.draw_pieces(ctx, s, sn, a * (1 - a_close))
        self.close_table(ctx, t, a_close)
        self.readouts(ctx, s, sn, a, ys)
        self.footer(ctx, a_all)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
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
    frames = [0, 200, 330, 500, 700, 900, 1100, 1300, 1450, 1650, 1850, 2000, 2200, 2400, 2500, 2700, 2900, 2999]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (0, 500, 1300, 1850, 2400, 2700):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
