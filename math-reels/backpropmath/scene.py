"""Backpropagation with nothing but the maths: no network diagram, no field. Dark theme of reels 1-22 and
31-33. The one-neuron network of reel 33 (bpsim.py: reel 31's training, stopped at step 300) and its
example #396. Nine pages of typeset equations, every number computed here from the trained weights:
the network, the forward pass, the loss, the chain rule from the loss down to every parameter, a
finite-difference check of one derivative, the update with step size 0.3, the forward pass again, and the
general rules for any depth. A table under the page keeps the five parameters, their derivatives and their
new values. Numbers are shown to 3 decimals and computed in full precision; a relation is written "=" when
the rounded numbers on screen reproduce the rounded result and "≈" when they miss it in the last digit.
60 s. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backprop"))
from bpsim import DATA_SEED, grads, spiral, train_to  # noqa: E402
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba,  # noqa: E402
                     tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 3600                       # 60 s
LOOP_FADE = (59.3, 59.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6"
GOLD = "#F6C56E"
seg = gds.seg
TAU = 2 * math.pi
BG_TOP, BG_BOT = "#0E1226", "#05060C"
GRID_C, GRID_A = "#FFFFFF", 0.045
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"
PANEL = (50, 500, 980, 920)           # the page: x, y, w, h
BOARD = (50, 1446, 980, 262)          # the table of parameters
PAD = 46                              # inner left margin of the page
S1, S2 = 43, 38                       # type sizes: symbolic rows, numeric rows
SEED, STOP, EXAMPLE, ETA = 1110, 300, 396, 0.3

# ------------------------------------------------------------------ tiny maths typesetter


def mfont(ctx, size, italic=False, bold=False):
    ctx.select_font_face("Inter", cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def T(s, c=None, bold=False, late=False):
    """Upright text: numbers, operators, function names."""
    return dict(k="t", s=s, c=c, it=False, bold=bold, late=late)


def V(s, sub=None, c=None, late=False):
    """An italic variable, with an optional upright subscript."""
    out = [dict(k="t", s=s, c=c, it=True, bold=False, late=late)]
    if sub is not None:
        out.append(dict(k="sub", e=[dict(k="t", s=sub, c=c, it=sub.isalpha() and len(sub) == 1, bold=False, late=late)], late=late))
    return out


def FR(num, den, c=None, late=False):
    return dict(k="frac", n=num, d=den, c=c, late=late)


def D(top, bot, c=None, late=False):
    """The partial derivative ∂top/∂bot; top and bot are a name or (name, subscript)."""
    def var(spec):
        return V(*spec, c=c, late=late) if isinstance(spec, tuple) else V(spec, c=c, late=late)
    return FR([T("∂", c, late=late)] + var(top), [T("∂", c, late=late)] + var(bot), c=c, late=late)


def SUP(elems, late=False):
    return dict(k="sup", e=elems, late=late)


def SP(em):
    return dict(k="sp", w=em)


def flat(*items):
    out = []
    for it in items:
        if isinstance(it, list):
            out.extend(it)
        else:
            out.append(it)
    return out


def typeset(ctx, elems, size, col, late=False):
    """Lay out a run of elements: glyph ops (x, dy, text, size, italic, bold, colour, late), fraction bars
    (x0, x1, dy, width, colour, late), total width, ascent and descent (dy is down-positive from the baseline)."""
    ops, bars = [], []
    x, asc, desc, prev_it = 0.0, 0.0, 0.0, False
    for e in elems:
        k = e["k"]
        lt = late or e.get("late", False)
        if k == "t":
            mfont(ctx, size, e["it"], e["bold"])
            ext = ctx.text_extents(e["s"])
            if prev_it and not e["it"]:
                x += 0.05 * size
            ops.append((x, 0.0, e["s"], size, e["it"], e["bold"], e["c"] or col, lt))
            x += ext.x_advance
            asc = max(asc, -ext.y_bearing)
            desc = max(desc, ext.height + ext.y_bearing)
            prev_it = e["it"]
        elif k in ("sub", "sup"):
            fs = size * 0.64
            sops, sbars, w, a, d = typeset(ctx, e["e"], fs, col, lt)
            dy = 0.22 * size if k == "sub" else -0.42 * size
            x0 = x + (0.01 * size if k == "sub" else 0.05 * size)
            ops += [(x0 + ox, dy + oy) + tuple(rest) for (ox, oy, *rest) in sops]
            bars += [(x0 + a0, x0 + a1, dy + by) + tuple(rest) for (a0, a1, by, *rest) in sbars]
            x = x0 + w + 0.03 * size
            asc = max(asc, a - dy)
            desc = max(desc, d + dy)
            prev_it = False
        elif k == "frac":
            fs = size * 0.80
            fc = e.get("c") or col
            nops, nbars, nw, na, nd = typeset(ctx, e["n"], fs, fc, lt)
            dops, dbars, dw, da, dd = typeset(ctx, e["d"], fs, fc, lt)
            pad = 0.10 * size
            wf = max(nw, dw) + 2 * pad
            x += 0.06 * size
            axis = -0.29 * size
            gap = 0.17 * size
            ny = axis - gap - nd
            dy_ = axis + gap + da
            nx, dx = x + (wf - nw) / 2, x + (wf - dw) / 2
            ops += [(nx + ox, ny + oy) + tuple(rest) for (ox, oy, *rest) in nops]
            ops += [(dx + ox, dy_ + oy) + tuple(rest) for (ox, oy, *rest) in dops]
            bars += [(nx + a0, nx + a1, ny + by) + tuple(rest) for (a0, a1, by, *rest) in nbars]
            bars += [(dx + a0, dx + a1, dy_ + by) + tuple(rest) for (a0, a1, by, *rest) in dbars]
            bars.append((x, x + wf, axis, max(1.6, 0.05 * size), fc, lt))
            x += wf + 0.06 * size
            asc = max(asc, -(ny - na))
            desc = max(desc, dy_ + dd)
            prev_it = False
        elif k == "sp":
            x += e["w"] * size
            prev_it = False
    return ops, bars, x, asc, desc


def draw_typeset(ctx, ops, bars, x0, base, a, a_late, glow=0.0):
    for (ox, oy, s, fs, it, bold, c, lt) in ops:
        al = a * (a_late if lt else 1.0)
        if al <= 0.003:
            continue
        mfont(ctx, fs, it, bold)
        if lt and glow > 0:
            ctx.set_source_rgba(*rgb(c), 0.22 * glow * al)
            for dx, dy in ((-1.6, 0), (1.6, 0), (0, -1.6), (0, 1.6)):
                ctx.move_to(x0 + ox + dx, base + oy + dy)
                ctx.show_text(s)
        ctx.set_source_rgba(*rgb(c), al)
        ctx.move_to(x0 + ox, base + oy)
        ctx.show_text(s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for (b0, b1, by, lw, c, lt) in bars:
        al = a * (a_late if lt else 1.0)
        if al <= 0.003:
            continue
        ctx.set_source_rgba(*rgb(c), al)
        ctx.set_line_width(lw)
        ctx.move_to(x0 + b0, base + by)
        ctx.line_to(x0 + b1, base + by)
        ctx.stroke()


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


# ------------------------------------------------------------------ numbers


def R(v, d=3):
    r = round(float(v), d)
    return 0.0 if r == 0 else r


def f3(v, d=3):
    """Rounded, with a true minus sign and no negative zero."""
    r = R(v, d)
    s = f"{abs(r):.{d}f}"
    return ("−" + s) if r < 0 else s


def pf(v, d=3):
    """As a factor: negative numbers in parentheses."""
    s = f3(v, d)
    return f"({s})" if s.startswith("−") else s


def rel(from_shown, true, d=3):
    """'=' when the numbers on screen reproduce the rounded result, '≈' when rounding makes them miss."""
    return "=" if R(from_shown, d) == R(true, d) else "≈"


def compute():
    rng = np.random.default_rng(DATA_SEED)
    X, y = spiral(rng)
    ps = train_to(SEED, [1], X, y, STOP)
    x1, x2 = (float(v) for v in X[EXAMPLE])
    t = float(y[EXAMPLE])
    w1, w2, b, v, c = float(ps[0][0, 0]), float(ps[0][1, 0]), float(ps[1][0]), float(ps[2][0, 0]), float(ps[3][0])

    def fwd(w1_, w2_, b_, v_, c_):
        z = w1_ * x1 + w2_ * x2 + b_
        h = max(0.0, z)
        s = v_ * h + c_
        p = 1 / (1 + math.exp(-s))
        return z, h, s, p, -math.log(p)

    z, h, s, p, L = fwd(w1, w2, b, v, c)
    dLds = p - t
    dLdc, dLdv, dLdh = dLds, dLds * h, dLds * v
    dhdz = 1.0 if z > 0 else 0.0
    dLdz = dLdh * dhdz
    dw1, dw2, db = dLdz * x1, dLdz * x2, dLdz
    eps = 1e-3
    Lp = fwd(w1 + eps, w2, b, v, c)[4]
    Lm = fwd(w1 - eps, w2, b, v, c)[4]
    fd = (Lp - Lm) / (2 * eps)
    new = dict(w1=w1 - ETA * dw1, w2=w2 - ETA * dw2, b=b - ETA * db, v=v - ETA * dLdv, c=c - ETA * dLdc)
    z2, h2, s2, p2, L2 = fwd(*new.values())
    _, _, pall = grads(ps, X, y)
    ps1 = [ps[0] - ETA * np.array([[dw1], [dw2]]), ps[1] - ETA * np.array([db]), ps[2] - ETA * np.array([[dLdv]]), ps[3] - ETA * np.array([dLdc])]
    _, _, pall1 = grads(ps1, X, y)
    right0 = int(np.sum((pall > 0.5) == (y > 0.5)))
    right1 = int(np.sum((pall1 > 0.5) == (y > 0.5)))
    # the backprop step must agree with reel 33's bpsim one_example (same network, same example, same step)
    from bpsim import one_example
    ex = one_example(ps, X[EXAMPLE], t)
    assert abs(ex["gW"][0][0, 0] - dw1) < 1e-12 and abs(ex["gW"][1][0, 0] - dLdv) < 1e-12 and abs(ex["gb"][1][0] - dLdc) < 1e-12
    return dict(x1=x1, x2=x2, y=t, w1=w1, w2=w2, b=b, v=v, c=c, z=z, h=h, s=s, p=p, L=L, dLds=dLds, dLdc=dLdc, dLdv=dLdv,
                dLdh=dLdh, dhdz=dhdz, dLdz=dLdz, dw1=dw1, dw2=dw2, db=db, Lp=Lp, Lm=Lm, fd=fd, new=new, z2=z2, h2=h2, s2=s2,
                p2=p2, L2=L2, right0=right0, right1=right1, n=len(y))


# ------------------------------------------------------------------ the scene


class Scene(gds.Scene):
    def __init__(self):
        self.F = compute()
        self.rels = []
        self.pages = self._pages()
        self._layout()
        self.cues = self._cues()
        self.facts = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.F.items() if k != "new"}
        self.facts["new"] = {k: round(v, 6) for k, v in self.F["new"].items()}
        self.facts["approx_lines"] = [r for r in self.rels if r[1] == "≈"]
        self.board_times = self._board_times()
        self._build_grid()

    # -------------------------------------------------------------- page content (every number from compute())

    def _pages(self):
        F = self.F
        n = F["new"]
        G, M, K, Y, N = GOLD, MAGENTA, INK, VERM, GREEN

        def r(t, lhs, rl, rhs, note=None, size=S1, gap=None):
            return dict(t=t, lhs=lhs, rel=rl, rhs=rhs, note=note, size=size, gap=gap)

        def chk(name, from_shown, true):
            s = rel(from_shown, true)
            self.rels.append((name, s, round(from_shown, 6), round(true, 6)))
            return s

        x1s, x2s = f3(F["x1"]), f3(F["x2"])
        w1, w2, b, v, c = F["w1"], F["w2"], F["b"], F["v"], F["c"]
        # forward, numbers shown
        z_rel = chk("z", R(w1) * R(F["x1"]) + R(w2) * R(F["x2"]) + R(b), F["z"])
        s_rel = chk("s", R(v) * R(F["h"]) + R(c), F["s"])
        p_rel = chk("p", 1 / (1 + math.exp(-R(F["s"]))), F["p"])
        L_rel = chk("L", -math.log(R(F["p"])), F["L"])
        dLds_rel = chk("dL/ds", R(F["p"]) - 1, F["dLds"])
        dLdv_rel = chk("dL/dv", R(F["dLds"]) * R(F["h"]), F["dLdv"])
        dLdh_rel = chk("dL/dh", R(F["dLds"]) * R(v), F["dLdh"])
        dLdz_rel = chk("dL/dz", R(F["dLdh"]) * 1, F["dLdz"])
        dw1_rel = chk("dL/dw1", R(F["dLdz"]) * R(F["x1"]), F["dw1"])
        dw2_rel = chk("dL/dw2", R(F["dLdz"]) * R(F["x2"]), F["dw2"])
        fd_rel = chk("finite difference", (R(F["Lp"], 6) - R(F["Lm"], 6)) / 0.002, F["fd"])
        up_rel = {k: chk("new " + k, R(old) - ETA * R(g), n[k]) for k, old, g in
                  (("w1", w1, F["dw1"]), ("w2", w2, F["dw2"]), ("b", b, F["db"]), ("v", v, F["dLdv"]), ("c", c, F["dLdc"]))}
        z2_rel = chk("z again", R(n["w1"]) * R(F["x1"]) + R(n["w2"]) * R(F["x2"]) + R(n["b"]), F["z2"])
        s2_rel = chk("s again", R(n["v"]) * R(F["h2"]) + R(n["c"]), F["s2"])
        p2_rel = chk("p again", 1 / (1 + math.exp(-R(F["s2"]))), F["p2"])
        L2_rel = chk("L again", -math.log(R(F["p2"])), F["L2"])

        def plus(v_):
            """'+ 0.474' or '− 0.627' for a term added at the end."""
            s = f3(v_)
            return f" − {s[1:]}" if s.startswith("−") else f" + {s}"

        pages = []
        # 1 · the network
        t0 = 2.6
        pages.append(dict(t0=t0, t1=9.4, label="THE NETWORK", col=GOLD, rows=[
            r(t0 + 0.4, V("z"), "=", flat(V("w", "1"), V("x", "1"), T(" + "), V("w", "2"), V("x", "2"), T(" + "), V("b")), "a weighted sum of the two inputs"),
            r(t0 + 1.3, V("h"), "=", flat(T("max(0, "), V("z"), T(")")), "the fold: below zero becomes 0 (ReLU)"),
            r(t0 + 2.2, V("s"), "=", flat(V("v"), SP(0.08), V("h"), T(" + "), V("c")), "the output neuron"),
            r(t0 + 3.1, V("p"), "=", flat(T("σ("), V("s"), T(")  =  "), FR([T("1")], flat(T("1 + "), T("e"), SUP(flat(T("−"), V("s")))))), "squashed into a probability"),
            r(t0 + 4.0, V("L"), "=", flat(T("−log "), V("p")), "the loss: zero only if p = 1, the answer"),
            r(t0 + 5.0, None, None, flat(T("Example:   "), V("x", "1"), T(f" = {x1s},   "), V("x", "2"), T(f" = {x2s},   "), V("y"), T(" = "), T("1", Y)), None, size=S2, gap=40),
        ]))
        # 2 · forward
        t0 = 9.4
        pages.append(dict(t0=t0, t1=17.4, label="FORWARD", col=GOLD, rows=[
            r(t0 + 0.4, V("z"), "=", flat(V("w", "1"), V("x", "1"), T(" + "), V("w", "2"), V("x", "2"), T(" + "), V("b"))),
            r(t0 + 0.9, None, "=", [T(f"{pf(w1)} · {pf(F['x1'])} + {f3(w2)} · {f3(F['x2'])}{plus(b)}")], size=S2),
            r(t0 + 1.5, None, z_rel, [T(f3(F["z"]), G, bold=True, late=True)], size=S2),
            r(t0 + 2.4, V("h"), "=", flat(T("max(0, "), T(f3(F["z"])), T(")  =  "), T(f3(F["h"]), G, bold=True, late=True)), size=S1, gap=30),
            r(t0 + 3.4, V("s"), "=", flat(V("v"), SP(0.08), V("h"), T(" + "), V("c"), T("  =  "), T(f"{f3(v)} · {f3(F['h'])}{plus(c)}")), size=S1, gap=30),
            r(t0 + 3.9, None, s_rel, [T(f3(F["s"]), G, bold=True, late=True)], size=S2),
            r(t0 + 4.8, V("p"), "=", flat(T("σ("), T(f3(F["s"])), T(f")  {p_rel}  "), T(f3(F["p"]), G, bold=True, late=True)), "the guess", gap=30),
            r(t0 + 5.9, V("L"), "=", flat(T("−log "), T(f3(F["p"])), T(f"  {L_rel}  "), T(f3(F["L"]), G, bold=True, late=True)), "the loss", gap=30),
        ]))
        # 3 · backward from the loss
        t0 = 17.4
        pages.append(dict(t0=t0, t1=23.4, label="BACKWARD · FROM THE LOSS", col=MAGENTA, rows=[
            r(t0 + 0.4, D("L", "s"), "=", flat(D("L", "p"), T(" · "), D("p", "s")), "the chain rule"),
            r(t0 + 1.5, None, "=", flat(T("("), T("−"), FR([T("1")], V("p")), T(")"), T(" · "), V("p"), T("(1 − "), V("p"), T(")")), "derivative of −log p, times derivative of σ", gap=18),
            r(t0 + 2.6, None, "=", flat(V("p"), T(" − 1")), "in general p − y; here the answer y = 1", gap=18),
            r(t0 + 3.6, None, "=", flat(T(f"{f3(F['p'])} − 1  {dLds_rel}  "), T(f3(F["dLds"]), M, bold=True, late=True)), size=S2, gap=18),
        ]))
        # 4 · the output layer
        t0 = 23.4
        pages.append(dict(t0=t0, t1=29.4, label="BACKWARD · THE OUTPUT NEURON", col=MAGENTA, rows=[
            r(t0 + 0.4, D("L", "c"), "=", flat(D("L", "s"), T(" · "), D("s", "c"), T("  =  "), D("L", "s"), T(" · 1"))),
            r(t0 + 0.9, None, "=", [T(f3(F["dLdc"]), M, bold=True, late=True)], size=S2),
            r(t0 + 1.9, D("L", "v"), "=", flat(D("L", "s"), T(" · "), D("s", "v"), T("  =  "), D("L", "s"), T(" · "), V("h")), gap=34),
            r(t0 + 2.4, None, "=", flat(T(f"{pf(F['dLds'])} · {f3(F['h'])}  {dLdv_rel}  "), T(f3(F["dLdv"]), M, bold=True, late=True)), size=S2),
            r(t0 + 3.4, D("L", "h"), "=", flat(D("L", "s"), T(" · "), D("s", "h"), T("  =  "), D("L", "s"), T(" · "), V("v")), gap=34),
            r(t0 + 3.9, None, "=", flat(T(f"{pf(F['dLds'])} · {f3(v)}  {dLdh_rel}  "), T(f3(F["dLdh"]), M, bold=True, late=True)), "the blame that reaches the neuron", size=S2),
        ]))
        # 5 · through the fold
        t0 = 29.4
        pages.append(dict(t0=t0, t1=36.6, label="BACKWARD · THROUGH THE FOLD", col=MAGENTA, rows=[
            r(t0 + 0.4, D("h", "z"), "=", flat(T("ReLU′("), V("z"), T(")  =  1")), f"z = {f3(F['z'])} > 0: the neuron was on, so the blame passes"),
            r(t0 + 1.5, D("L", "z"), "=", flat(D("L", "h"), T(" · "), D("h", "z"), T("  =  "), T(f"{pf(F['dLdh'])} · 1  {dLdz_rel}  "), T(f3(F["dLdz"]), M, bold=True, late=True)), size=40, gap=30),
            r(t0 + 2.6, D("L", ("w", "1")), "=", flat(D("L", "z"), T(" · "), V("x", "1"), T("  =  "), T(f"{pf(F['dLdz'])} · {pf(F['x1'])}  {dw1_rel}  "), T(f3(F["dw1"]), M, bold=True, late=True)), size=40, gap=30),
            r(t0 + 3.7, D("L", ("w", "2")), "=", flat(D("L", "z"), T(" · "), V("x", "2"), T("  =  "), T(f"{pf(F['dLdz'])} · {f3(F['x2'])}  {dw2_rel}  "), T(f3(F["dw2"]), M, bold=True, late=True)), size=40, gap=30),
            r(t0 + 4.8, D("L", "b"), "=", flat(D("L", "z"), T(" · 1  =  "), T(f3(F["db"]), M, bold=True, late=True)), "had z been negative, these three would all be 0", gap=30),
        ]))
        # 6 · check
        t0 = 36.6
        pages.append(dict(t0=t0, t1=41.2, label="CHECK · NUDGE w₁ AND MEASURE", col=GREEN, rows=[
            r(t0 + 0.4, D("L", ("w", "1")), "≈", [FR(flat(V("L"), T("("), V("w", "1"), T(" + 0.001) − "), V("L"), T("("), V("w", "1"), T(" − 0.001)")), [T("0.002")])], "the slope, measured directly"),
            r(t0 + 1.5, None, "=", [FR([T(f"{f3(F['Lp'], 6)} − {f3(F['Lm'], 6)}")], [T("0.002")])], gap=22),
            r(t0 + 2.4, None, fd_rel, [T(f3(F["fd"]), GREEN, bold=True, late=True)], size=S1, gap=18),
            r(t0 + 3.2, None, None, flat(T("Backpropagation gave  "), D("L", ("w", "1")), T("  =  "), T(f3(F["dw1"]), M, bold=True), T("   ✓", GREEN, bold=True, late=True)), None, size=S2, gap=40),
        ]))
        # 7 · update
        t0 = 41.2
        rows = [r(t0 + 0.4, V("θ"), "←", flat(V("θ"), T(" − "), V("η"), T(" · "), D("L", "θ")), f"every parameter θ steps against its derivative;  η = {ETA:g}")]
        for i, (k, sym, old, g) in enumerate((("w1", ("w", "1"), w1, F["dw1"]), ("w2", ("w", "2"), w2, F["dw2"]), ("b", ("b",), b, F["db"]),
                                              ("v", ("v",), v, F["dLdv"]), ("c", ("c",), c, F["dLdc"]))):
            rows.append(r(t0 + 1.4 + 0.8 * i, V(*sym), "←", flat(T(f3(old)), T(f" − {ETA:g} · "), T(pf(g), M), T(f"  {up_rel[k]}  "), T(f3(n[k]), N, bold=True, late=True)),
                          size=S1, gap=40 if i == 0 else 16))
        pages.append(dict(t0=t0, t1=47.6, label="UPDATE", col=GREEN, rows=rows))
        # 8 · forward again
        t0 = 47.6
        pages.append(dict(t0=t0, t1=53.0, label="FORWARD AGAIN · THE NEW NUMBERS", col=GOLD, rows=[
            r(t0 + 0.4, V("z"), "=", [T(f"{pf(n['w1'])} · {pf(F['x1'])} + {f3(n['w2'])} · {f3(F['x2'])}{plus(n['b'])}")], size=S2),
            r(t0 + 0.9, None, z2_rel, [T(f3(F["z2"]), G, bold=True, late=True)], size=S2),
            r(t0 + 1.7, V("s"), "=", flat(T(f"{f3(n['v'])} · {f3(F['h2'])}{plus(n['c'])}  {s2_rel}  "), T(f3(F["s2"]), G, bold=True, late=True)), size=S2, gap=26),
            r(t0 + 2.5, V("p"), "=", flat(T(f"σ({f3(F['s2'])})  {p2_rel}  "), T(f3(F["p2"]), G, bold=True, late=True)), size=S2, gap=26),
            r(t0 + 3.3, V("L"), "=", flat(T(f"−log {f3(F['p2'])}  {L2_rel}  "), T(f3(F["L2"]), G, bold=True, late=True)), size=S2, gap=26),
            r(t0 + 4.1, None, None, flat(T("Loss  "), T(f3(F["L"])), T("  →  "), T(f3(F["L2"]), GREEN, bold=True), T("        guess  "), T(f3(F["p"])), T("  →  "), T(f3(F["p2"]), GREEN, bold=True)), None, size=S2, gap=44),
            r(t0 + 4.5, None, None, flat(T(f"On all {F['n']} points:  "), T(str(F["right0"])), T("  →  "), T(str(F["right1"]), GREEN, bold=True), T("  right")), None, size=S2, gap=14),
        ]))
        # 9 · any network
        t0 = 53.0
        pages.append(dict(t0=t0, t1=60.6, label="ANY NETWORK, ANY DEPTH", col=GOLD, rows=[
            r(t0 + 0.4, flat(V("δ"), dict(k="sub", e=[T("out")])), "=", flat(V("p"), T(" − "), V("y")), "the error at the output"),
            r(t0 + 1.4, V("δ", "j"), "=", flat(T("ReLU′("), V("z", "j"), T(") · "), T("Σ"), dict(k="sub", e=V("k")), SP(0.1), V("w", "jk"), SP(0.04), V("δ", "k")),
              "one layer back: weights × the blame above; 0 if neuron j was off", gap=34),
            r(t0 + 2.4, D("L", ("w", "ij")), "=", flat(V("h", "i"), SP(0.06), V("δ", "j"), T(",      "), D("L", ("b", "j")), T("  =  "), V("δ", "j")),
              "a weight's blame: the number it multiplied × the blame where it lands", gap=34),
            r(t0 + 3.4, V("w"), "←", flat(V("w"), T(" − "), V("η"), T(" · "), D("L", "w")), "then every weight steps against its blame", gap=34),
        ]))
        return pages

    def _layout(self):
        surf, ctx = layer(64, 64)
        px, py, pw, ph = PANEL
        right = px + pw - 36
        for pg in self.pages:
            lhs_w = 0.0
            for row in pg["rows"]:
                lhs = row["lhs"]
                if lhs is not None and not isinstance(lhs, list):
                    lhs = [lhs]
                row["lhs"] = lhs
                if lhs:
                    ops, bars, w, a, d = typeset(ctx, lhs, row["size"], INK)
                    row["L"] = (ops, bars, w, a, d)
                    lhs_w = max(lhs_w, w)
                else:
                    row["L"] = None
                row["Rr"] = typeset(ctx, row["rhs"], row["size"], INK)
                if row["rel"]:
                    row["E"] = typeset(ctx, [T(row["rel"])], row["size"], INK2)
            x_rel = px + PAD + lhs_w + 18
            pg["x_rel"] = x_rel
            y = py + 108
            for i, row in enumerate(pg["rows"]):
                a = max(row["Rr"][3], row["L"][3] if row["L"] else 0, 0.75 * row["size"])
                d = max(row["Rr"][4], row["L"][4] if row["L"] else 0, 0.22 * row["size"])
                if i > 0:
                    y += row["gap"] if row["gap"] is not None else 14
                y += a
                row["base"] = y
                if row["rel"]:
                    rw = row["E"][2]
                    row["x_rhs"] = x_rel + rw + 0.35 * row["size"]
                else:
                    row["x_rhs"] = px + PAD
                end = row["x_rhs"] + row["Rr"][2]
                assert end <= right, (pg["label"], i, end)
                y += d
                if row["note"]:
                    mfont(ctx, 23)
                    nw = ctx.text_extents(row["note"]).x_advance
                    row["note_x"] = row["x_rhs"] if row["rel"] else px + PAD
                    if row["note_x"] + nw > right:
                        row["note_x"] = px + PAD
                    assert row["note_x"] + nw <= right, (pg["label"], row["note"])
                    y += 34
                    row["note_y"] = y
            assert y <= py + ph - 26, (pg["label"], y)
            shift = 0.38 * (py + ph - 26 - y)
            for row in pg["rows"]:
                row["base"] += shift
                if row["note"]:
                    row["note_y"] += shift
            pg["bottom"] = y + shift
        # the opening: the whole chain for one weight, with its numbers
        F = self.F
        chain = flat(D("L", ("w", "1")), T("  =  "), D("L", "s"), T(" · "), D("s", "h"), T(" · "), D("h", "z"), T(" · "), D("z", ("w", "1")))
        prod = R(F["dLds"]) * R(F["v"]) * 1 * R(F["x1"])
        hr = rel(prod, F["dw1"])
        self.rels.append(("opening chain", hr, round(prod, 6), round(F["dw1"], 6)))
        nums = flat(T(f"=  {pf(F['dLds'])} · {f3(F['v'])} · 1 · {pf(F['x1'])}  {hr}  "), T(f3(F["dw1"]), MAGENTA, bold=True))
        self.hook_rows = [typeset(ctx, chain, 58, INK), typeset(ctx, nums, 42, INK2)]
        for r_ in self.hook_rows:
            assert r_[2] < 960, r_[2]

    def _board_times(self):
        """When each derivative and each new value is first shown on the page (the table fills in then)."""
        pg = {p["label"]: p for p in self.pages}
        P4, P5, P7 = pg["BACKWARD · THE OUTPUT NEURON"], pg["BACKWARD · THROUGH THE FOLD"], pg["UPDATE"]
        late = 0.45
        grad_t = dict(c=P4["rows"][1]["t"] + late, v=P4["rows"][3]["t"] + late, w1=P5["rows"][2]["t"] + late,
                      w2=P5["rows"][3]["t"] + late, b=P5["rows"][4]["t"] + late)
        new_t = {k: P7["rows"][1 + i]["t"] + late for i, k in enumerate(("w1", "w2", "b", "v", "c"))}
        return grad_t, new_t

    # -------------------------------------------------------------- cues

    def _cues(self):
        F = self.F
        hook = [("Backpropagation, with nothing but the maths.", "black", 42, INK, 300, 1), ("One neuron, one example, every number.", "black", 38, INK2, 390, 1)]
        P = self.pages

        def c(i, a_, b_, s1=44, s2=36):
            return (P[i]["t0"], P[i]["t1"] - 0.3, [(a_, "black", s1, INK, 300, 1), (b_, "dbold", s2, INK2, 390, 0.95)], 10, 8, "pop")

        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            c(0, "The network: five numbers to learn.", "w₁, w₂, b in the neuron;  v, c at the output."),
            c(1, "Forward: put the numbers in.", f"The guess p = {f3(F['p'])}. The answer is 1."),
            c(2, "Backward: start at the loss.", "The chain rule turns −log σ(s) into p − y."),
            c(3, "Then one step down.", "Each derivative: the one above × one local factor."),
            c(4, "Through the fold.", "The neuron was on, so ReLU′ = 1 and the blame passes."),
            c(5, "Check one by brute force.", "Nudge w₁ by ±0.001 and measure the loss."),
            c(6, "Update: step against the gradient.", f"η = {ETA:g}. All five numbers move at once."),
            c(7, "Forward again.", f"The loss falls from {f3(F['L'])} to {f3(F['L2'])}."),
            (P[8]["t0"], 59.0, [("Any network, any depth: these four lines.", "black", 42, INK, 300, 1), ("Repeat a few thousand times. That is training.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (59.4, 60.1, hook, 12, 0, "fade"),
        ]

    # -------------------------------------------------------------- drawing

    def page_alpha(self, pg, t):
        return ease_out_cubic(seg(t, pg["t0"], pg["t0"] + 0.35)) * (1 - smooth(seg(t, pg["t1"] - 0.35, pg["t1"])))

    def draw_page(self, ctx, pg, t, fade):
        pa = self.page_alpha(pg, t) * fade
        if pa <= 0:
            return
        px, py, pw, ph = PANEL
        dy_out = -18 * smooth(seg(t, pg["t1"] - 0.35, pg["t1"]))
        # label and the page dots
        font(ctx, "bold", 20)
        tr = 0.14 * 20
        idx = self.pages.index(pg)
        lab = f"{idx + 1} · {pg['label']}"
        ctx.set_source_rgba(*rgb(pg["col"]), 0.95 * pa)
        show_tracked(ctx, lab, px + PAD, py + 56 + dy_out, tr)
        lw = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(pg["col"]), 0.35 * pa)
        ctx.set_line_width(1.2)
        ctx.move_to(px + PAD + lw + 18, py + 49 + dy_out)
        ctx.line_to(px + pw - 36 - 9 * 20 - 10, py + 49 + dy_out)
        ctx.stroke()
        for k in range(len(self.pages)):
            cx = px + pw - 36 - (len(self.pages) - 1 - k) * 20
            on = k == idx
            ctx.set_source_rgba(*rgb(pg["col"] if on else INK3), (1.0 if on else (0.7 if k < idx else 0.35)) * pa)
            ctx.arc(cx, py + 49 + dy_out, 5.0 if on else 3.2, 0, TAU)
            ctx.fill()
        # rows
        rows = pg["rows"]
        shown = [row for row in rows if t >= row["t"]]
        newest = shown[-1] if shown else None
        for row in rows:
            ra = ease_out_cubic(seg(t, row["t"], row["t"] + 0.35))
            if ra <= 0:
                continue
            a_late = ease_out_cubic(seg(t, row["t"] + 0.45, row["t"] + 0.8))
            glow = 1 - smooth(seg(t, row["t"] + 0.8, row["t"] + 2.2))
            dim = 1.0 if row is newest or t < row["t"] + 1.2 else 0.82
            al = ra * pa * dim
            dx = 14 * (1 - ra)
            base = row["base"] + dy_out
            # the accent bar beside the newest row
            if row is newest:
                ab = ra * pa * (1 - smooth(seg(t, row["t"] + 2.0, row["t"] + 3.0)))
                top = base - max(row["Rr"][3], row["L"][3] if row["L"] else 0, 0.75 * row["size"]) - 4
                bot = base + max(row["Rr"][4], row["L"][4] if row["L"] else 0, 0.22 * row["size"]) + 4
                ctx.set_source_rgba(*rgb(pg["col"]), 0.85 * ab)
                ctx.set_line_width(3.0)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(px + 22, top)
                ctx.line_to(px + 22, bot)
                ctx.stroke()
            if row["L"]:
                ops, bars, w, _, _ = row["L"]
                draw_typeset(ctx, ops, bars, pg["x_rel"] - 18 - w + dx, base, al, 1.0)
            if row["rel"]:
                ops, bars, w, _, _ = row["E"]
                draw_typeset(ctx, ops, bars, pg["x_rel"] + dx, base, al, 1.0)
            ops, bars, w, _, _ = row["Rr"]
            draw_typeset(ctx, ops, bars, row["x_rhs"] + dx, base, al, a_late, glow=glow)
            if row["note"]:
                mfont(ctx, 23)
                ctx.set_source_rgba(*rgb(INK3), 0.95 * al * a_late if any(o[7] for o in ops) else 0.95 * al)
                ctx.move_to(row["note_x"] + dx, row["note_y"] + dy_out)
                ctx.show_text(row["note"])

    def draw_panel_bg(self, ctx, a):
        if a <= 0:
            return
        for (x, y, w, h) in (PANEL, BOARD):
            ctx.set_source_rgba(*rgb(CARD_FILL), 0.62 * a)
            pill(ctx, x, y, w, h, 20)
            ctx.fill_preserve()
            ctx.set_source_rgba(*rgb(CARD_EDGE), 0.85 * a)
            ctx.set_line_width(1.2)
            ctx.stroke()
            ctx.set_source_rgba(1, 1, 1, 0.06 * a)
            ctx.set_line_width(1.0)
            ctx.move_to(x + 20, y + 1.5)
            ctx.line_to(x + w - 20, y + 1.5)
            ctx.stroke()

    def draw_board(self, ctx, t, a):
        if a <= 0:
            return
        F = self.F
        x0, y0, w, h = BOARD
        grad_t, new_t = self.board_times
        keys = ("w1", "w2", "b", "v", "c")
        syms = (("w", "1"), ("w", "2"), ("b",), ("v",), ("c",))
        grads_ = dict(w1=F["dw1"], w2=F["dw2"], b=F["db"], v=F["dLdv"], c=F["dLdc"])
        col_w = (w - 210) / 5
        cxs = [x0 + 210 + col_w * (i + 0.5) for i in range(5)]
        a_vals = ease_out_cubic(seg(t, 3.0, 3.5)) * a
        surf_ctx = ctx
        # header: the five parameters
        for cx, sym in zip(cxs, syms):
            ops, bars, ww, _, _ = typeset(surf_ctx, V(*sym), 34, INK)
            draw_typeset(ctx, ops, bars, cx - ww / 2, y0 + 58, a_vals, 1.0)
        ctx.set_source_rgba(*rgb(CARD_EDGE), 0.9 * a_vals)
        ctx.set_line_width(1.0)
        ctx.move_to(x0 + 30, y0 + 80)
        ctx.line_to(x0 + w - 30, y0 + 80)
        ctx.stroke()
        rows = (("value", y0 + 128, INK), ("∂L/∂θ", y0 + 180, MAGENTA), ("new value", y0 + 232, GREEN))
        for name, yy, col in rows:
            if name == "value":
                ra = a_vals
            elif name == "∂L/∂θ":
                ra = ease_out_cubic(seg(t, min(grad_t.values()) - 0.4, min(grad_t.values()))) * a
            else:
                ra = ease_out_cubic(seg(t, min(new_t.values()) - 0.4, min(new_t.values()))) * a
            if ra <= 0:
                continue
            font(ctx, "bold", 18)
            tr = 0.12 * 18
            ctx.set_source_rgba(*rgb(INK3 if name != "value" else INK2), 0.95 * ra)
            if name == "∂L/∂θ":
                ops, bars, ww, _, _ = typeset(ctx, flat(T("∂", INK3), V("L", c=INK3), T("/∂", INK3), V("θ", c=INK3)), 24, INK3)
                draw_typeset(ctx, ops, bars, x0 + 34, yy - 2, ra, 1.0)
            else:
                show_tracked(ctx, name.upper(), x0 + 34, yy - 6, tr)
            for k, cx in zip(keys, cxs):
                if name == "value":
                    val, ca = F[k], 1.0
                elif name == "∂L/∂θ":
                    ca = ease_out_cubic(seg(t, grad_t[k], grad_t[k] + 0.4))
                    val = grads_[k]
                else:
                    ca = ease_out_cubic(seg(t, new_t[k], new_t[k] + 0.4))
                    # the new value ticks from the old one as it appears
                    val = F[k] + (F["new"][k] - F[k]) * smooth(seg(t, new_t[k], new_t[k] + 0.6))
                if ca <= 0:
                    continue
                s = f3(val)
                font(ctx, "dbold", 30)
                ww = ctx.text_extents(s).x_advance
                if name != "value":
                    glow = 1 - smooth(seg(t, (grad_t if name == "∂L/∂θ" else new_t)[k] + 0.4, (grad_t if name == "∂L/∂θ" else new_t)[k] + 1.8))
                    if glow > 0:
                        ctx.set_source_rgba(*rgb(col), 0.25 * glow * ra * ca)
                        for ddx, ddy in ((-1.6, 0), (1.6, 0), (0, -1.6), (0, 1.6)):
                            ctx.move_to(cx - ww / 2 + ddx, yy + ddy)
                            ctx.show_text(s)
                ctx.set_source_rgba(*rgb(col if name != "value" else INK), ra * ca)
                ctx.move_to(cx - ww / 2, yy)
                ctx.show_text(s)

    def draw_footer(self, ctx, a):
        if a <= 0:
            return
        font(ctx, "bold", 19)
        tr = 0.12 * 19
        lab = f"ONE ReLU NEURON, {STOP} TRAINING STEPS ON THE SPIRAL  ·  EXAMPLE #{EXAMPLE}"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        lab = "NUMBERS ROUNDED TO 3 DECIMALS, COMPUTED IN FULL  ·  ≈ WHERE ROUNDING SHOWS"
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
        for (cx, cy, rad, col, amp) in ((300, 760, 620.0, MAGENTA, 0.05), (800, 1200, 620.0, BLUE, 0.06), (540, 1570, 420.0, GOLD, 0.03)):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
            g = np.clip(1 - d, 0, 1) ** 2 * amp
            base = base * (1 - g)[..., None] + np.array(rgb(col), np.float32) * g[..., None]
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        base = base * vig[..., None]
        self.bg = np.ascontiguousarray(base)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        a_all = ease_out_cubic(seg(t, 2.3, 2.9)) * fade
        surf, ctx = layer()
        self.draw_panel_bg(ctx, a_all)
        for pg in self.pages:
            if pg["t0"] - 0.01 <= t <= pg["t1"] + 0.01:
                self.draw_page(ctx, pg, t, fade)
        self.draw_board(ctx, t, a_all)
        self.draw_footer(ctx, a_all)
        a_hook = max(1 - smooth(seg(t, 2.0, 2.5)), ease_out_cubic(seg(t, 59.45, 59.95)))
        if a_hook > 0:
            for k, (ops, bars, w, asc, desc) in enumerate(self.hook_rows):
                base = (930, 1040)[k]
                draw_typeset(ctx, ops, bars, 540 - w / 2, base, a_hook * (1.0 if k == 0 else 0.95), 1.0)
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
    for name, s, a, b in sc.rels:
        print(f"   {name:18s} {s}   from the numbers shown {a:+.6f}   true {b:+.6f}")
    print("right", sc.F["right0"], "->", sc.F["right1"], " loss", round(sc.F["L"], 4), "->", round(sc.F["L2"], 4))
    for pg in sc.pages:
        print(pg["label"], "bottom", round(pg["bottom"]), "rows", len(pg["rows"]))
    frames = [0, 300, 520, 860, 1000, 1300, 1380, 1650, 1760, 2100, 2160, 2440, 2560, 2840, 3100, 3200, 3480, 3590]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (520, 1000, 1380, 1760, 2160, 2440, 2840, 3480):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
