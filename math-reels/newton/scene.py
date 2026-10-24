"""Newton's root-finding method. Part 1: x^2 - 2 = 0 on the real line, tangent by tangent, the camera
zooming onto sqrt 2 while the correct digits double (1, 3, 6, 12, 24, 48). Part 2: z^3 = 1 in the
complex plane, every starting point coloured by the root it ends at, built iteration by iteration
into the Newton fractal, then a zoom onto a boundary point where all three basins meet.
frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys
from decimal import Decimal, getcontext

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, linear_to_srgb,  # noqa: E402
                     over, rgb, show_cells, show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1440                     # 24 s
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
CX = 470
CHART = (90, 720, 880, 1440)
MAX_TEXT_W = 820
N_STEPS = 6
STEP_T = [2.6, 3.9, 5.0, 5.9, 6.6, 7.1]        # when Newton step n (1..6) starts on the real line
TANGENT_DT, RISE_DT = 0.35, 0.25
SHOWN_DECIMALS = 30
PLANE_T0 = 8.2                                  # crossfade real line -> complex plane
ITER_T0, ITER_DT, ITER_MAX = 10.4, 0.26, 14     # iteration-by-iteration reveal
ZOOM_T0, ZOOM_T1 = 14.4, 20.8
ZOOM_CENTER = (-(0.5 ** (1 / 3)), 0.0)          # Newton maps this point onto 0, where the three basins meet
HW0, HW1 = 1.5, 1.5e-5                          # half-width of the view in the complex plane
LOOP_FADE = (23.3, 23.85)
ROOTS = np.array([1.0 + 0j, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])


@njit(cache=True)
def newton_plane(idx, cnt, cx, cy, hw, maxit, tol):
    """Newton for z^3 - 1 from every pixel. idx: nearest root after the iterations (0..2);
    cnt: iterations used (maxit + 1 if not converged within maxit)."""
    h, w = idx.shape
    r1 = complex(1.0, 0.0)
    r2 = complex(-0.5, math.sqrt(3) / 2)
    r3 = complex(-0.5, -math.sqrt(3) / 2)
    for j in range(h):
        y = cy - (j + 0.5 - h / 2) / (w / 2) * hw
        for i in range(w):
            x = cx + (i + 0.5 - w / 2) / (w / 2) * hw
            z = complex(x, y)
            n = 0
            done = False
            while n < maxit:
                z2 = z * z
                fz = z2 * z - 1
                if abs(fz) < tol:
                    done = True
                    break
                if z2 == 0:
                    break
                z = z - fz / (3 * z2)
                n += 1
            if not done:
                z2 = z * z
                if abs(z2 * z - 1) < tol:
                    done = True
            d1 = abs(z - r1)
            d2 = abs(z - r2)
            d3 = abs(z - r3)
            k = 0
            if d2 < d1 and d2 <= d3:
                k = 1
            elif d3 < d1 and d3 < d2:
                k = 2
            idx[j, i] = k
            cnt[j, i] = n if done else maxit + 1


def newton_iterates():
    getcontext().prec = 120
    r = Decimal(2).sqrt()
    rs = str(r).replace(".", "")
    xs, digits, errs = [], [], []
    x = Decimal(1)
    for n in range(N_STEPS + 1):
        s = str(x)
        k = 0
        for a, b in zip(s.replace(".", ""), rs):
            if a != b:
                break
            k += 1
        xs.append(x)
        digits.append(k)
        errs.append(float(abs(x - r)))
        x = (x + 2 / x) / 2
    return r, xs, digits, errs


class Scene:
    def __init__(self):
        self.r, self.xs, self.digits, self.errs = newton_iterates()
        self.rf = float(self.r)
        self.off = [float(x - self.r) for x in self.xs]           # x_n - sqrt2, exact to float precision
        self.root_str = str(self.r).replace(".", "")
        self.bg = _ruin.Scene._background(self)
        self.root_cols = np.array([srgb_to_linear(rgb(c)) for c in (AMBER, CYAN, MAGENTA)], np.float32)
        self.cues = self._cues()
        self._vig = None

    def _cues(self):
        d = self.digits
        hook = [("Guess. Slide down the tangent. Repeat.", "black", 50, WHITE, 300, 1), ("Newton's method, 1669. Watch the digits.", "dbold", 42, WHITE, 392, 0.9)]
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.6, 5.2, [("Where the tangent hits zero: next guess.", "black", 46, AMBER, 350, 1)], 8, 6, "pop"),
            (5.4, 7.0, [("The error squares itself every step.", "black", 48, WHITE, 350, 1)], 8, 6, "pop"),
            (7.2, 8.6, [(f"Correct digits: {', '.join(str(k) for k in d[1:])}.", "black", 50, GREEN, 312, 1), ("They double. Six steps, 48 digits.", "dbold", 42, WHITE, 396, 0.9)], 6, 6, "chaos"),
            (8.8, 10.3, [("Now z³ = 1. Three roots.", "black", 60, WHITE, 312, 1), ("Colour every start by the root it finds.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (12.0, 14.2, [("Every step, the borders grow borders.", "black", 50, WHITE, 350, 1)], 8, 6, "pop"),
            (14.6, 16.8, [("Zoom in on a border.", "black", 64, WHITE, 350, 1)], 8, 6, "pop"),
            (17.2, 19.6, [("Every border point touches all three.", "black", 48, WHITE, 312, 1), ("Guesses a hair apart find different roots.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (20.0, 21.6, [("Infinitely deep. From one tangent line.", "black", 48, AMBER, 350, 1)], 8, 6, "pop"),
            (21.8, 23.5, [("Newton wrote it in 1669.", "black", 56, WHITE, 312, 1), ("Nobody saw this picture until computers.", "dbold", 40, WHITE, 396, 0.9)], 8, 6, "pop"),
            (23.6, 24.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ real line

    def step_state(self, t):
        """(completed steps n, phase, u) where phase is 'idle' | 'tangent' | 'rise' and u in [0,1]."""
        n = 0
        for k, ts in enumerate(STEP_T):
            if t < ts:
                break
            if t < ts + TANGENT_DT:
                return k, "tangent", (t - ts) / TANGENT_DT
            if t < ts + TANGENT_DT + RISE_DT:
                return k + 1, "rise", (t - ts - TANGENT_DT) / RISE_DT
            n = k + 1
        return n, "idle", 0.0

    def camera(self, t):
        """(centre offset from sqrt2, half-width) of the real-line window."""
        keys = [(0.0, 1.25 - self.rf, 0.95), (STEP_T[1] + 0.5, 1.25 - self.rf, 0.95), (STEP_T[2] + 0.2, 0.0, 0.012),
                (STEP_T[3] + 0.2, 0.0, 1.0e-5), (STEP_T[4] + 0.2, 0.0, 8.0e-12), (STEP_T[5] + 0.2, 0.0, 8.0e-12)]
        if t <= keys[0][0]:
            return keys[0][1], keys[0][2]
        for (ta, ca, ha), (tb, cb, hb) in zip(keys, keys[1:]):
            if ta <= t < tb:
                u = smooth((t - ta) / (tb - ta))
                hw = math.exp(math.log(ha) + (math.log(hb) - math.log(ha)) * u)
                # interpolate the root's screen position (centre offset in units of the half-width), so it never leaves the frame
                s_pos = (ca / ha) + (cb / hb - ca / ha) * u
                return s_pos * hw, hw
        return keys[-1][1], keys[-1][2]

    def _real_line(self, img, t):
        surf, ctx = layer()
        x0, y0, x1, y1 = CHART
        c, hw = self.camera(t)
        n, phase, u = self.step_state(t)

        def f(d):                                  # f(x) with x = sqrt2 + d, exact in d
            return d * (d + 2 * self.rf)

        def px(d):
            return x0 + (d - (c - hw)) / (2 * hw) * (x1 - x0)

        ds = np.linspace(c - hw, c + hw, 400)
        fs = f(ds)
        fmin, fmax = min(fs.min(), 0.0), max(fs.max(), 0.0)
        pad = 0.12 * (fmax - fmin)
        fmin, fmax = fmin - pad, fmax + pad

        def py(v):
            return y1 - (v - fmin) / (fmax - fmin) * (y1 - y0)

        # axis
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(1, 1, 1, 0.28)
        ctx.move_to(x0, py(0) + 0.5)
        ctx.line_to(x1, py(0) + 0.5)
        ctx.stroke()
        # curve
        ctx.set_line_width(4)
        ctx.set_source_rgba(*rgb(WHITE), 0.9)
        ctx.move_to(px(ds[0]), py(fs[0]))
        for d, v in zip(ds[1:], fs[1:]):
            ctx.line_to(px(d), py(v))
        ctx.stroke()
        # the root
        ctx.set_source_rgba(*rgb(GREEN), 0.9)
        ctx.arc(px(0.0), py(0), 7, 0, 2 * math.pi)
        ctx.fill()
        font(ctx, "bold", 22)
        ctx.set_source_rgba(*rgb(GREEN), 0.9)
        s = "√2"
        ctx.move_to(px(0.0) - ctx.text_extents(s).x_advance / 2, py(0) + 40)
        ctx.show_text(s)
        # completed tangents (faded) and current one
        for k in range(n + (1 if phase == "tangent" else 0)):
            d0, d1 = self.off[k], self.off[k + 1]
            if not (c - 3 * hw < d0 < c + 3 * hw):
                continue
            is_cur = (phase == "tangent" and k == n)
            is_last = is_cur or k == n - 1
            uu = ease_out_cubic(u) if is_cur else 1.0
            xa, ya = px(d0), py(f(d0))
            xb, yb = px(d1), py(0)
            # the tangent as a line: a little beyond the touching point, and (after landing) a little past the axis
            ex = 0.35
            xs_, ys_ = xa - (xb - xa) * ex, ya - (yb - ya) * ex
            xe, ye = xa + (xb - xa) * (uu * (1 + ex)), ya + (yb - ya) * (uu * (1 + ex))
            ctx.set_line_width(4 if is_last else 2.5)
            ctx.set_source_rgba(*rgb(AMBER), 0.95 if is_last else 0.4)
            ctx.move_to(xs_, ys_)
            ctx.line_to(xe, ye)
            ctx.stroke()
            ctx.set_source_rgba(*rgb(AMBER), 0.95 if is_last else 0.5)
            ctx.arc(xa, ya, 7, 0, 2 * math.pi)
            ctx.fill()
            font(ctx, "bold", 22)
            lab = f"x{chr(0x2080 + k)}"
            ctx.set_source_rgba(*rgb(AMBER), 0.9 if is_last else 0.5)
            ctx.move_to(px(d0) - ctx.text_extents(lab).x_advance / 2, py(0) - 24)
            ctx.show_text(lab)
        # rise: dotted vertical from the new x up to the curve
        if phase == "rise" or (phase == "idle" and n > 0):
            k = n
            d1 = self.off[k]
            if c - 3 * hw < d1 < c + 3 * hw:
                uu = ease_out_cubic(u) if phase == "rise" else 1.0
                xb, yb, yc = px(d1), py(0), py(f(d1))
                ctx.set_line_width(2)
                ctx.set_dash([5, 7])
                ctx.set_source_rgba(1, 1, 1, 0.45)
                ctx.move_to(xb, yb)
                ctx.line_to(xb, yb + (yc - yb) * uu)
                ctx.stroke()
                ctx.set_dash([])
                ctx.set_source_rgba(*rgb(AMBER), 0.95)
                ctx.arc(xb, yb, 7, 0, 2 * math.pi)
                ctx.fill()
                font(ctx, "bold", 22)
                lab = f"x{chr(0x2080 + k)}"
                ctx.move_to(xb - ctx.text_extents(lab).x_advance / 2, py(0) - 24)
                ctx.show_text(lab)
        # scale label
        font(ctx, "bold", 20)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        if hw < 1.0:
            s = f"WINDOW  ±{hw:.0e}".replace("e-0", "e-").replace("e-", " × 10⁻")
            s = s.replace("1 × ", "")
            show_tracked(ctx, s, x0, y1 + 50, 0.08 * 20)
        else:
            show_tracked(ctx, "f(x) = x² − 2", x0, y1 + 50, 0.08 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._digits_hud(img, t, n, phase)

    def _digits_hud(self, img, t, n, phase):
        surf, ctx = layer()
        shown = n
        x = self.xs[shown]
        s = str(x)
        if "." not in s:
            s += "."
        ip, dp = s.split(".")
        dp = (dp + "0" * SHOWN_DECIMALS)[:SHOWN_DECIMALS]
        k_ok = self.digits[shown]
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "CORRECT DIGITS", 0.14 * 24)
        show_tracked(ctx, "CORRECT DIGITS", 910 - w, 470, 0.14 * 24)
        big = {str(c): 50 for c in range(10)}
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, str(shown), 70, 545, big)
        sv = str(k_ok)
        ctx.set_source_rgb(*rgb(GREEN))
        show_cells(ctx, sv, 910 - cells_width(ctx, sv, big), 545, big)
        # the number, correct digits in green
        font(ctx, "dbold", 34)
        cw = 25
        chars = list(ip) + ["."] + list(dp)
        total = len(chars) * cw - 8
        sx = CX - total / 2
        idx_digit = 0
        for ch in chars:
            if ch == ".":
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.move_to(sx + cw / 2 - ctx.text_extents(ch).x_advance / 2 - 6, 660)
                ctx.show_text(ch)
                sx += cw - 12
                continue
            ok = idx_digit < k_ok
            ctx.set_source_rgba(*(rgb(GREEN) if ok else (1, 1, 1)), 0.95 if ok else 0.35)
            ctx.move_to(sx + cw / 2 - ctx.text_extents(ch).x_advance / 2, 660)
            ctx.show_text(ch)
            sx += cw
            idx_digit += 1
        if k_ok > SHOWN_DECIMALS + 1:
            font(ctx, "bold", 20)
            ctx.set_source_rgba(*rgb(GREEN), 0.8)
            show_tracked(ctx, f"… {k_ok} DIGITS CORRECT", sx + 10, 660, 0.08 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ complex plane

    def plane_params(self, t):
        """(cx, cy, hw, maxit, mode) for the complex-plane picture at time t."""
        if t < ITER_T0:
            return 0.0, 0.0, HW0, 0, "iter"
        if t < ZOOM_T0:
            k = min(ITER_MAX, 1 + int((t - ITER_T0) / ITER_DT))
            return 0.0, 0.0, HW0, k, "iter"
        u = min(1.0, (t - ZOOM_T0) / (ZOOM_T1 - ZOOM_T0))
        u = smooth(u) if u < 1 else 1.0
        hw = math.exp(math.log(HW0) + (math.log(HW1) - math.log(HW0)) * u)
        return ZOOM_CENTER[0], ZOOM_CENTER[1], hw, 120, "zoom"

    def _plane(self, t):
        cx, cy, hw, maxit, mode = self.plane_params(t)
        idx = np.zeros((H, W), np.int8)
        cnt = np.zeros((H, W), np.int32)
        newton_plane(idx, cnt, cx, cy, hw, maxit, 1e-9 if mode == "zoom" else 1e-4)
        col = self.root_cols[idx]                      # (H, W, 3) linear
        if mode == "iter":
            conv = cnt <= maxit
            shade = np.where(conv, 0.95, 0.22).astype(np.float32)
        else:
            c = np.minimum(cnt, maxit).astype(np.float32)
            shade = (0.18 + 0.82 * np.exp(-c / 9.0)).astype(np.float32)
            shade[cnt > maxit] = 0.05
        img_lin = col * shade[..., None]
        if self._vig is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
            self._vig = np.clip(1 - 0.45 * np.clip(r - 0.55, 0, 1) ** 1.5, 0, 1)[..., None].astype(np.float32)
        img_lin *= self._vig
        return img_lin, idx, cnt, (cx, cy, hw, maxit, mode)

    def _plane_hud(self, img, t, params):
        cx, cy, hw, maxit, mode = params
        surf, ctx = layer()
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.6)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22})
        if mode == "iter":
            show_tracked(ctx, "ITERATION", 70, 470, 0.14 * 24)
            font(ctx, "dbold", 76)
            ctx.set_source_rgb(*rgb(WHITE))
            show_cells(ctx, str(maxit), 70, 545, big)
        else:
            show_tracked(ctx, "ZOOM", 70, 470, 0.14 * 24)
            font(ctx, "dbold", 76)
            ctx.set_source_rgb(*rgb(WHITE))
            z = HW0 / hw
            s = f"{z:,.0f}×" if z < 1e6 else f"{z:.1e}×"
            show_cells(ctx, s, 70, 545, big)
        # legend: the three roots
        font(ctx, "bold", 22)
        labs = (("1", AMBER), ("−½ + i√3/2", CYAN), ("−½ − i√3/2", MAGENTA))
        y = 478
        for lab, colr in labs:
            w = tracked_width(ctx, lab, 0.06 * 22)
            ctx.set_source_rgba(*rgb(colr), 0.95)
            ctx.arc(910 - w - 22, y - 7, 7, 0, 2 * math.pi)
            ctx.fill()
            show_tracked(ctx, lab, 910 - w, y, 0.06 * 22)
            y += 34
        if mode == "iter":
            # mark the roots on the plane
            for k, r in enumerate(ROOTS):
                x = W / 2 + (r.real - cx) / hw * (W / 2)
                yy = H / 2 - (r.imag - cy) / hw * (W / 2)
                ctx.set_source_rgba(1, 1, 1, 0.95)
                ctx.arc(x, yy, 7, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_line_width(2)
                ctx.set_source_rgba(0, 0, 0, 0.6)
                ctx.arc(x, yy, 9, 0, 2 * math.pi)
                ctx.stroke()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        mix = 0.0 if t < PLANE_T0 - 0.4 else smooth((t - (PLANE_T0 - 0.4)) / 0.4)
        if fade < 1.0:
            mix *= fade
        out = None
        if mix < 1.0:
            img = linear_to_srgb(self.bg).astype(np.float32)
            self._real_line(img, t if t < PLANE_T0 else (0.0 if fade < 1 else t))
            out = img
        if mix > 0.0:
            lin, idx, cnt, params = self._plane(t)
            small = cv2.resize(lin, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
            lin = lin + 0.25 * glow
            pimg = linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
            self._plane_hud(pimg, t, params)
            out = pimg if out is None else out * (1 - mix) + pimg * mix
        _ruin.Scene._headlines(self, out, f, t)
        return dither(out, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s digits {sc.digits} errs {['%.1e' % e for e in sc.errs]}")
    frames = [0, 160, 250, 320, 370, 440, 560, 650, 720, 800, 880, 930, 980, 1050, 1120, 1200, 1300, 1439]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (250, 440, 1150):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
