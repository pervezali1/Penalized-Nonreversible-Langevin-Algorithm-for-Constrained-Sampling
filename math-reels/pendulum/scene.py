"""Frame renderer for "10,000 Pendulums. Then I Rewound Time."  frame(f) -> uint8 1080x1920 RGB."""

import json
import math

import cairo
import cv2
import numpy as np

import timemap as tm
from raster import (H, W, linear_to_srgb, pendulum_colours, splat_pendulums, splat_trails,
                    srgb_to_linear)
from sim import N, PIVOT, state_at

AMBER = "#FFC94D"
MAGENTA = "#FF3D7F"
CYAN = "#5CE1FF"
WHITE = "#F5F7FF"
CX = 470                     # visual centre column
LENS_C = (470, 708)
LENS_R = 140
LENS_WIN = 360               # local window around the lens centre
WEIGHT = 0.045
TRAIL = 16
MAX_TEXT_W = 820


def rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def font(ctx, kind, size):
    fam, wt = {"black": ("Inter Display Black", cairo.FONT_WEIGHT_NORMAL),
               "dbold": ("Inter Display", cairo.FONT_WEIGHT_BOLD),
               "bold": ("Inter", cairo.FONT_WEIGHT_BOLD),
               "regular": ("Inter", cairo.FONT_WEIGHT_NORMAL)}[kind]
    ctx.select_font_face(fam, cairo.FONT_SLANT_NORMAL, wt)
    ctx.set_font_size(size)


def tracked_width(ctx, s, tr):
    return sum(ctx.text_extents(ch).x_advance for ch in s) + tr * (len(s) - 1)


def show_tracked(ctx, s, x, y, tr):
    for ch in s:
        ctx.move_to(x, y)
        ctx.show_text(ch)
        x += ctx.text_extents(ch).x_advance + tr


def ease_out_cubic(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_out_back(u, s=1.70158):
    u = min(max(u, 0.0), 1.0) - 1
    return u * u * ((s + 1) * u + s) + 1


def layer():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def surf_rgba(surf):
    """Premultiplied float32 RGB and alpha from a cairo ARGB32 surface."""
    surf.flush()
    h, w = surf.get_height(), surf.get_width()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return buf[..., 2::-1].astype(np.float32) / 255, buf[..., 3].astype(np.float32) / 255


def over(img, rgb_premul, a):
    img *= (1 - a)[..., None]
    img += rgb_premul


# ------------------------------------------------------------------ text cues

HOOK = [("This is", "dbold", 54, WHITE, 292, 0.8), ("10,000 pendulums.", "black", 88, WHITE, 384, 1.0)]


def build_cues(t_chaos):
    """(t_in, t_out, lines, enter_frames, exit_frames, style)."""
    return [
        (0.0, 1.50, HOOK, 0, 6, "none"),
        (1.62, 3.70, [("Guess the second", "black", 80, WHITE, 312, 1), ("they split.", "black", 80, WHITE, 396, 1)], 8, 6, "pop"),
        (3.88, 5.72, [("Watch the zeros vanish ↓", "black", 66, WHITE, 350, 1)], 8, 6, "pop"),
        (5.90, 6.98, [("Here it comes…", "black", 84, MAGENTA, 350, 1)], 8, 6, "pop"),
        (t_chaos, 8.85, [("CHAOS.", "black", 170, MAGENTA, 385, 1)], 6, 6, "chaos"),
        (8.97, 9.92, [("180 million×", "black", 96, AMBER, 312, 1), ("bigger in under 8.3 seconds.", "dbold", 52, WHITE, 396, 1)], 8, 6, "pop"),
        (10.05, 10.95, [("Now rewind.", "black", 88, CYAN, 350, 1)], 8, 6, "pop"),
        (11.55, 13.70, [("Every single one", "black", 80, WHITE, 312, 1), ("comes back.", "black", 80, WHITE, 396, 1)], 8, 6, "pop"),
        (14.25, 16.80, [("Chaos isn't random.", "black", 80, WHITE, 312, 1)], 8, 6, "pop"),
        (14.85, 16.80, [("It's just sensitive.", "dbold", 60, AMBER, 396, 1)], 8, 6, "pop"),
        (17.20, 18.10, HOOK, 10, 0, "fade"),
    ]


class Scene:
    def __init__(self, sim_path, frames_path):
        sim = np.load(sim_path)
        self.snaps = np.ascontiguousarray(sim["snaps"])
        self.dt, self.snap_every = float(sim["dt"]), int(sim["snap_every"])
        fr = np.load(frames_path)
        self.d, self.speed, self.tips = fr["d"], fr["speed"], fr["tips"]
        self.lens_pts, self.mean_tip, self.since = fr["lens_pts"], fr["mean_tip"], fr["since"]
        with open(frames_path.replace(".npz", ".json")) as fh:
            meta = json.load(fh)
        self.meta = meta
        self.f_chaos = meta["chaos_frame"]
        self.t_chaos = meta["t_chaos"]
        self.gap_str = meta["gap_str"]
        self.lens_k = meta["lens_k"]
        self.lens_tween = meta["lens_tween"]
        self.cols = pendulum_colours(N)
        sel = np.r_[np.arange(0, N, 10), N - 1]
        disp = self.cols[sel] / self.cols[sel].max(axis=1, keepdims=True)
        self.lens_cols = linear_to_srgb(disp)
        self.cues = build_cues(self.t_chaos)
        self.bg = self._background()
        yy, xx = np.mgrid[0:LENS_WIN, 0:LENS_WIN].astype(np.float32)
        self.lens_rr = np.sqrt((xx - LENS_WIN / 2) ** 2 + (yy - LENS_WIN / 2) ** 2)
        self.glass = np.array(srgb_to_linear(rgb("#070A14")) ** (1 / 2.2), np.float32)  # approx sRGB
        self.glass = np.array(rgb("#070A14"), np.float32)
        self._check_text()

    # ---------------------------------------------------------------- setup

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - PIVOT[0]) ** 2 + (yy - PIVOT[1]) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#0E1428")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060B")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        bg = c0 * (1 - u) + c1 * u
        rc = math.hypot(PIVOT[0], PIVOT[1])
        v = np.clip((r - 700) / (rc - 700), 0, 1)
        v = v * v * (3 - 2 * v)
        bg *= (1 - 0.15 * v)[..., None]
        return bg.astype(np.float32)

    def _check_text(self):
        _, ctx = layer()
        self.text_scale = {}
        for cue in self.cues:
            for (s, kind, size, _c, _b, _a) in cue[2]:
                font(ctx, kind, size)
                w = ctx.text_extents(s).x_advance
                self.text_scale[(s, size)] = min(1.0, MAX_TEXT_W / w)
                assert w * self.text_scale[(s, size)] <= MAX_TEXT_W + 1

    # ---------------------------------------------------------------- frame

    def frame(self, f):
        f %= tm.N_FRAMES
        t = f / tm.FPS
        d = self.d[f]
        cam = tm.camera_scale(t, self.t_chaos)
        sx, sy = tm.shake(f, self.f_chaos)

        acc = np.zeros((H, W, 3), np.float32)
        sp = abs(self.speed[f])
        n_sub = 3 if sp <= 1.2 else (5 if sp <= 2.2 else 8)
        dd = (self.d[(f + 1) % tm.N_FRAMES] - self.d[(f - 1) % tm.N_FRAMES]) / 2
        for lam in np.linspace(-0.25, 0.25, n_sub):
            st = state_at(self.snaps, abs(d + lam * dd), self.dt, self.snap_every)
            splat_pendulums(acc, st, self.cols, WEIGHT / n_sub, cam, sx, sy)
        idx = [(f - k) % tm.N_FRAMES for k in range(TRAIL + 1)]
        splat_trails(acc, self.tips[idx], self.cols, WEIGHT, cam, sx, sy)

        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 2)
        b2 = cv2.GaussianBlur(small, (0, 0), 8)
        glow = cv2.resize(0.45 * b1 + 0.30 * b2, (W, H), interpolation=cv2.INTER_LINEAR)
        fl = tm.flash(f, self.f_chaos)
        e = 1 - np.exp(-1.5 * (1 + 20 * fl) * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        if fl:
            lin += fl
        img = linear_to_srgb(lin).astype(np.float32)

        self._lens(img, f, d, cam, sx, sy)
        self._hud(img, f, t, d, cam, sx, sy)
        if 612 <= f <= 617:
            img = self._glitch(img, f)
        rng = np.random.default_rng(f)
        noise = (rng.random((H, W, 1), np.float32) + rng.random((H, W, 1), np.float32) - 1) / 255
        return np.clip((img + noise) * 255 + 0.5, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- lens

    def _lens(self, img, f, d, cam, sx, sy):
        scale, alpha = tm.lens_presence(d)
        if alpha <= 0:
            return
        R = LENS_R * scale
        x0, y0 = LENS_C[0] - LENS_WIN // 2, LENS_C[1] - LENS_WIN // 2
        win = img[y0:y0 + LENS_WIN, x0:x0 + LENS_WIN]
        mask = np.clip((R - 4) - self.lens_rr + 0.5, 0, 1)[..., None] * alpha
        # glass
        win *= 1 - 0.88 * mask
        win += 0.88 * mask * self.glass
        # contents
        mag = 10 ** self.lens_k[f] * scale
        pts = self.lens_pts[f] * mag + LENS_WIN / 2
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, LENS_WIN, LENS_WIN)
        ctx = cairo.Context(surf)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_width(4 * scale)
        lim = LENS_WIN * 3
        for j in range(len(pts) - 1):
            xa, ya = pts[j]
            xb, yb = pts[j + 1]
            if max(abs(xa), abs(ya), abs(xb), abs(yb)) > lim:
                continue
            ctx.move_to(xa, ya)
            ctx.line_to(xb, yb)
            ctx.set_source_rgb(*self.lens_cols[j])
            ctx.stroke()
        crgb, ca = surf_rgba(surf)
        gl = cv2.GaussianBlur(crgb, (0, 0), 6 * scale) * 0.6
        win += gl * mask
        a = ca[..., None] * mask
        win *= 1 - a
        win += crgb * mask
        np.clip(win, 0, 1, out=win)

    # ---------------------------------------------------------------- HUD

    def _hud(self, img, f, t, d, cam, sx, sy):
        surf, ctx = layer()
        # reach guide + pivot
        ctx.set_line_width(1)
        ctx.set_source_rgba(1, 1, 1, 0.04)
        ctx.arc(PIVOT[0], PIVOT[1], 330 * cam, 0, 2 * math.pi)
        ctx.stroke()
        ctx.set_line_width(2)
        ctx.set_source_rgba(1, 1, 1, 0.8)
        ctx.arc(PIVOT[0] + sx, PIVOT[1] + sy, 8, 0, 2 * math.pi)
        ctx.stroke()

        self._lens_chrome(ctx, f, d, cam, sx, sy)
        self._counter(ctx, f)
        self._timer(ctx, f, t, d)
        self._pill(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

    def _lens_chrome(self, ctx, f, d, cam, sx, sy):
        scale, alpha = tm.lens_presence(d)
        if alpha <= 0:
            return
        cx, cy = LENS_C
        R = LENS_R * scale
        # leader line to the on-screen tip
        tx = PIVOT[0] + cam * (self.mean_tip[f, 0] - PIVOT[0]) + sx
        ty = PIVOT[1] + cam * (self.mean_tip[f, 1] - PIVOT[1]) + sy
        ang = math.atan2(ty - cy, tx - cx)
        ex, ey = cx + R * math.cos(ang), cy + R * math.sin(ang)
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(1, 1, 1, 0.30 * alpha)
        ctx.move_to(ex, ey)
        dist = math.hypot(tx - ex, ty - ey)
        if dist > 12:
            ctx.line_to(tx - 12 * math.cos(ang), ty - 12 * math.sin(ang))
            ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.6 * alpha)
        ctx.arc(tx, ty, 12, 0, 2 * math.pi)
        ctx.stroke()
        # outer soft glow + ring + crosshair
        for i in range(8, 0, -1):
            ctx.set_line_width(2.5 + 2 * i)
            ctx.set_source_rgba(1, 1, 1, 0.25 * alpha / 8 * 0.5)
            ctx.arc(cx, cy, R, 0, 2 * math.pi)
            ctx.stroke()
        ctx.set_line_width(2.5)
        ctx.set_source_rgba(1, 1, 1, 0.70 * alpha)
        ctx.arc(cx, cy, R, 0, 2 * math.pi)
        ctx.stroke()
        ctx.set_line_width(1)
        ctx.set_source_rgba(1, 1, 1, 0.10 * alpha)
        ctx.move_to(cx - R + 6, cy); ctx.line_to(cx + R - 6, cy)
        ctx.move_to(cx, cy - R + 6); ctx.line_to(cx, cy + R - 6)
        ctx.stroke()
        # label
        dim = 0.45 if self.lens_tween[f] else 1.0
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
        w = tracked_width(ctx, "ZOOM", 0.14 * 22)
        show_tracked(ctx, "ZOOM", 312 - w, 700, 0.14 * 22)
        font(ctx, "dbold", 34)
        s = tm.fmt_mag(self.lens_k[f])
        ctx.set_source_rgba(*rgb(WHITE), alpha * dim)
        ctx.move_to(312 - ctx.text_extents(s).x_advance, 740)
        ctx.show_text(s)

    def _cells(self, ctx, s, x, base, widths):
        for ch in s:
            cw = widths.get(ch)
            adv = ctx.text_extents(ch).x_advance
            if cw is None:
                cw = adv
            ctx.move_to(x + (cw - adv) / 2, base)
            ctx.show_text(ch)
            x += cw

    def _cell_width(self, ctx, s, widths):
        return sum(widths.get(ch, ctx.text_extents(ch).x_advance) for ch in s)

    def _counter(self, ctx, f):
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "LARGEST GAP SO FAR", 70, 440, 0.14 * 24)
        s = self.gap_str[f]
        val = float(s.rstrip("°+"))
        col = np.array(rgb(AMBER if val < 1 else MAGENTA))
        since = self.since[f]
        if since < 4:
            col = col + (1 - col) * 0.6 * (1 - since / 4)
        sc = 1 + 0.08 * (1 - since / 6) if since < 6 else 1.0
        font(ctx, "dbold", 76)
        ctx.save()
        ctx.translate(70, 515)
        ctx.scale(sc, sc)
        ctx.set_source_rgb(*col)
        widths = {str(k): 50 for k in range(10)}
        widths.update({".": 22, "°": 36, "+": 46})
        self._cells(ctx, s, 0, 0, widths)
        ctx.restore()

    def _timer(self, ctx, f, t, d):
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        w = tracked_width(ctx, "TIME", 0.14 * 24)
        show_tracked(ctx, "TIME", 910 - w, 440, 0.14 * 24)
        shown = d
        if 0 <= f - self.f_chaos < 30:
            shown = self.d[self.f_chaos]
        s = f"{shown:.2f} s"
        sc = 1.0
        if t < 9.8:  # pulse on whole simulated seconds during the forward run
            frac = d - math.floor(d)
            k = frac * tm.FPS
            if k < 12:
                sc = 1 + 0.10 * math.sin(math.pi * k / 12)
        font(ctx, "dbold", 52)
        widths = {str(k): 34 for k in range(10)}
        widths.update({".": 15, " ": 14})
        tw = self._cell_width(ctx, s, widths)
        ctx.save()
        ctx.translate(910, 515)
        ctx.scale(sc, sc)
        ctx.set_source_rgb(*rgb(WHITE))
        self._cells(ctx, s, -tw, 0, widths)
        ctx.restore()

    def _pill(self, ctx, t):
        a = min(1.0, (t - 10.30) * 10, (17.80 - t) * 10)
        if a <= 0:
            return
        font(ctx, "bold", 28)
        s = "◀◀ REWIND"
        w = ctx.text_extents(s).x_advance
        x0, y0 = 910 - w - 12, 562 - 28 - 6 + 4
        bw, bh = w + 24, 28 + 12
        r = 18
        ctx.new_sub_path()
        ctx.arc(x0 + bw - r, y0 + r, r, -math.pi / 2, 0)
        ctx.arc(x0 + bw - r, y0 + bh - r, r, 0, math.pi / 2)
        ctx.arc(x0 + r, y0 + bh - r, r, math.pi / 2, math.pi)
        ctx.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        ctx.set_source_rgba(*rgb(CYAN), 0.14 * a)
        ctx.fill()
        ctx.set_source_rgba(*rgb(CYAN), a)
        ctx.move_to(910 - w, 562)
        ctx.show_text(s)

    def _headlines(self, img, f, t):
        for (t_in, t_out, lines, n_in, n_out, style) in self.cues:
            if not (t_in - 1e-6 <= t < t_out + n_out / tm.FPS):
                continue
            k_in = (t - t_in) * tm.FPS
            k_out = (t - t_out) * tm.FPS
            alpha = 1.0
            if n_in and k_in < n_in:
                alpha = k_in / n_in if style == "fade" else ease_out_cubic(k_in / n_in)
            if k_out >= 0:
                alpha = min(alpha, 1 - k_out / n_out) if n_out else alpha
            if alpha <= 0:
                continue
            sc, dy = 1.0, 0.0
            if style == "pop" and k_in < n_in:
                u = ease_out_cubic(k_in / n_in)
                sc, dy = 0.92 + 0.08 * u, 18 * (1 - u)
            if style == "chaos" and k_in < n_in:
                sc = 1.30 - 0.30 * ease_out_back(k_in / n_in)
            surf, ctx = layer()
            tops, bots = [], []
            for (s, kind, size, colr, base, la) in lines:
                font(ctx, kind, size)
                fs = self.text_scale[(s, size)]
                ext = ctx.text_extents(s)
                w = ext.x_advance * fs
                cy = base - size * 0.36
                ctx.save()
                ctx.translate(CX, cy + dy)
                ctx.scale(sc * fs, sc * fs)
                ctx.translate(-CX, -cy)
                ctx.set_source_rgba(*rgb(colr), la)
                ctx.move_to(CX - w / 2 / fs, base)
                ctx.show_text(s)
                ctx.restore()
                tops.append(base - size)
                bots.append(base + size * 0.3)
            y0, y1 = max(0, int(min(tops) - 60)), min(H, int(max(bots) + 60))
            trgb, ta = surf_rgba(surf)
            band = slice(y0, y1)
            a_band = ta[band]
            if style == "chaos":
                g = cv2.GaussianBlur(a_band, (0, 0), 14) * 0.8 * alpha
                img[band] += g[..., None] * np.array(rgb(MAGENTA), np.float32)
            sh = cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha
            sh = np.roll(sh, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)

    # ---------------------------------------------------------------- glitch

    def _glitch(self, img, f):
        rng = np.random.default_rng(1000 + f)
        out = img.copy()
        for _ in range(8):
            h = int(rng.integers(24, 61))
            y = int(rng.integers(0, H - h))
            sh = int(rng.integers(12, 31)) * (1 if rng.random() < 0.5 else -1)
            out[y:y + h] = np.roll(img[y:y + h], sh, axis=1)
        out[:, 4:, 0] = out[:, :-4, 0]
        out[::3] *= 0.85
        return out
