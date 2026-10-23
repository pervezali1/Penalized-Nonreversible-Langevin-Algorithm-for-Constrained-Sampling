"""Shared cairo text/layer helpers for the reels."""

import cairo
import numpy as np

W, H = 1080, 1920


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


def show_cells(ctx, s, x, base, widths):
    """Fixed-advance glyph cells so counters don't jitter."""
    for ch in s:
        adv = ctx.text_extents(ch).x_advance
        cw = widths.get(ch, adv)
        ctx.move_to(x + (cw - adv) / 2, base)
        ctx.show_text(ch)
        x += cw


def cells_width(ctx, s, widths):
    return sum(widths.get(ch, ctx.text_extents(ch).x_advance) for ch in s)


def ease_out_cubic(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_out_back(u, s=1.70158):
    u = min(max(u, 0.0), 1.0) - 1
    return u * u * ((s + 1) * u + s) + 1


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def layer(w=W, h=H):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
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


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def dither(img, f):
    rng = np.random.default_rng(f)
    noise = (rng.random(img.shape[:2] + (1,), np.float32) + rng.random(img.shape[:2] + (1,), np.float32) - 1) / 255
    return np.clip((img + noise) * 255 + 0.5, 0, 255).astype(np.uint8)
