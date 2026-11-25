"""Facebook page cover for Math in Motion, drawn from the reels' own computations.

Left: the Galton board's 10,000 balls in their 17 piles (galton/gsim.py, seed 68), each ball a dot.
Right: Riemann's zeta function along the critical line, zeta(1/2 + i t) for t from 0 to 33 (mpmath), the curve
looping through 0 at the first five zeros, with those zeros as beads on the line.
Centre: the name, the tagline, the topics.

The image is 1640 x 924 (16:9). Facebook shows the whole of it on phones (640 x 360) and a 1640 x 624 band through the
middle on desktop (820 x 312). Everything sits inside that band.
    python fbcover.py <out.png>  -> out.png and a preview of the crops next to it"""

import math
import os
import sys

import cairo
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "galton"))
from textkit import font, linear_to_srgb, rgb, show_tracked, srgb_to_linear, tracked_width  # noqa: E402
import gsim  # noqa: E402

W, H = 1640, 924
SAFE = (0, 150, 1640, 774)                        # x0, y0, x1, y1: the desktop band
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"


def layer():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def rgba(surf):
    surf.flush()
    buf = np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :W]
    return buf[..., 2::-1].astype(np.float32) / 255, buf[..., 3].astype(np.float32) / 255


def galton(g):
    """The 17 piles, one dot per ball, 14 to a layer: bottom-left corner."""
    counts = np.bincount(gsim.flips().sum(1), minlength=gsim.ROWS + 1)
    bw, x0, floor = 26.0, 34.0, 742.0
    per, rh = 8, 2.1
    for k, n in enumerate(counts):
        u = abs(k - 8) / 8
        col = [a + (b - a) * min(1.0, u * 1.6) for a, b in zip(rgb(CYAN), rgb(MAGENTA))]
        g.set_source_rgba(*col, 0.95)
        for i in range(int(n)):
            lay, c = divmod(i, per)
            g.arc(x0 + k * bw + 3 + (c + 0.5) * (bw - 6) / per, floor - (lay + 0.5) * rh, 0.95, 0, 2 * math.pi)
            g.fill()
    # the pegs above, fading out
    for j in range(16):
        for i in range(j + 1):
            x = x0 + 8 * bw + bw / 2 + (i - j / 2) * bw
            y = 176 + j * 14.5
            g.set_source_rgba(*rgb("#7C88B5"), 0.15 + 0.2 * j / 15)
            g.arc(x, y, 2.2, 0, 2 * math.pi)
            g.fill()
    return counts


def zeta_curve(g):
    """zeta(1/2 + it) for t in [0, 33]: right side, with the first five zeros on the line beside it."""
    import mpmath as mp
    mp.mp.dps = 15
    ts = np.arange(0, 33.0, 0.02)
    z = np.array([complex(mp.zeta(mp.mpc(0.5, t))) for t in ts])
    zeros = [float(mp.zetazero(n).imag) for n in range(1, 6)]
    ox, oy, sc = 1440.0, 462.0, 78.0
    n = len(z)
    K = 24
    for c in range(K):
        j0, j1 = c * (n - 1) // K, (c + 1) * (n - 1) // K + 1
        u = (c + 0.5) / K
        g.set_source_rgba(*rgb(CYAN), 0.25 + 0.75 * u)
        g.set_line_width(2.0 + 2.0 * u)
        g.move_to(ox + sc * z[j0].real, oy - sc * z[j0].imag)
        for p in z[j0 + 1:j1]:
            g.line_to(ox + sc * p.real, oy - sc * p.imag)
        g.stroke()
    g.set_source_rgba(*rgb(WHITE), 0.9)
    g.arc(ox, oy, 5, 0, 2 * math.pi)
    g.fill()
    # the critical line with its first five zeros
    lx, y0, y1 = 1612.0, 750.0, 172.0
    g.set_source_rgba(*rgb(AMBER), 0.7)
    g.set_line_width(2.5)
    g.move_to(lx, y0)
    g.line_to(lx, y1)
    g.stroke()
    for gam in zeros:
        g.set_source_rgba(*rgb(CYAN), 1)
        g.arc(lx, y0 - (y0 - y1) * gam / 34.0, 5, 0, 2 * math.pi)
        g.fill()
    return zeros


def main(out):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((xx - W / 2) ** 2 + ((yy - H / 2) * 1.6) ** 2)
    c0 = np.array(srgb_to_linear(rgb("#121830")), np.float32)
    c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
    u = np.clip(r / 900, 0, 1)[..., None]
    bg = (c0 * (1 - u) + c1 * u).astype(np.float32)
    gs, g = layer()
    counts = galton(g)
    zeros = zeta_curve(g)
    g_rgb, g_a = rgba(gs)
    emis = srgb_to_linear(g_rgb / np.maximum(g_a, 1e-4)[..., None]).astype(np.float32) * g_a[..., None] * 1.1
    small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(0.45 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                      (W, H), interpolation=cv2.INTER_LINEAR)
    e = 1 - np.exp(-1.4 * (emis + glow))
    img = linear_to_srgb(1 - (1 - bg) * (1 - e)).astype(np.float32)
    # a soft dark plate behind the words, so they read over the art
    plate = np.exp(-(((xx - W / 2) / 430) ** 2 + ((yy - 462) / 230) ** 2) ** 1.6)[..., None]
    img *= 1 - 0.55 * plate
    # the words
    ts, t = layer()
    cx = W / 2
    font(t, "bold", 24)
    s = "NEURAL NETWORKS  ·  PROBABILITY  ·  PRIMES  ·  LINEAR ALGEBRA"
    w = tracked_width(t, s, 0.14 * 24)
    t.set_source_rgba(1, 1, 1, 0.7)
    show_tracked(t, s, cx - w / 2, 318, 0.14 * 24)
    font(t, "black", 104)
    s = "MATH IN MOTION"
    w = t.text_extents(s).x_advance
    t.set_source_rgba(*rgb(WHITE), 1)
    t.move_to(cx - w / 2, 452)
    t.show_text(s)
    font(t, "dbold", 40)
    s = "Every number on screen is computed, not drawn."
    w = t.text_extents(s).x_advance
    t.set_source_rgba(*rgb(CYAN), 1)
    t.move_to(cx - w / 2, 526)
    t.show_text(s)
    font(t, "black", 28)
    s = "NEW REELS EVERY WEEK"
    w = tracked_width(t, s, 0.12 * 28)
    t.set_source_rgba(*rgb(AMBER), 1)
    t.set_line_width(2.5)
    pad = 22
    t.new_sub_path()
    rx, ry, rw, rh = cx - w / 2 - pad, 576, w + 2 * pad, 56
    rr = rh / 2
    t.arc(rx + rw - rr, ry + rr, rr, -math.pi / 2, math.pi / 2)
    t.arc(rx + rr, ry + rr, rr, math.pi / 2, 3 * math.pi / 2)
    t.close_path()
    t.stroke()
    show_tracked(t, s, cx - w / 2, ry + 38, 0.12 * 28)
    t_rgb, t_a = rgba(ts)
    sh = cv2.GaussianBlur(t_a, (0, 0), 8) * 0.7
    img *= (1 - np.roll(sh, 3, axis=0))[..., None]
    img = img * (1 - t_a[..., None]) + t_rgb
    rng = np.random.default_rng(1)
    noise = (rng.random((H, W, 1), np.float32) + rng.random((H, W, 1), np.float32) - 1) / 255
    out_img = np.clip((img + noise) * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(out, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    # preview: what a computer shows (the band) and what a phone shows (all of it), at Facebook's display sizes
    desk = out_img[(H - 624) // 2:(H - 624) // 2 + 624]
    rows = [cv2.resize(desk, (820, 312), interpolation=cv2.INTER_AREA), cv2.resize(out_img, (820, 462), interpolation=cv2.INTER_AREA)]
    prev = np.vstack([np.vstack([r, np.full((14, 820, 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(out.replace(".png", "_crops.png"), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
    print("galton piles", counts.tolist(), "| zeros", [round(z, 3) for z in zeros])


if __name__ == "__main__":
    main(sys.argv[1])
