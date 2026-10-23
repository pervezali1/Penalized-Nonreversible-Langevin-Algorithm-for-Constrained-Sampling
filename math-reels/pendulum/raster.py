"""Numba splat rasteriser: 10,000 emissive pendulums into a float32 linear-light buffer."""

import math

import numpy as np
from numba import njit

from sim import L_PX, PIVOT

W, H = 1080, 1920


@njit(cache=True, inline="always")
def _bilinear(acc, x, y, r, g, b, w):
    if x < 0 or y < 0 or x >= W - 1 or y >= H - 1:
        return
    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0
    w00 = (1 - fx) * (1 - fy) * w
    w10 = fx * (1 - fy) * w
    w01 = (1 - fx) * fy * w
    w11 = fx * fy * w
    acc[y0, x0, 0] += r * w00; acc[y0, x0, 1] += g * w00; acc[y0, x0, 2] += b * w00
    acc[y0, x0 + 1, 0] += r * w10; acc[y0, x0 + 1, 1] += g * w10; acc[y0, x0 + 1, 2] += b * w10
    acc[y0 + 1, x0, 0] += r * w01; acc[y0 + 1, x0, 1] += g * w01; acc[y0 + 1, x0, 2] += b * w01
    acc[y0 + 1, x0 + 1, 0] += r * w11; acc[y0 + 1, x0 + 1, 1] += g * w11; acc[y0 + 1, x0 + 1, 2] += b * w11


@njit(cache=True, inline="always")
def _segment(acc, x0, y0, x1, y1, r, g, b, w, step):
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    n = int(length / step) + 1
    ws = w * step
    for k in range(n):
        u = (k + 0.5) / n
        _bilinear(acc, x0 + dx * u, y0 + dy * u, r, g, b, ws)


@njit(cache=True, inline="always")
def _gauss(acc, cx, cy, sigma, r, g, b, w):
    rad = int(sigma * 2.5) + 1
    x0 = int(cx)
    y0 = int(cy)
    inv = -0.5 / (sigma * sigma)
    for yy in range(y0 - rad, y0 + rad + 2):
        if yy < 0 or yy >= H:
            continue
        for xx in range(x0 - rad, x0 + rad + 2):
            if xx < 0 or xx >= W:
                continue
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            v = w * math.exp(d2 * inv)
            acc[yy, xx, 0] += r * v
            acc[yy, xx, 1] += g * v
            acc[yy, xx, 2] += b * v


@njit(cache=True)
def splat_pendulums(acc, state, cols, w, cam, sx, sy):
    """Rods + joint + bob for every pendulum. cam = zoom about the pivot; (sx, sy) = shake."""
    px = PIVOT[0] + sx
    py = PIVOT[1] + sy
    L = L_PX * cam
    for i in range(state.shape[0]):
        t1 = state[i, 0]
        t2 = state[i, 1]
        jx = px + L * math.sin(t1)
        jy = py + L * math.cos(t1)
        tx = jx + L * math.sin(t2)
        ty = jy + L * math.cos(t2)
        r = cols[i, 0]; g = cols[i, 1]; b = cols[i, 2]
        _segment(acc, px, py, jx, jy, r, g, b, w, 0.6)
        _segment(acc, jx, jy, tx, ty, r, g, b, w, 0.6)
        _gauss(acc, jx, jy, 1.2, r, g, b, 1.5 * w)
        _gauss(acc, tx, ty, 1.8, r, g, b, 3.0 * w)


@njit(cache=True)
def splat_trails(acc, tips, cols, w, cam, sx, sy):
    """tips: (K+1, N, 2) unscaled tip positions, index 0 = current frame, k = k frames ago."""
    px = PIVOT[0]
    py = PIVOT[1]
    K = tips.shape[0] - 1
    for i in range(tips.shape[1]):
        r = cols[i, 0]; g = cols[i, 1]; b = cols[i, 2]
        x0 = px + cam * (tips[0, i, 0] - px) + sx
        y0 = py + cam * (tips[0, i, 1] - py) + sy
        for k in range(K):
            x1 = px + cam * (tips[k + 1, i, 0] - px) + sx
            y1 = py + cam * (tips[k + 1, i, 1] - py) + sy
            _segment(acc, x0, y0, x1, y1, r, g, b, 0.5 * w * 0.82 ** k, 1.0)
            x0 = x1
            y0 = y1


@njit(cache=True)
def tips_of(state):
    n = state.shape[0]
    out = np.empty((n, 2))
    for i in range(n):
        out[i, 0] = PIVOT[0] + L_PX * (math.sin(state[i, 0]) + math.sin(state[i, 1]))
        out[i, 1] = PIVOT[1] + L_PX * (math.cos(state[i, 0]) + math.cos(state[i, 1]))
    return out


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


ANCHORS = ["#FF3B3B", "#FF9F1C", "#FFE14D", "#3DFF8F", "#2EE6FF", "#3D7BFF", "#A64DFF"]


def pendulum_colours(n):
    """Spectrum in linear light, normalised so the n colours sum to neutral white."""
    anc = np.array([srgb_to_linear([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]) for h in ANCHORS])
    u = np.arange(n) / (n - 1)
    pos = u * (len(ANCHORS) - 1)
    k = np.minimum(pos.astype(int), len(ANCHORS) - 2)
    f = (pos - k)[:, None]
    cols = anc[k] * (1 - f) + anc[k + 1] * f
    cols /= cols.mean(axis=0, keepdims=True)
    return cols.astype(np.float32)
