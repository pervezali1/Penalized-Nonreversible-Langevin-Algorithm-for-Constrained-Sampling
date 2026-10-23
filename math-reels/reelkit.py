"""Tiny render kit for vertical math reels.

cairo draws anti-aliased vectors, numpy/OpenCV do bloom + grading,
a process pool renders frames in parallel and ffmpeg encodes them.
"""

import math
import os
import subprocess
from multiprocessing import get_context

import cairo
import cv2
import numpy as np

W, H = 1080, 1920
FPS = 60


# ---------------------------------------------------------------- colour


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else float(x >= e1)
    return t * t * (3 - 2 * t)


def ease_in_out(t):
    t = clamp01(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out_cubic(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.70158):
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


# ---------------------------------------------------------------- surfaces


def new_layer(w=W, h=H):
    """Transparent ARGB32 surface + context."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def layer_rgb(surf):
    """Premultiplied cairo surface -> float32 HxWx3 RGB (premultiplied, 0..1)."""
    surf.flush()
    h, w = surf.get_height(), surf.get_width()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    # little-endian ARGB32 is stored B, G, R, A
    return buf[..., 2::-1].astype(np.float32) * (1.0 / 255.0)


def layer_alpha(surf):
    surf.flush()
    h, w = surf.get_height(), surf.get_width()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return buf[..., 3].astype(np.float32) * (1.0 / 255.0)


# ---------------------------------------------------------------- effects


def bloom(img, radii=(6, 18, 48, 120), weights=(0.9, 0.6, 0.45, 0.35), threshold=0.0):
    """Multi-scale additive glow. img: float32 HxWx3 linear-ish RGB."""
    src = img if threshold <= 0 else np.maximum(img - threshold, 0)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for r, wt in zip(radii, weights):
        # blur at reduced resolution for speed, then upsample
        f = max(1, int(r // 6))
        small = cv2.resize(src, (w // f, h // f), interpolation=cv2.INTER_AREA) if f > 1 else src
        sig = r / f
        k = int(sig * 3) * 2 + 1
        b = cv2.GaussianBlur(small, (k, k), sig)
        if f > 1:
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        out += wt * b
    return out


def tonemap(img, white=1.0):
    """Soft-clip highlights so glow saturates gracefully instead of hard clipping."""
    x = np.maximum(img, 0)
    knee = 0.75
    over = x > knee
    y = x.copy()
    y[over] = knee + (white - knee) * (1 - np.exp(-(x[over] - knee) / (white - knee)))
    return y


def vignette(strength=0.35, w=W, h=H):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    r2 = dx * dx * 0.9 + dy * dy * 0.55
    return (1 - strength * np.clip(r2, 0, 1.6) / 1.6)[..., None].astype(np.float32)


_rng = np.random.default_rng(7)
_GRAIN = (_rng.random((H, W, 1), dtype=np.float32) - 0.5)


def finish(img, grain=1.6, frame_index=0):
    """float RGB 0..1 -> uint8 with temporal dither (kills banding after IG recompression)."""
    sh = (frame_index * 7919) % H
    g = np.roll(_GRAIN, sh, axis=0) * (grain / 255.0)
    return np.clip((img + g) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- text


def set_font(ctx, size, family="Inter", weight="bold"):
    w = cairo.FONT_WEIGHT_BOLD if weight == "bold" else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face(family, cairo.FONT_SLANT_NORMAL, w)
    ctx.set_font_size(size)
    fo = cairo.FontOptions()
    fo.set_antialias(cairo.ANTIALIAS_GRAY)
    fo.set_hint_style(cairo.HINT_STYLE_NONE)
    fo.set_hint_metrics(cairo.HINT_METRICS_OFF)
    ctx.set_font_options(fo)


def text_width(ctx, s, tracking=0.0):
    if not s:
        return 0.0
    ext = ctx.text_extents(s)
    return ext.x_advance + tracking * (len(s) - 1)


def draw_text(ctx, s, x, y, size, rgb=(1, 1, 1), alpha=1.0, family="Inter", weight="bold",
              align="center", tracking=0.0, shadow=0.0, outline=0.0, outline_rgb=(0, 0, 0)):
    """Draw one line with baseline at y. align: left|center|right. tracking in px."""
    set_font(ctx, size, family, weight)
    tw = text_width(ctx, s, tracking)
    x0 = x - tw / 2 if align == "center" else (x - tw if align == "right" else x)

    def path():
        cx = x0
        if tracking == 0:
            ctx.move_to(cx, y)
            ctx.text_path(s)
            return
        for ch in s:
            ctx.move_to(cx, y)
            ctx.text_path(ch)
            cx += ctx.text_extents(ch).x_advance + tracking

    if shadow > 0:
        for i, (dx, dy, a) in enumerate(((0, size * 0.04, 0.55), (0, size * 0.08, 0.3))):
            ctx.save()
            ctx.translate(dx, dy)
            path()
            ctx.set_source_rgba(0, 0, 0, alpha * a * shadow)
            ctx.fill()
            ctx.restore()
    if outline > 0:
        path()
        ctx.set_source_rgba(*outline_rgb, alpha)
        ctx.set_line_width(outline)
        ctx.stroke()
    path()
    ctx.set_source_rgba(*rgb, alpha)
    ctx.fill()
    return tw


# ---------------------------------------------------------------- encode


def _worker_init(fn):
    global _FRAME_FN
    _FRAME_FN = fn


def _worker(i):
    return i, _FRAME_FN(i).tobytes()


def render_video(frame_fn, n_frames, out_path, fps=FPS, workers=None, crf=14, preset="slow",
                 audio_path=None, frames=None):
    """frame_fn(i) -> uint8 HxWx3 RGB. Frames rendered in a fork pool, piped to x264 in order."""
    workers = workers or os.cpu_count()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    idx = list(frames) if frames is not None else list(range(n_frames))
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(frame_fn,)) as pool:
        for k, (i, data) in enumerate(pool.imap(_worker, idx, chunksize=2)):
            proc.stdin.write(data)
            if k % 120 == 0:
                print(f"  frame {k}/{len(idx)}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def save_png(rgb_u8, path):
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
