"""Gradient descent in two dimensions, defined as simply as possible, on a white background.

The cut hill: the real bowl f = 0.5u^2 + 1.8v^2 (axes rotated 32 degrees) is cut open along its valley
line and ray-cast once, per pixel, into a softly shaded blue ceramic surface with crisp contour lines, a
pale cut face whose top edge is the parabola profile, and a soft grounding shadow. One ball (a real sphere
resting on the surface), and at every step one vermilion arrow: the exact negative gradient times the
learning rate, drawn on the surface from the ball's contact point to the landing point. It grows out from
under the ball, the ball slides along it, and the arrow stays in place as a faded ghost, so the walked path
is a chain of shrinking arrows from the rim to the trough (every one the real past step). From step 3 a
magnifier loupe re-renders the trough from a second ray-cast at a continuous magnification (x4 -> x12,
growing as the square root of the shrinking step) so every one of the 24 arrows stays a visible arrow +
slide while still shrinking step by step. At the end the two table axes (the two weights) and the error
axis fade in. Every number shown is computed from the same function. Slow pacing, 30 s, loop.
frame(f) -> uint8 1080x1920 RGB; a pure function of f (everything static is cached in __init__)."""

import math
import os
import sys

import cairo
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import H, W, cells_width, dither, ease_out_cubic, font, layer, over, rgb, show_cells, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402

FPS = 60
N_FRAMES = 1800                     # 30 s
# light theme: one accent only
INK, INK2, INK3, GRID_C = "#111827", "#6B7280", "#4B5563", "#9CA3AF"
BLUE, AMBER, MAGENTA, GREEN = "#2563EB", "#E4572E", "#DB2777", "#059669"   # AMBER is the vermilion accent
ACC = AMBER
SURF_LO, SURF_HI = "#6FA0E8", "#DCE4F3"       # bowl floor (deepest blue) -> rim (pale blue-grey, still a solid on white)
FACE_C = "#EEF1F6"                            # cut face: pale cool grey
EDGE_C = "#3A4A6B"                            # silhouette / profile stroke (ink-blue, never black)
SHADOW_C = "#2B3A55"                          # ground shadow, contact shadows
CX = 540
# the bowl: an elongated bowl rotated so the path curves
ROT = math.radians(32)
_cr, _sr = math.cos(ROT), math.sin(ROT)
A_, B_ = 0.5, 1.8
LR = 0.09                           # learning rate: 24 steps stay visible (first 87 px, last 4 px on screen)
START_UV = (2.85, -0.64)            # start near the rim, out along the long axis (f = 4.80); the path keeps v <= 0
START = (_cr * START_UV[0] - _sr * START_UV[1], _sr * START_UV[0] + _cr * START_UV[1])
N_STEPS = 24
# timeline
T_SURF = 2.6                        # surface draws in
T_BALL = 6.0
T_FIRST = 9.6                       # first arrow
STEP_DT = 0.62                      # per step: arrow 0.26 s, move 0.26 s, rest
T_END = T_FIRST + N_STEPS * STEP_DT  # 24.48 s: the ball is at its last position
LOOP_FADE = (29.3, 29.85)
MAX_TEXT_W = 880
# ---------------------------------------------------------------- camera (fixed perspective, long lens)
Z_RIM = 5.6                         # the hill is drawn for f <= Z_RIM
KZ = 0.45                           # world height z = KZ * f (display scale only)
V_CUT = 0.22                        # cut plane v = V_CUT: the drawn domain is v <= V_CUT (ball path has v <= 0)
CAM_EL, CAM_AZ, CAM_D = 38.0, 144.0, 24.0
F_PX = 3600.0                       # focal length in px
PCX, PCY = 520.0, 1010.0            # principal point
CROP = (10, 1070, 600, 1330)        # ray-cast crop x0, x1, y0, y1 (1x px)
BALL_R = 21.0                       # sphere radius in px at the camera distance CAM_D (small enough that the arrows clear it)
GHOST_A = 0.22                      # the bowl's faint silhouette under the title (frame 0 and frame 1799)
ARROW_GHOST_A = 0.62                # a walked arrow stays in place at this alpha (no halo)
ARROW_MIN_PX = 12.0                 # arrows shorter than this on screen are not drawn (the hairline and dots carry them)
# ---------------------------------------------------------------- the loupe (magnifier inset, lower left)
LOUPE_C = (255.0, 1200.0)           # disc centre; the bottom rim at 1340 leaves 45 px of air above the STEP readout
LOUPE_R = 140.0
LOUPE_BALL = 26.0                   # the ball is a fixed-size marker inside the loupe (never smaller than the real one)
K_LOUPE = 3                         # the loupe opens for step 3 and is fully open before that arrow starts
T_LOUPE = T_FIRST + K_LOUPE * STEP_DT - 0.5
MAG_LO, MAG_HI = 4.0, 12.0          # continuous magnification MAG_LO * sqrt(step[K_LOUPE] / step now), clamped
LP_BANDS = ((6, 6.0), (9, 9.0), (12, MAG_HI + 1))   # (supersampling, magnification up to which that cache is used)
T_LOUPE_OUT = (T_END + 0.45, T_END + 0.9)   # closes before the weights cue (25.6 s)
LABEL_Y = 604                       # rule height of the leader labels (annotation band above the bowl)

_el, _az = math.radians(CAM_EL), math.radians(CAM_AZ)
CAM_T = np.array([0.0, 0.0, 0.9])
CAM_C = CAM_T + CAM_D * np.array([math.cos(_el) * math.cos(_az), math.cos(_el) * math.sin(_az), math.sin(_el)])
CAM_FW = (CAM_T - CAM_C) / np.linalg.norm(CAM_T - CAM_C)
CAM_RIGHT = np.cross(CAM_FW, np.array([0.0, 0.0, 1.0]))
CAM_RIGHT /= np.linalg.norm(CAM_RIGHT)
CAM_UP = np.cross(CAM_RIGHT, CAM_FW)
LIGHT = -0.35 * CAM_RIGHT + 0.45 * CAM_UP - 0.82 * CAM_FW
LIGHT /= np.linalg.norm(LIGHT)
TAU = 2 * math.pi


def f_xy(x, y):
    u = _cr * x + _sr * y
    v = -_sr * x + _cr * y
    return A_ * u * u + B_ * v * v


def grad_xy(x, y):
    u = _cr * x + _sr * y
    v = -_sr * x + _cr * y
    gu, gv = 2 * A_ * u, 2 * B_ * v
    return (_cr * gu - _sr * gv, _sr * gu + _cr * gv)


def normal_xy(x, y):
    gx, gy = grad_xy(x, y)
    n = np.array([-KZ * gx, -KZ * gy, 1.0])
    return n / np.linalg.norm(n)


def project_w(P):
    """World point (x, y, z) -> screen (sx, sy, depth)."""
    d = np.asarray(P, float) - CAM_C
    dd = float(d @ CAM_FW)
    return PCX + F_PX * float(d @ CAM_RIGHT) / dd, PCY - F_PX * float(d @ CAM_UP) / dd, dd


def project_arr(X, Y, Z):
    """Vectorised projection of world arrays -> screen arrays."""
    dX, dY, dZ = X - CAM_C[0], Y - CAM_C[1], Z - CAM_C[2]
    dd = dX * CAM_FW[0] + dY * CAM_FW[1] + dZ * CAM_FW[2]
    sx = PCX + F_PX * (dX * CAM_RIGHT[0] + dY * CAM_RIGHT[1] + dZ * CAM_RIGHT[2]) / dd
    sy = PCY - F_PX * (dX * CAM_UP[0] + dY * CAM_UP[1] + dZ * CAM_UP[2]) / dd
    return sx, sy


def project(x, y, zf):
    """(x, y, f) -> screen (sx, sy)."""
    sx, sy, _ = project_w((x, y, zf * KZ))
    return sx, sy


def surf_pt(x, y):
    return project(x, y, f_xy(x, y))


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def lerp(a, b, k):
    return a + (b - a) * k


def _fmt(v, dec=None):
    if dec is None:
        dec = 2 if v >= 0.1 else 3
    return f"{v:.{dec}f}"


def poly_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))


def poly_prefix(pts, frac):
    """The first `frac` of a polyline's length."""
    if frac >= 1:
        return list(pts)
    tot = poly_len(pts) * frac
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= tot:
            fr = (tot - acc) / max(1e-9, L)
            out.append((a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr))
            return out
        acc += L
        out.append(b)
    return out


class Scene:
    def __init__(self):
        # the path of the ball
        pts = [START]
        for _ in range(N_STEPS):
            x, y = pts[-1]
            gx, gy = grad_xy(x, y)
            pts.append((x - LR * gx, y - LR * gy))
        self.path = pts
        self.heights = [f_xy(*p) for p in pts]
        self.slopes = [math.hypot(*grad_xy(*p)) for p in pts]
        self.cues = self._cues()
        # static geometry caches (computed once, before the fork pool; frame() only composites them)
        self.spt = [surf_pt(*p) for p in pts]
        self.seg_px = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.spt[:-1], self.spt[1:])]
        self.dstep = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.spt[:-1], self.spt[1:])] + [(0.0, 0.0)]
        self.step_poly = [self.surf_poly(a, b) for a, b in zip(pts[:-1], pts[1:])]   # each step's arrow, on the surface
        self._build_bowl()
        self._build_shadow()
        self._build_loupe()

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("Gradient descent,", "black", 64, INK, 300, 1), ("in the simplest way I can.", "black", 50, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 5.8, [("A hill.", "black", 64, INK, 300, 1), ("Height = how wrong you are.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.2, 9.2, [("A ball, dropped anywhere.", "black", 54, INK, 300, 1), ("It wants the lowest point. It can't see the hill.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.6, 13.4, [("One rule.", "black", 64, AMBER, 300, 1), ("Feel the slope under your feet. Step downhill.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (13.8, 17.4, [("The arrow is the slope.", "black", 50, INK, 300, 1), ("Steep hill, long arrow, big step.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.8, 21.4, [("Near the bottom the slope fades.", "black", 46, INK, 300, 1), ("Smaller arrows, smaller steps. It settles.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.8, 25.2, [("That's all gradient descent is.", "black", 50, INK, 300, 1), ("Gradient = the slope. Descent = walk down it.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (25.6, 29.0, [("In a neural network", "black", 44, INK, 278, 1), ("the two axes are two weights.", "black", 44, INK, 334, 1), ("Real networks: millions of axes. Same rule.", "dbold", 36, AMBER, 402, 0.95)], 10, 8, "pop"),
            (29.4, 30.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ schedule

    def state(self, t):
        """(step index k of the ball's current position, phase, u): phase 'rest' | 'arrow' | 'move'."""
        if t < T_FIRST:
            return 0, "rest", 0.0
        k = int((t - T_FIRST) / STEP_DT)
        if k >= N_STEPS:
            return N_STEPS, "rest", 0.0
        u = (t - T_FIRST - k * STEP_DT) / STEP_DT
        if u < 0.42:
            return k, "arrow", u / 0.42
        if u < 0.84:
            return k, "move", (u - 0.42) / 0.42
        return k + 1, "rest", 0.0

    def ball_xy(self, t):
        """The ball's world position at t (eased slide during the move phase) and the slide fraction."""
        k, phase, u = self.state(t)
        x, y = self.path[min(k, N_STEPS)]
        mm = 0.0
        if phase == "move" and k < N_STEPS:
            xn, yn = self.path[k + 1]
            mm = smooth(u)
            x, y = x + (xn - x) * mm, y + (yn - y) * mm
        return x, y, mm

    def seg_now(self, t):
        """The current step's length on screen (1x px), interpolated through the slide: continuous in t."""
        k, phase, u = self.state(t)
        if k >= N_STEPS:
            return self.seg_px[-1]
        s0 = self.seg_px[k]
        if phase == "move":
            return lerp(s0, self.seg_px[min(k + 1, N_STEPS - 1)], smooth(u))
        return s0

    def mag(self, t):
        """Loupe magnification: continuous, MAG_LO * sqrt(step[K_LOUPE] / step now), clamped to [MAG_LO, MAG_HI]. It only
        partially compensates the shrinking step, so the magnified arrow still shrinks from step to step (as the square
        root of the real step) and nothing on screen ever grows while the hill flattens."""
        return min(MAG_HI, max(MAG_LO, MAG_LO * math.sqrt(self.seg_px[K_LOUPE] / self.seg_now(t))))

    def loupe_target(self, t):
        """Screen point the loupe is centred on: the ball plus 0.45 of the coming step (continuous in t)."""
        k, phase, u = self.state(t)
        x, y, mm = self.ball_xy(t)
        bx, by = surf_pt(x, y)
        if k >= N_STEPS:
            return bx, by
        d0 = self.dstep[k]
        d1 = self.dstep[k + 1] if phase == "move" else d0
        return bx + 0.45 * lerp(d0[0], d1[0], mm), by + 0.45 * lerp(d0[1], d1[1], mm)

    # ------------------------------------------------------------ the ray-cast

    def _raycast(self, px, py):
        """Ray-cast the solid hill {v <= V_CUT, f <= Z_RIM, 0 <= z <= KZ f} through pixel-centre offsets px, py
        (1x px from the principal point). Returns shaded colour (no contours), hit, is_face, exterior, Zf, Zrev."""
        PX, PY = np.meshgrid(px, py)
        dx = CAM_FW[0] + PX / F_PX * CAM_RIGHT[0] - PY / F_PX * CAM_UP[0]
        dy = CAM_FW[1] + PX / F_PX * CAM_RIGHT[1] - PY / F_PX * CAM_UP[1]
        dz = CAM_FW[2] + PX / F_PX * CAM_RIGHT[2] - PY / F_PX * CAM_UP[2]
        del PX, PY
        nn = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= nn
        dy /= nn
        dz /= nn
        del nn
        u0, v0, z0 = _cr * CAM_C[0] + _sr * CAM_C[1], -_sr * CAM_C[0] + _cr * CAM_C[1], CAM_C[2]
        du, dv = _cr * dx + _sr * dy, -_sr * dx + _cr * dy
        # quadratic KZ f(u0 + t du, v0 + t dv) = z0 + t dz, solved in float64 with the stable root form
        a = KZ * (A_ * du * du + B_ * dv * dv)
        b = KZ * (2 * A_ * u0 * du + 2 * B_ * v0 * dv) - dz
        c = KZ * (A_ * u0 * u0 + B_ * v0 * v0) - z0
        disc = b * b - 4 * a * c
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        q = -0.5 * (b + np.where(b >= 0, 1.0, -1.0) * sq)
        q = np.where(np.abs(q) < 1e-12, 1e-12, q)
        ta, tb = q / a, c / q
        t1, t2 = np.minimum(ta, tb), np.maximum(ta, tb)
        del disc, sq, q, ta, tb, a, b
        zr = Z_RIM * KZ

        def valid(tt):
            return ok & (tt > 0) & (z0 + tt * dz <= zr + 1e-9) & (v0 + tt * dv <= V_CUT)

        v1, v2 = valid(t1), valid(t2)
        # cut face: plane v = V_CUT, region 0 <= z <= KZ f(u, V_CUT), f <= Z_RIM
        dvs = np.where(np.abs(dv) < 1e-9, 1e-9, dv)
        tp = (V_CUT - v0) / dvs
        up_ = u0 + tp * du
        zp = z0 + tp * dz
        fp = A_ * up_ * up_ + B_ * V_CUT * V_CUT
        vp = (tp > 0) & (zp >= 0) & (zp <= KZ * fp) & (fp <= Z_RIM)
        BIG = 1e9
        T1 = np.where(v1, t1, BIG)
        T2 = np.where(v2, t2, BIG)
        TP = np.where(vp, tp, BIG)
        t = np.minimum(np.minimum(T1, T2), TP)
        hit = t < BIG
        is_face = vp & (TP <= T1) & (TP <= T2)
        exterior = v1 & (T1 < TP) & (T1 <= T2) & ~is_face
        del T1, T2, TP, up_, zp, fp, vp, tp, dvs, t1, t2, v1, v2
        X = CAM_C[0] + t * dx
        Y = CAM_C[1] + t * dy
        Z = CAM_C[2] + t * dz
        U = _cr * X + _sr * Y
        V = -_sr * X + _cr * Y
        Zf = np.where(is_face, Z / KZ, A_ * U * U + B_ * V * V)      # f at the hit; on the face its real height
        Zrev = np.where(is_face, A_ * U * U + B_ * V_CUT * V_CUT, Zf)   # for the draw-in: the face grows with its profile
        f32 = np.float32
        Zf = Zf.astype(f32)
        Zrev = Zrev.astype(f32)
        U, V, Z = U.astype(f32), V.astype(f32), Z.astype(f32)
        dx, dy, dz = dx.astype(f32), dy.astype(f32), dz.astype(f32)
        del X, Y, t, du, dv, ok
        # shading: surface normal, cool-tinted wrap lighting, trough ambient occlusion, a barely-there satin sheen
        gx = KZ * (2 * A_ * U * _cr - 2 * B_ * V * _sr)
        gy = KZ * (2 * A_ * U * _sr + 2 * B_ * V * _cr)
        nrm = np.sqrt(gx * gx + gy * gy + 1)
        nx, ny, nz = -gx / nrm, -gy / nrm, 1 / nrm
        del gx, gy, nrm
        sgn = np.where(exterior, -1.0, 1.0).astype(f32)
        nx, ny, nz = nx * sgn, ny * sgn, nz * sgn
        nx = np.where(is_face, f32(-_sr), nx)
        ny = np.where(is_face, f32(_cr), ny)
        nz = np.where(is_face, f32(0.0), nz)
        L = LIGHT.astype(f32)
        ndl = nx * L[0] + ny * L[1] + nz * L[2]
        lam = np.clip(ndl, 0, 1)
        wrap = np.clip(ndl * 0.5 + 0.5, 0, 1)
        del ndl
        zk = np.clip(Zf / Z_RIM, 0, 1)
        k = zk ** 0.6
        lo_, hi_, face_ = (np.array(rgb(h_), f32) for h_ in (SURF_LO, SURF_HI, FACE_C))
        base = lo_ + (hi_ - lo_) * k[..., None]
        shade01 = 0.20 * wrap + 0.80 * lam
        tint = np.array([0.44, 0.53, 0.74], f32)                           # the shadow side stays cool, never grey
        mult = tint + (1 - tint) * shade01[..., None]
        mult *= (1 - 0.16 * (1 - zk) ** 2)[..., None]
        col = base * mult
        hx, hy, hz = L[0] - dx, L[1] - dy, L[2] - dz
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** 28
        col += (0.08 * spec)[..., None]
        del hx, hy, hz, hn, spec, base, mult, shade01, wrap, lam, nx, ny, nz, dx, dy, dz, k
        # the cut face: pale, a vertical gradient (darker at the table), no sheen, no contours; exterior slivers likewise
        zfrac = np.clip(Z / zr, 0, 1)
        fcol = face_ * (0.92 + 0.08 * zfrac)[..., None]
        col = np.where(is_face[..., None], fcol, col)
        col = np.where(exterior[..., None], face_ * 0.90, col)
        del fcol, zfrac, Z, U, V
        return col.astype(f32), hit, is_face, exterior, Zf, Zrev

    # ------------------------------------------------------------ the cached bowl

    def _build_bowl(self):
        """Ray-cast at 2x inside CROP, bake the contour lines, downsample to 1x, then stroke the silhouette edges
        with cairo. Cached as premultiplied rgb, alpha, a height map (f at the hit) and its pixel gradient (for the
        anti-aliased draw-in)."""
        ss = 2
        f32 = np.float32
        x0, x1, y0, y1 = CROP
        w1, h1 = x1 - x0, y1 - y0
        px = (np.arange(w1 * ss) + 0.5) / ss + x0 - PCX
        py = (np.arange(h1 * ss) + 0.5) / ss + y0 - PCY
        col, hit, is_face, exterior, Zf, Zrev = self._raycast(px, py)
        # contour lines every 0.4 of height: 1.0 px at 1x, surface colour x 0.74; none through the trough or on the face
        Zs = np.where(hit & ~is_face, Zf, np.nan)
        gyv, gxv = np.gradient(Zs)
        gpx = np.sqrt(np.nan_to_num(gxv) ** 2 + np.nan_to_num(gyv) ** 2) + 1e-6
        del Zs, gxv, gyv
        DZ = 0.4
        frac = np.abs(Zf / DZ - np.round(Zf / DZ)) * DZ / gpx
        line = np.clip(1.0 * ss / 2 + 0.5 - frac, 0, 1)
        line = np.where((Zf < DZ * 0.5) | is_face | exterior, 0, line)
        col *= (1 - 0.26 * line)[..., None]
        del frac, line, gpx
        alpha = hit.astype(f32)
        rgbp = col * alpha[..., None]
        del col
        rs = lambda im: cv2.resize(im, (w1, h1), interpolation=cv2.INTER_AREA)  # noqa: E731
        rgb1 = rs(rgbp)
        a1 = rs(alpha)
        a_core = a1.copy()
        z1 = rs(np.where(hit, Zrev, f32(Z_RIM)).astype(f32))
        face1 = rs(is_face.astype(f32))
        self.n_exterior_px = int(rs(exterior.astype(f32)).sum())
        del rgbp, alpha, Zf, Zrev, hit, is_face, exterior
        # silhouette strokes (cairo, 1x): the far rim, the parabola profile edge, and the face outline
        surf, ctx = layer(w1, h1)
        ctx.translate(-x0, -y0)
        ec = rgb(EDGE_C)
        ua = math.sqrt((Z_RIM - B_ * V_CUT * V_CUT) / A_)                 # rim ends: f(u, V_CUT) = Z_RIM
        th0 = math.asin(V_CUT / math.sqrt(Z_RIM / B_))
        # rim: f = Z_RIM, v <= V_CUT (left end -> far side -> right end)
        ths = np.linspace(math.pi - th0, 2 * math.pi + th0, 160)
        rim_pts = []
        for th in ths:
            u, v = math.sqrt(Z_RIM / A_) * math.cos(th), math.sqrt(Z_RIM / B_) * math.sin(th)
            rim_pts.append(project(_cr * u - _sr * v, _sr * u + _cr * v, Z_RIM))
        for i, p in enumerate(rim_pts):
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        ctx.set_source_rgba(*ec, 0.70)
        ctx.set_line_width(2.2)
        ctx.stroke()
        # profile edge: z = KZ f(u, V_CUT) along the cut plane (right end -> trough -> left end)
        prof_pts = [project(_cr * u - _sr * V_CUT, _sr * u + _cr * V_CUT, A_ * u * u + B_ * V_CUT * V_CUT) for u in np.linspace(ua, -ua, 160)]
        for i, p in enumerate(prof_pts):
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        ctx.set_source_rgba(*ec, 0.70)
        ctx.set_line_width(2.2)
        ctx.stroke()
        self.surf_clip = rim_pts + prof_pts          # the drawn surface on screen: overlays lying on it are clipped to this
        self.prof_pts = prof_pts
        self.ua = ua
        # face outline (two vertical ends and the table line)
        corners = [project(_cr * u - _sr * V_CUT, _sr * u + _cr * V_CUT, zf) for u, zf in ((-ua, Z_RIM), (-ua, 0.0), (ua, 0.0), (ua, Z_RIM))]
        self.corners = corners
        ctx.move_to(*corners[0])
        for p in corners[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgba(*ec, 0.45)
        ctx.set_line_width(1.2)
        ctx.stroke()
        srgb, sa = surf_rgba(surf)
        rgb1 = rgb1 * (1 - sa)[..., None] + srgb
        a1 = a1 * (1 - sa) + sa
        # height map for the reveal: extend into the stroke pixels just outside the hit mask, then take the pixel
        # gradient on a nearest-hit-extended map so boundary gradients are the surface's own (no stray early edges)
        hit1 = a1 > 0.0
        z_ext = cv2.erode(z1, np.ones((7, 7), np.uint8))
        z1 = np.where(face1 > 0.5, z1, np.where(hit1 & (z1 >= Z_RIM - 1e-4), z_ext, z1)).astype(f32)
        zfill = z1.copy()
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(10):
            zfill = np.where(hit1, z1, cv2.erode(zfill, k3)).astype(f32)
        gyv, gxv = np.gradient(zfill)
        gpx1 = (np.sqrt(gxv * gxv + gyv * gyv) + 1e-5).astype(f32)
        gpx1[~hit1] = 1e9
        self.b_rgb = np.ascontiguousarray(rgb1.astype(f32))
        self.b_a = np.ascontiguousarray(a1.astype(f32))
        self.b_z = np.ascontiguousarray(z1)
        self.b_gpx = np.ascontiguousarray(gpx1)
        self.b_stroke = np.ascontiguousarray(hit1 & (a_core < 0.5))
        ys, xs = np.where(self.b_a > 0.5)
        self.bbox = (int(xs.min()) + x0, int(xs.max()) + x0, int(ys.min()) + y0, int(ys.max()) + y0)

    def _build_shadow(self):
        """Soft ground shadow of the footprint on the table z = 0 (alpha only, crop-sized)."""
        x0, x1, y0, y1 = CROP
        surf, ctx = layer(x1 - x0, y1 - y0)
        ctx.translate(-x0, -y0)
        au, av = math.sqrt(Z_RIM / A_) * 0.94, math.sqrt(Z_RIM / B_) * 0.94
        for i in range(37):
            th = math.pi + math.pi * i / 36
            u, v = au * math.cos(th), av * math.sin(th)
            p = project(_cr * u - _sr * v, _sr * u + _cr * v, 0.0)
            (ctx.move_to if i == 0 else ctx.line_to)(p[0] + 6, p[1] + 8)
        ctx.close_path()
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.fill()
        _, a = surf_rgba(surf)
        self.sh_a = (cv2.GaussianBlur(a, (0, 0), 10) * 0.20).astype(np.float32)
        self.sh_rgb = np.array(rgb(SHADOW_C), np.float32)

    def _build_loupe(self):
        """Second ray-cast of the trough for the loupe, in three magnification bands (supersampling 6, 9 and 12 for m up
        to 6, 9 and 12) so the warpAffine sampling stays sharp; each band is a tight window around every loupe target it
        will be asked for. Stored as premultiplied BGRA uint8."""
        self.lp = []
        frames = range(int((T_LOUPE - 0.5) * FPS), int((T_LOUPE_OUT[1] + 0.1) * FPS) + 2)
        mt = [(self.mag(f / FPS), self.loupe_target(f / FPS)) for f in frames]
        m_lo = -1.0
        for ss, m_hi in LP_BANDS:
            xs, ys = [], []
            for m, (tx, ty) in mt:
                if m_lo < m <= m_hi:
                    r = LOUPE_R / m + 6
                    xs += [tx - r, tx + r]
                    ys += [ty - r, ty + r]
            m_lo = m_hi
            if not xs:
                continue
            x0, x1 = int(math.floor(min(xs))), int(math.ceil(max(xs)))
            y0, y1 = int(math.floor(min(ys))), int(math.ceil(max(ys)))
            w, h = x1 - x0, y1 - y0
            px = (np.arange(w * ss) + 0.5) / ss + x0 - PCX
            py = (np.arange(h * ss) + 0.5) / ss + y0 - PCY
            col, hit, is_face, exterior, Zf, Zrev = self._raycast(px, py)
            a = hit.astype(np.float32)
            prem = col * a[..., None]
            bgra = np.empty((h * ss, w * ss, 4), np.uint8)
            bgra[..., 0] = np.clip(prem[..., 2] * 255 + 0.5, 0, 255)
            bgra[..., 1] = np.clip(prem[..., 1] * 255 + 0.5, 0, 255)
            bgra[..., 2] = np.clip(prem[..., 0] * 255 + 0.5, 0, 255)
            bgra[..., 3] = np.clip(a * 255 + 0.5, 0, 255)
            self.lp.append((ss, m_hi, x0, y0, np.ascontiguousarray(bgra)))

    def _lp_surface(self, tx, ty, m):
        """The loupe's surface patch (2R x 2R, opaque over white) as a cairo surface, sampled from the band's cache."""
        for ss, m_hi, x0, y0, bgra in self.lp:
            if m <= m_hi:
                break
        n = int(round(2 * LOUPE_R))
        s = ss / m
        offx = (tx - x0 - LOUPE_R / m) * ss + 0.5 * s - 0.5
        offy = (ty - y0 - LOUPE_R / m) * ss + 0.5 * s - 0.5
        M = np.array([[s, 0, offx], [0, s, offy]], np.float32)
        patch = cv2.warpAffine(bgra, M, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = np.empty((n, n, 4), np.uint8)
        white = 255 - patch[..., 3:4].astype(np.int16)
        out[..., :3] = np.clip(patch[..., :3].astype(np.int16) + white, 0, 255)
        out[..., 3] = 255
        surf = cairo.ImageSurface.create_for_data(memoryview(out), cairo.FORMAT_ARGB32, n, n, n * 4)
        return surf, out

    # ------------------------------------------------------------ geometry helpers

    def surf_poly(self, p, q, n=16, lift=0.0):
        """Polyline on the surface from p to q (xy chord), screen coords; `lift` world units along the surface normal
        (lift = ball radius gives the path of the ball's centre)."""
        s = np.linspace(0, 1, n + 1)
        X = p[0] + (q[0] - p[0]) * s
        Y = p[1] + (q[1] - p[1]) * s
        Z = f_xy(X, Y) * KZ
        if lift:
            U = _cr * X + _sr * Y
            V = -_sr * X + _cr * Y
            gu, gv = 2 * A_ * U, 2 * B_ * V
            gx, gy = _cr * gu - _sr * gv, _sr * gu + _cr * gv
            nn = np.sqrt(KZ * KZ * (gx * gx + gy * gy) + 1)
            X, Y, Z = X - lift * KZ * gx / nn, Y - lift * KZ * gy / nn, Z + lift / nn
        sx, sy = project_arr(X, Y, Z)
        return list(zip(sx.tolist(), sy.tolist()))

    def ball_geom(self, x, y):
        """Contact point P, normal n, sphere radius (px, world), sphere centre on screen, contact on screen."""
        P = np.array([x, y, f_xy(x, y) * KZ])
        n = normal_xy(x, y)
        sx, sy, d = project_w(P)
        rb_px = BALL_R * CAM_D / d
        rb_w = rb_px * d / F_PX
        cx, cy, _ = project_w(P + n * rb_w)
        return P, n, rb_px, rb_w, (cx, cy), (sx, sy)

    @staticmethod
    def tangent_basis(n):
        t1 = np.array([1.0, 0.0, -n[0] / n[2]])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return t1, t2

    def ring_pts(self, P, n, r_w, off=None, npts=36, lift=0.004):
        """Screen points of a circle of world radius r_w in the tangent plane at P (slightly lifted off the surface)."""
        t1, t2 = self.tangent_basis(n)
        th = np.linspace(0, TAU, npts, endpoint=False)
        Q = P[None, :] + np.outer(np.cos(th), t1) * r_w + np.outer(np.sin(th), t2) * r_w + n[None, :] * lift
        if off is not None:
            Q = Q + off[None, :]
        sx, sy = project_arr(Q[:, 0], Q[:, 1], Q[:, 2])
        return list(zip(sx.tolist(), sy.tolist()))

    def level_pts(self, h, npts=240):
        """The contour f = h on the drawn half (screen arrays + mask of the drawn part)."""
        th = np.linspace(0, TAU, npts + 1)
        u, v = math.sqrt(h / A_) * np.cos(th), math.sqrt(h / B_) * np.sin(th)
        ok = v <= V_CUT - 0.004
        X, Y = _cr * u - _sr * v, _sr * u + _cr * v
        sx, sy = project_arr(X, Y, np.full_like(X, h * KZ))
        return sx, sy, ok

    @staticmethod
    def _path(ctx, pts, close=False):
        for i, p in enumerate(pts):
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        if close:
            ctx.close_path()

    # ------------------------------------------------------------ drawing: the ball

    def contact_shadow(self, ctx, P, n, rb_w, alpha, scale=1.0, M=None):
        lt = LIGHT - float(LIGHT @ n) * n
        off = -lt * 0.5 * rb_w
        sc = rgb(SHADOW_C)
        for s, a in ((1.0, 0.13), (1.35, 0.06), (1.8, 0.03)):
            pts = self.ring_pts(P, n, rb_w * 0.95 * s * scale, off=off, npts=28, lift=0.003)
            if M:
                pts = [M(p) for p in pts]
            self._path(ctx, pts, close=True)
            ctx.set_source_rgba(*sc, a * alpha)
            ctx.fill()

    def sphere(self, ctx, cx, cy, r, alpha, contact=None, squash=0.0):
        ctx.save()
        if squash > 0 and contact is not None:
            ctx.translate(contact[0], contact[1])
            ctx.scale(1 + 0.10 * squash, 1 - 0.12 * squash)
            ctx.translate(-contact[0], -contact[1])
        g = cairo.RadialGradient(cx - 0.32 * r, cy - 0.34 * r, r * 0.05, cx, cy, r)
        g.add_color_stop_rgba(0.0, *rgb("#64748B"), alpha)
        g.add_color_stop_rgba(0.45, *rgb("#1F2937"), alpha)
        g.add_color_stop_rgba(1.0, *rgb("#030712"), alpha)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        g2 = cairo.RadialGradient(cx + 0.45 * r, cy + 0.5 * r, r * 0.2, cx + 0.3 * r, cy + 0.3 * r, r)
        g2.add_color_stop_rgba(0.0, *rgb("#93C5FD"), 0.0)
        g2.add_color_stop_rgba(0.75, *rgb("#93C5FD"), 0.0)
        g2.add_color_stop_rgba(1.0, *rgb("#93C5FD"), 0.45 * alpha)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.set_source(g2)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
        ctx.arc(cx - 0.36 * r, cy - 0.38 * r, 0.2 * r, 0, TAU)
        ctx.fill()
        ctx.restore()

    def ghost(self, ctx, p, r_w, alpha, M=None):
        """'Where it will land': a dotted ring lying flat on the surface at the landing point p."""
        x, y = p
        P = np.array([x, y, f_xy(x, y) * KZ])
        n = normal_xy(x, y)
        pts = self.ring_pts(P, n, r_w, npts=40)
        if M:
            pts = [M(p_) for p_ in pts]
        self._path(ctx, pts, close=True)
        ctx.set_source_rgba(1, 1, 1, 0.18 * alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(INK), 0.45 * alpha)
        ctx.set_line_width(1.4)
        ctx.set_dash([3.0, 4.0])
        ctx.stroke()
        ctx.set_dash([])

    # ------------------------------------------------------------ drawing: arrow, pulse, patch

    def arrow(self, ctx, pts, alpha, L_full, halo=1.0):
        """Polyline arrow (kite head) along pts; weights from the full step length L_full (px in the drawn space).
        `halo` scales the white outline: 1 for the live arrow, 0 for a ghost left on the trail."""
        tot = poly_len(pts)
        if tot < 4 or L_full < 4 or alpha <= 0.002:
            return
        head = min(20.0, max(9.0, 0.45 * L_full))
        wmain = min(6.0, max(4.0, 1.5 + 0.12 * L_full))
        tip = pts[-1]
        # direction of the head: from the point `head` back along the polyline to the tip
        back = max(0.0, tot - head)
        acc = 0.0
        pb = pts[0]
        for a, b in zip(pts[:-1], pts[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if acc + L >= back:
                fr = (back - acc) / max(1e-9, L)
                pb = (a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr)
                break
            acc += L
        ddx, ddy = tip[0] - pb[0], tip[1] - pb[1]
        nn = math.hypot(ddx, ddy) or 1.0
        ux, uy = ddx / nn, ddy / nn
        for col, al, wd, hd, ext in (("#FFFFFF", 0.85 * alpha * halo, wmain + 4.0, head + 3.6, 2.0), (ACC, 0.95 * alpha, wmain, head, 0.0)):
            if al <= 0.002:
                continue
            tx, ty = tip[0] + ux * ext, tip[1] + uy * ext
            hh = min(hd, tot + ext)
            ctx.set_source_rgba(*rgb(col), al)
            ctx.set_line_width(wd)
            # shaft: the polyline up to 0.7 head behind the tip
            shaft = poly_prefix(pts, max(0.0, (tot - hh * 0.7) / tot))
            if len(shaft) >= 2 and poly_len(shaft) > 0.5:
                self._path(ctx, shaft)
                ctx.stroke()
            hw = hh * 0.42
            ctx.move_to(tx, ty)
            ctx.line_to(tx - ux * hh - uy * hw, ty - uy * hh + ux * hw)
            ctx.line_to(tx - ux * hh * 0.82, ty - uy * hh * 0.82)
            ctx.line_to(tx - ux * hh + uy * hw, ty - uy * hh - ux * hw)
            ctx.close_path()
            ctx.fill()

    def pulse(self, ctx, p, q, alpha, r_w, M=None):
        """Landing ripple: a ring in the tangent plane at p, age fraction q in [0, 1), world radius r_w at q = 1."""
        x, y = p
        P = np.array([x, y, f_xy(x, y) * KZ])
        n = normal_xy(x, y)
        pts = self.ring_pts(P, n, r_w * ease_out_cubic(q), npts=48)
        if M:
            pts = [M(p_) for p_ in pts]
        self._path(ctx, pts, close=True)
        ctx.set_source_rgba(*rgb(ACC), 0.5 * (1 - q) * alpha)
        ctx.set_line_width(3.0 - 2.0 * q)
        ctx.stroke()

    # ------------------------------------------------------------ drawing: the actors on the surface (main view and loupe)

    def step_arrow(self, ctx, st, M, m, ball_px, al):
        """The live arrow while it grows: the exact step -LR * grad f, drawn on the surface from the ball's contact point
        to the landing point (M maps 1x screen -> drawn space, m = scale). Drawn under the sphere, so it emerges from
        under the ball; during the slide actors() keeps it in place and settles it into its ghost."""
        k, phase, u = st
        if k >= N_STEPS or phase != "arrow":
            return
        L_full = self.seg_px[k] * m
        if L_full < ball_px + 8:                      # too short to clear the ball in this view: the loupe shows it
            return
        Mp = (lambda p: M(p)) if M else (lambda p: p)
        poly = poly_prefix(self.step_poly[k], smooth(u))
        self.arrow(ctx, [Mp(p) for p in poly], al, L_full)

    def actors(self, ctx, t, st, bxy, geom, M, m, ball_px, al, a_ball, drop, in_loupe):
        """Trail hairline, the ghost arrows of the walked steps, pulses, landing ring, contact shadow and trail dots in
        the drawn space (M maps 1x screen -> drawn space, m = scale). Everything here lies under the ball."""
        k, phase, u = st
        kk = min(k, N_STEPS)
        x, y = bxy
        P, n, rb_w = geom
        path = self.path
        ink = rgb(INK)
        Mp = (lambda p: M(p)) if M else (lambda p: p)
        hair_w, hair_a, dot_r = (2.0, 0.4, 4.0) if in_loupe else (2.2, 0.4, 5.0)
        # ---- trail hairline: completed steps, plus the walked part of the current step
        ctx.set_source_rgba(*ink, hair_a * al)
        ctx.set_line_width(hair_w)
        for i in range(1, kk + 1):
            self._path(ctx, [Mp(p) for p in self.step_poly[i - 1]])
            ctx.stroke()
        if phase == "move" and k < N_STEPS:
            self._path(ctx, [Mp(p) for p in self.surf_poly(path[k], (x, y), n=8)])
            ctx.stroke()
        # ---- the walked arrows stay in place as ghosts (each one the real past step); the current one is left behind by
        #      the sliding ball and settles into its ghost (its white halo fades out with it)
        for i in range(0, kk):
            L = self.seg_px[i] * m
            if L >= ARROW_MIN_PX:
                self.arrow(ctx, [Mp(p) for p in self.step_poly[i]], ARROW_GHOST_A * al, L, halo=0.0)
        if phase == "move" and k < N_STEPS:
            L = self.seg_px[k] * m
            if L >= ARROW_MIN_PX:
                w = smooth(seg(u, 0.15, 0.85))
                pts = [Mp(p) for p in self.step_poly[k]]
                if L >= ball_px + 8:
                    self.arrow(ctx, pts, lerp(1.0, ARROW_GHOST_A, w) * al, L, halo=1.0 - w)
                else:
                    self.arrow(ctx, pts, ARROW_GHOST_A * w * al, L, halo=0.0)
        # ---- landing pulses (main view: the first five; loupe: all)
        tl = (t - T_FIRST - 0.84 * STEP_DT) / STEP_DT
        il = int(math.floor(tl))
        age = (tl - il) * STEP_DT
        d_ = project_w((x, y, f_xy(x, y) * KZ))[2]
        r_pulse = (1.7 * LOUPE_BALL / m * d_ / F_PX) if in_loupe else 0.27
        if 0 <= il < N_STEPS and age < 0.35 and (in_loupe or il < 5):
            self.pulse(ctx, path[il + 1], age / 0.35, al, r_pulse, M)
        if not in_loupe and T_BALL + 0.5 <= t < T_BALL + 0.85:
            self.pulse(ctx, path[0], (t - T_BALL - 0.5) / 0.35, al, 0.27, M)
        # ---- landing ring: a dotted ring at the landing point, in while the arrow grows, out as the ball covers it
        if k < N_STEPS and phase in ("arrow", "move") and self.seg_px[k] * m >= 3.0 * ball_px:
            ga = min(1.0, max(0.0, (smooth(u) - 0.45) / 0.4)) if phase == "arrow" else 1 - smooth(seg(u, 0.5, 0.92))
            if ga > 0:
                self.ghost(ctx, path[k + 1], rb_w, ga * al, M)
        # ---- the ball's contact shadow
        self.contact_shadow(ctx, P, n, rb_w, a_ball * (0.35 + 0.65 * (1 - drop)), scale=0.6 + 0.4 * (1 - drop), M=M)
        # ---- trail dots at every past position (drawn last so the arrows never hide them; smaller where steps are tiny)
        for i in range(0, kk + 1):
            r = dot_r if i == 0 else min(dot_r, max(2.2, 0.36 * self.seg_px[i - 1] * m))
            ad = a_ball if i == 0 else 1.0
            ctx.set_source_rgba(*ink, 0.65 * al * ad)
            pa = Mp(self.spt[i])
            ctx.arc(pa[0], pa[1], r, 0, TAU)
            ctx.fill()

    # ------------------------------------------------------------ drawing: the loupe

    def loupe(self, ctx, t, fade, st, bxy, geom, cxy, bsc, rb_px, squash):
        a_in = ease_out_cubic(seg(t, T_LOUPE, T_LOUPE + 0.45))
        a_out = 1 - smooth(seg(t, *T_LOUPE_OUT))
        al = a_in * a_out * fade
        if al <= 0:
            return
        # on the way out the actors inside the glass (ball, trail, arrows, badge) leave first; the glass follows
        a_act = a_in * fade * (1 - smooth(seg(t, T_LOUPE_OUT[0], T_LOUPE_OUT[0] + 0.4 * (T_LOUPE_OUT[1] - T_LOUPE_OUT[0]))))
        m = self.mag(t)
        tx, ty = self.loupe_target(t)
        Cx, Cy, R = LOUPE_C[0], LOUPE_C[1], LOUPE_R
        sc = (0.72 + 0.28 * a_in) * (0.6 + 0.4 * a_out)
        M = lambda p: (Cx + (p[0] - tx) * m, Cy + (p[1] - ty) * m)  # noqa: E731
        ink = rgb(INK)
        # ---- the magnified spot in the main view: a thin ring around the ball and, when there is room, a leader to the glass
        cx, cy = cxy
        r_ring = max(R / m, rb_px + 8)
        ddx, ddy = Cx - cx, Cy - cy
        dn = math.hypot(ddx, ddy) or 1.0
        ux, uy = ddx / dn, ddy / dn
        pa = (cx + ux * r_ring, cy + uy * r_ring)
        pb = (Cx - ux * (R + 14) * sc, Cy - uy * (R + 14) * sc)
        leader = dn - r_ring - (R + 14) * sc >= 30
        for col, lw, a_ in (("#FFFFFF", 4.0, 0.8 * al), (INK, 1.3, 0.5 * al)):
            ctx.set_source_rgba(*rgb(col), a_)
            ctx.set_line_width(lw)
            ctx.arc(cx, cy, r_ring, 0, TAU)
            ctx.stroke()
            if leader:
                ctx.move_to(*pa)
                ctx.line_to(*pb)
                ctx.stroke()
        ctx.save()
        ctx.translate(Cx, Cy)
        ctx.scale(sc, sc)
        ctx.translate(-Cx, -Cy)
        # ---- soft drop shadow
        g = cairo.RadialGradient(Cx, Cy + 8, R - 4, Cx, Cy + 8, R + 22)
        g.add_color_stop_rgba(0.0, *rgb(SHADOW_C), 0.22 * al)
        g.add_color_stop_rgba(0.45, *rgb(SHADOW_C), 0.08 * al)
        g.add_color_stop_rgba(1.0, *rgb(SHADOW_C), 0.0)
        ctx.arc(Cx, Cy + 8, R + 22, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        # ---- content, clipped to the disc
        ctx.save()
        ctx.arc(Cx, Cy, R, 0, TAU)
        ctx.clip()
        ctx.set_source_rgba(1, 1, 1, al)
        ctx.paint()
        psurf, keep = self._lp_surface(tx, ty, m)
        ctx.set_source_surface(psurf, Cx - R, Cy - R)
        ctx.paint_with_alpha(al)
        # profile edge of the cut (the surface ends there)
        self._path(ctx, [M(p) for p in self.prof_pts])
        ctx.set_source_rgba(*rgb(EDGE_C), 0.7 * al)
        ctx.set_line_width(2.2)
        ctx.stroke()
        # fine contour lines (real level sets): spacing 0.1 / 0.05 / 0.02 / 0.01, crossfaded as the magnification grows
        w1, w2, w3 = seg(m, 4.6, 6.4), seg(m, 8.0, 10.5), seg(m, 10.7, 11.9)
        for dz, aa in ((0.1, 1 - w1), (0.05, w1 * (1 - w2)), (0.02, w2 * (1 - w3)), (0.01, w3)):
            if aa <= 0.01:
                continue
            ctx.set_source_rgba(*ink, 0.22 * aa * al)
            ctx.set_line_width(1.4)
            h = dz
            while h <= 3.0:
                sx, sy, ok = self.level_pts(h)
                lx, ly = Cx + (sx - tx) * m, Cy + (sy - ty) * m
                if (np.abs(lx - Cx) < R + 30).any() and (np.abs(ly - Cy) < R + 30).any():
                    pen = False
                    for i in range(len(lx)):
                        if not ok[i]:
                            if pen:
                                ctx.stroke()
                                pen = False
                            continue
                        if not pen:
                            ctx.move_to(lx[i], ly[i])
                            pen = True
                        else:
                            ctx.line_to(lx[i], ly[i])
                    if pen:
                        ctx.stroke()
                h += dz
        # a soft lens highlight from the upper left (under the actors), so the magnified trough never reads as a flat field
        g = cairo.RadialGradient(Cx - 0.42 * R, Cy - 0.46 * R, 0.0, Cx - 0.30 * R, Cy - 0.32 * R, 1.08 * R)
        g.add_color_stop_rgba(0.0, 1, 1, 1, 0.17 * al)
        g.add_color_stop_rgba(0.55, 1, 1, 1, 0.04 * al)
        g.add_color_stop_rgba(1.0, 1, 1, 1, 0.0)
        ctx.arc(Cx, Cy, R, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        # the actors at the magnified scale, then the live arrow, then the ball (a fixed-size marker) on top of its tail
        P, n, rb_w = geom
        d_ = project_w(P)[2]
        rb_w_l = LOUPE_BALL / m * d_ / F_PX
        bc = M(bsc)
        ldx, ldy = (cxy[0] - bsc[0]) / max(1e-6, rb_px), (cxy[1] - bsc[1]) / max(1e-6, rb_px)
        lc = (bc[0] + ldx * LOUPE_BALL, bc[1] + ldy * LOUPE_BALL)
        self.actors(ctx, t, st, bxy, (P, n, rb_w_l), M, m, LOUPE_BALL, a_act, a_act, 0.0, True)
        self.step_arrow(ctx, st, M, m, LOUPE_BALL, a_act)
        self.sphere(ctx, lc[0], lc[1], LOUPE_BALL, a_act, contact=bc, squash=squash)
        ctx.restore()
        # ---- rim ring and the magnification badge (counts up with the continuous zoom)
        ctx.arc(Cx, Cy, R, 0, TAU)
        ctx.set_source_rgba(*rgb(INK2), 0.85 * al)
        ctx.set_line_width(2.0)
        ctx.stroke()
        font(ctx, "bold", 22)
        s = f"×{int(round(m))}"
        tw = tracked_width(ctx, s, 1.0)
        bx_, by_ = Cx + R * 0.72, Cy + R * 0.72
        ctx.set_source_rgba(1, 1, 1, a_act)
        ctx.arc(bx_, by_, 24, 0, TAU)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a_act)
        ctx.set_line_width(1.5)
        ctx.stroke()
        ctx.set_source_rgba(*ink, a_act)
        show_tracked(ctx, s, bx_ - tw / 2, by_ + 8, 1.0)
        ctx.restore()
        del keep

    # ------------------------------------------------------------ drawing: annotations and readouts

    def label(self, ctx, A, K, word, value, alpha, grow, side="r", dot=True):
        """Leader: anchor A -> knee K -> horizontal rule under the text (word regular 30, value bold 36)."""
        ax, ay = A
        kx, ky = K
        font(ctx, "regular", 30)
        w1 = ctx.text_extents(word).x_advance
        font(ctx, "dbold", 36)
        w2 = ctx.text_extents(value).x_advance if value else 0
        tw = w1 + (14 + w2 if value else 0)
        tx = kx + 14 if side == "r" else kx - 14 - tw
        x_end = (tx + tw + 10) if side == "r" else (tx - 10)
        g1 = min(1.0, grow / 0.6)
        for col, lw, al in (("#FFFFFF", 3.6, 0.9 * alpha), (INK, 1.2, 0.6 * alpha)):
            ctx.set_source_rgba(*rgb(col), al)
            ctx.set_line_width(lw)
            ctx.move_to(ax, ay)
            ctx.line_to(ax + (kx - ax) * g1, ay + (ky - ay) * g1)
            if grow > 0.6:
                ctx.line_to(kx + (x_end - kx) * min(1.0, (grow - 0.6) / 0.4), ky)
            ctx.stroke()
        if dot:
            ctx.set_source_rgba(1, 1, 1, alpha)
            ctx.arc(ax, ay, 4.6, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(INK), 0.8 * alpha)
            ctx.arc(ax, ay, 3.0, 0, TAU)
            ctx.fill()
        ta = alpha * max(0.0, (grow - 0.7) / 0.3)
        if ta <= 0:
            return
        font(ctx, "regular", 30)
        ctx.set_source_rgba(*rgb(INK3), ta)
        ctx.move_to(tx, ky - 10)
        ctx.show_text(word)
        if value:
            font(ctx, "dbold", 36)
            ctx.set_source_rgba(*rgb(INK), ta)
            ctx.move_to(tx + w1 + 14, ky - 10)
            ctx.show_text(value)

    def readouts(self, ctx, k, z, slope, dec_h, dec_s, a_h, a_s):
        if a_h <= 0 and a_s <= 0:
            return
        Y_LAB, Y_VAL, Y_BAR = 1404, 1466, 1489
        font(ctx, "bold", 24)
        tr = 0.14 * 24
        for s, x, al, align in (("STEP", 70, a_s, "l"), ("HEIGHT (ERROR)", CX, a_h, "c"), ("SLOPE", 1010, a_s, "r")):
            if al <= 0:
                continue
            w = tracked_width(ctx, s, tr)
            xx = x if align == "l" else (x - w / 2 if align == "c" else x - w)
            ctx.set_source_rgba(*rgb(INK2), 0.9 * al)
            show_tracked(ctx, s, xx, Y_LAB, tr)
        font(ctx, "dbold", 46)
        adv0 = ctx.text_extents("0").x_advance
        widths = {ch: adv0 for ch in "0123456789"}
        for s, x, al, align, col in ((f"{k}", 70, a_s, "l", INK), (_fmt(z, dec_h), CX, a_h, "c", INK), (_fmt(slope, dec_s), 1010, a_s, "r", ACC)):
            if al <= 0:
                continue
            w = cells_width(ctx, s, widths)
            xx = x if align == "l" else (x - w / 2 if align == "c" else x - w)
            ctx.set_source_rgba(*rgb(col), al)
            show_cells(ctx, s, xx, Y_VAL, widths)
        ctx.set_line_width(6)
        # HEIGHT: ink-grey bar centred under the number, proportional to height / initial height
        if a_h > 0:
            ctx.set_source_rgba(*rgb("#E5E7EB"), a_h)
            ctx.move_to(CX - 110, Y_BAR)
            ctx.line_to(CX + 110, Y_BAR)
            ctx.stroke()
            L = 220 * min(1.0, z / self.heights[0])
            if L > 0.5:
                ctx.set_source_rgba(*rgb(INK2), a_h)
                ctx.move_to(CX - L / 2, Y_BAR)
                ctx.line_to(CX + L / 2, Y_BAR)
                ctx.stroke()
        if a_s > 0:
            # SLOPE: accent bar anchored at the right, proportional to slope / initial slope
            ctx.set_source_rgba(*rgb(ACC), 0.18 * a_s)
            ctx.move_to(790, Y_BAR)
            ctx.line_to(1010, Y_BAR)
            ctx.stroke()
            L = 220 * min(1.0, slope / self.slopes[0])
            if L > 0.5:
                ctx.set_source_rgba(*rgb(ACC), a_s)
                ctx.move_to(1010 - L, Y_BAR)
                ctx.line_to(1010, Y_BAR)
                ctx.stroke()
            # STEP: 24 dots, ink for completed steps
            for i in range(N_STEPS):
                ctx.set_source_rgba(*rgb(INK if i < k else "#D1D5DB"), a_s)
                ctx.arc(74 + 11 * i, Y_BAR, 3.5, 0, TAU)
                ctx.fill()

    # ------------------------------------------------------------ the weight axes (end cue)

    AX_X = ((-0.68, 0.0), (-2.6, 0.0))    # x axis on the table: from just outside the table line (v > V_CUT) to the near end (weight 1)
    AX_Y = ((0.0, 0.42), (0.0, 2.6))      # y axis on the table, likewise (weight 2)
    AX_W, AX_A, AX_HEAD = 2.4, 0.75, 18.0

    @staticmethod
    def _arrowhead(ctx, p, q, size, alpha, col=INK):
        """Filled head at q of the segment p -> q."""
        ux, uy = q[0] - p[0], q[1] - p[1]
        nn = math.hypot(ux, uy) or 1.0
        ux, uy = ux / nn, uy / nn
        ctx.set_source_rgba(*rgb(col), alpha)
        ctx.move_to(q[0] + ux * 1.5, q[1] + uy * 1.5)
        ctx.line_to(q[0] - ux * size - uy * size * 0.42, q[1] - uy * size + ux * size * 0.42)
        ctx.line_to(q[0] - ux * size * 0.8, q[1] - uy * size * 0.8)
        ctx.line_to(q[0] - ux * size + uy * size * 0.42, q[1] - uy * size - ux * size * 0.42)
        ctx.close_path()
        ctx.fill()

    def axes_under(self, img, alpha):
        """The two weight axes drawn on the table plane z = 0, composited under the hill so it occludes them."""
        surf, ctx = layer()
        ctx.set_line_width(self.AX_W)
        for (P0, P1) in (self.AX_X, self.AX_Y):
            a, b = project(P0[0], P0[1], 0.0), project(P1[0], P1[1], 0.0)
            ctx.set_source_rgba(*rgb(INK), self.AX_A * alpha)
            ctx.move_to(*a)
            ctx.line_to(*b)
            ctx.stroke()
            self._arrowhead(ctx, a, b, self.AX_HEAD, 0.85 * alpha)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def axes_over(self, ctx, alpha):
        """Labels of the weight axes (readout style: tracked caps) and the vertical error axis along the cut face's left edge."""
        font(ctx, "bold", 24)
        tr = 0.14 * 24
        ctx.set_source_rgba(*rgb(INK2), 0.95 * alpha)
        ex = project(*self.AX_X[1], 0.0)
        show_tracked(ctx, "WEIGHT 1", ex[0] + 18, ex[1] + 9, tr)
        ey = project(*self.AX_Y[1], 0.0)
        w = tracked_width(ctx, "WEIGHT 2", tr)
        show_tracked(ctx, "WEIGHT 2", ey[0] - w + 12, ey[1] + 42, tr)
        # error: up the left edge of the cut face, stopping short of the corner where the rim and profile strokes meet
        c_bot, c_top = self.corners[2], self.corners[3]
        ux, uy = c_top[0] - c_bot[0], c_top[1] - c_bot[1]
        nn = math.hypot(ux, uy)
        ux, uy = ux / nn, uy / nn
        nx, ny = -uy, ux                      # to the right of the edge (into the face)
        a = (c_bot[0] + nx * 14 + ux * 10, c_bot[1] + ny * 14 + uy * 10)
        b = (c_top[0] + nx * 14 - ux * 30, c_top[1] + ny * 14 - uy * 30)
        ctx.set_source_rgba(*rgb(INK), self.AX_A * alpha)
        ctx.set_line_width(self.AX_W)
        ctx.move_to(*a)
        ctx.line_to(*b)
        ctx.stroke()
        self._arrowhead(ctx, a, b, self.AX_HEAD, 0.85 * alpha)
        ctx.set_source_rgba(*rgb(INK2), 0.95 * alpha)
        mx, my = a[0] + 0.24 * (b[0] - a[0]) + 16, a[1] + 0.24 * (b[1] - a[1]) + 9
        show_tracked(ctx, "ERROR", mx, my, tr)

    # ------------------------------------------------------------ the animated layer

    def _win(self, t, t0, t1):
        return min(ease_out_cubic((t - t0) / 0.5), 1 - seg(t, t1 - 0.3, t1)) if t0 <= t < t1 else 0.0

    def _draw(self, ctx, t, fade):
        if t < T_BALL:
            return
        path = self.path
        k, phase, u = self.state(t)
        kk = min(k, N_STEPS)
        # ---- the ball's position (eased slide), the true height and slope under it, drop-in and landing squash
        x, y, mm = self.ball_xy(t)
        z = f_xy(x, y)
        slope_now = math.hypot(*grad_xy(x, y))
        P, n, rb_px, rb_w, (cx, cy), (bx, by) = self.ball_geom(x, y)
        drop_u = seg(t, T_BALL, T_BALL + 0.5)
        a_ball = min(1.0, drop_u / 0.4) * fade
        drop = (1 - drop_u) ** 2                       # ease-in fall from 80 px above
        squash = 0.0
        tl = (t - T_FIRST - 0.84 * STEP_DT) / STEP_DT
        il = int(math.floor(tl))
        age = (tl - il) * STEP_DT
        if 0 <= il < N_STEPS and age < 0.16:
            squash = 1 - ease_out_cubic(age / 0.16)
        if T_BALL + 0.5 <= t < T_BALL + 0.66:
            squash = 1 - ease_out_cubic((t - T_BALL - 0.5) / 0.16)
        # ---- everything lying on the surface is clipped to the surface's screen region (the cut face is in front of it)
        ctx.save()
        self._path(ctx, self.surf_clip, close=True)
        ctx.clip()
        self.actors(ctx, t, (k, phase, u), (x, y), (P, n, rb_w), None, 1.0, rb_px, fade, a_ball, drop, False)
        self.step_arrow(ctx, (k, phase, u), None, 1.0, rb_px, fade)
        ctx.restore()
        # ---- leader labels (each quantity named once); both hang off the ball, in the band above the bowl
        A = (cx + 0.72 * rb_px + 4, cy - 0.72 * rb_px - 4)
        a1 = self._win(t, 6.6, 9.3)
        if a1 > 0:
            self.label(ctx, A, (A[0] + 110, LABEL_Y), "height", _fmt(z), a1 * fade, ease_out_cubic(seg(t, 6.6, 7.3)))
        a2 = self._win(t, 9.4, 11.4)
        if a2 > 0:
            kx = self.spt[0][0] + 0.72 * rb_px + 4 + 140      # the knee stays put while the ball slides
            self.label(ctx, A, (kx, LABEL_Y), "slope", _fmt(slope_now, 2), a2 * fade, ease_out_cubic(seg(t, 9.4, 10.04)))
        a3 = self._win(t, 11.5, 13.4)
        if a3 > 0:
            K3 = (self.spt[2][0] + 150, LABEL_Y)
            d3 = math.hypot(K3[0] - self.spt[2][0], K3[1] - self.spt[2][1])
            A3 = (self.spt[2][0] + 6 * (K3[0] - self.spt[2][0]) / d3, self.spt[2][1] + 6 * (K3[1] - self.spt[2][1]) / d3)
            self.label(ctx, A3, K3, "each dot: one step", "", a3 * fade, ease_out_cubic(seg(t, 11.5, 12.1)), dot=False)
        # ---- the ball: a sphere resting on the surface, over the tail of its arrow (with drop-in and landing squash)
        self.sphere(ctx, cx, cy - 80 * drop, rb_px, a_ball, contact=(bx, by), squash=squash)
        # ---- the loupe
        self.loupe(ctx, t, fade, (k, phase, u), (x, y), (P, n, rb_w), (cx, cy), (bx, by), rb_px, squash)
        # ---- readouts (the format switches at a landing, or at the start of the slide that crosses 0.1) and the axis labels
        kh = k + 1 if (phase == "move" and k < N_STEPS) else kk
        dec_h = 2 if self.heights[kh] >= 0.1 else 3
        dec_s = 2 if self.slopes[kh] >= 0.1 else 3
        self.readouts(ctx, kk, z, slope_now, dec_h, dec_s, fade * ease_out_cubic(seg(t, 6.4, 6.9)), fade * ease_out_cubic(seg(t, 9.6, 10.0)))
        a4 = self._win(t, 25.6, 29.0)
        if a4 > 0:
            self.axes_over(ctx, a4 * fade)

    def _headlines(self, ctx, t):
        for (t_in, t_out, lines, n_in, n_out, style) in self.cues:
            if not (t_in - 1e-6 <= t < t_out + n_out / FPS):
                continue
            k_in = (t - t_in) * FPS
            k_out = (t - t_out) * FPS
            alpha = 1.0
            if n_in and k_in < n_in:
                alpha = ease_out_cubic(k_in / n_in)
            if k_out >= 0 and n_out:
                alpha = min(alpha, 1 - k_out / n_out)
            if alpha <= 0:
                continue
            dy = 14 * (1 - ease_out_cubic(k_in / n_in)) if (style == "pop" and n_in and k_in < n_in) else 0.0
            for (s, kind, size, colr, base, la) in lines:
                font(ctx, kind, size)
                w = ctx.text_extents(s).x_advance
                fs = min(1.0, MAX_TEXT_W / w)
                ctx.save()
                ctx.translate(CX, base + dy)
                ctx.scale(fs, fs)
                ctx.translate(-CX, -base)
                ctx.set_source_rgba(*rgb(colr), la * alpha)
                ctx.move_to(CX - w / 2, base)
                ctx.show_text(s)
                ctx.restore()

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = np.ones((H, W, 3), np.float32)
        x0, x1, y0, y1 = CROP
        sub = img[y0:y1, x0:x1]
        # the faint silhouette of the hill is always on the page (frame 0 = frame 1799): the hook has a subject
        over(sub, self.b_rgb * GHOST_A, self.b_a * GHOST_A)
        a4 = self._win(t, 25.6, 29.0) * fade
        if a4 > 0:
            self.axes_under(img, a4)
        reveal = seg(t, T_SURF, T_SURF + 2.6)
        if reveal > 0 and fade > 0:
            sh = self.sh_a * (smooth(seg(reveal, 0.45, 1.0)) * fade)
            sub *= (1 - sh)[..., None]
            sub += sh[..., None] * self.sh_rgb
            if reveal < 1:
                # the hill rises out of the page by height; the frontier is a contour, starting from a shallow dish
                # (f = 0.4) that already has walls, and the whole thing fades up over its first 0.2 s
                zf = 0.4 + (Z_RIM - 0.38) * ease_out_cubic(reveal) ** 2
                m = np.clip((zf - self.b_z) / self.b_gpx + 0.5, 0, 1)
                m = np.where(self.b_stroke & (self.b_z > zf + 0.05), 0.0, m).astype(np.float32)
                front = np.clip(1.2 - np.abs(self.b_z - zf) / self.b_gpx, 0, 1) * (1 - reveal) ** 0.5
                m *= fade * min(1.0, reveal / 0.08)
                over(sub, self.b_rgb * (m * (1 - 0.35 * front))[..., None], self.b_a * m)
            elif fade < 1:
                over(sub, self.b_rgb * fade, self.b_a * fade)
            else:
                over(sub, self.b_rgb, self.b_a)
        if fade > 0 and t >= T_BALL:
            surf, ctx = layer()
            self._draw(ctx, t, fade)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
        surf2, ctx2 = layer()
        self._headlines(ctx2, t)
        trgb, ta = surf_rgba(surf2)
        over(img, trgb, ta)
        return dither(img, f)



if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s heights {sc.heights[0]:.2f} -> {sc.heights[-1]:.4f} slopes {sc.slopes[0]:.2f} -> {sc.slopes[-1]:.4f}")
    print(f"bowl bbox x {sc.bbox[0]}..{sc.bbox[1]} y {sc.bbox[2]}..{sc.bbox[3]}; exterior px {sc.n_exterior_px}")
    print("loupe caches", [(ss, m_hi, x0, y0, b.shape, round(b.nbytes / 1e6, 1)) for ss, m_hi, x0, y0, b in sc.lp])
    print("mag per step", [round(sc.mag(T_FIRST + k * STEP_DT + 0.05), 2) for k in range(N_STEPS)])
    print("loupe arrow px", [round(sc.seg_px[k] * sc.mag(T_FIRST + k * STEP_DT + 0.05), 1) for k in range(N_STEPS)])
    print("start", tuple(round(v) for v in sc.spt[0]), "landings", [tuple(round(v) for v in p) for p in sc.spt[1:7]], "min", tuple(round(v) for v in surf_pt(0, 0)))
    print("step px", [round(v, 1) for v in sc.seg_px])
    frames = [0, 160, 220, 300, 400, 560, 600, 640, 700, 780, 900, 960, 1100, 1250, 1400, 1520, 1650, 1799]
    ims = []
    times = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        dt = time.time() - t0
        times.append(dt)
        print(f"  frame {fr} {dt:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s (first {times[0]:.2f}s)")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (300, 620, 700, 760, 900, 1100, 1300, 1480, 1700):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
