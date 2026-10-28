"""A general-surface version of the reel-23 cut hill: the same camera, lighting, ceramic shading, contour
lines, cut face, silhouette strokes and ground shadow as gdsimple/scene.py, but for ANY height function
f(u, v) given as numba-compatible scalar functions (f, df/du, df/dv) in the bowl's own rotated axes. The
analytic paraboloid ray-cast is replaced by a numba ray-marcher with bisection. Install into a scene that
subclasses gdsimple/lr3 by calling `install(gds, F, FU, FV)`: it patches the module's f_xy / grad_xy /
normal_xy and returns the mixin methods (_raycast, _build_bowl, _build_shadow) to bind on the Scene."""

import math

import cv2
import numba as nb
import numpy as np


def install(gds, F, FU, FV, textkit):
    """F, FU, FV: @njit scalar functions of (u, v). Returns a dict of methods for the Scene subclass."""
    _cr, _sr = gds._cr, gds._sr
    KZ, Z_RIM, V_CUT = gds.KZ, gds.Z_RIM, gds.V_CUT
    CAM_C, CAM_FW, CAM_RIGHT, CAM_UP, F_PX, PCX, PCY, CROP = gds.CAM_C, gds.CAM_FW, gds.CAM_RIGHT, gds.CAM_UP, gds.F_PX, gds.PCX, gds.PCY, gds.CROP
    LIGHT = gds.LIGHT
    SURF_LO, SURF_HI, FACE_C, EDGE_C, SHADOW_C = gds.SURF_LO, gds.SURF_HI, gds.FACE_C, gds.EDGE_C, gds.SHADOW_C
    project, project_arr = gds.project, gds.project_arr
    rgb, layer, surf_rgba = textkit.rgb, textkit.layer, textkit.surf_rgba

    # ---------------------------------------------------------------- the patched height field (module-level, used everywhere)
    def f_xy(x, y):
        u = _cr * x + _sr * y
        v = -_sr * x + _cr * y
        return f_arr(u, v)

    def f_arr(u, v):
        if np.ndim(u) == 0 and np.ndim(v) == 0:
            return F(float(u), float(v))
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return _f_vec(u.ravel(), v.ravel()).reshape(u.shape)

    def grad_xy(x, y):
        u = _cr * x + _sr * y
        v = -_sr * x + _cr * y
        if np.ndim(u) == 0:
            gu, gv = FU(float(u), float(v)), FV(float(u), float(v))
        else:
            u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
            gu = _fu_vec(u.ravel(), v.ravel()).reshape(u.shape)
            gv = _fv_vec(u.ravel(), v.ravel()).reshape(u.shape)
        return (_cr * gu - _sr * gv, _sr * gu + _cr * gv)

    def normal_xy(x, y):
        gx, gy = grad_xy(x, y)
        n = np.array([-KZ * gx, -KZ * gy, 1.0])
        return n / np.linalg.norm(n)

    @nb.njit(cache=False)
    def _f_vec(u, v):
        out = np.empty_like(u)
        for i in range(u.size):
            out[i] = F(u[i], v[i])
        return out

    @nb.njit(cache=False)
    def _fu_vec(u, v):
        out = np.empty_like(u)
        for i in range(u.size):
            out[i] = FU(u[i], v[i])
        return out

    @nb.njit(cache=False)
    def _fv_vec(u, v):
        out = np.empty_like(u)
        for i in range(u.size):
            out[i] = FV(u[i], v[i])
        return out

    gds.f_xy, gds.grad_xy, gds.normal_xy = f_xy, grad_xy, normal_xy

    # ---------------------------------------------------------------- the ray-marcher
    @nb.njit(parallel=True, cache=False)
    def march(px, py, cam, fw, right, up, f_px, kz, z_rim, v_cut, cr, sr, t0, t1, dt):
        """For every pixel: first entry of the ray into the solid {v <= v_cut, f <= z_rim, 0 <= z <= kz f}.
        kind: 0 miss, 1 surface, 2 cut face, 3 rim wall (outer side, f = z_rim). Returns kind, t, U, V, Z."""
        H, W = py.size, px.size
        kind = np.zeros((H, W), np.int8)
        T = np.zeros((H, W), np.float64)
        U = np.zeros((H, W), np.float64)
        V = np.zeros((H, W), np.float64)
        Z = np.zeros((H, W), np.float64)
        zr = z_rim * kz
        for j in nb.prange(H):
            for i in range(W):
                dx = fw[0] + px[i] / f_px * right[0] - py[j] / f_px * up[0]
                dy = fw[1] + px[i] / f_px * right[1] - py[j] / f_px * up[1]
                dz = fw[2] + px[i] / f_px * right[2] - py[j] / f_px * up[2]
                nn = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= nn
                dy /= nn
                dz /= nn
                # entry through the cut plane v = v_cut (analytic): if that point is inside the other constraints
                u0 = cr * cam[0] + sr * cam[1]
                v0 = -sr * cam[0] + cr * cam[1]
                du = cr * dx + sr * dy
                dv = -sr * dx + cr * dy
                t_face = 1e18
                if abs(dv) > 1e-12:
                    tp = (v_cut - v0) / dv
                    if tp > 0:
                        up_ = u0 + tp * du
                        zp = cam[2] + tp * dz
                        fp = F(up_, v_cut)
                        if zp >= 0.0 and zp <= kz * fp and fp <= z_rim:
                            t_face = tp
                # march for the surface / rim wall
                t = t0
                was_in = False
                t_hit = 1e18
                k_hit = 0
                prev_t = t
                prev_f = 1e18
                while t < t1:
                    u = u0 + t * du
                    v = v0 + t * dv
                    z = cam[2] + t * dz
                    inside = False
                    fv_ = F(u, v)
                    if v <= v_cut and z >= -1.0:
                        if fv_ <= z_rim and z <= kz * fv_:
                            inside = True
                    if (not inside) and fv_ > z_rim and z < zr and prev_f <= z_rim and v <= v_cut:
                        # the ray left the footprint below the rim height: it must have passed under the rim's lip
                        # between the last two samples (a sliver thinner than dt). Sub-sample to find that entry.
                        for q in range(1, 64):
                            tq = prev_t + (t - prev_t) * q / 64.0
                            uq = u0 + tq * du
                            vq = v0 + tq * dv
                            zq = cam[2] + tq * dz
                            fq = F(uq, vq)
                            if vq <= v_cut and zq >= -1.0 and fq <= z_rim and zq <= kz * fq:
                                inside = True
                                t = tq
                                prev_t = prev_t + (t - prev_t) * (q - 1) / 64.0 if q > 1 else prev_t
                                break
                    if inside:
                        # bisect between prev_t (outside) and t (inside)
                        lo = prev_t
                        hi = t
                        for _ in range(18):
                            mid = 0.5 * (lo + hi)
                            um = u0 + mid * du
                            vm = v0 + mid * dv
                            zm = cam[2] + mid * dz
                            ins = False
                            if vm <= v_cut and zm >= -1.0:
                                fm = F(um, vm)
                                if fm <= z_rim and zm <= kz * fm:
                                    ins = True
                            if ins:
                                hi = mid
                            else:
                                lo = mid
                        t_hit = hi
                        uh = u0 + hi * du
                        vh = v0 + hi * dv
                        zh = cam[2] + hi * dz
                        fh = F(uh, vh)
                        # which constraint is tight at the entry; an entry below the table (through the cut plane
                        # under the face) is not a hit at all
                        if zh < -1e-4:
                            k_hit = 0
                            t_hit = 1e18
                        else:
                            # what the ray was doing just outside: above the surface -> it came in through the top
                            # (a surface hit); under the surface's extension but outside the rim -> through the rim wall
                            ul_ = u0 + lo * du
                            vl_ = v0 + lo * dv
                            zl_ = cam[2] + lo * dz
                            fl_ = F(ul_, vl_)
                            if zl_ > kz * fl_:
                                k_hit = 1
                            elif fl_ > z_rim:
                                k_hit = 3
                            else:
                                k_hit = 1
                        break
                    prev_t = t
                    prev_f = fv_
                    t += dt
                if t_face < t_hit:
                    t_hit = t_face
                    k_hit = 2
                if k_hit > 0:
                    kind[j, i] = k_hit
                    T[j, i] = t_hit
                    U[j, i] = u0 + t_hit * du
                    V[j, i] = v0 + t_hit * dv
                    Z[j, i] = cam[2] + t_hit * dz
        return kind, T, U, V, Z

    def _raycast(self, px, py):
        """Same outputs as gdsimple.Scene._raycast: shaded colour, hit, is_face, exterior, Zf, Zrev."""
        f32 = np.float32
        cam = np.asarray(CAM_C, float)
        # march range: the solid lies within |u| <= 4.5, |v| <= 2.5, 0 <= z <= zr; use the distance to that box
        t0, t1 = CAM_D_RANGE
        kind, T, U, V, Z = march(np.asarray(px, float), np.asarray(py, float), cam, np.asarray(CAM_FW, float),
                                 np.asarray(CAM_RIGHT, float), np.asarray(CAM_UP, float), float(F_PX), float(KZ),
                                 float(Z_RIM), float(V_CUT), float(_cr), float(_sr), float(t0), float(t1), MARCH_DT)
        hit = kind > 0
        is_face = kind == 2
        exterior = kind == 3
        PX, PY = np.meshgrid(np.asarray(px, float), np.asarray(py, float))
        dx = CAM_FW[0] + PX / F_PX * CAM_RIGHT[0] - PY / F_PX * CAM_UP[0]
        dy = CAM_FW[1] + PX / F_PX * CAM_RIGHT[1] - PY / F_PX * CAM_UP[1]
        dz = CAM_FW[2] + PX / F_PX * CAM_RIGHT[2] - PY / F_PX * CAM_UP[2]
        del PX, PY
        nn = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = (dx / nn).astype(f32), (dy / nn).astype(f32), (dz / nn).astype(f32)
        del nn
        Fh = f_arr(U, V)
        Zf = np.where(is_face, Z / KZ, Fh)
        Zrev = np.where(is_face, f_arr(U, np.full_like(U, V_CUT)), Zf)
        Zf, Zrev = Zf.astype(f32), Zrev.astype(f32)
        # normals from the analytic gradient (in world xy)
        gu = _fu_vec(U.ravel(), V.ravel()).reshape(U.shape)
        gv = _fv_vec(U.ravel(), V.ravel()).reshape(U.shape)
        gx = KZ * (_cr * gu - _sr * gv)
        gy = KZ * (_sr * gu + _cr * gv)
        nrm = np.sqrt(gx * gx + gy * gy + 1)
        nx, ny, nz = (-gx / nrm).astype(f32), (-gy / nrm).astype(f32), (1 / nrm).astype(f32)
        del gx, gy, nrm, gu, gv
        # rim wall: outward horizontal normal (gradient direction, no z)
        if exterior.any():
            gu = _fu_vec(U.ravel(), V.ravel()).reshape(U.shape)
            gv = _fv_vec(U.ravel(), V.ravel()).reshape(U.shape)
            wx, wy = _cr * gu - _sr * gv, _sr * gu + _cr * gv
            wn = np.sqrt(wx * wx + wy * wy) + 1e-9
            nx = np.where(exterior, (wx / wn).astype(f32), nx)
            ny = np.where(exterior, (wy / wn).astype(f32), ny)
            nz = np.where(exterior, f32(0.0), nz)
        nx = np.where(is_face, f32(-_sr), nx)
        ny = np.where(is_face, f32(_cr), ny)
        nz = np.where(is_face, f32(0.0), nz)
        L = np.asarray(LIGHT, f32)
        ndl = nx * L[0] + ny * L[1] + nz * L[2]
        lam = np.clip(ndl, 0, 1)
        wrap = np.clip(ndl * 0.5 + 0.5, 0, 1)
        del ndl
        zk = np.clip(Zf / Z_RIM, 0, 1)
        k = zk ** 0.6
        lo_, hi_, face_ = (np.array(rgb(h_), f32) for h_ in (SURF_LO, SURF_HI, FACE_C))
        base = lo_ + (hi_ - lo_) * k[..., None]
        shade01 = 0.20 * wrap + 0.80 * lam
        tint = np.array([0.44, 0.53, 0.74], f32)
        mult = tint + (1 - tint) * shade01[..., None]
        mult *= (1 - 0.16 * (1 - zk) ** 2)[..., None]
        col = base * mult
        hx, hy, hz = L[0] - dx, L[1] - dy, L[2] - dz
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** 28
        col += (0.08 * spec)[..., None]
        del hx, hy, hz, hn, spec, base, mult, shade01, wrap, lam, nx, ny, nz, dx, dy, dz, k
        zr = Z_RIM * KZ
        zfrac = np.clip(Z / zr, 0, 1).astype(f32)
        fcol = face_ * (0.92 + 0.08 * zfrac)[..., None]
        col = np.where(is_face[..., None], fcol, col)
        col = np.where(exterior[..., None], face_ * 0.90, col)
        return col.astype(f32), hit, is_face, exterior, Zf, Zrev

    # ---------------------------------------------------------------- rim and profile (level set f = Z_RIM, numerically)
    def _outer_roots(v, u_span=6.0, n=2401):
        """Outermost u with f(u, v) = Z_RIM (left, right) or None if the level does not reach this v."""
        us = np.linspace(-u_span, u_span, n)
        fs = f_arr(us, np.full_like(us, v)) - Z_RIM
        below = np.where(fs <= 0)[0]
        if below.size == 0:
            return None
        i0, i1 = below[0], below[-1]
        if i0 == 0 or i1 == n - 1:
            return None

        def bis(ia, ib):
            a, b = us[ia], us[ib]
            fa = F(float(a), float(v)) - Z_RIM
            for _ in range(40):
                m = 0.5 * (a + b)
                fm = F(float(m), float(v)) - Z_RIM
                if (fm > 0) == (fa > 0):
                    a, fa = m, fm
                else:
                    b = m
            return 0.5 * (a + b)
        return bis(i0 - 1, i0), bis(i1, i1 + 1)

    def rim_profile(self):
        """rim_pts: screen polyline of the drawn rim, the level set f = Z_RIM restricted to v <= V_CUT, ordered from
        the left end of the cut around the far side to the right end (found as the boundary of the mask f <= Z_RIM
        on a fine (u, v) grid, with the straight cut segment removed); prof_pts: the profile z = KZ f(u, V_CUT)
        from the right end to the left end. Also returns the rim in (u, v) for the ground shadow."""
        res = 0.004
        us = np.arange(-5.0, 5.0 + res, res)
        vs = np.arange(-3.0, V_CUT + res, res)
        UU, VV = np.meshgrid(us, vs)
        mask = (f_arr(UU, VV) <= Z_RIM).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=len)[:, 0, :]                       # (col, row) = (u index, v index)
        pts = [(us[i], vs[j]) for i, j in c]
        keep = [p for p in pts if p[1] < V_CUT - 2.5 * res]
        # rotate so the list starts just after the gap left by the removed cut segment
        gap = max(range(len(keep)), key=lambda i: (keep[i][0] - keep[i - 1][0]) ** 2 + (keep[i][1] - keep[i - 1][1]) ** 2)
        keep = keep[gap:] + keep[:gap]
        if keep[0][0] > keep[-1][0]:                        # start at the left end (u < 0), finish at the right end
            keep = keep[::-1]
        rim_uv = keep[::3]
        rim_pts = [project(_cr * u - _sr * v, _sr * u + _cr * v, Z_RIM) for u, v in rim_uv]
        r = _outer_roots(V_CUT)
        ul, ur = r
        us_p = np.linspace(ur, ul, 240)
        prof_pts = [project(_cr * u - _sr * V_CUT, _sr * u + _cr * V_CUT, F(float(u), float(V_CUT))) for u in us_p]
        return rim_pts, prof_pts, ul, ur, rim_uv

    def _build_bowl(self):
        ss = 2
        f32 = np.float32
        x0, x1, y0, y1 = CROP
        w1, h1 = x1 - x0, y1 - y0
        px = (np.arange(w1 * ss) + 0.5) / ss + x0 - PCX
        py = (np.arange(h1 * ss) + 0.5) / ss + y0 - PCY
        col, hit, is_face, exterior, Zf, Zrev = self._raycast(px, py)
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
        surf, ctx = layer(w1, h1)
        ctx.translate(-x0, -y0)
        ec = rgb(EDGE_C)
        rim_pts, prof_pts, ul, ur, rim_uv = rim_profile(self)
        for i, p in enumerate(rim_pts):
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        ctx.set_source_rgba(*ec, 0.70)
        ctx.set_line_width(2.2)
        ctx.stroke()
        for i, p in enumerate(prof_pts):
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        ctx.set_source_rgba(*ec, 0.70)
        ctx.set_line_width(2.2)
        ctx.stroke()
        self.surf_clip = rim_pts + prof_pts
        self.prof_pts = prof_pts
        self.ua = ur
        self.rim_uv = rim_uv
        corners = [project(_cr * u - _sr * V_CUT, _sr * u + _cr * V_CUT, zf) for u, zf in ((ul, Z_RIM), (ul, 0.0), (ur, 0.0), (ur, Z_RIM))]
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
        x0, x1, y0, y1 = CROP
        surf, ctx = layer(x1 - x0, y1 - y0)
        ctx.translate(-x0, -y0)
        rim_uv = self.rim_uv if hasattr(self, "rim_uv") else rim_profile(self)[4]
        for i, (u, v) in enumerate(rim_uv[::4]):
            p = project(_cr * u * 0.94 - _sr * v * 0.94, _sr * u * 0.94 + _cr * v * 0.94, 0.0)
            (ctx.move_to if i == 0 else ctx.line_to)(p[0] + 6, p[1] + 8)
        ctx.close_path()
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.fill()
        _, a = surf_rgba(surf)
        self.sh_a = (cv2.GaussianBlur(a, (0, 0), 10) * 0.20).astype(np.float32)
        self.sh_rgb = np.array(rgb(SHADOW_C), np.float32)

    return dict(_raycast=_raycast, _build_bowl=_build_bowl, _build_shadow=_build_shadow, f_arr=f_arr, rim_profile=rim_profile)


CAM_D_RANGE = (12.0, 40.0)      # march range in world units along the ray (camera distance is 24)
MARCH_DT = 0.025
