"""Video time -> signed simulation time, camera, and display rules. All pure functions."""

import math

FPS = 60
N_FRAMES = 1080            # 18.000 s
S0 = 0.2                   # sim time shown on frame 0


def _S(u):
    return 3 * u * u - 2 * u ** 3


def sigma(t):
    """Signed sim time for video time t (seconds). Displayed state is state(|sigma|)."""
    if t < 9.8:
        return S0 + t
    if t < 10.2:
        u = (t - 9.8) / 0.4
        return 10.0 + 0.4 * (u - u ** 3 + u ** 4 / 2)
    if t < 10.3:
        return 10.2
    if t < 10.7:
        u = (t - 10.3) / 0.4
        return 10.2 - 1.2 * (u ** 3 - u ** 4 / 2)
    if t < 10.8:
        return 9.6 - 3 * (t - 10.7)
    if t < 11.1:
        u = (t - 10.8) / 0.3
        return 9.3 - 0.3 * (3 * u - 2 * (u ** 3 - u ** 4 / 2))
    if t < 13.8:
        return 8.7 - (t - 11.1)
    if t < 14.1:
        u = (t - 13.8) / 0.3
        return 6.0 - 0.3 * (u + u ** 3 - u ** 4 / 2)
    if t < 15.75:
        return 5.55 - 2 * (t - 14.1)
    if t < 16.15:
        u = (t - 15.75) / 0.4
        return 2.25 - 0.4 * (2 * u - (u ** 3 - u ** 4 / 2))
    return 1.65 - (t - 16.15)


def speed(t, h=1e-4):
    return (sigma(t + h) - sigma(t - h)) / (2 * h)


def d_of_frame(f):
    return abs(sigma((f % N_FRAMES) / FPS))


def camera_scale(t, t_chaos):
    """Zoom about the pivot. t_chaos is the CHAOS frame time."""
    if t <= t_chaos:
        return 1 + 0.05 * _S(t / t_chaos)
    f = t * FPS
    fc = t_chaos * FPS
    if f < fc + 8:
        u = (f - fc) / 8
        return 1.05 - 0.05 * (1 - (1 - u) ** 3)
    if t < t_chaos + 0.1333:
        return 1.0
    if t < 10.2:
        return 1.0 + 0.06 * _S((t - (t_chaos + 0.1333)) / (10.2 - (t_chaos + 0.1333)))
    if t < 10.3:
        return 1.06
    if t < 11.5:
        return 1.06 - 0.06 * _S((t - 10.3) / 1.2)
    return 1.0


def shake(f, f_chaos):
    k = f - f_chaos
    if 0 <= k < 10:
        a = 6 * math.exp(-k / 3)
        return a * math.sin(1.7 * k), a * math.cos(2.3 * k)
    return 0.0, 0.0


def flash(f, f_chaos):
    """Brief impact flash: a small linear-light wash plus an emissive boost, decaying over 3 frames."""
    k = f - f_chaos
    return (0.015, 0.006, 0.002)[k] if 0 <= k < 3 else 0.0


def gap_display(M):
    """Counter string for the running-max gap M (degrees)."""
    if M >= 180:
        return "180°+"
    if M >= 100:
        return f"{int(M // 10) * 10}°"
    if M >= 95:
        return "100°"
    e = math.floor(math.log10(M))
    mant = M / 10 ** e
    r = math.floor(mant + 0.5)
    if r >= 10:
        r, e = 1, e + 1
    v = r * 10 ** e
    if e >= 0:
        return f"{int(round(v))}°"
    return f"{v:.{-e}f}°"


def lens_log_mag(s, change_points):
    """log10 magnification at sim time s, with smoothstep tweens ending at each change point."""
    k = 7
    for cp in change_points:
        if s >= cp:
            k -= 1
        elif s >= cp - 0.2:
            u = (s - (cp - 0.2)) / 0.2
            return k - _S(u), True
        else:
            break
    return float(max(k, 0)), False


def lens_presence(s):
    """(scale, alpha) of the lens as a function of sim time."""
    if s < 0.7 or s > 7.45:
        return 1.0, 0.0
    if s < 0.867:
        u = (s - 0.7) / 0.167
        b = u - 1
        eo = b * b * ((1.70158 + 1) * b + 1.70158) + 1
        return 0.6 + 0.4 * eo, u
    if s > 7.30:
        u = (s - 7.30) / 0.15
        return 1 + 0.15 * u, 1 - u
    return 1.0, 1.0


def fmt_mag(k):
    return "×" + f"{10 ** int(round(k)):,}"
