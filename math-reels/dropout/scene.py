"""Three identical wide ReLU networks train on a spiral where 83 of the 400 labels were flipped:
plain, with dropout, with weight decay. Three live decision fields (flipped labels drawn as diamonds,
filled when the network believes the lie), a lies-believed count and clean-test score under each,
and a race chart of clean-test accuracy. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from textkit import (H, W, cells_width, dither, font, layer, linear_to_srgb, over, rgb, show_cells,  # noqa: E402
                     show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)
from regsim import DROP_P, NAMES, RECORD, STEPS  # noqa: E402

_spec = importlib.util.spec_from_file_location("ruin_scene", os.path.join(ROOT, "ruin", "scene.py"))
_ruin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruin)

FPS = 60
N_FRAMES = 1380                     # 23 s
F0, F1, EXPO = 30, 900, 1.3
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
NET_COL = {"plain": MAGENTA, "dropout": AMBER, "l2": CYAN}
LABEL = {"plain": "PLAIN", "dropout": "DROPOUT", "l2": "WEIGHT DECAY"}
SUB = {"plain": "nothing held back", "dropout": f"{int(DROP_P * 100)}% of neurons off, every step", "l2": "weights kept small"}
CX = 470
PANEL_Y0, PANEL_S = 690, 310
PANEL_X0 = [60, 385, 710]
CHART = (90, 1300, 990, 1450)
LOOP_FADE = (22.3, 22.85)
MAX_TEXT_W = 820
TRAIN_PATH = os.path.join(HERE, "dropout_train.npz")


def frame_step(f):
    if f < F0:
        return 0
    if f >= F1:
        return STEPS
    u = (f - F0) / (F1 - F0)
    return int(round(STEPS * (1 - (1 - u) ** EXPO)))


def step_frame(s):
    u = 1 - (1 - s / STEPS) ** (1 / EXPO)
    return int(math.ceil(F0 + u * (F1 - F0)))


class Scene:
    def __init__(self, train_path=TRAIN_PATH):
        if not os.path.exists(train_path):
            import regsim
            regsim.main(train_path)
        d = np.load(train_path)
        with open(train_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.X, self.y, self.yn, self.flip = d["X"], d["y"], d["yn"], d["flip"]
        self.n_flip = int(self.flip.sum())
        self.fields = {n: d[f"{n}_fields"] for n in NAMES}
        self.pred = {n: d[f"{n}_pred"] for n in NAMES}
        self.acc_train = {n: np.array(self.meta["acc_train"][n]) for n in NAMES}
        self.acc_test = {n: np.array(self.meta["acc_test"][n]) for n in NAMES}
        self.lies = {n: np.array(self.meta["lies"][n]) for n in NAMES}
        self.final = self.meta["final"]
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_base = np.array(srgb_to_linear(rgb("#070A16")), np.float32)
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    def rec(self, step):
        return int(round(step / RECORD))

    # ------------------------------------------------------------ cues

    def _cues(self):
        nf = self.n_flip
        hook = [(f"I flipped {nf} of 400 labels. Lies in the data.", "black", 46, WHITE, 300, 1), ("Which network believes them?", "dbold", 44, WHITE, 392, 0.9)]
        cues = [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.8, 5.6, [("Plain. Dropout: half the neurons off, every step. Weight decay.", "black", 38, WHITE, 312, 1), ("Diamonds are the lies. Filled = the network believes it.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
        ]
        t_end = F1 / FPS
        half = nf // 2
        hit = np.where(self.lies["plain"] >= half)[0]
        if len(hit):
            s_half = int(hit[0]) * RECORD
            t_h = max(step_frame(s_half) / FPS + 0.1, 6.0)
            cues.append((t_h, t_h + 2.4, [(f"Plain believes {half} lies already.", "black", 52, MAGENTA, 312, 1), ("Its clean score peaked. Now it's falling.", "dbold", 40, WHITE, 396, 0.9)], 6, 6, "chaos"))
            t_mid0 = t_h + 2.7
        else:
            t_mid0 = 6.0
        if t_end - 0.4 - t_mid0 > 1.6:
            cues.append((t_mid0, t_end - 0.4, [("Dropout can't memorise a point:", "black", 48, AMBER, 312, 1), ("the neuron that knew it is switched off half the time.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"))
        fp, fd, fl = (self.final[n] for n in NAMES)
        cues += [
            (t_end + 0.3, t_end + 2.6, [(f"Lies believed: {fp['lies']}, {fd['lies']}, {fl['lies']} of {nf}.", "black", 48, WHITE, 312, 1), (f"Clean test: {fp['test'] * 100:.0f}%, {fd['test'] * 100:.0f}%, {fl['test'] * 100:.0f}%.", "black", 48, GREEN, 404, 1)], 6, 6, "chaos"),
            (t_end + 2.8, t_end + 5.2, [("Memorising a lie takes one neuron that owns one point.", "black", 40, WHITE, 312, 1), ("Dropout kills that neuron half the time. Decay shrinks it.", "dbold", 38, WHITE, 396, 0.9)], 8, 6, "pop"),
            (t_end + 5.4, 22.2, [("A network that can memorise anything", "black", 48, AMBER, 312, 1), ("needs a reason not to.", "black", 60, AMBER, 404, 1)], 8, 6, "pop"),
            (22.5, 23.1, hook, 10, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        step = frame_step(f)
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        lin = self.bg.copy()
        pts = np.zeros((H, W, 3), np.float32)
        r = self.rec(step)
        for i, name in enumerate(NAMES):
            p_small = self.fields[name][r].astype(np.float32)
            if mix > 0:
                p_small = (1 - mix) * p_small + mix * self.fields[name][0].astype(np.float32)
            self._panel(lin, pts, i, name, p_small, 0 if mix > 0.5 else r)
        small = cv2.resize(pts, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.6 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.6 * (pts + glow))
        lin = 1 - (1 - np.clip(lin, 0, 1)) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, 0 if mix > 0.5 else step)
        _ruin.Scene._headlines(self, img, f, t)
        return dither(img, f)

    def _panel(self, lin, pts, i, name, p_small, r):
        x0, y0, s = PANEL_X0[i], PANEL_Y0, PANEL_S
        p = cv2.resize(p_small, (s, s), interpolation=cv2.INTER_CUBIC)
        p = np.clip(p, 0, 1)
        conf = np.abs(2 * p - 1)
        k = (conf ** 0.9 * 0.12)[..., None]
        tint = p[..., None] * self.c_mag + (1 - p[..., None]) * self.c_cyan
        region = self.c_base + k * tint
        edge = np.exp(-((p - 0.5) / 0.03) ** 2)
        gmag = np.hypot(cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3))
        edge *= np.clip(gmag / 0.02, 0, 1)
        region += (edge * 0.16)[..., None] * (0.55 + 0.45 * tint)
        lin[y0:y0 + s, x0:x0 + s] = region
        pred = self.pred[name][r]
        surf, ctx = layer()
        cx, cy = x0 + s / 2, y0 + s / 2
        sc = (s / 2) / 1.05
        for j in range(len(self.X)):
            lab = self.yn[j] > 0.5                       # the label the network was shown
            col = rgb(MAGENTA) if lab else rgb(CYAN)
            x, y = cx + self.X[j, 0] * sc, cy - self.X[j, 1] * sc
            agrees = pred[j] == lab
            if self.flip[j]:
                # a lie: diamond in the lie's colour; filled when the network believes it
                ctx.move_to(x, y - 5.5)
                ctx.line_to(x + 5.5, y)
                ctx.line_to(x, y + 5.5)
                ctx.line_to(x - 5.5, y)
                ctx.close_path()
                if agrees:
                    ctx.set_source_rgba(*col, 1)
                    ctx.fill_preserve()
                    ctx.set_source_rgba(1, 1, 1, 0.95)
                    ctx.set_line_width(1.6)
                    ctx.stroke()
                else:
                    ctx.set_source_rgba(*col, 0.7)
                    ctx.set_line_width(1.6)
                    ctx.stroke()
            elif agrees:
                ctx.set_source_rgba(*col, 1)
                ctx.arc(x, y, 2.7, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.arc(x, y, 1.2, 0, 2 * math.pi)
                ctx.fill()
            else:
                ctx.set_source_rgba(*col, 0.55)
                ctx.set_line_width(1.4)
                ctx.arc(x, y, 3.4, 0, 2 * math.pi)
                ctx.stroke()
        prgb, _ = surf_rgba(surf)
        pts += (prgb ** 2.2 * 0.7).astype(np.float32)

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, step):
        surf, ctx = layer()
        r = self.rec(step)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, "%": 60, ".": 22})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "BEST ON CLEAN DATA", 0.14 * 24)
        show_tracked(ctx, "BEST ON CLEAN DATA", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{step:,}", 70, 545, big)
        tests = {n: self.acc_test[n][r] for n in NAMES}
        leader = max(NAMES, key=lambda n: (tests[n], -NAMES.index(n))) if step > 0 else None
        font(ctx, "dbold", 48)
        s = LABEL[leader] if leader else "—"
        ctx.set_source_rgb(*(rgb(NET_COL[leader]) if leader else (1, 1, 1)))
        ctx.move_to(910 - ctx.text_extents(s).x_advance, 545)
        ctx.show_text(s)
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, f"2 → 256 → 256 → 1   ·   ReLU   ·   ADAM 0.01   ·   {self.n_flip} OF 400 LABELS FLIPPED", 70, 612, 0.08 * 22)
        lie_cells = {str(c): 36 for c in range(10)} | {"/": 24}
        for i, name in enumerate(NAMES):
            x0, y0, s_ = PANEL_X0[i], PANEL_Y0, PANEL_S
            col = NET_COL[name]
            ctx.set_line_width(1.5)
            ctx.set_source_rgba(*rgb(col), 0.5)
            ctx.rectangle(x0 + 0.5, y0 + 0.5, s_, s_)
            ctx.stroke()
            font(ctx, "black", 26)
            ctx.set_source_rgba(*rgb(col), 0.98)
            ctx.move_to(x0, y0 - 20)
            ctx.show_text(LABEL[name])
            font(ctx, "bold", 16)
            ctx.set_source_rgba(1, 1, 1, 0.5)
            show_tracked(ctx, SUB[name], x0, y0 - 50, 0.04 * 16)
            # lies believed
            lies = int(self.lies[name][r])
            font(ctx, "bold", 17)
            ctx.set_source_rgba(1, 1, 1, 0.5)
            lab = "LIES BELIEVED"
            show_tracked(ctx, lab, x0 + s_ / 2 - tracked_width(ctx, lab, 0.12 * 17) / 2, y0 + s_ + 34, 0.12 * 17)
            font(ctx, "dbold", 52)
            s = f"{lies}/{self.n_flip}"
            ctx.set_source_rgb(*rgb(col))
            tw = cells_width(ctx, s, lie_cells)
            show_cells(ctx, s, x0 + s_ / 2 - tw / 2, y0 + s_ + 90, lie_cells)
            font(ctx, "bold", 17)
            tr, te = self.acc_train[name][r], tests[name]
            lab_a, lab_b = f"TRAIN {tr * 100:.0f}%", f"CLEAN {te * 100:.0f}%"
            wa, wb, wd = tracked_width(ctx, lab_a, 0.08 * 17), tracked_width(ctx, lab_b, 0.08 * 17), 30
            sx = x0 + s_ / 2 - (wa + wd + wb) / 2
            ctx.set_source_rgba(1, 1, 1, 0.55)
            show_tracked(ctx, lab_a, sx, y0 + s_ + 128, 0.08 * 17)
            ctx.set_source_rgba(1, 1, 1, 0.35)
            show_tracked(ctx, "\u00b7", sx + wa + wd / 2 - 3, y0 + s_ + 128, 0)
            ctx.set_source_rgba(*rgb(GREEN), 0.95)
            show_tracked(ctx, lab_b, sx + wa + wd, y0 + s_ + 128, 0.08 * 17)
        # race chart: clean test accuracy
        cx0, cy0, cx1, cy1 = CHART

        def px_(st):
            return cx0 + (cx1 - cx0) * st / STEPS

        def py_(a):
            return cy1 - (cy1 - cy0) * (a - 0.6) / 0.4

        ctx.set_line_width(1)
        for a, lab in ((0.6, "60%"), (0.8, "80%"), (1.0, "100%")):
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.5 if a == 1 else 0.14)
            ctx.set_dash([5, 7] if a == 1 else [])
            ctx.move_to(cx0, py_(a) + 0.5)
            ctx.line_to(cx1, py_(a) + 0.5)
            ctx.stroke()
            ctx.set_dash([])
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*(rgb(GREEN) if a == 1 else (1, 1, 1)), 0.7 if a == 1 else 0.45)
            ctx.move_to(cx0 - ctx.text_extents(lab).x_advance - 10, py_(a) + 6)
            ctx.show_text(lab)
        font(ctx, "bold", 17)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        for st in (0, 1000, 2000, 3000):
            lab = f"{st:,}"
            ctx.move_to(px_(st) - ctx.text_extents(lab).x_advance / 2, cy1 + 26)
            ctx.show_text(lab)
        lab = "STEPS  ·  ACCURACY ON CLEAN DATA"
        ctx.move_to(cx1 - ctx.text_extents(lab).x_advance, cy1 + 48)
        ctx.show_text(lab)
        for name in NAMES:
            acc = self.acc_test[name]
            ctx.set_line_width(3)
            ctx.set_source_rgba(*rgb(NET_COL[name]), 0.95)
            ctx.move_to(px_(0), py_(max(0.6, acc[0])))
            for k in range(1, r + 1):
                ctx.line_to(px_(k * RECORD), py_(max(0.6, acc[k])))
            ctx.stroke()
            if r > 0:
                ctx.arc(px_(r * RECORD), py_(max(0.6, acc[r])), 6, 0, 2 * math.pi)
                ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s flips {sc.n_flip} final {sc.final}")
    frames = [0, 120, 200, 260, 320, 380, 440, 500, 560, 640, 720, 800, 900, 980, 1080, 1180, 1280, 1379]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (440, 980):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
