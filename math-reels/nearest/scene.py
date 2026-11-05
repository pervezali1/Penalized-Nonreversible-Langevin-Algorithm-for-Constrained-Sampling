"""The laziest algorithm beats the handwriting network. Dark theme of reels 31-56, maths in Computer Modern
(stage3d.tex), narrated by Kokoro (voicekit). Nearest neighbour keeps all 1,257 training pictures and copies the label
of the closest one: 533 of the 540 unseen digits right (98.7%) against the network's 522 (96.7%), and it wins on six
other random splits too; but it does 80,448 multiplications per digit read against the network's 2,368 (nnsim.py).
On screen: the new digit and the closest training picture, all 1,257 training pictures scanned and lit by closeness,
the two scores, and the work per digit. frame(f) -> uint8 1080x1920 RGB."""

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
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import paste  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg
FPS = 60
TAU = 2 * math.pi
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA, GOLD, CYAN = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6", "#F6C56E", "#38BDF8"
BG_TOP, BG_BOT = "#0E1226", "#05060C"
PANEL_BASE = "#161C33"

sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "nn_results.json")):
    import nnsim
    nnsim.main()
with open(os.path.join(HERE, "nn_results.json")) as fh:
    R = json.load(fh)
import nnsim  # noqa: E402
XTR, YTR, XTE, YTE = nnsim.split(0)
assert YTR.tolist() == R["ytr"]
SHOW = R["show"]
NTR, NTE = R["n_train"], R["n_test"]
DS = [np.array(d) for d in R["show_d"]]
NEAR = R["show_near"]
MIST = next(m for m in R["mistakes"] if m["i"] == SHOW[9])
OTH = R["others"]
Q3, Q9 = 3, 9                                   # the shown 3 (the hook) and the shown 9 (the mistake)
SAME9 = int(np.flatnonzero(YTR == 9)[np.argmin(DS[Q9][YTR == 9])])        # the training 9 closest to the shown 9


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert int(DS[Q9][SAME9]) == MIST["d_same"]
assert (NTR, R["right_net"], pc(R["acc_net"]), R["right_nn"], pc(R["acc_nn"]), len(R["mistakes"]), round(R["mults_nn"] / R["mults_net"]),
        sum(o["nearest"] > o["net"] for o in OTH), len(OTH), R["show_nn_read"][Q9], MIST["d"] < MIST["d_same"], NTR * 64, R["mults_net"]) == \
    (1257, 522, "96.7", 533, "98.7", 7, 34, 6, 6, 3, True, 80448, 2368), "script numbers changed"
SCRIPT = [
    "The laziest algorithm in machine learning just beat my neural network. And it doesn't learn anything at all.",
    "This is my handwriting network. It trained for three hundred steps, and reads ninety-six point seven percent of digits it has never seen.",
    "The lazy way: keep all one thousand two hundred and fifty-seven training pictures. For a new digit, find the one that looks most alike, pixel by pixel, and copy its answer.",
    "That's it. Five hundred and thirty-three of the five hundred and forty unseen digits right: ninety-eight point seven percent. And on six other random splits of the data, it won all six.",
    "It still makes seven mistakes. This nine is closer, pixel by pixel, to a three than to any nine it kept.",
    "And here's the catch. To read one digit, it compares it with every picture it kept: eighty thousand multiplications, thirty-four times the network's work. With a million pictures, sixty-four million, every single time.",
    "So always try the simple method first. And nearest neighbors are everywhere: AI search and recommendations use them every day, just not on pixels, on features a network learned.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 the lazy way, 3 the score, 4 a mistake, 5 the catch, 6 close
SEG_MIN = [6.5, 8.0, 10.5, 10.0, 7.0, 13.0, 10.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_SCAN = (word_t(2, "For a new digit"), word_t(2, "and copy its answer"))
T_SCORE = word_t(3, "Five hundred")
T_CYCLE = (SEG[3] + 0.25, T_SCORE + 5.0)
T_SCAN2 = (word_t(5, "it compares"), word_t(5, "eighty thousand"))
T_MULT = word_t(5, "eighty thousand")
T_MILLION = word_t(5, "With a million")

PANEL = 256
PL, PR, PY = (120, PANEL), (704, PANEL), 500       # the new digit (left) and the closest picture (right)
MOS_X, MOS_Y, THUMB, GAPX, ROWS_PER, BAND_GAP = 131, 852, 12, 1, 2, 6
COLS = 66
SCORE_Y = 1196
CARD = (70, 1430, 940, 250)

# the training pictures sorted by digit, two rows per digit: their place in the mosaic and the scan order
ORDER = np.argsort(YTR, kind="stable")
BOX = np.zeros((NTR, 2), int)
RANK = np.zeros(NTR, int)                        # scan position of each training picture
for pos, i in enumerate(ORDER):
    d = YTR[i]
    k = pos - int(np.sum(YTR < d))
    r, c = divmod(k, COLS)
    assert r < ROWS_PER
    BOX[i] = (MOS_X + c * (THUMB + GAPX), MOS_Y + d * (ROWS_PER * (THUMB + GAPX) + BAND_GAP) + r * (THUMB + GAPX))
    RANK[i] = pos
MOS_W = COLS * (THUMB + GAPX)
MOS_H = 10 * ROWS_PER * (THUMB + GAPX) + 9 * BAND_GAP


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("net", "lazy", "score", "mistake", "catch", "close")[k - 1]
    return "close"


def state(t):
    """(shown digit 0..9, scan progress 0..1 or None for no scan yet)."""
    md = mode(t)
    if md == "net":
        return Q3, None
    if md == "lazy":
        return Q3, (smooth(seg(t, *T_SCAN)) if t >= T_SCAN[0] else None)
    if md == "score":
        k = int(np.clip((t - T_CYCLE[0]) / ((T_CYCLE[1] - T_CYCLE[0]) / 10), 0, 9))
        return (Q3 if t < T_CYCLE[0] else k), 1.0
    if md == "mistake":
        return Q9, 1.0
    if md == "catch":
        return Q3, (seg(t, *T_SCAN2) if t >= T_SCAN2[0] else 1.0)
    return Q3, 1.0


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((0.5 * (T_SCAN[0] + T_SCAN[1])) * FPS), int((T_SCORE + 1.0) * FPS), int((T_MILLION + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(right_net=R["right_net"], right_nn=R["right_nn"], acc_net=R["acc_net"], acc_nn=R["acc_nn"], mistake=MIST,
                          others=OTH, mults=(R["mults_net"], R["mults_nn"]), segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self._build_mosaic()

    def _cues(self):
        S_ = SEG
        hook = [("The laziest algorithm", "black", 54, INK, 300, 1), ("just beat my neural network.", "dbold", 44, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("My handwriting network.", "black", 50, INK, 300, 1), (f"300 training steps: {pc(R['acc_net'])}%.", "dbold", 44, BLUE, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("The lazy way:", "black", 54, INK, 300, 1), ("find the closest picture. Copy it.", "dbold", 42, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("No training at all:", "black", 52, INK, 300, 1), (f"{pc(R['acc_nn'])}%.", "dbold", 50, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"It makes {len(R['mistakes'])} mistakes.", "black", 52, INK, 300, 1), ("This 9 is closest to a 3.", "dbold", 44, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The catch:", "black", 54, INK, 300, 1), (f"{round(R['mults_nn'] / R['mults_net'])}\u00d7 the work, for every digit.", "dbold", 44, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("Try the simple method first.", "black", 50, INK, 300, 1), ("Nearest neighbors still run AI search.", "dbold", 40, GREEN, 390, 0.95)], 10, 8, "pop"),
            (T_END - 0.3, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    def _build_bg(self):
        surf, ctx = layer()
        ctx.set_source_rgba(1, 1, 1, 0.045)
        ctx.set_line_width(1.0)
        for x in range(54, W, 54):
            ctx.move_to(x + 0.5, 0)
            ctx.line_to(x + 0.5, H)
        for y in range(54, H, 54):
            ctx.move_to(0, y + 0.5)
            ctx.line_to(W, y + 0.5)
        ctx.stroke()
        grgb, ga = surf_rgba(surf)
        c0 = np.array(rgb(BG_TOP), np.float32) ** 2.2
        c1 = np.array(rgb(BG_BOT), np.float32) ** 2.2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = np.clip(np.sqrt((xx - W / 2) ** 2 + (yy - 1000) ** 2) / 1100, 0, 1)[..., None]
        base = (c0 * (1 - u) + c1 * u) ** (1 / 2.2)
        base = base * (1 - ga)[..., None] + grgb
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        self.bg = np.ascontiguousarray(base * vig[..., None])

    def _build_mosaic(self):
        """The ink of every training picture at thumbnail size, and which picture owns each pixel of the mosaic."""
        self.ink = np.zeros((MOS_H, MOS_W), np.float32)
        self.owner = -np.ones((MOS_H, MOS_W), int)
        for i in range(NTR):
            x, y = BOX[i, 0] - MOS_X, BOX[i, 1] - MOS_Y
            big = np.kron(XTR[i].reshape(8, 8), np.ones((3, 3)))
            self.ink[y:y + THUMB, x:x + THUMB] = cv2.resize(big.astype(np.float32), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            self.owner[y:y + THUMB, x:x + THUMB] = i
        self.inside = self.owner >= 0

    # ------------------------------------------------------------ pieces

    def label(self, ctx, s, x, y, a, col=INK3, size=16, center=False):
        font(ctx, "bold", size)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        tr = 0.1 * size
        if center:
            x -= tracked_width(ctx, s, tr) / 2
        show_tracked(ctx, s, x, y, tr)

    def nearest_now(self, q, p):
        """The closest training picture among those scanned so far (scan progress p), or None."""
        d = DS[q]
        if p is None:
            return None
        if p >= 1:
            return NEAR[q]
        done = RANK < p * NTR
        if not done.any():
            return None
        cand = np.flatnonzero(done)
        return int(cand[np.argmin(d[cand])])

    def mosaic(self, img, q, p, a):
        """All 1,257 training pictures; scanned ones lit by how close they are to the new digit."""
        d = DS[q].astype(np.float64)
        base = np.array(rgb(PANEL_BASE), np.float32)
        col = np.tile(np.array(rgb(INK), np.float32) * 0.78, (NTR, 1))
        if p is not None:
            done = RANK < p * NTR if p < 1 else np.ones(NTR, bool)
            dmin, dmed = d.min(), np.median(d)
            near = (1 - np.clip((d - dmin) / (dmed - dmin), 0, 1)) ** 3
            lit = np.array(rgb(INK3), np.float32)[None] * 0.35 * (1 - near[:, None]) + np.array(rgb(GOLD), np.float32)[None] * near[:, None]
            col[done] = lit[done]
            best = self.nearest_now(q, p)
            if best is not None:
                ok = YTR[best] == YTE[SHOW[q]]
                col[best] = np.array(rgb(GREEN if ok else VERM), np.float32)
        pix = base[None, None] * (1 - self.ink[..., None]) + col[np.maximum(self.owner, 0)] * self.ink[..., None]
        sub = img[MOS_Y:MOS_Y + MOS_H, MOS_X:MOS_X + MOS_W]
        m = self.inside[..., None] * a
        sub[:] = sub * (1 - m) + pix * m

    def mosaic_marks(self, ctx, q, p, a, mode_now=None):
        font(ctx, "dbold", 17)
        for dgt in range(10):
            y = MOS_Y + dgt * (ROWS_PER * (THUMB + GAPX) + BAND_GAP) + 17
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(MOS_X - 26, y)
            ctx.show_text(str(dgt))
        self.label(ctx, f"ALL {NTR:,} TRAINING PICTURES, BY DIGIT", MOS_X - 26, MOS_Y - 16, a)
        if p is not None and 0 < p < 1:                 # the scan front
            k = min(NTR - 1, int(p * NTR))
            i = int(ORDER[k])
            ctx.set_source_rgba(1, 1, 1, 0.9 * a)
            ctx.set_line_width(1.6)
            ctx.rectangle(BOX[i, 0] - 2, BOX[i, 1] - 2, THUMB + 4, THUMB + 4)
            ctx.stroke()
        if mode_now == "mistake":                        # where the closest 9 was
            j = SAME9
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.set_line_width(2.0)
            ctx.new_sub_path()
            ctx.arc(BOX[j, 0] + THUMB / 2, BOX[j, 1] + THUMB / 2, THUMB * 1.05, 0, TAU)
            ctx.stroke()
            font(ctx, "dbold", 17)
            s = f"closest 9: {MIST['d_same']}"
            ctx.move_to(min(BOX[j, 0] + THUMB + 10, MOS_X + MOS_W - ctx.text_extents(s).x_advance), BOX[j, 1] - 8)
            ctx.show_text(s)
        best = self.nearest_now(q, p)
        if best is not None:
            ok = YTR[best] == YTE[SHOW[q]]
            ctx.set_source_rgba(*rgb(GREEN if ok else VERM), a)
            ctx.set_line_width(2.4)
            ctx.new_sub_path()
            ctx.arc(BOX[best, 0] + THUMB / 2, BOX[best, 1] + THUMB / 2, THUMB * 1.05, 0, TAU)
            ctx.stroke()

    def digit(self, ctx, x0, y0, s, img8, a, frame_col=INK3):
        c = s / 8
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        for p in range(64):
            v = float(img8[p])
            ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
            ctx.rectangle(x0 + (p % 8) * c, y0 + (p // 8) * c, c + 0.4, c + 0.4)
            ctx.fill()
        ctx.set_source_rgba(*rgb(frame_col), 0.8 * a)
        ctx.set_line_width(2.0)
        ctx.rectangle(x0, y0, s, s)
        ctx.stroke()

    def panels(self, ctx, t, q, p, a):
        md = mode(t)
        i = SHOW[q]
        true = int(YTE[i])
        self.label(ctx, "A NEW DIGIT (NEVER SEEN)", PL[0] + PANEL / 2, PY - 18, a, center=True)
        self.digit(ctx, PL[0], PY, PANEL, XTE[i], a)
        font(ctx, "dbold", 24)
        s = f"it's a {true}"
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(PL[0] + PANEL / 2 - ctx.text_extents(s).x_advance / 2, PY + PANEL + 36)
        ctx.show_text(s)
        cx = PR[0] + PANEL / 2
        if md == "net":
            self.label(ctx, "THE NETWORK READS", cx, PY - 18, a, center=True)
            ctx.set_source_rgba(*rgb(PANEL_BASE), a)
            ctx.rectangle(PR[0], PY, PANEL, PANEL)
            ctx.fill()
            font(ctx, "black", 170)
            s = str(R["show_net_read"][q])
            ctx.set_source_rgba(*rgb(BLUE), a)
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, PY + PANEL / 2 + 62)
            ctx.show_text(s)
            font(ctx, "dbold", 22)
            s = "64 \u2192 32 \u2192 10 neurons"
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, PY + PANEL + 36)
            ctx.show_text(s)
            return
        best = self.nearest_now(q, p)
        final = p is not None and p >= 1
        self.label(ctx, f"CLOSEST OF ALL {NTR:,}" if final else "CLOSEST SO FAR", cx, PY - 18, a, center=True)
        if best is None:
            ctx.set_source_rgba(*rgb(PANEL_BASE), a)
            ctx.rectangle(PR[0], PY, PANEL, PANEL)
            ctx.fill()
            font(ctx, "black", 120)
            ctx.set_source_rgba(*rgb(INK3), 0.6 * a)
            ctx.move_to(cx - ctx.text_extents("?").x_advance / 2, PY + PANEL / 2 + 44)
            ctx.show_text("?")
            return
        ok = YTR[best] == true
        col = GREEN if ok else VERM
        self.digit(ctx, PR[0], PY, PANEL, XTR[best], a, col if final else GOLD)
        font(ctx, "dbold", 24)
        s = f"a {YTR[best]}: answer {YTR[best]}" if final else f"a {YTR[best]}"
        ctx.set_source_rgba(*rgb(col if final else INK2), a)
        ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, PY + PANEL + 36)
        ctx.show_text(s)
        # the distance, between the two
        dd = int(DS[q][best])
        font(ctx, "dbold", 22)
        ctx.set_source_rgba(*rgb(INK3), a)
        s1 = "difference"
        ctx.move_to(540 - ctx.text_extents(s1).x_advance / 2, PY + PANEL / 2 - 30)
        ctx.show_text(s1)
        font(ctx, "black", 50)
        s2 = f"{dd:,}"
        ctx.set_source_rgba(*rgb(GOLD), a)
        ctx.move_to(540 - ctx.text_extents(s2).x_advance / 2, PY + PANEL / 2 + 26)
        ctx.show_text(s2)
        if final:
            font(ctx, "dbold", 19)
            ctx.set_source_rgba(*rgb(INK3), a)
            s3 = f"the smallest of {NTR:,}"
            ctx.move_to(540 - ctx.text_extents(s3).x_advance / 2, PY + PANEL / 2 + 62)
            ctx.show_text(s3)

    def scores(self, ctx, t, a):
        md = mode(t)
        if md == "catch":
            return self.costs(ctx, t, a)
        self.label(ctx, f"RIGHT ON THE SAME {NTE} UNSEEN DIGITS", 540, SCORE_Y, a, center=True)
        cols = [(300, "NEURAL NETWORK", BLUE, R["acc_net"], R["right_net"], "trained 300 steps"),
                (780, "NEAREST NEIGHBOR", GOLD, R["acc_nn"], R["right_nn"], "trained: never")]
        reveal = {"net": [1, 0], "lazy": [1, 0]}.get(md, [1, 1])
        if md == "score":
            reveal = [1, smooth(seg(t, T_SCORE, T_SCORE + 0.5))]
        for (cx, name, col, acc, right, note), r in zip(cols, reveal):
            self.label(ctx, name, cx, SCORE_Y + 42, a, col=col, size=18, center=True)
            if r <= 0:
                font(ctx, "black", 64)
                ctx.set_source_rgba(*rgb(INK3), 0.5 * a)
                ctx.move_to(cx - ctx.text_extents("?").x_advance / 2, SCORE_Y + 112)
                ctx.show_text("?")
                continue
            font(ctx, "black", 68)
            s = f"{pc(acc)}%"
            ctx.set_source_rgba(*rgb(col), r * a)
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, SCORE_Y + 114)
            ctx.show_text(s)
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), r * a)
            s = f"{right} of {NTE} right"
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, SCORE_Y + 150)
            ctx.show_text(s)
            ctx.set_source_rgba(*rgb(INK3), r * a)
            ctx.move_to(cx - ctx.text_extents(note).x_advance / 2, SCORE_Y + 180)
            ctx.show_text(note)

    def costs(self, ctx, t, a):
        self.label(ctx, "MULTIPLICATIONS TO READ ONE DIGIT", 540, SCORE_Y, a, center=True)
        g = smooth(seg(t, T_MULT - 0.4, T_MULT + 0.8))
        x0, wbar = 110, 860
        rows = [("neural network", BLUE, R["mults_net"], SCORE_Y + 36), ("nearest neighbor", GOLD, R["mults_nn"], SCORE_Y + 104)]
        for name, col, v, y in rows:
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x0, y + 6)
            ctx.show_text(name)
            font(ctx, "black", 30)
            s = f"{v:,}"
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x0 + wbar - ctx.text_extents(s).x_advance, y + 8)
            ctx.show_text(s)
            ctx.set_source_rgba(*rgb(PANEL_BASE), a)
            ctx.rectangle(x0, y + 18, wbar, 16)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), a)
            ctx.rectangle(x0, y + 18, wbar * g * v / R["mults_nn"], 16)
            ctx.fill()
        font(ctx, "dbold", 21)
        s = f"numbers it keeps: network {R['numbers_net']:,}, nearest neighbor {R['numbers_nn']:,}"
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(540 - ctx.text_extents(s).x_advance / 2, SCORE_Y + 190)
        ctx.show_text(s)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        q, p = state(t)
        md = mode(t)
        self.mosaic(img, q, None if md == "net" else p, a)
        surf, ctx = layer()
        self.panels(ctx, t, q, p, a)
        self.mosaic_marks(ctx, q, None if md == "net" else p, a, md)
        self.scores(ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close", "lazy"):
            paste(img, r"$\mathrm{difference} = \sum_{i=1}^{64}\,(x_i - x'_i)^2$", 32, 540, y + 90, GOLD, a, align="c")
            paste(img, r"$x_i,\ x'_i:\ \mathrm{the\ ink\ (0\ to\ 16)\ in\ pixel}\ i\ \mathrm{of\ the\ new\ digit\ and\ of\ a\ training\ picture}$", 19, 540, y + 182, INK2, a, align="c")
            paste(img, r"$\mathrm{answer} = \mathrm{the\ label\ of\ the\ training\ picture\ with\ the\ smallest\ difference}$", 21, 540, y + h - 18, INK, a, align="c")
        elif md == "net":
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$2{,}368\ \mathrm{weights,\ tuned\ by\ 300\ steps\ of\ training\ on\ the\ 1{,}257\ pictures}$", 22, 540, y + 162, BLUE, a, align="c")
        elif md == "score":
            lo_n, hi_n = min(o["net"] for o in OTH), max(o["net"] for o in OTH)
            lo_k, hi_k = min(o["nearest"] for o in OTH), max(o["nearest"] for o in OTH)
            paste(img, rf"$\mathrm{{6\ other\ random\ splits:\ nearest\ neighbor\ wins\ all\ 6}}$", 26, 540, y + 96, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{network}}\ {pc(lo_n)}\ \mathrm{{to}}\ {pc(hi_n)}\%;\ \ \mathrm{{nearest\ neighbor}}\ {pc(lo_k)}\ \mathrm{{to}}\ {pc(hi_k)}\%$", 24, 540, y + 156, INK2, a, align="c")
            paste(img, rf"$\mathrm{{this\ split:\ {R['right_net']}\ vs\ {R['right_nn']}\ right\ out\ of\ {NTE}}}$", 20, 540, y + h - 22, INK3, a, align="c")
        elif md == "mistake":
            paste(img, rf"$\mathrm{{difference\ to\ the\ closest\ 3:}}\ {MIST['d']}\qquad \mathrm{{to\ the\ closest\ 9:}}\ {MIST['d_same']}$", 24, 540, y + 100, VERM, a, align="c")
            paste(img, rf"$\mathrm{{(the\ network\ reads\ this\ one\ right:\ {MIST['net_read']})}}$", 22, 540, y + 162, INK2, a, align="c")
            paste(img, rf"$\mathrm{{{len(R['mistakes'])}\ mistakes\ out\ of\ {NTE};\ the\ network\ makes\ {NTE - R['right_net']}}}$", 20, 540, y + h - 22, INK3, a, align="c")
        else:
            paste(img, r"$1{,}257\ \mathrm{pictures} \times 64\ \mathrm{pixels} = 80{,}448\ \mathrm{multiplications}$", 28, 540, y + 96, GOLD, a, align="c")
            paste(img, r"$\mathrm{network:}\ 64 \times 32 + 32 \times 10 = 2{,}368,\ \mathrm{however\ much\ data}$", 23, 540, y + 152, BLUE, a, align="c")
            am = smooth(seg(t, T_MILLION - 0.2, T_MILLION + 0.4))
            paste(img, r"$1{,}000{,}000\ \mathrm{pictures} \times 64 = 64{,}000{,}000\ \mathrm{per\ digit}$", 24, 540, y + h - 26, VERM, a * am, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        self.draw_main(img, t, a)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "REAL HANDWRITTEN DIGITS   \u00b7   540 UNSEEN   \u00b7   ALL COMPUTED"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sc = Scene()
    print("facts", {k: v for k, v in sc.facts.items() if k != "others"})
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        ims.append(cv2.resize(sc.frame(fr)[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]))
    for fr in frames:
        cv2.imwrite(f"{out}/f{fr:04d}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
