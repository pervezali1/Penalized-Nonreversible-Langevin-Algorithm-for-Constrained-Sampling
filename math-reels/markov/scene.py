"""How long is a game of snakes and ladders? One matrix answers it exactly (an absorbing Markov chain). Dark theme of
reels 31-63, maths in Computer Modern (stage3d.tex), narrated by Kokoro (voicekit). A board made up for the reel, 36
squares, 3 ladders, 4 snakes; Q = the chance of going from each square to each square in one turn; the expected
length of a game is the first row of (I - Q)^-1 added up: 16.25 turns. 100,000 simulated games average 16.24 and their
lengths match the matrix's prediction; without the snakes and ladders, 10.76; the snake from 32 to 6 adds 4.65 turns
(boardsim.py). On screen: the board with a sample game, the matrix Q, the expected visits to each square, the
histogram of game lengths against the matrix's prediction. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

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
if not os.path.exists(os.path.join(HERE, "board_results.json")):
    import boardsim
    boardsim.main()
with open(os.path.join(HERE, "board_results.json")) as fh:
    R = json.load(fh)
N = R["n"]
LAD = {int(k): v for k, v in R["ladders"].items()}
SNK = {int(k): v for k, v in R["snakes"].items()}
Q = np.array(R["Q"])
VIS = np.array(R["visits"])
DIST = np.array(R["dist"])
HIST = np.array(R["hist"])
SAMPLE = R["sample"]
WITHOUT = R["without"]

assert (f"{R['expected']:.2f}", f"{R['sim_mean']:.2f}", R["sim_longest"], f"{R['plain']:.2f}", f"{-WITHOUT['32-6']:.2f}", R["games"], len(LAD), len(SNK), R["mode_turns"]) == \
    ("16.25", "16.24", 142, "10.76", "4.65", 100000, 3, 4, 9), "script numbers changed"
SCRIPT = [
    "One of these hundred thousand games of snakes and ladders lasted a hundred and forty-two turns. How long is a typical game? You don't have to play them. One matrix answers it exactly.",
    "Here's a board I made up: thirty-six squares, three ladders, four snakes. Each turn you roll a die, so from any square there are six places to land, each with a chance of one in six.",
    "Write those chances as a matrix, Q: one row for where you are, one column for where you land. Ladders and snakes just move a chance to a different column.",
    "Now invert one matrix: I minus Q. Its first row counts how often you'll visit each square, and adding it up gives the expected length of a game: sixteen point two five turns.",
    "A hundred thousand simulated games average sixteen point two four, and their lengths match the matrix's prediction. Without the snakes and ladders, a game would take ten point seven six.",
    "It even says which snake hurts most: the one from thirty-two down to six adds four point six five turns to the average game.",
    "That's an absorbing Markov chain: one inverse matrix, for anything that wanders until it stops.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the board, 2 the matrix, 3 the inverse, 4 the check, 5 the worst snake, 6 close
SEG_MIN = [9.5, 11.5, 10.0, 11.5, 11.0, 8.5, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_ROLL = word_t(1, "Each turn")
T_MAT = word_t(2, "Write those")
T_JUMP = word_t(2, "Ladders and snakes")
T_INV = word_t(3, "Its first row")
T_SUM = word_t(3, "adding it up")
T_SIM = word_t(4, "A hundred thousand")
T_PLAIN = word_t(4, "Without the snakes")
T_SNAKE = word_t(5, "the one from")
STEP_DT = 0.55

SQ = 112
BX0, BY0 = (W - 6 * SQ) // 2, 470
PLOT = (140, 1196, 820, 160)
CARD = (70, 1430, 940, 250)


def square_xy(s):
    """Centre of square s (1..36), numbered back and forth from the bottom left; 0 is just left of square 1."""
    if s <= 0:
        return BX0 - SQ * 0.45, BY0 + 5.5 * SQ
    s = min(s, N)
    r, c = divmod(s - 1, 6)
    if r % 2 == 1:
        c = 5 - c
    return BX0 + (c + 0.5) * SQ, BY0 + (5 - r + 0.5) * SQ


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("board", "matrix", "inverse", "check", "snake", "close")[k - 1]
    return "close"


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_ROLL + 2.0) * FPS), int((T_SUM + 1.0) * FPS), int((T_SNAKE + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(expected=R["expected"], exact=R["exact"], sim_mean=R["sim_mean"], longest=R["sim_longest"], plain=R["plain"], without=WITHOUT,
                          segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("How long is a game?", "black", 56, INK, 300, 1), ("One matrix knows exactly.", "dbold", 48, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("36 squares, 3 ladders, 4 snakes.", "black", 46, INK, 300, 1), ("6 places to land, 1/6 each.", "dbold", 46, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Write the chances as a matrix.", "black", 46, INK, 300, 1), ("Q: from → to.", "dbold", 48, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Invert I − Q.", "black", 56, INK, 300, 1), (f"Expected: {R['expected']:.2f} turns.", "dbold", 48, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"{R['games']:,} games: {R['sim_mean']:.2f}.", "black", 50, INK, 300, 1), (f"No snakes or ladders: {R['plain']:.2f}.", "dbold", 44, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The worst snake: 32 → 6.", "black", 50, INK, 300, 1), (f"+{-WITHOUT['32-6']:.2f} turns on average.", "dbold", 48, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("An absorbing Markov chain.", "black", 50, INK, 300, 1), ("One inverse matrix.", "dbold", 48, GOLD, 390, 0.95)], 10, 8, "pop"),
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

    # ------------------------------------------------------------ pieces

    @staticmethod
    def label(ctx, s, x, y, a, col=INK3, size=16, center=False):
        font(ctx, "bold", size)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        tr = 0.1 * size
        if center:
            x -= tracked_width(ctx, s, tr) / 2
        show_tracked(ctx, s, x, y, tr)

    def token_pos(self, t):
        """The sample game's token: (x, y, square, turn) at time t in the board segment, or None."""
        md = mode(t)
        if md != "board" or t < T_ROLL:
            return None
        path = SAMPLE["path"]
        k = (t - T_ROLL) / STEP_DT
        i = int(min(k, len(path) - 1))
        f = smooth(min(1.0, (k - i) * 1.6)) if i < len(path) - 1 else 0.0
        x0, y0 = square_xy(path[i])
        x1, y1 = square_xy(path[min(i + 1, len(path) - 1)])
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f - 26 * math.sin(math.pi * f), path[i], i

    def board(self, ctx, t, a, heat=None, hl_snake=None):
        base = np.array(rgb(PANEL_BASE))
        for s in range(1, N + 1):
            cx, cy = square_xy(s)
            col = base * (1.0 if (s // 1 + (s - 1) // 6) % 2 else 1.25)
            if heat is not None and s < N:
                v = min(1.0, heat[s] / 1.05)
                col = col * (1 - v) + np.array(rgb(GOLD)) * v * 0.85
            ctx.set_source_rgba(*np.clip(col, 0, 1), a)
            ctx.rectangle(cx - SQ / 2 + 1, cy - SQ / 2 + 1, SQ - 2, SQ - 2)
            ctx.fill()
        # finish
        cx, cy = square_xy(N)
        font(ctx, "black", 22)
        ctx.set_source_rgba(*rgb(GREEN), a)
        ctx.move_to(cx - 26, cy + 10)
        ctx.show_text("END")
        # ladders
        for f_, to in LAD.items():
            (x0, y0), (x1, y1) = square_xy(f_), square_xy(to)
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            nx, ny = -dy / L * 9, dx / L * 9
            ctx.set_source_rgba(*rgb(GREEN), 0.95 * a)
            ctx.set_line_width(4)
            for sgn in (-1, 1):
                ctx.move_to(x0 + sgn * nx, y0 + sgn * ny)
                ctx.line_to(x1 + sgn * nx, y1 + sgn * ny)
            nr = int(L / 26)
            for k in range(1, nr):
                u = k / nr
                ctx.move_to(x0 + dx * u - nx, y0 + dy * u - ny)
                ctx.line_to(x0 + dx * u + nx, y0 + dy * u + ny)
            ctx.stroke()
        # snakes
        for f_, to in SNK.items():
            (x0, y0), (x1, y1) = square_xy(f_), square_xy(to)
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            nx, ny = -dy / L, dx / L
            hot = hl_snake is not None and (f_, to) == hl_snake
            ctx.set_source_rgba(*rgb(VERM if (hl_snake is None or hot) else INK3), (0.95 if hl_snake is None or hot else 0.5) * a)
            ctx.set_line_width(7 if hot else 5)
            pts = [(x0 + dx * u + nx * 16 * math.sin(u * TAU * max(1.5, L / 170)), y0 + dy * u + ny * 16 * math.sin(u * TAU * max(1.5, L / 170))) for u in np.linspace(0, 1, 60)]
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.stroke()
            ctx.arc(x0, y0, 9 if hot else 7, 0, TAU)
            ctx.fill()
        # numbers last, outlined, so the ladders and snakes never hide them
        for s in range(1, N + 1):
            cx, cy = square_xy(s)
            texts = [(str(s), "dbold", 18, INK3, cx - SQ / 2 + 7, cy - SQ / 2 + 22, False)]
            if heat is not None and s < N and heat[s] > 0.005:
                texts.append((f"{heat[s]:.2f}", "dbold", 17, INK, cx + SQ / 2 - 8, cy + SQ / 2 - 10, True))
            for sv, fw, fs, col, tx, ty, right in texts:
                font(ctx, fw, fs)
                if right:
                    tx -= ctx.text_extents(sv).x_advance
                ctx.move_to(tx, ty)
                ctx.text_path(sv)
                ctx.set_source_rgba(0.04, 0.05, 0.09, 0.85 * a)
                ctx.set_line_width(4 if right else 2.5)
                ctx.stroke_preserve()
                ctx.set_source_rgba(*rgb(col), a)
                ctx.fill()
        self.label(ctx, "A BOARD MADE UP FOR THIS REEL  ·  GREEN: LADDERS UP  ·  RED: SNAKES DOWN", 70, BY0 - 16, a)

    def board_layer(self, ctx, t, a):
        md = mode(t)
        heat = VIS if md in ("inverse", "check", "hook", "close") and (md != "inverse" or t >= T_INV - 0.2) else None
        hl = (32, 6) if md == "snake" else None
        self.board(ctx, t, a, heat, hl)
        tp = self.token_pos(t)
        if tp is not None:
            x, y, s, i = tp
            # the six squares the current roll could reach
            if i < len(SAMPLE["path"]) - 1:
                for d in range(1, 7):
                    tgt = s + d
                    if tgt > N:
                        continue
                    cx, cy = square_xy(min(tgt, N))
                    ctx.set_source_rgba(*rgb(CYAN), 0.5 * a)
                    ctx.set_line_width(2.5)
                    ctx.rectangle(cx - SQ / 2 + 4, cy - SQ / 2 + 4, SQ - 8, SQ - 8)
                    ctx.stroke()
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.arc(x, y, 17, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(0, 0, 0, 0.6 * a)
            ctx.set_line_width(2)
            ctx.arc(x, y, 17, 0, TAU)
            ctx.stroke()
            font(ctx, "dbold", 20)
            ctx.set_source_rgba(*rgb(INK2), a)
            s_ = f"sample game: turn {min(i + 1, len(SAMPLE['rolls']))}, rolled {SAMPLE['rolls'][min(i, len(SAMPLE['rolls']) - 1)]}"
            ctx.move_to(70, BY0 + 6 * SQ + 32)
            ctx.show_text(s_)
        if md == "snake":
            font(ctx, "black", 30)
            ctx.set_source_rgba(*rgb(VERM), a)
            x, y = square_xy(32)
            s_ = f"+{-WITHOUT['32-6']:.2f} turns"
            # left of the snake's head, so the snake does not cross it
            ctx.move_to(x - 24 - ctx.text_extents(s_).x_advance, y + 12)
            ctx.show_text(s_)

    def matrix_view(self, img, ctx, t, a):
        """Q as a 36 x 36 grid of cells (rows: from, columns: to)."""
        cell = 18
        x0, y0 = (W - 36 * cell) // 2 + 20, 500
        hl_row = int((t - T_MAT) / 0.35) % 36 if t >= T_MAT else 0
        jumps_on = smooth(seg(t, T_JUMP - 0.2, T_JUMP + 0.5))
        for i in range(36):
            for j in range(36):
                v = Q[i, j]
                if v <= 0:
                    col, al = rgb(PANEL_BASE), 0.8
                else:
                    is_jump = (j in LAD.values() or j in SNK.values()) and any(i < k <= i + 6 for k in list(LAD) + list(SNK) if (LAD.get(k) == j or SNK.get(k) == j))
                    col = rgb(VERM if is_jump and j in SNK.values() and jumps_on > 0.5 else (GREEN if is_jump and jumps_on > 0.5 else CYAN))
                    al = 0.4 + 0.6 * min(1, v * 6)
                ctx.set_source_rgba(*col, al * a)
                ctx.rectangle(x0 + j * cell + 1, y0 + i * cell + 1, cell - 2, cell - 2)
                ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.9 * a)
        ctx.set_line_width(2)
        ctx.rectangle(x0, y0 + hl_row * cell, 36 * cell, cell)
        ctx.stroke()
        font(ctx, "dbold", 18)
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(x0, y0 - 14)
        ctx.show_text("to square 0 → 35")
        ctx.save()
        ctx.translate(x0 - 14, y0 + 36 * cell)
        ctx.rotate(-math.pi / 2)
        ctx.move_to(0, 0)
        ctx.show_text("from square 0 → 35")
        ctx.restore()
        row = Q[hl_row]
        targets = [j for j in range(36) if row[j] > 0]
        font(ctx, "dbold", 20)
        ctx.set_source_rgba(*rgb(INK), a)
        s = f"row {hl_row}: " + ", ".join(f"{j}" for j in targets) + f"  (each 1/6; {6 - int(round(row.sum() * 6))} of 6 reach the end)"
        ctx.move_to(x0, y0 + 36 * cell + 34)
        ctx.show_text(s)

    def plot(self, ctx, t, a):
        md = mode(t)
        x0, y0, w, h = PLOT
        kmax = 60
        top = max(HIST[:kmax].max(), DIST[:kmax].max()) * 1.1

        def gx(k):
            return x0 + (k - 0.5) / kmax * w

        def gy(v):
            return y0 + h - v / top * h
        ctx.set_source_rgba(*rgb(INK3), 0.8 * a)
        ctx.set_line_width(1.2)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        font(ctx, "dbold", 16)
        ctx.set_source_rgba(*rgb(INK2), a)
        for k in (1, 10, 20, 30, 40, 50, 60):
            s = str(k)
            ctx.move_to(gx(k) - ctx.text_extents(s).x_advance / 2, y0 + h + 20)
            ctx.show_text(s)
        show_sim = md in ("hook", "check", "snake", "close")
        show_m = md in ("check", "snake", "close", "hook") and (md != "check" or t >= T_SIM + 1.2)
        if show_sim:
            gs = smooth(seg(t, T_SIM - 0.2, T_SIM + 1.0)) if md == "check" else 1.0
            bw = w / kmax
            ctx.set_source_rgba(*rgb(BLUE), 0.85 * a)
            for k in range(1, kmax + 1):
                v = HIST[k - 1] * gs
                ctx.rectangle(gx(k) - bw / 2 + 1, gy(v), bw - 2, y0 + h - gy(v))
            ctx.fill()
        if show_m:
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.set_line_width(2.6)
            for k in range(1, kmax + 1):
                (ctx.move_to if k == 1 else ctx.line_to)(gx(k), gy(DIST[k - 1]))
            ctx.stroke()
            e = R["expected"]
            ctx.set_source_rgba(*rgb(GREEN), a)
            ctx.set_line_width(2.4)
            ctx.move_to(gx(e), y0 - 6)
            ctx.line_to(gx(e), y0 + h)
            ctx.stroke()
        if md in ("check",) and t >= T_PLAIN - 0.2 or md in ("snake",):
            e = R["plain"]
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.set_dash([6, 6])
            ctx.move_to(gx(e), y0 - 6)
            ctx.line_to(gx(e), y0 + h)
            ctx.stroke()
            ctx.set_dash([])
        self.label(ctx, f"LENGTH OF A GAME, IN TURNS ({R['games']:,} SIMULATED GAMES; THE LONGEST: {R['sim_longest']})", x0 - 70, y0 - 30, a)
        font(ctx, "dbold", 17)
        xx = x0 + w - 330
        show_plain = md in ("check",) and t >= T_PLAIN - 0.2 or md in ("snake",)
        for s, col, on, mark in ((f"simulated games", BLUE, show_sim, "box"), ("the matrix's prediction", GOLD, show_m, "line"),
                                 (f"expected: {R['expected']:.2f}", GREEN, show_m, "bar"),
                                 (f"no snakes or ladders: {R['plain']:.2f}", INK2, show_plain, "dash")):
            if not on:
                continue
            ctx.set_source_rgba(*rgb(col), a)
            yy = y0 + 6
            if mark == "box":
                ctx.rectangle(xx, yy - 12, 12, 12)
                ctx.fill()
            elif mark == "line":
                ctx.set_line_width(2.6)
                ctx.move_to(xx - 1, yy - 6)
                ctx.line_to(xx + 14, yy - 6)
                ctx.stroke()
            else:
                ctx.set_line_width(2.4)
                if mark == "dash":
                    ctx.set_dash([4, 3])
                ctx.move_to(xx + 6, yy - 15)
                ctx.line_to(xx + 6, yy + 2)
                ctx.stroke()
                ctx.set_dash([])
            ctx.move_to(xx + 22, yy)
            ctx.show_text(s)
            y0 += 24

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        surf, ctx = layer()
        if md == "matrix":
            self.matrix_view(img, ctx, t, a)
        else:
            self.board_layer(ctx, t, a)
        if md in ("hook", "check", "snake", "close"):
            self.plot(ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        num, den = R["exact"].split("/")
        if md in ("hook", "close", "inverse"):
            paste(img, r"$t = (I - Q)^{-1}\,\mathbf{1},\qquad t_0 = 16.2469\ldots\ \mathrm{turns}$", 32, 540, y + 96, GREEN, a, align="c")
            paste(img, r"$\mathrm{row\ 0\ of}\ (I - Q)^{-1}:\ \mathrm{expected\ visits\ to\ each\ square}$", 22, 540, y + 158, INK2, a, align="c")
            paste(img, rf"$\mathrm{{exactly}}\ \ {num}\,/\,{den}$", 21, 540, y + h - 20, INK3, a, align="c")
        elif md == "board":
            paste(img, r"$\mathrm{start\ off\ the\ board;\ roll\ 1\ to\ 6;\ ladders\ up,\ snakes\ down}$", 26, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{reach\ or\ pass\ 36\ and\ the\ game\ is\ over}$", 24, 540, y + 162, CYAN, a, align="c")
        elif md == "matrix":
            paste(img, r"$Q_{ij} = P(\mathrm{from\ square}\ i\ \mathrm{to\ square}\ j\ \mathrm{in\ one\ turn})$", 30, 540, y + 100, GOLD, a, align="c")
            paste(img, r"$\mathrm{a\ ladder\ or\ snake\ moves\ that\ 1/6\ to\ the\ column\ of\ its\ other\ end}$", 22, 540, y + 162, INK2, a, align="c")
        elif md == "check":
            paste(img, rf"$\mathrm{{{R['games']:,}\ games:\ mean}}\ {R['sim_mean']:.2f},\ \ \mathrm{{matrix}}\ {R['expected']:.2f}$".replace(",", "{,}"), 28, 540, y + 96, BLUE, a, align="c")
            paste(img, rf"$\mathrm{{most\ likely\ length}}\ {R['mode_turns']},\ \ \mathrm{{median}}\ {R['median_turns']},\ \ \mathrm{{mean}}\ {R['expected']:.2f}:\ \mathrm{{long\ games\ pull\ it\ up}}$", 21, 540, y + 156, INK2, a, align="c")
            paste(img, rf"$\mathrm{{no\ snakes\ or\ ladders:}}\ {R['plain']:.2f}\ \mathrm{{turns}}$", 21, 540, y + h - 20, INK3, a, align="c")
        else:
            paste(img, r"$\mathrm{remove\ one\ snake\ or\ ladder,\ invert\ again,\ compare}$", 26, 540, y + 96, INK, a, align="c")
            items = [("32\\to6", WITHOUT["32-6"]), ("34\\to23", WITHOUT["34-23"]), ("27\\to10", WITHOUT["27-10"])]
            s = ",\\ \\ ".join(f"{k}:\\ {-v:+.2f}" for k, v in items)
            paste(img, rf"$\mathrm{{snakes:}}\ {s}\ \mathrm{{turns}}$", 21, 540, y + 158, VERM, a, align="c")
            lad = ",\\ \\ ".join(f"{k}:\\ {-v:+.2f}" for k, v in [("20\\to29", WITHOUT["20-29"]), ("8\\to26", WITHOUT["8-26"]), ("3\\to16", WITHOUT["3-16"])])
            paste(img, rf"$\mathrm{{ladders:}}\ {lad}\ \mathrm{{turns}}$", 21, 540, y + h - 20, GREEN, a, align="c")

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
        lab = f"AN ABSORBING MARKOV CHAIN   ·   {R['games']:,} SIMULATED GAMES   ·   ALL COMPUTED"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sc = Scene()
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)[..., ::-1]
        cv2.imwrite(f"{out}/f{fr:04d}.png", im)
        ims.append(cv2.resize(im, (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
