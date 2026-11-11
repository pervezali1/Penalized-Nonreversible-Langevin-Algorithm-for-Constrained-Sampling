"""Build the Euler Sunday carousel slides as HTML (rendered to JPEG by render.js).

Every number and picture comes from solutions.py, so nothing on a slide is typed
in by hand that the code has not checked.

    python3 make_slides.py        # writes build/html/*.html + build/manifest.json
    node render.js                # writes images/pNN_sK.jpg (1080 x 1350)
"""
import html
import json
import math
import re
from pathlib import Path

import solutions as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "html"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
INK_ON_COLOR = "#0e1013"


def fmt(n):
    return f"{n:,}"


# --------------------------------------------------------------------------- page
CSS = """
:root{--bg:#121417;--panel:#1b1e24;--panel2:#0c0e11;--line:#2b2f37;--text:#f2f0eb;
--text2:#bfc2c9;--muted:#858a94;--blue:#3987e5;--orange:#d95926;--aqua:#199e70;--violet:#9085e9;}
*{box-sizing:border-box;font-variant-ligatures:none;font-feature-settings:'liga' 0,'calt' 0}
html,body{margin:0;width:1080px;height:1350px;background:var(--bg);overflow:hidden}
body{font-family:'Inter',sans-serif;color:var(--text);position:relative;
background:radial-gradient(900px 700px at -5% -5%,rgba(57,135,229,.13),transparent 65%),
radial-gradient(800px 600px at 105% 105%,rgba(217,89,38,.08),transparent 60%),var(--bg)}
.frame{position:absolute;inset:0;padding:60px 76px 54px;display:flex;flex-direction:column}
.top{display:flex;justify-content:space-between;align-items:center;font:600 23px 'Inter';
letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.top b{color:var(--text);font-weight:700}
.top .dot{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--blue);margin-right:14px;vertical-align:1px}
.kicker{font:700 26px 'Inter';letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-top:46px}
h1{font:800 66px/1.08 'Inter Display';letter-spacing:-.022em;margin:14px 0 0}
h1.big{font-size:76px}
.sub{font:400 33px/1.42 'Inter';color:var(--text2);margin-top:22px}
.sub b,.body b{color:var(--text);font-weight:650}
.body{font:400 33px/1.45 'Inter';color:var(--text2)}
.stage{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:0}
.foot{display:flex;justify-content:space-between;align-items:center;font:600 23px 'Inter';
letter-spacing:.14em;text-transform:uppercase;color:var(--muted);padding-top:18px}
.foot .swipe{color:var(--text2)}
.code{background:var(--panel2);border:1.5px solid var(--line);border-radius:22px;padding:34px 40px;
font:400 29px/1.58 'JetBrains Mono';white-space:pre;color:#e6e3dc;width:100%}
.code .k{color:#8fb9f2}.code .n{color:#f0a27e}.code .c{color:#7d8390;font-style:italic}.code .s{color:#7fd1ae}.code .f{color:#e9c46a}
.card{background:var(--panel);border:1.5px solid var(--line);border-radius:22px;padding:30px 36px;width:100%}
.mono{font-family:'JetBrains Mono'}
.eq{font:500 36px/1.6 'JetBrains Mono';color:var(--text);white-space:pre-wrap}
.row{display:flex;gap:22px;width:100%}
.stat{flex:1;background:var(--panel);border:1.5px solid var(--line);border-radius:22px;padding:26px 30px}
.stat .v{font:800 54px/1.1 'Inter Display';letter-spacing:-.01em;color:var(--text)}
.stat .l{font:500 25px/1.35 'Inter';color:var(--muted);margin-top:8px}
.legend{display:flex;gap:34px;justify-content:center;font:500 27px 'Inter';color:var(--text2);margin-top:26px}
.legend i{display:inline-block;width:26px;height:26px;border-radius:7px;margin-right:12px;vertical-align:-5px}
.answer{font:800 150px/1 'Inter Display';letter-spacing:-.03em;color:var(--text);margin:8px 0 0}
.answer-u{height:10px;width:220px;border-radius:5px;background:var(--blue);margin:26px 0 0}
.turn{border:2px solid var(--blue);border-radius:24px;padding:30px 36px;background:rgba(57,135,229,.08);width:100%}
.turn .t{font:800 26px 'Inter';letter-spacing:.14em;color:var(--blue);text-transform:uppercase}
.turn .q{font:600 35px/1.4 'Inter';color:var(--text);margin-top:12px}
.turn .h{font:400 27px/1.4 'Inter';color:var(--text2);margin-top:12px}
.note{font:400 27px/1.45 'Inter';color:var(--muted)}
.callout{border-left:6px solid var(--orange);padding:6px 0 6px 28px;font:400 31px/1.45 'Inter';color:var(--text2)}
.callout b{color:var(--text)}
"""

KW = r"\b(for|in|if|elif|else|while|def|return|print|import|from|and|or|not|del|range|sum|len|lambda)\b"


def hl(code):
    """Tiny Python syntax highlighter -> HTML."""
    out = []
    for line in code.strip("\n").split("\n"):
        m = re.search(r"#.*$", line)
        body, comment = (line[: m.start()], m.group(0)) if m else (line, "")
        body = html.escape(body)
        body = re.sub(r"(&#x27;[^&]*&#x27;|\"[^\"]*\")", r'<span class="s">\1</span>', body)
        body = re.sub(KW, r'<span class="k">\1</span>', body)
        body = re.sub(r"(?<![\w#])(\d[\d_]*)", r'<span class="n">\1</span>', body)
        if comment:
            body += f'<span class="c">{html.escape(comment)}</span>'
        out.append(body)
    return "\n".join(out)


def page(num, k, n_slides, title_text, inner):
    swipe = '<span class="swipe">swipe &rarr;</span>' if k < n_slides else "<span>see you next sunday</span>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title_text)}</title>
<style>{CSS}</style></head><body><div class="frame">
<div class="top"><span><span class="dot"></span><b>Euler Sunday</b> &nbsp;·&nbsp; Project Euler #{num}</span><span>{k} / {n_slides}</span></div>
{inner}
<div class="foot"><span>Math in Motion</span>{swipe}</div>
</div></body></html>"""


def svg(w, h, body, extra=""):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" {extra}>{body}</svg>'


def t(x, y, s, size=28, fill="#f2f0eb", weight=600, anchor="middle", family="Inter", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="central" {extra}>{html.escape(str(s))}</text>')


# ======================================================================= PROBLEM 1
def p1():
    num, slides = 1, []
    # --- slide 1: 1..100 grid
    cell, gap = 70, 7
    body = []
    for i in range(100):
        v = i + 1
        x, y = (i % 10) * (cell + gap), (i // 10) * (cell + gap)
        if v % 15 == 0:
            c = AQUA
        elif v % 3 == 0:
            c = BLUE
        elif v % 5 == 0:
            c = ORANGE
        else:
            c = None
        body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="10" fill="{c or "#1b1e24"}"/>')
        body.append(t(x + cell / 2, y + cell / 2 + 1, v, 28, INK_ON_COLOR if c else "#6f7580", 650 if c else 500))
    side = 10 * cell + 9 * gap
    grid = svg(side, side, "".join(body))
    slides.append(("Multiples of 3 or 5", f"""
<div class="kicker">Problem 1 · Multiples of 3 or 5</div>
<h1>Add up every multiple of 3 or 5 below 1000.</h1>
<div class="sub">Below 10 it's 3 + 5 + 6 + 9 = <b>23</b>. Here are the first 100:</div>
<div class="stage">{grid}
<div class="legend"><span><i style="background:{BLUE}"></i>multiple of 3</span><span><i style="background:{ORANGE}"></i>multiple of 5</span><span><i style="background:{AQUA}"></i>both</span></div></div>"""))

    # --- slide 2: the loop
    code = """y = []
for i in range(1, 1000):
    if i % 3 == 0:
        y.append(i)
    elif i % 5 == 0:
        y.append(i)

print(sum(y))"""
    slides.append(("The loop", f"""
<div class="kicker">Way 1 · The loop</div>
<h1>Check all 999 numbers. Keep the hits.</h1>
<div class="stage" style="gap:40px">
<div class="code">{hl(code)}</div>
<div class="callout">The <b>elif</b> matters. 15, 30, 45, … are multiples of <b>both</b> 3 and 5.
The elif adds each of them once, not twice. Remember that: it comes back in the shortcut.</div>
</div>
<div class="row" style="margin-bottom:6px"><div class="stat"><div class="v">999</div><div class="l">numbers checked</div></div>
<div class="stat"><div class="v">{len([i for i in range(1, 1000) if i % 3 == 0 or i % 5 == 0])}</div><div class="l">of them are hits</div></div></div>"""))

    # --- slide 3: Gauss staircase
    n, u = 8, 62
    body = []
    for r in range(n):              # row r has r+1 blue cells then n-r dim cells
        for c in range(n + 1):
            x, y = c * u, r * u
            filled = c <= r
            body.append(f'<rect x="{x + 3}" y="{y + 3}" width="{u - 6}" height="{u - 6}" rx="8" '
                        f'fill="{BLUE if filled else "#262a33"}"/>')
    stair = svg((n + 1) * u, n * u, "".join(body)) + (
        f'<div class="note" style="margin-top:18px">{n} rows × {n + 1} columns = {n * (n + 1)} squares, and the blue staircase 1 + … + {n} is half: {n * (n + 1) // 2}</div>')
    S3, S5, S15 = S.p1_S(3), S.p1_S(5), S.p1_S(15)
    slides.append(("Gauss's trick", f"""
<div class="kicker">Way 2 · The shortcut, part 1</div>
<h1>1 + 2 + … + n = n(n + 1) ÷ 2</h1>
<div class="sub">Two copies of the staircase fill a rectangle, so one staircase is half of it.</div>
<div class="stage">{stair}</div>
<div class="card eq" style="font-size:28px">3 + 6 + … + 999 = 3 × (1 + 2 + … + 333)
                = 3 × 333 × 334 ÷ 2 = {fmt(S3)}</div>"""))

    # --- slide 4: inclusion-exclusion Venn
    vb = []
    R, cx1, cx2, cy = 250, 330, 600, 300
    vb.append(f'<circle cx="{cx1}" cy="{cy}" r="{R}" fill="{BLUE}" fill-opacity=".22" stroke="{BLUE}" stroke-width="4"/>')
    vb.append(f'<circle cx="{cx2}" cy="{cy}" r="{R}" fill="{ORANGE}" fill-opacity=".22" stroke="{ORANGE}" stroke-width="4"/>')
    vb.append(t(cx1 - 105, cy - 40, "multiples of 3", 27, "#f2f0eb", 650))
    vb.append(t(cx1 - 105, cy + 4, "333 numbers", 25, "#bfc2c9", 500))
    vb.append(t(cx1 - 105, cy + 52, fmt(S3), 36, "#f2f0eb", 700, family="JetBrains Mono"))
    vb.append(t(cx2 + 105, cy - 40, "multiples of 5", 27, "#f2f0eb", 650))
    vb.append(t(cx2 + 105, cy + 4, "199 numbers", 25, "#bfc2c9", 500))
    vb.append(t(cx2 + 105, cy + 52, fmt(S5), 36, "#f2f0eb", 700, family="JetBrains Mono"))
    mid = (cx1 + cx2) / 2
    vb.append(t(mid, cy - 40, "of 15", 26, "#f2f0eb", 650))
    vb.append(t(mid, cy + 4, "66 numbers", 23, "#bfc2c9", 500))
    vb.append(t(mid, cy + 52, fmt(S15), 30, "#f2f0eb", 700, family="JetBrains Mono"))
    venn = svg(930, 600, "".join(vb))
    slides.append(("Count the overlap once", f"""
<div class="kicker">Way 2 · The shortcut, part 2</div>
<h1>Add both circles. Then take the overlap out once.</h1>
<div class="sub">Multiples of 15 sit in both circles, so adding the two sums counts them twice. Same job as the elif.</div>
<div class="stage">{venn}</div>
<div class="card eq" style="text-align:center">{fmt(S3)} + {fmt(S5)} − {fmt(S15)}</div>"""))

    # --- slide 5: answer + your turn
    big = S.p1_formula(10**6)
    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(S.p1_formula())}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px">The loop checks 999 numbers. The formula does <b>3 short calculations</b>, whatever the limit.
Below one million: <b>{fmt(big)}</b>, still 3 calculations.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">Add up every multiple of 4 or 6 below 1000.</div>
<div class="h">Careful: which numbers get counted twice? (It isn't the multiples of 24.) Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 2, the even Fibonacci numbers.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 2
def fib_tiling(sizes):
    """Place Fibonacci squares in a spiral. Returns list of (x, y, s) in unit coords (y down)."""
    rects = [(0, 0, sizes[0])]
    x0, y0, x1, y1 = 0, 0, sizes[0], sizes[0]
    dirs = ["right", "down", "left", "up"]
    for i, s in enumerate(sizes[1:]):
        d = dirs[i % 4]
        if d == "right":
            r = (x1, y0, s)
        elif d == "down":
            r = (x0, y1, s)
        elif d == "left":
            r = (x0 - s, y1 - s, s)
        else:
            r = (x1 - s, y0 - s, s)
        rects.append(r)
        x0, y0 = min(x0, r[0]), min(y0, r[1])
        x1, y1 = max(x1, r[0] + s), max(y1, r[1] + s)
    return rects, (x0, y0, x1, y1)


def p2():
    num, slides = 2, []
    seq = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    rects, (x0, y0, x1, y1) = fib_tiling(seq)
    # transpose so the 34 x 55 tiling lies landscape (55 wide)
    rects = [(y, x, s) for x, y, s in rects]
    x0, y0, x1, y1 = y0, x0, y1, x1
    sc = 900 / (x1 - x0)
    body, arcs = [], []
    # spiral: in each square, a quarter arc between the two corners shared with neighbours
    for idx, (x, y, s) in enumerate(rects):
        X, Y, Sz = (x - x0) * sc, (y - y0) * sc, s * sc
        even = s % 2 == 0 and idx > 0
        fill = BLUE if even else "#1b1e24"
        body.append(f'<rect x="{X + 2:.1f}" y="{Y + 2:.1f}" width="{Sz - 4:.1f}" height="{Sz - 4:.1f}" rx="{min(10, Sz / 6):.1f}" fill="{fill}"/>')
    # quarter arcs: square i placed in direction d -> arc corners
    pts = []
    dirs = ["right", "down", "left", "up"]
    for idx, (x, y, s) in enumerate(rects):
        if idx == 0:
            continue
        d = dirs[(idx - 1) % 4]
        X, Y, Sz = (x - x0) * sc, (y - y0) * sc, s * sc
        # corners in the untransposed frame, then swap x/y (a reflection flips the sweep)
        if d == "right":
            p, q = (Y, X), (Y + Sz, X + Sz)
        elif d == "down":
            p, q = (Y + Sz, X), (Y, X + Sz)
        elif d == "left":
            p, q = (Y + Sz, X + Sz), (Y, X)
        else:
            p, q = (Y, X + Sz), (Y + Sz, X)
        p, q = (p[1], p[0]), (q[1], q[0])
        arcs.append(f'M {p[0]:.1f} {p[1]:.1f} A {Sz:.1f} {Sz:.1f} 0 0 0 {q[0]:.1f} {q[1]:.1f}')
    body.append(f'<path d="{" ".join(arcs)}" fill="none" stroke="#f2f0eb" stroke-opacity=".55" stroke-width="3"/>')
    for idx, (x, y, s) in enumerate(rects):
        X, Y, Sz = (x - x0) * sc, (y - y0) * sc, s * sc
        if s >= 2:
            even = s % 2 == 0 and idx > 0
            body.append(t(X + Sz / 2, Y + Sz / 2, s, max(22, min(64, Sz * 0.32)), INK_ON_COLOR if even else "#bfc2c9", 750))
    tile = svg(900, (y1 - y0) * sc, "".join(body))
    slides.append(("Even Fibonacci numbers", f"""
<div class="kicker">Problem 2 · Even Fibonacci numbers</div>
<h1>Add the even Fibonacci numbers up to four million.</h1>
<div class="sub">1, 2, 3, 5, 8, 13, 21, 34, … each one is the sum of the two before it. The even squares are blue.</div>
<div class="stage">{tile}</div>"""))

    terms = S.p2_terms()
    evens = S.p2_evens()
    code = """f = [1, 2]
while f[-1] < 4_000_000:
    f.append(f[-2] + f[-1])
if f[-1] > 4_000_000:
    del f[-1]          # one step too far

total = 0
for x in f:
    if x % 2 == 0:
        total += x
print(total)"""
    slides.append(("The loop", f"""
<div class="kicker">Way 1 · The loop</div>
<h1>Build every term. Keep the even ones.</h1>
<div class="stage"><div class="code">{hl(code)}</div></div>
<div class="row" style="margin-bottom:6px"><div class="stat"><div class="v">{len(terms)}</div><div class="l">terms up to 4,000,000</div></div>
<div class="stat"><div class="v">{len(evens)}</div><div class="l">of them are even</div></div></div>"""))

    # parity strip
    first = terms[:12]
    cw, gap = 74, 9
    body = []
    for i, v in enumerate(first):
        x = i * (cw + gap)
        ev = v % 2 == 0
        body.append(f'<rect x="{x}" y="0" width="{cw}" height="{cw}" rx="12" fill="{BLUE if ev else "#1b1e24"}"/>')
        body.append(t(x + cw / 2, cw / 2 + 1, v, 26 if v < 100 else 22, INK_ON_COLOR if ev else "#bfc2c9", 700))
        body.append(t(x + cw / 2, cw + 38, "even" if ev else "odd", 21, "#f2f0eb" if ev else "#858a94", 600))
    strip = svg(12 * cw + 11 * gap, cw + 60, "".join(body))
    slides.append(("Every third one", f"""
<div class="kicker">The pattern</div>
<h1>Every third Fibonacci number is even.</h1>
<div class="stage" style="gap:46px">{strip}
<div class="card" style="font:500 33px/1.75 'Inter';color:var(--text2)">
odd + even = <b style="color:var(--text)">odd</b><br>even + odd = <b style="color:var(--text)">odd</b><br>odd + odd = <b style="color:var(--text)">even</b>
<div style="font:400 29px/1.45 'Inter';color:var(--muted);margin-top:14px">So after each even term come two odd ones, then an even one again. Forever.</div></div>
</div>"""))

    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #2b2f37">'
        f'<span style="color:var(--muted)">E{i + 1}</span><span>{fmt(v)}</span></div>' for i, v in enumerate(evens))
    nxt = 4 * evens[-1] + evens[-2]
    slides.append(("Skip the odd ones", f"""
<div class="kicker">Way 2 · Skip the odd ones</div>
<h1>Each even term is 4 × the last + the one before.</h1>
<div class="sub" class="mono"><span class="mono" style="color:var(--text)">34 = 4·8 + 2 &nbsp; 144 = 4·34 + 8</span></div>
<div class="stage"><div class="card" style="font:500 31px 'JetBrains Mono';color:var(--text);padding:18px 36px">{rows}
<div style="padding:9px 0 0;color:var(--muted);font-size:25px;font-family:Inter">next: {fmt(nxt)}, over four million. Stop.</div></div></div>
<div class="note">{len(evens)} terms instead of {len(terms)}, and none of them odd.</div>"""))

    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(sum(evens))}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px">2 + 8 + 34 + 144 + … + 3,524,578. The loop builds {len(terms)} terms; the shortcut builds only the {len(evens)} even ones.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">In 1, 2, 3, 5, 8, 13, … is the 100th term even or odd?</div>
<div class="h">No need to compute it: the pattern on slide 3 is enough. Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 3, the largest prime factor.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 3
def tree_svg(nodes, w, h):
    """nodes: list of (x, y, label, kind, parent_index). kind: 'prime' | 'comp' | 'root'"""
    body = []
    for x, y, lab, kind, par in nodes:
        if par is not None:
            px, py = nodes[par][0], nodes[par][1]
            body.append(f'<line x1="{px}" y1="{py + 34}" x2="{x}" y2="{y - 34}" stroke="#4a505c" stroke-width="3"/>')
    for x, y, lab, kind, par in nodes:
        size = 34 if len(lab) < 10 else 30
        wbox = max(110, len(lab) * size * 0.66 + 44)
        fill = {"prime": BLUE, "comp": "#1b1e24", "root": "#1b1e24", "last": ORANGE}[kind]
        stroke = "#3a3f4a" if kind in ("comp", "root") else fill
        body.append(f'<rect x="{x - wbox / 2:.1f}" y="{y - 34}" width="{wbox:.1f}" height="68" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        body.append(t(x, y + 1, lab, size, INK_ON_COLOR if kind in ("prime", "last") else "#f2f0eb", 700, family="JetBrains Mono"))
    return svg(w, h, "".join(body))


def p3():
    num, slides = 3, []
    small = tree_svg([
        (420, 50, "13,195", "root", None),
        (220, 190, "5", "prime", 0), (560, 190, "2,639", "comp", 0),
        (420, 330, "7", "prime", 2), (700, 330, "377", "comp", 2),
        (560, 470, "13", "prime", 4), (840, 470, "29", "prime", 4)], 940, 520)
    slides.append(("Largest prime factor", f"""
<div class="kicker">Problem 3 · Largest prime factor</div>
<h1>What is the largest prime factor of <span class="mono" style="font-size:60px">600,851,475,143</span>?</h1>
<div class="sub">Warm-up: 13,195 = 5 × 7 × 13 × 29, so its answer is <b>29</b>.</div>
<div class="stage">{small}
<div class="legend"><span><i style="background:{BLUE}"></i>prime</span><span><i style="background:#1b1e24;border:2px solid #3a3f4a"></i>still to split</span></div></div>"""))

    fac, steps, last_d, chain = S.p3_factor()
    code = """n = 600851475143
d = 2
while d * d <= n:      # only up to √n
    if n % d == 0:
        n //= d        # divide it out
    else:
        d += 1
print(n)"""
    slides.append(("The idea", f"""
<div class="kicker">The method · Trial division</div>
<h1>Try 2, 3, 4, … and divide out every factor you find.</h1>
<div class="stage"><div class="code">{hl(code)}</div></div>
<div class="callout" style="margin-bottom:8px">Two tricks make this fast: the number <b>shrinks</b> every time a factor comes out,
and the search <b>stops at the square root</b> of what's left.</div>"""))

    big = tree_svg([
        (470, 50, "600,851,475,143", "root", None),
        (150, 190, "71", "prime", 0), (600, 190, "8,462,696,833", "comp", 0),
        (330, 330, "839", "prime", 2), (720, 330, "10,086,647", "comp", 2),
        (520, 470, "1471", "prime", 4), (840, 470, "6857", "last", 4)], 940, 520)
    slides.append(("It shrinks", f"""
<div class="kicker">Trick 1 · The number shrinks</div>
<h1>Each factor you divide out makes the rest smaller.</h1>
<div class="sub">12 digits, then 10, then 8, then 4.</div>
<div class="stage">{big}</div>
<div class="note">After 1471 comes out, only <b style="color:var(--text)">6857</b> is left. But is 6857 prime, or does it split further?</div>"""))

    r = math.isqrt(6857)
    slides.append(("Stop at the square root", f"""
<div class="kicker">Trick 2 · Stop at the square root</div>
<h1>If n = a × b, the smaller of a and b is at most √n.</h1>
<div class="sub">So if nothing up to √n divides n, n is prime. √6857 ≈ {math.sqrt(6857):.1f}, and every number up to {last_d} has already been tried.</div>
<div class="stage" style="gap:22px">
<div class="row"><div class="stat"><div class="v">{fmt(last_d)}<sup style="font-size:32px">2</sup></div><div class="l">= {fmt(last_d * last_d)}, bigger than 6857: the loop stops</div></div></div>
<div class="row"><div class="stat"><div class="v">{fmt(steps)}</div><div class="l">trial divisions in total</div></div>
<div class="stat"><div class="v">600 billion</div><div class="l">tries if you tested every number</div></div></div>
</div>"""))

    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">6857</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px;font-family:'JetBrains Mono';color:var(--text);font-size:33px">600,851,475,143 = 71 × 839 × 1471 × 6857</div>
<div class="sub">About {fmt(steps)} tries instead of 600 billion.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">What is the largest prime factor of 2026? And of 2027?</div>
<div class="h">One of them is a lot easier than it looks. Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 4, the largest palindrome product.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 4
def p4():
    num, slides = 4, []
    allp = S.p4_all()
    best = max(allp, key=lambda r: r[2])
    size, pad, rpad = 680, 100, 30
    sc = (size - pad - rpad) / 900
    body = [f'<rect x="{pad}" y="0" width="{size - pad - rpad}" height="{size - pad - rpad}" fill="#16191e" stroke="#2b2f37" stroke-width="2"/>']
    for a, b, p in allp:
        if (a, b) == best[:2]:
            continue
        X, Y = pad + (a - 100) * sc, (999 - b) * sc
        body.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="3.2" fill="{BLUE}" fill-opacity=".85"/>')
    X, Y = pad + (best[0] - 100) * sc, (999 - best[1]) * sc
    body.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="15" fill="none" stroke="{ORANGE}" stroke-width="4"/>')
    body.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="{ORANGE}"/>')
    body.append(t(X - 30, Y + 44, "the biggest one", 25, "#f2f0eb", 650, anchor="end",
                  extra='stroke="#121417" stroke-width="7" paint-order="stroke"'))
    for v in (100, 400, 700, 999):
        body.append(t(pad + (v - 100) * sc, size - pad - rpad + 30, v, 22, "#858a94", 500))
        body.append(t(pad - 14, (999 - v) * sc, v, 22, "#858a94", 500, anchor="end"))
    body.append(t(pad + (size - pad - rpad) / 2, size - 30, "first number", 23, "#bfc2c9", 600))
    body.append(t(20, (size - pad - rpad) / 2, "second number", 23, "#bfc2c9", 600,
                  extra=f'transform="rotate(-90 20 {(size - pad - rpad) / 2})"'))
    plot = svg(size, size + 20, '<g transform="translate(0,20)">' + "".join(body) + "</g>")
    slides.append(("Largest palindrome product", f"""
<div class="kicker">Problem 4 · Largest palindrome product</div>
<h1>Find the largest palindrome made from two 3-digit numbers.</h1>
<div class="sub">Like 9009 = 91 × 99, which reads the same both ways.</div>
<div class="stage">{plot}</div>
<div class="note">Each dot is a pair whose product is a palindrome: {fmt(len(allp))} pairs in all.</div>"""))

    code = """best = 0
for a in range(100, 1000):
    for b in range(a, 1000):
        p = a * b
        if str(p) == str(p)[::-1] and p > best:
            best = p
print(best)"""
    slides.append(("Brute force", f"""
<div class="kicker">Way 1 · Try every pair</div>
<h1>Multiply every pair. Keep the biggest palindrome.</h1>
<div class="stage"><div class="code">{hl(code)}</div></div>
<div class="row" style="margin-bottom:6px"><div class="stat"><div class="v">{fmt(900 * 901 // 2)}</div><div class="l">pairs to multiply and check</div></div>
<div class="stat"><div class="v">{fmt(len(allp))}</div><div class="l">of them give a palindrome</div></div></div>"""))

    slides.append(("The 11 trick", f"""
<div class="kicker">The insight</div>
<h1>Every 6-digit palindrome is a multiple of 11.</h1>
<div class="stage" style="gap:34px">
<div class="card eq" style="font-size:33px;line-height:1.75">abccba = 100001a + 10010b + 1100c
      = 11 × (9091a + 910b + 100c)</div>
<div class="callout">11 is prime, so if a × b is a multiple of 11, then <b>a or b is</b>.
Example: 888,888 = 924 × 962, and 924 = 11 × 84.</div>
<div class="note" style="width:100%">So one of the two numbers only needs to run over 990, 979, 968, … : the multiples of 11.</div>
</div>"""))

    best_p, ba, bb, checks = S.p4_smart()
    big, small = 640, 640 * math.sqrt(checks / (900 * 901 // 2))
    body = [f'<rect x="0" y="0" width="{big}" height="{big}" rx="10" fill="#1b1e24" stroke="#3a3f4a" stroke-width="2"/>',
            f'<rect x="{big - small - 24:.1f}" y="24" width="{small:.1f}" height="{small:.1f}" rx="4" fill="{ORANGE}"/>',
            t(big / 2, big / 2 + 20, f"{fmt(900 * 901 // 2)} pairs", 40, "#bfc2c9", 700),
            t(big / 2, big / 2 + 72, "(area to scale)", 26, "#858a94", 500),
            t(big - small - 44, 24 + small / 2, f"{checks} pairs", 32, "#f2f0eb", 700, anchor="end")]
    sq = svg(big, big, "".join(body))
    slides.append(("Search from the top", f"""
<div class="kicker">Way 2 · Start at the top, stop early</div>
<h1>Count down, use the 11s, quit when you can't win.</h1>
<div class="sub">Once a × 999 is smaller than the best palindrome so far, no smaller a can beat it.</div>
<div class="stage">{sq}</div>"""))

    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(best_p)}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px;font-family:'JetBrains Mono';color:var(--text);font-size:34px">{fmt(best_p)} = {ba} × {bb}, and {ba} = 11 × {ba // 11}</div>
<div class="sub">{checks} pairs checked instead of {fmt(900 * 901 // 2)}.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">Is every 4-digit palindrome a multiple of 11 too? Prove it, or find one that isn't.</div>
<div class="h">Hint: write abba the way slide 3 writes abccba. Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 5, the smallest multiple.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 5
def p5():
    num, slides = 5, []
    R, C = 330, 400
    body = [f'<circle cx="{C}" cy="{C}" r="{R}" fill="none" stroke="#2b2f37" stroke-width="3"/>']
    for k in range(1, 21):
        ang = -math.pi / 2 + (k - 1) * 2 * math.pi / 20
        x, y = C + R * math.cos(ang), C + R * math.sin(ang)
        body.append(f'<line x1="{C + 92 * math.cos(ang):.1f}" y1="{C + 92 * math.sin(ang):.1f}" x2="{C + (R - 40) * math.cos(ang):.1f}" y2="{C + (R - 40) * math.sin(ang):.1f}" stroke="{BLUE}" stroke-opacity=".35" stroke-width="2.5"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="38" fill="#1b1e24" stroke="{BLUE}" stroke-width="3"/>')
        body.append(t(x, y + 1, k, 30, "#f2f0eb", 700))
    body.append(f'<circle cx="{C}" cy="{C}" r="88" fill="{ORANGE}"/>')
    body.append(t(C, C + 4, "?", 90, INK_ON_COLOR, 800, family="Inter Display"))
    ring = svg(2 * C, 2 * C, "".join(body))
    slides.append(("Smallest multiple", f"""
<div class="kicker">Problem 5 · Smallest multiple</div>
<h1>2520 is the smallest number that 1 to 10 all divide. What about 1 to 20?</h1>
<div class="stage">{ring}</div>"""))

    tries = S.p5_brute_count()
    code = """n = 20
while any(n % k for k in range(1, 21)):
    n += 20      # a multiple of 20
print(n)"""
    slides.append(("Brute force", f"""
<div class="kicker">Way 1 · Count up</div>
<h1>Try 20, 40, 60, … until all 20 numbers divide it.</h1>
<div class="stage"><div class="code">{hl(code)}</div></div>
<div class="row" style="margin-bottom:6px"><div class="stat"><div class="v">{fmt(tries)}</div><div class="l">numbers tried before it stops</div></div></div>
<div class="note" style="margin-top:18px">It works, but it takes millions of tries. Prime factors do it in one line.</div>"""))

    primes = [p for p, _ in S.prime_powers(20)]
    cw, rh = 92, 38
    left = 92
    body = []
    for j, p in enumerate(primes):
        body.append(t(left + j * cw + cw / 2, 20, p, 27, "#f2f0eb", 750))
    best_e = {p: max(S.exponent(p, k) for k in range(1, 21)) for p in primes}
    for i, k in enumerate(range(2, 21)):
        y = 50 + i * rh
        body.append(t(left - 22, y + rh / 2, k, 25, "#bfc2c9", 600, anchor="end"))
        for j, p in enumerate(primes):
            e = S.exponent(p, k)
            if e:
                top = e == best_e[p] and (p ** e == k)
                x = left + j * cw
                body.append(f'<rect x="{x + 6}" y="{y + 5}" width="{cw - 12}" height="{rh - 10}" rx="8" fill="{ORANGE if top else BLUE}" fill-opacity="{1 if top else .35}"/>')
                body.append(t(x + cw / 2, y + rh / 2 + 1, f"{p}" if e == 1 else f"{p}" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[e], 21, INK_ON_COLOR if top else "#f2f0eb", 700))
    table = svg(left + len(primes) * cw, 50 + 19 * rh, "".join(body))
    slides.append(("Prime powers", f"""
<div class="kicker">The insight</div>
<h1>Factor 2 to 20. Keep the biggest power of each prime.</h1>
<div class="stage">{table}</div>
<div class="note">Orange: the highest power each prime reaches, 16 = 2⁴, 9 = 3², then 5, 7, 11, 13, 17, 19.</div>"""))

    L = math.lcm(*range(1, 21))
    f20 = math.factorial(20)
    slides.append(("Multiply the champions", f"""
<div class="kicker">Way 2 · One multiplication</div>
<h1>Multiply the champions, and nothing more.</h1>
<div class="stage" style="gap:30px">
<div class="card eq" style="font-size:34px;text-align:center;line-height:1.8">2⁴ × 3² × 5 × 7 × 11 × 13 × 17 × 19<br>= 16 × 9 × 5 × 7 × 11 × 13 × 17 × 19</div>
<div class="code" style="font-size:30px">{hl('from math import lcm' + chr(10) + 'print(lcm(*range(1, 21)))')}</div>
<div class="callout">Why not just 1 × 2 × … × 20 = 20!? It works too, but it's
<b>{fmt(f20 // L)} times</b> too big.</div>
</div>"""))

    L25 = math.lcm(*range(1, 26))
    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(L)}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px">Every number from 1 to 20 divides it, and nothing smaller works: drop any one prime power and some number stops dividing.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">What is the smallest number that 1 to 25 all divide?</div>
<div class="h">Hint: only two of 21, 22, 23, 24, 25 bring something new. Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 6, the sum square difference.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 6
def square_picture(n, side):
    total = n * (n + 1) // 2
    u = side / total
    body = []
    pos = [0]
    for k in range(1, n + 1):
        pos.append(pos[-1] + k)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x, y = pos[j - 1] * u, pos[i - 1] * u
            w, h = j * u, i * u
            c = ORANGE if i == j else BLUE
            op = 1 if i == j else .78
            body.append(f'<rect x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" width="{w - 3:.1f}" height="{h - 3:.1f}" rx="2" fill="{c}" fill-opacity="{op}"/>')
    for k in range(4, n + 1):
        x = (pos[k - 1] + k / 2) * u
        body.append(t(x, x, f"{k}²", max(16, k * u * 0.3), INK_ON_COLOR, 750))
    return svg(side, side, "".join(body))


def p6():
    num, slides = 6, []
    s10, sq10_, sumsq10, d10 = S.p6(10)
    pic = square_picture(10, 760)
    slides.append(("Sum square difference", f"""
<div class="kicker">Problem 6 · Sum square difference</div>
<h1 style="font-size:56px">(1 + 2 + … + 100)²<br>− (1² + 2² + … + 100²) = ?</h1>
<div class="sub">For 1 to 10: a {s10} × {s10} square ({fmt(sq10_)}) minus the orange squares ({sumsq10}) leaves the blue area: <b>{fmt(d10)}</b>.</div>
<div class="stage">{pic}</div>"""))

    code = """sum_sq = 0
total = 0
for k in range(1, 101):
    sum_sq += k * k
    total += k
print(total * total - sum_sq)"""
    slides.append(("The loop", f"""
<div class="kicker">Way 1 · The loop</div>
<h1>Add them up, square, subtract.</h1>
<div class="stage"><div class="code">{hl(code)}</div></div>
<div class="note" style="margin-bottom:8px">Fine for 100. But both sums have a formula, so the loop isn't needed at all.</div>"""))

    s, sq, ss, d = S.p6(100)
    slides.append(("Two formulas", f"""
<div class="kicker">Way 2 · Two formulas</div>
<h1>Both sums have a closed form.</h1>
<div class="stage" style="gap:28px">
<div class="card"><div class="note">sum of the first n numbers</div>
<div class="eq" style="font-size:38px">n(n + 1) ÷ 2 = 100 × 101 ÷ 2 = {fmt(s)}</div>
<div class="eq" style="font-size:32px;color:var(--text2)">squared: {fmt(s)}² = {fmt(sq)}</div></div>
<div class="card"><div class="note">sum of the first n squares</div>
<div class="eq" style="font-size:36px">n(n + 1)(2n + 1) ÷ 6</div>
<div class="eq" style="font-size:32px;color:var(--text2)">= 100 × 101 × 201 ÷ 6 = {fmt(ss)}</div></div>
</div>"""))

    slides.append(("What the picture says", f"""
<div class="kicker">What the picture says</div>
<h1>The difference is every rectangle off the diagonal.</h1>
<div class="sub">Each i × j with i ≠ j shows up twice, once on each side of the diagonal. Add them all and you get one formula:</div>
<div class="stage" style="gap:28px">
<div class="card eq" style="font-size:40px;text-align:center">(n − 1) n (n + 1)(3n + 2) ÷ 12</div>
<div class="card eq" style="font-size:30px;line-height:1.75">n = 10:  9 × 10 × 11 × 32 ÷ 12 = {fmt(d10)} ✓
n = 100: 99 × 100 × 101 × 302 ÷ 12</div>
</div>"""))

    d20 = S.p6(20)[3]
    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(d)}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px;font-family:'JetBrains Mono';color:var(--text);font-size:33px">{fmt(sq)} − {fmt(ss)}</div>
<div class="sub">The square of the sum grows like n⁴, the sum of squares only like n³, so the difference is mostly the square.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">Do the same for 1 to 20.</div>
<div class="h">Use either formula and check with the other. Answer in the first comment.</div></div>
</div>
<div class="note">Next Sunday: Problem 7, the 10,001st prime.</div>"""))
    return num, slides


# ======================================================================= PROBLEM 7
def p7():
    num, slides = 7, []
    cols, rows, cell, gap = 12, 10, 70, 7

    def spf(v):
        for p in (2, 3, 5, 7):
            if v % p == 0 and v != p:
                return p
        return None

    def grid(mode):
        body = []
        for i in range(cols * rows):
            v = i + 1
            x, y = (i % cols) * (cell + gap), (i // cols) * (cell + gap)
            prime = S.is_prime(v)
            if mode == "primes":
                c, ink = (BLUE, INK_ON_COLOR) if prime else ("#1b1e24", "#6f7580")
            else:
                f = spf(v)
                cmap = {2: "#262a33", 3: "#262a33", 5: "#262a33", 7: "#262a33"}
                if prime:
                    c, ink = BLUE, INK_ON_COLOR
                elif v == 1:
                    c, ink = "#1b1e24", "#6f7580"
                else:
                    c, ink = cmap[f], "#6f7580"
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="10" fill="{c}"/>')
            body.append(t(x + cell / 2, y + cell / 2 + 1, v, 25, ink, 650 if prime else 500))
            if mode == "sieve" and not prime and v > 1:
                f = spf(v)
                col = {2: "#858a94", 3: ORANGE, 5: AQUA, 7: VIOLET}[f]
                body.append(f'<line x1="{x + 12}" y1="{y + cell - 12}" x2="{x + cell - 12}" y2="{y + 12}" stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
        return svg(cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap, "".join(body))

    n_pr = sum(1 for v in range(1, 121) if S.is_prime(v))
    slides.append(("The 10,001st prime", f"""
<div class="kicker">Problem 7 · The 10,001st prime</div>
<h1>The first six primes are 2, 3, 5, 7, 11 and 13. What is the 10,001st?</h1>
<div class="stage">{grid("primes")}</div>
<div class="note">The {n_pr} primes up to 120, in blue. We need ten thousand more.</div>"""))

    slides.append(("The sieve", f"""
<div class="kicker">The method · Sieve of Eratosthenes</div>
<h1>Cross out the multiples of 2, 3, 5, 7. What survives is prime.</h1>
<div class="stage">{grid("sieve")}
<div class="legend" style="gap:26px"><span><i style="background:#858a94"></i>× 2</span><span><i style="background:{ORANGE}"></i>× 3</span><span><i style="background:{AQUA}"></i>× 5</span><span><i style="background:{VIOLET}"></i>× 7</span></div></div>
<div class="note">Up to 120 you can stop at 7: the next prime, 11, has 11² = 121 &gt; 120.</div>"""))

    bound = S.p7_bound()
    lim = int(bound) + 1
    pr = S.sieve(lim)
    target = pr[10000]
    # pi(x) chart
    Wc, Hc, padl, padb = 900, 560, 110, 70
    padr = 60
    xmax, ymax = 120000, 12000
    pts, cnt, j = [], 0, 0
    for x in range(0, xmax + 1, 1000):
        while j < len(pr) and pr[j] <= x:
            j += 1
        pts.append((x, j))
    # extend beyond lim with a fresh sieve for the chart's right edge
    pr_full = S.sieve(xmax)
    pts, j = [], 0
    for x in range(0, xmax + 1, 500):
        while j < len(pr_full) and pr_full[j] <= x:
            j += 1
        pts.append((x, j))
    X = lambda v: padl + v / xmax * (Wc - padl - padr)
    Y = lambda v: (Hc - padb) - v / ymax * (Hc - padb - 20)
    body = []
    for gy in range(0, ymax + 1, 3000):
        body.append(f'<line x1="{padl}" y1="{Y(gy):.1f}" x2="{Wc - padr}" y2="{Y(gy):.1f}" stroke="#2b2f37" stroke-width="1.5"/>')
        body.append(t(padl - 16, Y(gy), fmt(gy), 22, "#858a94", 500, anchor="end"))
    for gx in range(0, xmax + 1, 40000):
        body.append(t(X(gx), Hc - padb + 32, fmt(gx), 22, "#858a94", 500))
    body.append(f'<line x1="{X(bound):.1f}" y1="{Y(0):.1f}" x2="{X(bound):.1f}" y2="{Y(ymax):.1f}" stroke="{ORANGE}" stroke-width="2.5" stroke-dasharray="8 7"/>')
    body.append(t(X(bound) - 12, Y(ymax) + 20, f"bound ≈ {fmt(int(bound))}", 23, "#f2f0eb", 650, anchor="end"))
    body.append(f'<line x1="{padl}" y1="{Y(10001):.1f}" x2="{X(target):.1f}" y2="{Y(10001):.1f}" stroke="#bfc2c9" stroke-width="2" stroke-dasharray="4 6"/>')
    body.append(t(padl + 14, Y(10001) - 22, "10,001 primes", 23, "#bfc2c9", 600, anchor="start"))
    d = "M " + " L ".join(f"{X(x):.1f} {Y(v):.1f}" for x, v in pts)
    body.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="3" stroke-linejoin="round"/>')
    body.append(f'<circle cx="{X(target):.1f}" cy="{Y(10001):.1f}" r="9" fill="{BLUE}" stroke="#121417" stroke-width="3"/>')
    body.append(t(X(68000), Y(3000), "number of primes ≤ x", 24, "#bfc2c9", 600, anchor="start"))
    body.append(t((padl + Wc) / 2, Hc - 10, "x", 24, "#bfc2c9", 600))
    chart = svg(Wc, Hc, "".join(body))
    slides.append(("How far to sieve", f"""
<div class="kicker">How far do we sieve?</div>
<h1>The prime number theorem tells us where to stop.</h1>
<div class="sub">For n ≥ 6, the n-th prime is below n(ln n + ln ln n). For n&nbsp;=&nbsp;10,001 that's about <b>{fmt(int(bound))}</b>.</div>
<div class="stage">{chart}</div>
<div class="note">The blue curve crosses 10,001 primes before the dashed line, so a sieve up to {fmt(lim)} is enough.</div>"""))

    code = f"""N = {lim}                    # the bound
is_p = [True] * (N + 1)
is_p[0] = is_p[1] = False
for i in range(2, int(N ** 0.5) + 1):
    if is_p[i]:
        for m in range(i * i, N + 1, i):
            is_p[m] = False
primes = [i for i in range(N + 1) if is_p[i]]
print(primes[10000])    # 10,001st (counting from 0)"""
    slides.append(("The code", f"""
<div class="kicker">The code</div>
<h1>One sieve, one list, one lookup.</h1>
<div class="stage"><div class="code" style="font-size:26px">{hl(code)}</div></div>
<div class="row" style="margin-bottom:6px"><div class="stat"><div class="v">{fmt(len(pr))}</div><div class="l">primes up to {fmt(lim)}</div></div>
<div class="stat"><div class="v">{fmt(math.isqrt(lim))}</div><div class="l">largest i the outer loop needs</div></div></div>"""))

    slides.append(("Answer", f"""
<div class="kicker">The answer</div>
<div class="answer">{fmt(target)}</div>
<div class="answer-u"></div>
<div class="sub" style="margin-top:30px">The 10,001st prime. The estimate n ln n ≈ {fmt(round(10001 * math.log(10001)))} lands within {round(abs(target - 10001 * math.log(10001)) / target * 100)}% of it.</div>
<div class="stage">
<div class="turn"><div class="t">Your turn</div>
<div class="q">What is the 100th prime?</div>
<div class="h">The grid on slide 1 gets you to the 30th. Answer in the first comment.</div></div>
</div>
<div class="note">Thanks for solving along. More Euler problems are on the way.</div>"""))
    return num, slides


# ======================================================================= build
def main():
    manifest = []
    for fn in (p1, p2, p3, p4, p5, p6, p7):
        num, slides = fn()
        for k, (title, inner) in enumerate(slides, 1):
            name = f"p{num:02d}_s{k}"
            (OUT / f"{name}.html").write_text(page(num, k, len(slides), title, inner), encoding="utf-8")
            manifest.append({"problem": num, "slide": k, "name": name, "title": title})
    (HERE / "build" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} slides")


if __name__ == "__main__":
    main()
