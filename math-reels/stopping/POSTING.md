# Posting kit — *When do you stop looking? The 37% rule, 10,000 lives.* (dark theme, no narration, 26 s)

## Files

* Reel: `output/stopping_reel.mp4` — 26.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the curve with the catch: the "ended up alone" line crossing its peak at 37.
* Simulation: `stopsim.py` (seed 67; `python stopsim.py out.json` writes every number below).

## What's on screen (all of it is real)

The "secretary problem" (optimal stopping). 100 people arrive one at a time in a random order. Each gets a yes or
a no on the spot, and a no is final. The rule: say no to the first 37, remember the best of them, then say yes to
the first person better than all of them. If nobody is, you end up alone.

* **One life (life 8 of the 10,000):** the bars are the 100 people in the order you meet them, height = their rank
  from 1 to 100 (taller = better match). The best of the first 37 is the 4th best overall (rank 97); the first
  person after them to beat that is person 60, who turns out to be the best of all 100. Life 8 is the first of the
  10,000 in which the rule finds the very best; lives 1–7 did not, and lives 1–6 are shown next, one by one.
* **10,000 lives:** one cell per life, 100 × 100.

| Outcome with the 37% rule | Lives | Share | Exact |
| --- | --- | --- | --- |
| Found the very best | 3,708 | **37.1%** | 37.10% |
| Settled for someone else | 2,613 | 26.1% | 25.90% |
| Ended up alone (the best was in the first 37) | 3,679 | **36.8%** | 37% |

  Lives 1–6: settled for the 2nd best, alone, alone, alone, settled for the 6th best, alone. When the rule settles,
  it is usually close: the 2nd best in 52% of those lives, a top-3 person in 77%.
* **Picking at random** (yes to the first person): 1% exact; 1.2% in these 10,000 lives (117).
* **Every cutoff 0–99** (the curve), all from the same 10,000 lives, with the exact formula dashed:
  P(R) = (R/100) × (1/R + 1/(R+1) + … + 1/99). Cutoff 10: 23.9%, 37: 37.1%, 70: 25.9%, 99: 0.9%.
* **The catch:** the share who end up alone with cutoff R is the chance the best is among the first R, which is
  R/100: a straight line. It crosses the curve at its peak.

## Honesty notes

* The top of the curve is flat: every cutoff from 30 to 44 finds the best 36.4–37.1% of the time (exact). In these
  10,000 lives the highest simulated point is 36 (37.2%), one step from the exact best, 37, by chance; the reel
  marks 37 and the value the 10,000 lives give there (37.1%).
* The rule assumes you know how many people there will be, they come in a random order, you can tell how each one
  compares with everyone so far, nobody can be recalled, and only the very best counts as a win. If "good enough"
  counts, an earlier cutoff does better.
* "Alone" is this reel's word for nobody beating the first 37. In the textbook version you must then take the last
  person, who is almost never the best, so the 37.1% is the same either way.
* The heights are ranks, 1 to 100: the rule only ever compares people, never measures them.
* As the number of people grows, the best cutoff tends to n/e ≈ 36.8% of them and the chance to 1/e ≈ 36.8%.

## Caption (paste as-is)

```
When do you stop looking and choose?

Say you'll meet 100 people, one at a time, and every no is final. The 37% rule: say no to the first 37. Then say yes to the first one better than all of them.

I ran 10,000 lives. It found the very best person 37.1% of the time. Picking at random: 1%.

Why 37? Look at fewer and you say yes too soon. Look at more and the best has usually already gone by.

The catch: 37% of the time, the best one was in the first 37. Nobody after them is better, and you end up alone.

Would you take those odds? 👇

(A real simulation: 10,000 random orders of 100 people. The exact answer is 37.1%. Same maths for hiring and flat hunting.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #probability #statistics #datascience #optimalstopping
```

Exactly five hashtags.

## YouTube

Title:

```
The 37% Rule: When to Stop Looking (10,000 Lives Simulated) #shorts
```

Description:

```
When do you stop looking and choose? 100 people arrive one at a time in a random order, each gets a yes or no on the spot, and a no is final.

The 37% rule: say no to the first 37, then say yes to the first one better than all of them. In 10,000 simulated lives it found the very best person 37.1% of the time (exact: 37.10%), settled for someone else 26.1%, and ended up alone 36.8% (exact: 37%), every time the best one was among the first 37. Picking at random finds the best 1% of the time.

Trying every cutoff from 0 to 99 on the same 10,000 lives draws the curve: too few and you say yes too soon, too many and the best has already gone. Its peak, at 37, is exactly where it crosses the chance of ending up alone.

#math #probability #optimalstopping #shorts
```

## Pinned comment

```
Why exactly 37%? With n people and cutoff r, you find the very best when they come after the first r AND the best of everyone before them is in the first r. Add that up and the chance is (r/n)(1/r + 1/(r+1) + ... + 1/(n−1)), which for large n is −x·ln(x) with x = r/n. That's largest at x = 1/e ≈ 0.368, and there it equals x itself: the chance of finding the best equals the chance the best was in the first r and you end up alone. That's why the two lines cross at the top of the curve.
```

## Audio

Keep the built-in track: every person in the example life plucks a note pitched by how good a match they are
(quiet while just looking), a bell on the yes and a bell and impact when it turns out to be the best; the first six
lives end on their outcome (bell, pluck or a low blip for alone); a rattle follows the lives; while the curve draws,
a tone follows it and chimes at 37; a falling tone for the catch.

## Replies to expect

* "Nobody dates 100 people" → the rule is a fraction: with 20 candidates, look at the first 7 (exact best: 7, 38.4%).
* "Why not just pick someone good?" → right: the 37% rule is only best if nothing but the very best counts; the
  honesty notes say so.
* "This is the secretary problem" → yes; the reel calls it the 37% rule.
