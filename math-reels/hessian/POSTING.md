# Posting kit — *Gradient descent vs saddle points: why second derivatives and the Hessian matter.* (white style, maths on screen, 50 s)

## Files

* Reel: `output/hessian_reel.mp4` — 50.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the hill with the curvature drawn on it (−2.2 along the pass, +3.6 across it,
  +4.4 and +3.6 at a valley bottom) and the two Hessians with their verdicts.

## What's on screen (all of it is real)

The two-valley hill of reel 30, f(u, v) = 2.2·((u/2)² − 1)² + 1.8·v², rendered from its actual height by
`../valleys/hillgen.py` in the reel-23 look: two valleys with bottoms at u = ±2 (height 0) and a pass
between them whose top (0, 0) is a saddle point at height 2.2.

| Time | What | The maths |
| --- | --- | --- |
| 0–9 s | The second derivative in 1D | two small curves on the card: f″ > 0 curves up (a valley), f″ < 0 curves down (a hilltop); at both turning points the slope f′ is zero |
| 9.6–13 s | The Hessian | H = [∂²f/∂u², ∂²f/∂u∂v; ∂²f/∂v∂u, ∂²f/∂v²]: curvature along u, along v, and the twist between them. Strokes drawn on the hill along u and across v at the pass |
| 13–17 s | Read the signs | at the pass H = [−2.2, 0; 0, 3.6]: one down, one up, a saddle; at the bottom (−2, 0) H = [4.4, 0; 0, 3.6]: both up, a minimum (strokes and their curvatures on the hill) |
| 17–21 s | Three rules | gradient descent: step = −0.1 × slope; Newton: step = −H⁻¹ × slope; saddle-free Newton: step = −\|H\|⁻¹ × slope (negative curvature flipped to positive) |
| 21–39 s | The race | three balls, one start (u = 0.01 off the line through the pass, v = −0.9 up the back wall), 36 steps of 0.5 s; readouts of height and distance off the pass for each; a live card with the step along the pass |
| 39–49 s | Why the Hessian matters | its signs tell a bottom from a saddle; Newton trusts it and steps onto the saddle; \|H\| escapes in 6 steps, not 20; n weights have n² second derivatives |

**Along the pass, one step.** Near the pass the slope along it is −2.2·u, so one step sends u to:

| Method | Step along the pass | Factor |
| --- | --- | --- |
| Gradient descent (0.1) | u − 0.1·(−2.2·u) | ×1.22 |
| Newton | u − (−2.2·u)/(−2.2) | ×0 |
| Saddle-free Newton | u − (−2.2·u)/\|−2.2\| | ×2 |

Across the pass (v), the curvature is +3.6: gradient descent keeps 64% of v per step, both Newtons solve
it exactly in one step (f is exactly quadratic in v).

**The run.**

| | Off the pass (\|u\| > 0.5) | At the bottom (within 0.01) | After 36 steps |
| --- | --- | --- | --- |
| Gradient descent | step 20 | step 35 | u = 1.995, height 0.0001 |
| Newton | never | never | u = 0, height 2.200, slope 0.000 |
| Saddle-free Newton | step 6 | step 10 | u = 2.000, height 0.000 |

Newton: u = 0.01 → −0.0000005 (step 1) → 6 × 10⁻²⁰ (step 2) → exactly 0, and v = 0 from step 1: it sits on the
saddle with zero slope, which a stopping rule would call converged. Saddle-free Newton: u = 0.01, 0.02,
0.04, 0.08, 0.160, 0.323, 0.664, 1.55, 2.33 (overshoot), 2.06, 2.003, 2.000. Gradient descent: 0.01,
0.0122, 0.0149, 0.0182 … (×1.22 a step), off the pass at step 20.

## Honesty notes

* This hill's Hessian has no twist (∂²f/∂u∂v = 0 everywhere), because its valleys line up with the
  axes. In general the off-diagonal entries are not zero, and the eigenvalues of H (the curvatures along
  its principal directions) are what decide bowl, peak or saddle; here they are simply the diagonal.
* Newton's method is drawn to any point where the slope is zero, saddles included: it jumps to where the
  local quadratic is flat. That is a textbook property, not a quirk of this hill.
* Saddle-free Newton here is the plain version: divide each direction's slope by the absolute value of its
  curvature (with a floor of 0.001 that is never reached). Practical versions add damping or a trust
  region; near an inflection the plain version can take a long step (here it overshoots to 2.33 and comes
  back).
* Real networks: the Hessian of n weights has n² entries (10¹² for a million weights), so second-order
  methods are used only through approximations, and most training runs on the slope alone.

## Caption (paste as-is)

```
Slope zero. Valley, or saddle?

The slope can't tell you. At the bottom of a valley and at the top of a mountain pass, the ground is flat in every direction. Gradient descent sees exactly the same thing at both: slope zero.

The second derivative can tell. In 1D it's the curvature: f″ > 0 curves up (a valley), f″ < 0 curves down (a hilltop). On a hill it becomes a matrix, the Hessian: the curvature in every direction at once.

On this hill, at the pass: H = [−2.2, 0; 0, 3.6]. Down one way, up the other. A saddle.
At a valley bottom: H = [4.4, 0; 0, 3.6]. Up both ways. A minimum.

Then a race. Three balls, same start, 0.01 off the line through the pass:

Gradient descent (step = −0.1 × slope): along the pass each step multiplies the offset by 1.22. Slow. Off the pass at step 20.

Newton's method (step = −H⁻¹ × slope): uses the curvature, so it should be smart. It jumps straight onto the saddle in one step, where the slope is zero, and stops. "Converged." To a saddle. It never moves again.

Saddle-free Newton (step = −|H|⁻¹ × slope): the same curvature, but with the negative one flipped. Now the offset doubles every step. Off the pass at step 6, at the bottom by step 10.

Same information, read two ways: trust the curvature and you walk into the saddle; flip it and you escape three times faster than gradient descent.

So why doesn't everyone use the Hessian? A network with a million weights has a trillion second derivatives. That's why training mostly runs on the slope alone, and pays for it on every plateau.

(The hill is the real function from the saddle reel; every step, every curvature and every readout is computed. Newton's method is attracted to any point with zero slope, saddles included; that's textbook, not a trick.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #optimization #gradientdescent #hessian #saddlepoint #newtonsmethod #calculus #ai
```

Short alternative:

```
At a valley bottom and at a saddle the slope is zero. Only the second derivatives, the Hessian, tell them apart: [−2.2, 0; 0, 3.6] is a saddle, [4.4, 0; 0, 3.6] a minimum. Race from the same start: gradient descent ×1.22 a step, off the pass at step 20. Newton's method jumps onto the saddle and stops. Saddle-free Newton flips the negative curvature: ×2 a step, off at step 6.

Follow for more.

#machinelearning #optimization #hessian #saddlepoint
```

## YouTube description

```
Gradient descent vs saddle points: why second derivatives and the Hessian matter.

At a valley bottom and at the top of a mountain pass the slope is zero in every direction, so gradient descent can't tell them apart. The second derivative can: in 1D, f″ > 0 curves up and f″ < 0 curves down. On a surface the second derivatives form a matrix, the Hessian, and the signs of its eigenvalues decide: all positive is a minimum, mixed is a saddle.

On the hill f(u, v) = 2.2((u/2)² − 1)² + 1.8v² (two valleys and a pass), the Hessian at the pass is [−2.2, 0; 0, 3.6] (a saddle) and at a valley bottom [4.4, 0; 0, 3.6] (a minimum).

Three optimizers start at the same point, 0.01 off the line through the pass:
- Gradient descent, step size 0.1: along the pass u → 1.22·u per step. Off the pass at step 20, at the bottom at step 35.
- Newton's method, step = −H⁻¹·slope: u → 0·u. It lands on the saddle in one step and stays there forever (height 2.2, slope 0).
- Saddle-free Newton, step = −|H|⁻¹·slope (negative curvature flipped): u → 2·u. Off the pass at step 6, at the bottom at step 10.

Why it isn't used everywhere: n parameters have n² second derivatives; a million weights means a trillion entries, so practical training approximates curvature or ignores it.

Notes: this hill's Hessian is diagonal because the valleys line up with the axes; in general the eigenvalues decide. The saddle-free step here is the plain version without damping; it overshoots once near the inflection.

#machinelearning #optimization #hessian #saddlepoint
```

## Pinned comment

```
Why Newton walks into saddles: it replaces the hill with its local quadratic and jumps to where that quadratic is flat. Along the pass the curvature is −2.2 (a hilltop in that direction), so "where it's flat" is the top: the saddle. Any point with zero slope attracts Newton, minimum or not.

Why |H| fixes it: dividing by the size of the curvature keeps Newton's sense of scale but always steps downhill. Along the pass it turns u into 2u instead of 0·u. This "saddle-free Newton" idea (Dauphin et al., 2014) came out of exactly this question for neural networks, where almost every flat point is a saddle.

Why gradient descent still escapes: its factor along the pass is 1 − 0.1 × (−2.2) = 1.22 > 1. Negative curvature always pushes it away; it's just slow when the curvature is small.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 50 s. A soft chord bed; a pluck as each
1D curve and each curvature stroke draws (falling for curving down, rising for curving up); during the
race one voice per ball, one pluck per landing while it moves (gradient descent left, Newton centre, saddle-
free Newton right, an octave up); Newton falls silent the moment it stops on the saddle; a low tone as each
ball reaches a bottom; bells at the headings.
