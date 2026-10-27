# Posting kit — *Momentum.* (white style, two balls)

## Files

* Reel: `output/momentumw_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the two balls mid-descent, momentum well ahead, the step inset open at lower left.

## What's on screen (all of it is real)

The same cut hill as reels 23 and 24 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour
lines), two balls, the same step size (0.03) and the same start (4.80). Plain descent: step = 0.03 ×
slope, every step from nothing. Momentum: step = 0.8 × last step + 0.03 × slope (heavy-ball momentum,
β = 0.8), computed exactly in the bowl's own axes. 24 steps every 0.62 s.

During each of the momentum ball's arrow phases the main view shows its two ingredients, a blue
"memory" arrow (0.8 × the last step) and a vermilion slope arrow from its tip, then the ball slides
along their sum. At this step size the real arrows are only a dozen pixels, so an inset at lower left
shows the same two screen vectors magnified ×4 (never redrawn, only scaled), with a dashed ring where
the step lands: you watch the blue part grow as speed builds and shrink as the ball settles.

| Step | Plain descent | Momentum 0.8 |
| --- | --- | --- |
| 0 | 4.80 | 4.80 |
| 1 | 4.41 | 4.41 |
| 5 | 3.23 | 1.87 |
| 10 | 2.28 | 0.339 |
| 15 | 1.65 | 0.0004 |
| 20 | 1.21 | 0.040 |
| 24 | **0.94** | **0.026** |

Facts the cues quote, all from the run: momentum's longest step is step 5, 3.3× plain's step at that
moment (0.285 vs 0.087 in the bowl's units); it crosses the bottom along the valley at step 15 (height
0.0004), runs on to −0.26 past it, the height rises to 0.040 at step 20, then falls again; after 24
steps plain is at 0.94 and would need 83 steps to get where momentum is at 24.

Why: along the valley the slope keeps pointing the same way, so the remembered part of each step
compounds; in the limit a constant slope gives a step of 0.03/(1 − 0.8) = 0.15 × slope, five times
plain's. Across the valley and past the bottom the slope flips sign, so the memory fights it: that is
the overshoot and the return. Readouts hold each landing's height; the words "crawling" and "settled"
appear after the last landing.

## Caption (paste as-is)

```
Momentum: the ball that remembers which way it was going.

Same hill as the last two reels. Same step size, same start, two balls.

The red ball is plain gradient descent: feel the slope, step 0.03 times the slope, and every step starts from nothing. After 24 steps it is at height 0.94. Crawling.

The blue ball does one extra thing: it keeps 80% of its last step and adds the slope to it. That's momentum. While the slope keeps pointing the same way, those remembered steps pile up: by step 5 its step is 3.3 times longer than the red ball's. By step 10 it is at 0.34 while red is at 2.28. By step 15 it is at the bottom.

Then the cost of remembering: it can't stop on a dime. It overshoots the bottom, the height creeps back up to 0.040 by step 20, and it settles back down, at 0.026 after 24 steps. Red would need 83 steps to get there.

The inset at the bottom left is the blue ball's next step, magnified 4 times: the blue arrow is what it remembers, the red arrow is the slope, the step is the two added together. Watch the blue arrow grow, then shrink.

This is why nearly every neural network is trained with momentum in some form. Adam has it built in.

(The hill is the real function 0.5u² + 1.8v² on rotated axes; every arrow is its actual gradient and every step is exactly the rule stated. Momentum 0.8, step size 0.03, for both balls.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #momentum #neuralnetworks #ai #math #deeplearning #optimization
```

Short alternative:

```
Same hill, same step size, same start. Plain descent starts every step from nothing: 4.80 → 0.94 in 24 steps. Momentum keeps 80% of its last step and adds the slope: 4.80 → 0.026, with a small overshoot on the way. The ball that remembers which way it was going.

Follow for more.

#machinelearning #gradientdescent #momentum #ai
```

## YouTube description

```
Two balls on the same hill, same step size (0.03), same start. The red one is plain gradient descent: step = 0.03 × slope, every step from nothing. The blue one is momentum: step = 0.8 × last step + 0.03 × slope.

Plain descent: 4.80 → 0.94 after 24 steps. It would need 83 steps to get where momentum is at 24.

Momentum: 4.80 → 1.87 by step 5 (its step is already 3.3× plain's), 0.34 by step 10, 0.0004 by step 15, then it overshoots the bottom, the height rises to 0.040 by step 20, and it settles to 0.026 by step 24.

Why it's faster: along the valley the slope keeps pointing the same way, so the remembered part of each step compounds. With a constant slope the step would grow to 0.03/(1 − 0.8) = 0.15 × slope, five times plain's. Why it overshoots: past the bottom the slope flips, but the memory still points forward, so the ball runs on until the slope has cancelled it. Across a narrow valley the same thing damps the zigzag that plain descent would make with a bigger step.

The inset at the bottom left shows the blue ball's next step magnified 4×: the blue arrow is 0.8 × its last step, the red arrow is the slope × 0.03, and the dashed ring is where their sum lands.

Every number on screen is computed, not drawn: the surface is the real function 0.5u² + 1.8v² on rotated axes, every arrow is its actual gradient, and every step is exactly the rule stated. Honest note: 0.8 is a gentle momentum; at the textbook 0.9 this same ball overshoots much further on this hill (back up to 0.59 at step 20) and is at 0.19 after 24 steps, still better than plain's 0.94, but no longer settled.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #momentum #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
One line of maths: v ← βv − η∇E, w ← w + v. Plain descent is the same with β = 0. Here η = 0.03 and β = 0.8.

Along a slope that keeps the same sign the step grows towards η/(1 − β) × slope, here 0.15 × slope, five times plain's. That's the speed-up. Past the bottom the slope flips, but v still points forward, so the ball overshoots and comes back; the height goes 0.0004 (step 15) → 0.040 (step 20) → 0.026 (step 24). That's the cost, and the reason momentum is also what damps zigzags across a narrow valley (the opposing gradients cancel in v).

Honesty check: β = 0.8 is gentle. At 0.9 this ball overshoots to 0.59 and ends at 0.19; at 0.7 it ends at 0.005 with almost no overshoot. The right amount depends on the hill, exactly like the step size in reel 24. Adam (reels 7 and 8) keeps this memory and adds a second one, of how big the gradients have been.

Next, same white style: what Adam adds to this. Want it?
```

## Audio

Keep the built-in track: the soft chord bed; at every landing one sound per ball. Momentum: a plucked
note walking down a scale as its height falls, struck harder as its steps get longer and softer as
they shrink. Plain: a quiet tick and a faint pluck that barely move down the scale. A low tone when the
run ends, bells at the headings. −16 LUFS.

## Replies to expect

* "Why 0.8 and not 0.9?" → 0.9 overshoots much further on this hill (0.59 at step 20, 0.19 at step 24). 0.8 shows the speed-up and the overshoot in one clean run. Both numbers are in the pinned comment.
* "Isn't the overshoot bad?" → it's the price of the speed-up, and it's small here: 0.040 at worst, against plain's 1.21 at the same step. In a narrow valley the same mechanism damps zigzags, which is the bigger win.
* "Is this Nesterov?" → no, plain heavy-ball momentum (Polyak). Nesterov looks ahead before measuring the slope; the picture is nearly identical at these settings.
* "Does Adam do this?" → yes, its first moment is exactly this memory (β₁, usually 0.9), plus a per-weight step-size scaling from a second memory of squared gradients.
* "Same step size is unfair to plain descent" → fair point: plain descent could use a bigger step. Reel 24 shows what happens when it does, and reel 25 keeps the step size fixed so the only difference is the memory.
