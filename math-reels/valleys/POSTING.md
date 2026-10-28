# Posting kit — *Two valleys.* (white style, four balls, plain descent)

## Files

* Reel: `output/valleys_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — step 12: A already at the deep bottom, B sliding down to join it, C and D settling in the shallow valley.

## What's on screen (all of it is real)

A hill with two bottoms, drawn in the reel-23 style by a general-surface ray-marcher (`hillgen.py`):
along the long axis a double well, f(u, v) = 2.2·((u/2)² − 1)² + 0.6u + c + 1.8v², lifted so the deep
bottom is at height 0. Ridge 3.52, shallow bottom 2.39, deep bottom 0. Cut plane at v = 0.75. The
background is a faint blue-grey tint with a light grid.

Four balls, plain gradient descent, the same rule for all: step = slope × 0.1. 24 steps of 0.62 s.
All four start up the back wall (v = −0.9) at four positions along the long axis:

| Ball | Start u | Start height | After 24 steps | Ends in |
| --- | --- | --- | --- | --- |
| A (vermilion) | −1.10 | 3.11 | **0.000** | deep valley |
| B (blue) | 0.18 (0.1 left of the ridge) | 4.97 | **0.000** | deep valley |
| C (green) | 0.38 (0.1 right of the ridge) | 4.97 | **2.395** | shallow valley |
| D (magenta) | 1.30 | 4.21 | **2.394** | shallow valley |

B and C start 0.2 apart, either side of the ridge at u = 0.28; they are 0.3 apart by step 3 and end
in different valleys. Nothing is added or tuned: the slope under each ball decides, and the two in the
shallow valley have slope zero there, so they will never move again.

## Caption (paste as-is)

```
Two valleys. Where you start decides where you end.

A hill with two bottoms, one shallow, one deep. Four balls, the same rule for all of them: feel the slope, step 0.1 times the slope, repeat. The only difference is where each one starts.

A starts on the deep side and rolls to the bottom: height 0. D starts on the shallow side and rolls to the bottom of that valley: 2.39, and stops, because the slope there is zero. B and C start 0.2 apart on the ridge, one a hair left of the top, one a hair right. By step 3 they've parted. B ends at 0. C ends at 2.40, next to D, and will never leave.

That's all there is to it. Plain gradient descent only sees the slope under its feet. It cannot see the other valley, it cannot compare, and once the ground is flat it stops. That's a local minimum, and it's why the starting point matters, why training the same network twice from different random starts can give different results, and why people care so much about initialisation (reels 19 and 20).

(The hill is a real double well, 2.2·((u/2)² − 1)² + 0.6u + 1.8v², rendered from its actual height; every arrow is its true gradient and every step is exactly slope × 0.1. Heights after 24 steps: 0.000, 0.000, 2.395, 2.394.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
A hill with two bottoms. Four balls, same rule, step 0.1 times the slope. Two end at the deep bottom, two at the shallow one, including two that started 0.2 apart on the ridge. Plain descent only sees the slope under its feet. Where you start decides where you end.

Follow for more.

#machinelearning #gradientdescent #localminimum #ai
```

## YouTube description

```
A hill with two bottoms: a shallow valley (height 2.4) and a deep one (height 0), with a ridge between them at 3.5. Four balls, plain gradient descent, the same rule for all: step = slope × 0.1. The only difference is the starting point.

A starts on the deep side (height 3.11) and reaches the bottom: 0.000 after 24 steps.
B starts 0.1 left of the ridge (4.97) and also reaches the deep bottom: 0.000.
C starts 0.1 right of the ridge (4.97) and ends in the shallow valley: 2.395.
D starts on the shallow side (4.21) and ends there too: 2.394.

B and C start 0.2 apart and are in different valleys by step 3. Nothing is tuned: the slope under each ball decides, and the two in the shallow valley stop because the slope there is zero. They will never move again, and they have no way of knowing that a deeper valley exists.

That is a local minimum, the simplest reason the starting point matters in training, and the reason the same network trained twice from different random initialisations can end up different. The next reels pick this up: what noise (small batches) does to a ball in the shallow valley, and why real loss surfaces in millions of dimensions have fewer true traps than this picture suggests.

Every number on screen is computed, not drawn: the hill is the real function 2.2·((u/2)² − 1)² + 0.6u + 1.8v² lifted so the deep bottom is 0, rendered from its actual height; every arrow is its true gradient and every step is exactly slope × 0.1.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
The maths in one line: gradient descent moves to wherever the slope under its feet is zero, and it has no way to compare that place with any other. On a hill with two bottoms, the ridge divides the starting points into two sets, and each set ends in its own valley: B (0.1 left of the ridge) and C (0.1 right of it) are the two sides of that line.

Honesty check: this hill is drawn to make the point; real loss surfaces in millions of dimensions have far fewer true traps like the shallow valley, and far more saddles and flat plateaus. Two things do get balls out of a valley like C's: noise in the slope (small batches, next reel) and momentum carrying speed over a small ridge.

Next, same white style: shake the shallow valley. What small-batch noise does to C.
```

## Audio

Keep the built-in track: the soft chord bed; at every landing one pluck per ball, walking down the scale
with its height; the two that reach the deep valley go to the bottom of the scale, the two in the shallow
valley stop part-way and repeat their note. A low tone when the run ends, bells at the headings. −16 LUFS.

## Replies to expect

* "Why did B go left and C go right?" → the ridge top is at u = 0.28; B starts at 0.18, C at 0.38. The ground under B tilts left, under C right. That's the whole decision.
* "Couldn't C just keep going?" → not with this rule: at the bottom of the shallow valley the slope is zero, so the step is zero. It needs something plain descent doesn't have: noise or momentum.
* "Is the shallow valley 'wrong'?" → on a training loss, lower is better by definition, but the lowest training loss isn't always the best test error (reel 21). Wide flat valleys often generalise better than deep narrow ones.
* "How do people deal with this?" → random restarts, noise from small batches, momentum, and good initialisation. Each gets its own reel.
* "Is this what happens in a real network?" → much less than the picture suggests: in very high dimensions most stationary points are saddles, not traps. That reel is coming.
