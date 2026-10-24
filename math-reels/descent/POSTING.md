# Posting kit — *10,000 AIs. Same lesson. Random starts.*

## Files

* Reel: `output/descent_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the momentum reveal ("85% found the best answer").

## What's on screen (all of it is real)

10,000 particles do gradient descent on a 2-D loss landscape with one deep well
("the best answer") and eight shallow traps (local minima), from the same 10,000
random starting points:

| Run | Optimiser | Reach the best answer |
| --- | --- | --- |
| 1 | plain gradient descent, step 0.002 | **34%** (3,370 / 10,000) |
| 2 | same, with momentum 0.97 | **85%** (8,480 / 10,000) |

"Found the best answer" = final loss within 0.25 of the global minimum, which only
the deep well satisfies. The landscape is a bowl plus Gaussian wells; the contour
lines and shading are the real function. The counter is computed per frame.

## Caption (paste as-is)

```
10,000 copies of the same AI, learning the same lesson from 10,000 random starting points. Every one of them just rolls downhill (that's gradient descent). Only 34% reach the best answer; the rest get stuck in shallow traps.

Then the same 10,000, from the same starts, with one change: momentum. They keep some of their speed, overshoot the shallow traps, and 85% make it.

That's why almost every modern optimiser (SGD+momentum, Adam) carries momentum: a plain downhill roll stops in the first dip it finds.

Where would YOU have started?

(Simulated in Python: a bowl plus 9 Gaussian wells, 450 steps, same seed for both runs. Colours are the real loss; every sound follows the particles.)

#machinelearning #gradientdescent #ai #deeplearning #datascience
```

Exactly five hashtags.

Short alternative:

```
Same AI, same lesson, 10,000 random starts. Plain gradient descent: 34% find the best answer. Add momentum: 85%.

Keep moving.

#machinelearning #gradientdescent #ai #deeplearning #datascience
```

## Pinned comment

```
For the optimisation folks: yes, this is a toy 2-D landscape built to show the effect, and momentum is not a guarantee (on some landscapes it overshoots the good answer too). In high dimensions the bigger win from momentum is speed through flat valleys and ravines rather than escaping sharp local minima. The point of the reel is the intuition: an optimiser that only ever goes downhill stops at the first dip. Next: the learning rate that's too big — watch the whole swarm explode. Want it?
```

## Audio

Keep the built-in track, named **"the sound of 10,000 AIs learning"**: a four-voice
pad whose pitch falls with the average loss, a flowing noise texture that follows the
particles' speed, a pop for every 100 particles that reach the best answer, a tremolo
on the momentum run, bells at the two reveals.

## Replies to expect

* "Local minima aren't the real problem in deep learning, saddle points are" → correct for high dimensions; the pinned comment says so.
* "Momentum can also overshoot the global minimum" → yes; that's why it's paired with a decaying learning rate in practice.
* "Why 0.97?" → chosen so the effect is visible at this step size; with step 0.02 and momentum 0.9 the same landscape gives 34% vs 85% too.

## Series ideas

1. "The learning rate that's too big" — the swarm explodes (chaos tie-in).
2. "Adam vs SGD" — same swarm, two optimisers, split screen.
3. "What a saddle point looks like" — 10,000 particles hesitating on a ridge.
