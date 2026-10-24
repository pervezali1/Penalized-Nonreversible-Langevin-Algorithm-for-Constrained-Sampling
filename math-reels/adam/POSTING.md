# Posting kit — *You said: "just use Adam."*

## Files

* Reel: `output/adam_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the Adam reveal.

## What's on screen (all of it is real)

Same landscape and the same 10,000 random starts as reel 04, with the step size set
to 0.1 (50× the one used in reel 04):

| Run | Optimiser | Step size | Reach the best answer |
| --- | --- | --- | --- |
| 1 | plain gradient descent | 0.1 | 42%, and never settles (the swarm keeps bouncing inside every well) |
| 2 | Adam (β₁ 0.9, β₂ 0.999) | 0.1 | **83%**, settled by step ~25 |
| caveat (measured, not animated) | Adam | 0.002 (reel 04's step) | 32% — no better than plain descent's 34% |

Adam's update is `p ← p − lr · m̂ / (√v̂ + ε)`, where m̂ is the running mean of the
gradient and v̂ the running mean of its square. Dividing by √v̂ makes the step roughly
`lr` in size whatever the slope: big where the gradient has been steady, tiny where it
has been noisy. That is why it can take a 0.1 step calmly where plain descent bounces,
and why, at a tiny step size, it gains nothing.

## Caption (paste as-is)

```
Last reel, everyone said "just use Adam." So I did. Same 10,000 random starts, same hill.

First, plain gradient descent with a big step (0.1): it overshoots inside every well and never settles. 42% are near the best answer, but none of them are still.

Then Adam, same step size: calm, straight in, 83% by step 25.

Here's the part nobody says: Adam at the tiny step size from last time gets 32%, no better than plain descent. Adam doesn't find better answers. It resizes every step (small where the slope has been wild, big where it's been calm), so you can afford a step 50× bigger, and the big step is what clears the traps.

Which optimiser do you reach for first?

(Adam with β₁ = 0.9, β₂ = 0.999, ε = 1e-8, bias-corrected, 150 steps; plain GD 120 steps. Every sound follows the particles.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #adam #optimizer #deeplearning #datascience
```

Short alternative:

```
"Just use Adam." Same hill, same 10,000 starts, step 0.1: plain descent bounces forever (42%), Adam settles in 25 steps (83%). But Adam at the old tiny step: 32%. The win is the big step.

Follow for more.

#machinelearning #adam #optimizer #deeplearning #datascience
```

## Pinned comment

```
For the optimiser crowd: Adam = momentum on the gradient (m) divided by an RMS of recent gradients (√v), both bias-corrected. The division normalises step length per parameter, which is the whole trick here: it tolerates a step size that makes plain GD unstable in the wells (curvature ~19, so GD needs step < 0.1). It is NOT a local-minimum escape mechanism; at a small step it converges to the nearest well like anything else. In real networks the per-parameter scaling matters more than this 2-D picture shows (sparse gradients, ravines). Next: AdamW and why weight decay was broken inside Adam. Want it?
```

## Audio

Keep the built-in track, **"the sound of Adam"**: the pad's pitch follows the average loss,
a harsh jitter texture under the bouncing plain-descent run, a smooth flow under Adam,
a pop for every 100 particles that reach the answer.

## Replies to expect

* "You cherry-picked the step size" → both runs use 0.1; the caveat run uses reel 04's 0.002; plain GD at 0.002 was reel 04 (34%). All four numbers are in the caption.
* "Adam's eps/β₂ matter" → defaults from the paper; feel free to request a sweep.
* "This landscape is 2-D" → yes, it's the intuition reel; the pinned comment says what changes in high dimensions.
