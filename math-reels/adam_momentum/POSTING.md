# Posting kit — *Same 10,000 AIs. Now Adam. Momentum: OFF / ON*

## Files

* Reel: `output/adam_momentum_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the momentum-on reveal.

## What's on screen (all of it is real)

Reel 04's structure (run → fly back to the same starts → run again) with Adam doing the
stepping both times. Adam's momentum is its β₁, the running mean of the gradient:

| Run | Optimiser | Step size | Reach the best answer |
| --- | --- | --- | --- |
| 1 | Adam, β₁ = 0 (momentum off), β₂ = 0.999 | 0.05 | **48%** |
| 2 | Adam, β₁ = 0.95 (momentum on), β₂ = 0.999 | 0.05 | **79%** |

Same landscape, same seed, same 10,000 random starts as reels 04, 05 and 07.
For reference (measured, not animated): with Adam's default β₁ = 0.9 the second run gets
70%; at step 0.1 it is 60% → 85%; at reel 04's tiny step 0.002 both are ~31%, because at
that step nothing can cross a trap.

## Caption (paste as-is)

```
Reel 4 with Adam instead of plain gradient descent. Same 10,000 random starts, same hill, Adam's momentum switched off, then on.

Momentum off (β₁ = 0): Adam still sizes every step sensibly, but it stops in whatever dip it reaches first. 48% find the best answer.

Momentum on (β₁ = 0.95): it remembers where it was heading for longer, coasts through the shallow dips, and 79% make it. (At Adam's default 0.9 it's 70%; the longer the memory, the more traps it clears.)

Adam's momentum is the same idea as plain momentum: keep a running average of the direction you've been moving. Last reel plain descent + momentum got 85%; the ingredient that clears traps is the memory, not the optimiser's name.

Which do you run with, default β₁ or something else?

(Adam: β₂ = 0.999, ε = 1e-8, bias-corrected, 150 steps, step size 0.05. Every sound follows the particles.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #adam #momentum #deeplearning #datascience
```

Short alternative:

```
Reel 4, but with Adam. Momentum off: 48% find the best answer. Momentum on (β₁ = 0.95): 79%. Same starts, same hill. Momentum is the ingredient.

Follow for more.

#machinelearning #adam #momentum #deeplearning #datascience
```

## Pinned comment

```
Why the gap: with β₁ = 0 Adam is basically RMSprop; each step points along the current gradient, normalised, so inside a shallow well it just walks to the bottom. With β₁ = 0.95 the step follows a running average of the last ~20 gradients, so a particle arriving at a dip still carries the downhill direction it came in with and can roll out the far side. At tiny step sizes neither version can cross a trap (both ~31% at 0.002); at 0.1 it's 60% vs 85%; with the default β₁ = 0.9 at this step it's 70%. Next: AdamW, and why weight decay inside Adam was quietly broken for years. Want it?
```

## Audio

Keep the built-in track, **"the sound of Adam's momentum"**: the pad's pitch follows the
average loss, the texture follows particle speed, a pop for every 100 particles that
reach the answer, bells at the two reveals.

## Replies to expect

* "β₁ = 0 isn't Adam" → right, it's RMSprop with bias correction; that is exactly the point of switching it off.
* "Different step size, different story" → yes: 0.002 → ~31/31, 0.05 → 48/79 (48/70 at β₁ = 0.9), 0.1 → 60/85; all measured, all in this kit.
* "2-D toy" → yes, the intuition reel; the pinned comment says what changes in high dimensions.
