# Posting kit — *Teaching an AI: how big a step?*

## Files

* Reel: `output/lr_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the explosion.

## What's on screen (all of it is real)

The same 10,000 particles and loss landscape as the gradient-descent reel, run three
times from the same starts with plain gradient descent and only the step size changed:

| Step size | What happens | Reach the best answer |
| --- | --- | --- |
| 0.002 | settles into the wells | 34% |
| 0.3 (150× bigger) | overshoots every well, never settles | 8% |
| 1.7 (850× bigger) | diverges: positions grow ~10% per step, loss → ∞ | 0% |

The bowl of the landscape has curvature 1.2, so plain gradient descent is unstable
once the step exceeds 2/1.2 ≈ 1.67; 1.7 is just past that edge, which is why the
blow-up is gradual rather than instant. The "average loss" counter is the real mean
loss above the global minimum; it reads ∞ once it passes 100.

## Caption (paste as-is)

```
Every AI learns by taking steps downhill. The size of each step is a single number, and it decides everything.

Step 0.002: the swarm settles and learns.
Step 0.3: it overshoots every answer and bounces forever.
Step 1.7: it explodes. Loss → infinity.

That number is the learning rate. Too small and training takes forever; too big and the model turns into noise. It's the first thing anyone training a neural network tunes, and the reason "it diverged" is the most common failure in the field.

Which one would you have picked?

(Same 10,000 starts and landscape as my last reel; plain gradient descent, only the step size changed. Every sound follows the particles.)

#machinelearning #learningrate #ai #deeplearning #datascience
```

Short alternative:

```
Same AI, three step sizes. 0.002: learns. 0.3: can't settle. 1.7: explodes. That's the learning rate.

#machinelearning #learningrate #ai #deeplearning #datascience
```

## Pinned comment

```
Why 1.7 explodes: on a bowl with curvature c, plain gradient descent only converges if the step size is below 2/c. Here c = 1.2, so the limit is about 1.67. At 1.7 each step overshoots by more than it fell, positions grow ~10% per step, and the loss runs off to infinity. Real training has the same cliff, just in millions of dimensions, which is why learning-rate schedules, warm-up and Adam exist. Next: Adam vs plain SGD on this exact landscape. Want it?
```

## Audio

Keep the built-in track, **"the sound of a learning rate"**: the pad's pitch follows the
average loss, the texture follows particle speed (harsh in the 0.3 run), a riser leads
into the detonation at the moment the loss passes 1,000.

## Replies to expect

* "Nobody uses a fixed learning rate" → right, schedules and adaptive optimisers exist precisely because of this; it's the intuition reel.
* "Loss can't be infinite" → in float64 it's 10¹² by the end of the run, and ∞ is what you see in a real training log once it overflows (NaN).
