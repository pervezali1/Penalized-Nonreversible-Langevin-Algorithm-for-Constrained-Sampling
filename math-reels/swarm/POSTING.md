# Posting kit — *10,000 AIs. This time, they talk to each other.* (dark theme, no narration, 20 s)

## Files

* Reel: `output/swarm_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.5 LUFS, true peak −1.3 dBTP.
* Cover: `output/cover.png` — the talking run near the end, every particle in the deep well.
* Simulation: `swarmsim.py` (positions and facts), `netpso.py` → `netpso_results.json` (the twist in the pinned comment).

## What's on screen (all of it is real)

The landscape, the 10,000 random starting points and the first run are exactly reel 04's (`../descent/landscape.py`):
a bowl plus nine Gaussian wells, one deep ("the best answer") and eight shallow traps. "Found the best answer" means a
final loss within 0.25 of the global minimum, which only the deep well satisfies.

| Run | How each AI moves | Reach the best answer |
| --- | --- | --- |
| 1, alone | plain gradient descent: step 0.002 down the slope, 450 steps | **34%** (3,368 / 10,000) |
| 2, talking | particle swarm optimisation from the same 10,000 starts, 80 steps | **100%** (10,000 / 10,000) |

Run 2 is particle swarm optimisation (Kennedy and Eberhart, 1995). Each step, every particle keeps 0.72 of its velocity
and is pulled toward two places, each with strength 1.49 times a fresh random number in [0, 1]: the best spot it has
found itself, and the best spot any of the 10,000 has found (Clerc's constriction values). Speed is capped at 0.05
per step. The particles never use the slope, only the heights where they land.

* Half the swarm is in the deep well by step 15, 90% by step 26, all 10,000 by step 44.
* The amber lines at the start of run 2 join 240 of the particles (a fixed random sample) to the swarm's best spot;
  the ring marks that spot. Both fade in over the first 0.25 s of the run and fade out between 2.6 s and 4 s.
* The step counter and the percentage are computed per frame from the recorded positions.

## Honesty notes

* This is a toy 2-D landscape built to have one deep well and eight traps. 228 of the 10,000 random starts (2.3%)
  already sit in the deep well, so the swarm's best spot is in the right place from the first step and everyone
  only has to follow. With fewer particles, or a trap that looks better from far away, a swarm can pile into the
  wrong spot together ("premature convergence").
* Cost per step is comparable: each step, run 1 evaluates the slope at 10,000 points and run 2 the height at
  10,000 points. Run 1 is shown over 450 steps, run 2 over 80.
* Reel 04 showed the fair comparison for gradient descent: with momentum, the same 10,000 reach 85%.
* The swarm's pulls are random (seed 64). Checked with seeds 1–7: always 50% by step 15 and 90% by step 26, all
  10,000 by step 41 to 46, 100% every time.
* **The twist (pinned comment):** on a real network the swarm loses badly. `netpso.py` runs the same swarm (10,000
  particles, inertia 0.72, pulls 1.49) on the 2 → 8 → 8 → 1 ReLU network of reels 31 and 33 (105 numbers) on the same
  400-point spiral, minimising the same cross-entropy loss. After 300 steps (3,010,000 runs of the network) its best
  network gets **79.5%** right. Gradient descent (Adam) gets **95.5%** after 1,500 steps, 3,000 passes counting the
  forward and backward pass. Best of 1,000,000 random networks: 69%.

## Caption (paste as-is)

```
10,000 AIs. This time, they talk to each other.

First, alone: each one rolls downhill from a random start (plain gradient descent). Only 34% reach the best answer. The rest get stuck in traps.

Same hill, same 10,000 starts. Now every AI can see the best spot any of them has found, and heads toward it. Half of them are there by step 15. All 10,000 by step 44.

They never felt the slope. They only compared heights.

So why don't we train real AI this way? I ran the same swarm on a real neural network with 105 numbers. After 3 million runs, its best got 79.5% right. Gradient descent got 95.5% after 3,000. Talking wins with 2 numbers. The slope wins with 105, and real AIs have billions.

Smart alone or average together? 👇

(Particle swarm optimisation in Python: 10,000 particles, inertia 0.72, pulls 1.49, same landscape and same starting points as the gradient descent run.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #ai #optimization #swarmintelligence #datascience
```

Exactly five hashtags.

## YouTube

Title:

```
10,000 AIs. This Time They Talk to Each Other. #shorts
```

Description:

```
10,000 AIs search the same landscape from the same 10,000 random starts, twice.

Alone (plain gradient descent, step 0.002, 450 steps): 34% (3,368) reach the best answer; the rest get stuck in shallow traps.

Talking (particle swarm optimisation: each AI keeps 0.72 of its speed and is pulled toward its own best spot and the best spot any of them has found, strength 1.49, speed cap 0.05): half are there by step 15, all 10,000 by step 44. The swarm never uses the slope, only heights.

The catch: on a real neural network (2 → 8 → 8 → 1, 105 numbers, two-arm spiral) the same swarm reaches 79.5% after 3,010,000 runs of the network; gradient descent reaches 95.5% after 3,000.

#machinelearning #ai #optimization #shorts
```

## Pinned comment

```
Why don't we train real networks this way? I tried: the same 10,000-particle swarm on a small real network (105 numbers, two spiral arms). After 300 steps, 3,010,000 runs of the network, its best was 79.5% right. Gradient descent: 95.5% after 1,500 steps (3,000 passes, forward + backward). Best of 1,000,000 random networks: 69%. The slope tells every one of the 105 numbers which way to move at once; the swarm has to find out by trial. In 2-D that's cheap. With 105 numbers it isn't, and real models have billions. Swarms still earn their keep where there is no slope at all.
```

## Audio

Keep the built-in track (reel 04's sound, driven by this swarm): a pad whose pitch falls with the average loss, a
flow texture and tremolo that follow the particles' speed, a pop for every batch of particles that reaches the best
answer, bells at the two reveals (7.6 s and 16.6 s).

## Replies to expect

* "That's just PSO" → yes, particle swarm optimisation, named in the caption; the settings are the textbook ones.
* "The swarm cheated, some started in the well" → true (2.3%), see the honesty notes; that's what sharing buys you:
  one lucky start is enough for everyone.
* "Gradient descent with momentum would do better" → reel 04: 85%.
* "Why not use swarms for AI then?" → the pinned comment: 79.5% vs 95.5% on 105 numbers.
