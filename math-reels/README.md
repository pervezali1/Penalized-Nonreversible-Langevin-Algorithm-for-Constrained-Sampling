# math-reels

Programmatically rendered vertical math animations for Instagram Reels
(1080×1920, 60 fps, H.264 + sonified audio). No manim, no LaTeX, no GPU:
numpy + numba simulate, pycairo draws vectors, OpenCV does bloom, ffmpeg encodes.

## Reel 01 — *10,000 Pendulums. Then I Rewound Time.*

One white double pendulum is secretly 10,000 pendulums stacked on top of each
other, all released within **0.000001°**. A live ×10,000,000 lens shows the hidden
rainbow needle; a counter watches the zeros vanish; at **8.28 s** on the timer the
gap passes 180° and the white line tears into a 10,000-rod rainbow. Then time runs
backwards and every single one comes home, landing exactly on frame 0 so the
reel loops without a seam.

Every number on screen is recomputed from the simulation and asserted by the
build; every sound is generated from the pendulums' motion.

| Output | Where |
| --- | --- |
| Reel | `pendulum/output/pendulums_reel.mp4` (18.000 s, 1080×1920, 60 fps) |
| Cover | `pendulum/output/cover.png` (frame 504, the CHAOS frame) |
| Contact sheet | `pendulum/output/contact_sheet.png` |
| Posting kit | [`pendulum/POSTING.md`](pendulum/POSTING.md) — caption, hashtags, pinned comment, timing, series plan |

## Reel 02 — *This AI has 2 neurons.*

Three real neural networks (2, 8 and 64 hidden neurons) are trained live in the
render on the same two-arm spiral. The whole background is the network's decision
field; hollow rings are the points it still gets wrong; a small diagram draws the
real weights. 2 neurons can't bend enough and stall at 65%, 8 reach 82%, 64 find
the spiral and hit 100%. Payoff: *Your brain has 86,000,000,000.* The audio is a
dissonant cluster that resolves into a major chord as accuracy rises.

| Output | Where |
| --- | --- |
| Reel | `neurons/output/neurons_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `neurons/output/cover.png` |
| Posting kit | [`neurons/POSTING.md`](neurons/POSTING.md) |

Build: `cd neurons && ../.venv/bin/python build.py /path/to/workdir` (~10 s to train, ~8 min to render).

## Reel 03 — *How many people until two share a birthday?*

One random room fills to 23 people on a 365-day ring while the exact probability
climbs to 50.7%; 10,000 simulated rooms of 23 light up a grid (5,067 had a match);
the 253 pairs are drawn as chords; the room fills to 70 (99.9%).

| Output | Where |
| --- | --- |
| Reel | `birthday/output/birthday_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `birthday/output/cover.png` |
| Posting kit | [`birthday/POSTING.md`](birthday/POSTING.md) |

Build: `cd birthday && ../.venv/bin/python build.py /path/to/workdir` (~7 min to render).

## Reel 04 — *10,000 AIs. Same lesson. Random starts.*

10,000 particles do gradient descent on a shaded loss landscape with one deep well and
eight shallow traps. Plain descent: 34% reach the best answer. Same starts with
momentum 0.97: 85%. Colours follow the real loss; the counter is computed per frame.

| Output | Where |
| --- | --- |
| Reel | `descent/output/descent_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `descent/output/cover.png` |
| Posting kit | [`descent/POSTING.md`](descent/POSTING.md) |

Build: `cd descent && ../.venv/bin/python build.py /path/to/workdir` (~5 min to render).

## Reel 05 — *Teaching an AI: how big a step?*

The descent swarm run three times with plain gradient descent and only the step size
changed: 0.002 learns (34%), 0.3 overshoots and never settles (8%), 1.7 diverges
and the loss counter runs to ∞.

| Output | Where |
| --- | --- |
| Reel | `lr/output/lr_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `lr/output/cover.png` |
| Posting kit | [`lr/POSTING.md`](lr/POSTING.md) |

## Reel 06 — *1693. Pepys asks Newton: which bet wins?*

The Newton–Pepys problem: at least one 6 in 6 dice vs two in 12 vs three in 18.
Three rows of real simulated dice roll 10,000 times each while running win rates
converge on 66.5 / 61.9 / 59.7%; exact binomial ticks appear at the reveal.

| Output | Where |
| --- | --- |
| Reel | `pepys/output/pepys_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `pepys/output/cover.png` |
| Posting kit | [`pepys/POSTING.md`](pepys/POSTING.md) |

## Reel 07 — *You said: "just use Adam."*

Reel 04's swarm at step size 0.1: plain descent bounces in every well and never settles
(42% near the answer); Adam at the same step size settles in ~25 steps (83%). The caveat
on screen: Adam at reel 04's tiny step size gets 32%, no better than plain descent.

| Output | Where |
| --- | --- |
| Reel | `adam/output/adam_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `adam/output/cover.png` |
| Posting kit | [`adam/POSTING.md`](adam/POSTING.md) |

## Reel 08 — *Same 10,000 AIs. Now Adam. Momentum: OFF / ON*

Reel 04's structure with Adam doing the stepping: β₁ = 0 (momentum off) reaches the best
answer 48% of the time; β₁ = 0.95 79%, at step size 0.05 from the same starts.

| Output | Where |
| --- | --- |
| Reel | `adam_momentum/output/adam_momentum_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `adam_momentum/output/cover.png` |
| Posting kit | [`adam_momentum/POSTING.md`](adam_momentum/POSTING.md) |

## Reel 06 v2 — *1693. A gambler asks Newton: which bet wins?*

Newton–Pepys redone so the reason is visible: each bet gets a live histogram of the number
of sixes, magenta below the target and cyan at or above it, so the win rate is the cyan
area. The exact binomial outline appears at the reveal and an amber marker shows that the
target is exactly the average (n/6).

| Output | Where |
| --- | --- |
| Reel | `pepys2/output/pepys2_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `pepys2/output/cover.png` |
| Posting kit | [`pepys2/POSTING.md`](pepys2/POSTING.md) |

## Reel 10 — *Can a neural network learn pure noise?*

Reel 2's 64-neuron network trained on the spiral with real labels (train 100%, test 92%),
then on the same points with the labels shuffled: it still reaches 97.8% on training while
the decision field turns into a patchwork, and a held-out test set scores 53%, a coin flip.

| Output | Where |
| --- | --- |
| Reel | `overfit/output/overfit_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `overfit/output/cover.png` |
| Posting kit | [`overfit/POSTING.md`](overfit/POSTING.md) |

## Reel 11 — *Roll a die. Move halfway to a corner.*

The chaos game: dice and three corners accumulate 50,000 points into a Sierpinski triangle
(coloured by the corner each point moved toward); then four slightly different random rules
grow a 200,000-point Barnsley fern, coloured by rule. Sonified by points landing.

| Output | Where |
| --- | --- |
| Reel | `fractal/output/fractal_reel.mp4` (20.000 s, 1080×1920, 60 fps) |
| Cover | `fractal/output/cover.png` |
| Posting kit | [`fractal/POSTING.md`](fractal/POSTING.md) |

## Reel 12 — *3 doors. 1 car. 2 goats. Switch or stay?*

Monty Hall: 10,000 simulated games with STAY and SWITCH side by side, each game filling a
cell of a 100×100 grid (cyan = switch won, magenta = stay won), then the 100-door version
where the host opens 98 goats. Stay 33.7%, switch 66.3%.

| Output | Where |
| --- | --- |
| Reel | `monty/output/monty_reel.mp4` (24.000 s, 1080×1920, 60 fps; opens with a 9 s walkthrough of the rules) |
| Cover | `monty/output/cover.png` |
| Posting kit | [`monty/POSTING.md`](monty/POSTING.md) |

## Reel 13 — *You have $100. Goal: $200.*

Gambler's ruin on roulette red/black: 1,000 bankrolls betting $1 at a time drawn as glowing
walks that the house edge drags to the floor (998 broke, 1 doubles; exact odds 1 in 37,650), then
the same 1,000 betting everything once (496 double). Every bet pays the house its edge.

| Output | Where |
| --- | --- |
| Reel | `ruin/output/ruin_reel.mp4` (22.000 s, 1080×1920, 60 fps) |
| Cover | `ruin/output/cover.png` |
| Posting kit | [`ruin/POSTING.md`](ruin/POSTING.md) |

## Reel 14 — *Lose? Double your bet.*

The Martingale on roulette: one gambler's bankroll climbs $1 a round to $1,364, then ten losses in a
row (bets 1, 2, 4 … 512) wipe the $1,023 stake in ten spins; then 1,000 Martingale players over
5,000 spins drawn as a climbing band that drops into a band of busts (978 busted, all 1,000 together
−$402,270). The math card: 99.84% chance of +$1 a round, 0.16% chance of −$1,023, average −67¢, the
house's 5.3% of everything wagered.

| Output | Where |
| --- | --- |
| Reel | `martingale/output/martingale_reel.mp4` (24.000 s, 1080×1920, 60 fps) |
| Cover | `martingale/output/cover.png` |
| Posting kit | [`martingale/POSTING.md`](martingale/POSTING.md) |

## Reel 15 — *Guess. Slide down the tangent. Repeat.*

Newton's method twice. On the real line, x² − 2 from the guess 1: tangent after tangent while the
camera zooms with the error and the correct digits double, 1, 3, 6, 12, 24, 48 (120-digit decimal
arithmetic, checked against √2). Then z³ = 1 in the complex plane: all 2 million pixels iterated
and coloured by the root they reach, shown iteration by iteration as the borders grow borders, then
a 100,000× zoom onto a point where all three basins meet.

| Output | Where |
| --- | --- |
| Reel | `newton/output/newton_reel.mp4` (24.000 s, 1080×1920, 60 fps) |
| Cover | `newton/output/cover.png` |
| Posting kit | [`newton/POSTING.md`](newton/POSTING.md) |

## Reel 16 — *Throw darts at random. That's π.*

Monte Carlo π: one million random darts rain into a square with a circle inside while the estimate
4 × inside ÷ total ticks live with its correct digits in green (three of them at a million). Then
300 independent dart-throwers drawn side by side form a funnel that narrows exactly like the 95%
band π ± 1.96 × 1.642/√N: each extra digit costs 100× more darts, ten digits would take 10¹⁹.

| Output | Where |
| --- | --- |
| Reel | `montecarlo/output/montecarlo_reel.mp4` (23.000 s, 1080×1920, 60 fps) |
| Cover | `montecarlo/output/cover.png` |
| Posting kit | [`montecarlo/POSTING.md`](montecarlo/POSTING.md) |

## Reel 17 — *Drop a needle on floorboards. That's π.*

Buffon's needle: 100,000 needles as long as the gap rain onto floorboards while π ≈ 2 × needles ÷
crossings ticks live (two digits at the end, as 1/√N predicts). Then why π appears: one needle
rotates while a graph fills with its crossing chance |sin θ|, whose average over all angles is 2/π.
Then Lazzarini's 1901 claim of six digits from 3,408 needles: one crossing more or fewer gives
3.1433 or 3.1399, and 1,000 simulated repeats of his experiment hit his 1,808 exactly 15 times.

| Output | Where |
| --- | --- |
| Reel | `buffon/output/buffon_reel.mp4` (25.000 s, 1080×1920, 60 fps) |
| Cover | `buffon/output/cover.png` |
| Posting kit | [`buffon/POSTING.md`](buffon/POSTING.md) |

## Reel 18 — *Same AI. Same data. Three activations.*

Three copies of reel 02's 2 → 32 → 32 → 1 network, identical starting weights, same Adam and the
same 1,000 steps, differing only in the activation: ReLU, tanh, sigmoid. Three live decision fields
side by side with an accuracy under each and a race chart beneath. ReLU hits 100% at step 470,
tanh ends at 99.8%, sigmoid at 71%: the vanishing gradient, with the reason on screen (sigmoid's
slope is at most ¼ per layer, ReLU's is 1).

| Output | Where |
| --- | --- |
| Reel | `activations/output/activations_reel.mp4` (23.000 s, 1080×1920, 60 fps) |
| Cover | `activations/output/cover.png` |
| Posting kit | [`activations/POSTING.md`](activations/POSTING.md) |

Build: `cd activations && ../.venv/bin/python actsim.py && ../.venv/bin/python build.py /path/to/workdir`
(the per-step weights file is regenerated, not committed).

## Reel 19 — *What if every weight starts at zero?*

Three copies of reel 18's ReLU network on the same spiral, same Adam, same 1,000 steps, differing
only in the starting weights: random, all zeros, all 0.1. Live decision fields, an accuracy under
each, and under that a board of the 32 first-layer neurons' response maps: 32 different tiles vs 32
dark tiles vs 32 identical tiles moving in lockstep, with the distinct-neuron count computed from the
weights at every step (32 / 1 / 1). Random 100% at step 470, zeros 50%, all-0.1 62%. Identical
neurons get identical gradients, so they stay identical forever.

| Output | Where |
| --- | --- |
| Reel | `init/output/init_reel.mp4` (23.000 s, 1080×1920, 60 fps) |
| Cover | `init/output/cover.png` |
| Posting kit | [`init/POSTING.md`](init/POSTING.md) |

Build: `cd init && ../.venv/bin/python initsim.py && ../.venv/bin/python build.py /path/to/workdir`
(the per-step weights file is regenerated, not committed).

## Reel 20 — *How random should the start be?*

Three copies of an 8-hidden-layer ReLU network on the spiral with plain gradient descent, same
seed, differing only in the spread of the random starting weights: std 0.01, He initialisation
(√(2/32) = 0.25), std 1.0. Under each decision field a bar per layer shows the activation std as the
signal passes through: a staircase down to 10⁻¹³, a flat row at the input level, a row that hits
7,000 before training and overflows at step 2. Final accuracies 57% / 98% / 50%.

| Output | Where |
| --- | --- |
| Reel | `scale/output/scale_reel.mp4` (23.000 s, 1080×1920, 60 fps) |
| Cover | `scale/output/cover.png` |
| Posting kit | [`scale/POSTING.md`](scale/POSTING.md) |

Build: `cd scale && ../.venv/bin/python scalesim.py && ../.venv/bin/python build.py /path/to/workdir`
(the per-step fields file is regenerated, not committed).

## Reel 21 — *I flipped 83 of 400 labels. Which network believes them?*

Three copies of a wide ReLU network (2 → 256 → 256 → 1) on the spiral with 83 labels flipped, same
start, same Adam, same 3,000 steps: plain, dropout 0.5, weight decay 1e-3. Flipped labels are drawn
as diamonds, filled when the network believes the lie, with a lies-believed count and clean-test
score under each field and a race chart of clean-test accuracy. Plain believes 76 of 83 and ends at
77% on clean data (after peaking at 83%); dropout 32 lies and 83%; weight decay 28 lies and 82%.

| Output | Where |
| --- | --- |
| Reel | `dropout/output/dropout_reel.mp4` (23.000 s, 1080×1920, 60 fps) |
| Cover | `dropout/output/cover.png` |
| Posting kit | [`dropout/POSTING.md`](dropout/POSTING.md) |

Build: `cd dropout && ../.venv/bin/python regsim.py && ../.venv/bin/python build.py /path/to/workdir`
(the per-step fields file is regenerated, not committed).

## Reel 22 — *3 doors. 1 car. 2 goats.* (Monty Hall v2, the explanatory cut)

Four beats, one argument. Hook: three doors, you pick 1, the host opens a goat, switch? Proof: the
three possible worlds (car behind 1, 2 or 3) side by side; the host opens a goat in each; switching
wins in two of them, STAY 1/3 / SWITCH 2/3. Scoreboard: 10,000 games played both ways, live bars,
33.7% vs 66.3%. Kicker: 100 doors, the host opens 98 goats one by one and skips one. Closing line:
the host's choice is information.

| Output | Where |
| --- | --- |
| Reel | `monty2/output/monty2_reel.mp4` (25.000 s, 1080×1920, 60 fps) |
| Cover | `monty2/output/cover.png` |
| Posting kit | [`monty2/POSTING.md`](monty2/POSTING.md) |

## Reel 23 — *Gradient descent, in the simplest way I can.* (white background)

The definitional one, in a light style designed by a judged panel and reviewed frame by frame: a
shaded ceramic cutaway bowl with contour lines and a soft ground shadow, a real sphere with a
contact shadow, and at every step one vermilion arrow, the slope under the ball pointing downhill,
as long as the step it is about to take, with a ghost circle at the landing spot. Walked arrows stay
behind as a shrinking chain; leader-line labels name the slope and the steps; a magnifier loupe
follows the ball from step 3 (×4 to ×12) so the arrow stays readable as the real steps shrink.
Readouts for step, height and slope; axes labelled weight 1, weight 2 and error at the close. Slow
30 s pacing, gentle soundtrack. Height 4.80 → 0.044 over 24 steps at
learning rate 0.09.

| Output | Where |
| --- | --- |
| Reel | `gdsimple/output/gdsimple_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `gdsimple/output/cover.png` |
| Posting kit | [`gdsimple/POSTING.md`](gdsimple/POSTING.md) |

## Reel 24 — *The step size.* (white style, three balls)

The learning rate on the cut hill of reel 23: three balls, same rule, step sizes 0.02, 0.2 and 2
times the slope (each ten times the last), each in its own colour with its number riding above it. Too
small keeps 98% of its distance each step and is still at height 1.33 after 24 steps; just right is
below 0.01 by step 14 and under 0.001 by step 24; too big jumps the whole valley to the mirror point at
the same height (4.50 → 4.50 → 4.51 → 4.91: along the valley 2 × slope is exactly the factor −1 knife
edge) while its 0.002 sideways offset grows 6.2× a step (factor 1 − 2·1.8·2) and throws it off the hill
on step 4. The ball fades out early on that slide, its readout says "left the hill" from the exit step,
its trail stays. The cue lines are generated from the run (knife-edge wording only when the along-valley
factor is exactly −1; the just-right line reports whether it settled). One sound per ball per landing:
a pluck walking down a scale, a quiet tick, a bouncy tone on one pitch, and a whoosh as the big one goes.

| Output | Where |
| --- | --- |
| Reel | `lr3/output/lr3_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `lr3/output/cover.png` |
| Posting kit | [`lr3/POSTING.md`](lr3/POSTING.md) |

## Reel 25 — *Momentum.* (white style, two balls)

Momentum on the cut hill of reels 23 and 24: two balls, the same step size (0.03) and start (4.80).
Plain descent steps by step size × slope from nothing each time and is at 0.94 after 24 steps;
momentum keeps 80% of its last step and adds the slope, so its steps grow while the slope agrees
(3.3× plain's by step 5), it is at the bottom by step 15 (0.0004), overshoots to 0.040 at step 20 and
settles to 0.026. During its arrow phases the main view shows the blue memory arrow and the vermilion
slope arrow it is adding; an inset at lower right shows the same two screen vectors magnified ×4 with
a ring where the step lands. All cue numbers come from the run (longest step, overshoot, heights).

| Output | Where |
| --- | --- |
| Reel | `momentumw/output/momentumw_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `momentumw/output/cover.png` |
| Posting kit | [`momentumw/POSTING.md`](momentumw/POSTING.md) |

## Reel 26 — *Adam.* (white style, three acts)

Adam's one real property, shown honestly: the step does not depend on how steep the hill is. Three
acts of 12 steps on the cut hill, two balls each, the same number 0.25 for both. Act 1, the hill as
before: plain descent 4.80 → 0.004, Adam 4.80 → 0.025 (plain wins). Act 2, the hill × 100: plain's
first step is about 14 bowl-lengths and it leaves on step 1; Adam walks the identical path (480 → 2.53, same
landings to the last digit). Act 3, the hill ÷ 100: plain drops 8% in 12 steps; Adam walks the identical
path again. Plain descent is exact; Adam is the published update with defaults β₁ = 0.9, β₂ = 0.999,
ε = 1e-8 and bias correction. The far exit is drawn towards a point 6 units out so the arrow leaves the
picture cleanly; the readout keeps the true heights.

| Output | Where |
| --- | --- |
| Reel | `adamw/output/adamw_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `adamw/output/cover.png` |
| Posting kit | [`adamw/POSTING.md`](adamw/POSTING.md) |

## Reel 27 — *The noisy slope.* (white style, mini-batches)

The hill is the real mean squared error of a two-weight least-squares fit over 1,000 data points,
built so that average is exactly the bowl of reels 23–26 plus a floor of 0.10 from the labels' noise
(`batchw/batchsim.py`: inputs whitened to the exact second moment, label noise orthogonal to the inputs
and the constant). Three balls, same rule (slope × 0.2) and start; before each step they measure the
slope on 4, 32 or all 1,000 points (random samples, fixed seeds). After 24 steps: 0.111 (still
bouncing, 0.119 ± 0.019 over steps 16–24), 0.101 (a wobble), 0.100 (still). The 4-example ball's
measured slope is 51° off the true one on average; a grey arrow shows the true slope beside the
measured one. Cues state that the expected path is the smooth path, that nothing can go below the
floor, and the cost: 96 vs 768 vs 24,000 examples looked at.

| Output | Where |
| --- | --- |
| Reel | `batchw/output/batchw_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `batchw/output/cover.png` |
| Posting kit | [`batchw/POSTING.md`](batchw/POSTING.md) |

## Reel 28 — *Two valleys.* (white style, four balls, plain descent)

A hill with two bottoms, rendered in the reel-23 style by a general-surface ray-marcher
(`valleys/hillgen.py`, numba, with the same camera, shading, contours, cut face and strokes; the rim is a
true level-set contour): a double well along the long axis, f = 2.2·((u/2)² − 1)² + 0.6u + c + 1.8v²,
ridge 3.52, shallow bottom 2.39, deep bottom 0. Four balls, plain gradient descent with step size 0.1,
24 steps of 0.62 s, four starts along the back wall: A (u = −1.1) and B (0.1 left of the ridge) reach
the deep bottom (0.000); C (0.1 right of the ridge) and D (u = 1.3) settle in the shallow valley (2.395,
2.394). B and C start 0.2 apart and are in different valleys by step 3. Background: a faint tint with a
light grid.

| Output | Where |
| --- | --- |
| Reel | `valleys/output/valleys_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `valleys/output/cover.png` |
| Posting kit | [`valleys/POSTING.md`](valleys/POSTING.md) |

## Reel 29 — *The wrong valley, shaken.* (white style, batch sizes on the two-valley hill)

The hill of reel 28 as the average error over 1,000 examples, each example's error being the hill plus a
tilt with the tilts summing to zero (`valleys2/tiltsim.py`), so a batch of B examples feels the exact
slope plus real sampling noise of size 7/√B. Ball C (all 1,000) sits at the shallow bottom with slope
exactly zero and never moves; three groups of eight copies measure the slope on 4, 8 or 32 examples a
step for 140 steps of 0.1 s (kicks 0.35, 0.25, 0.12 against a ridge 1.13 high), with a live count of
how many are over the ridge: 5 of 8, 1 of 8, 0 of 8 (first crossings at steps 10 and 30), matching 60%,
12% and 0% over 500 random runs each (stated on screen). The eight shown per group are the first runs
that stay below the rim of the drawing. Rendered by `valleys/hillgen.py` on the tinted grid.

| Output | Where |
| --- | --- |
| Reel | `valleys2/output/valleys2_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `valleys2/output/cover.png` |
| Posting kit | [`valleys2/POSTING.md`](valleys2/POSTING.md) |

## Reel 30 — *The saddle.* (white style, one ball, the maths on screen, 40 s)

The two-valley hill with no tilt, f = 2.2·((u/2)² − 1)² + 1.8v²: equal valleys at 0 and a pass at 2.2
whose top is a true saddle point (curvature −2.2 along it, +3.6 across it). One ball dropped 0.01 off
the line through the pass, step size 0.1, 40 steps of 0.5 s: on the pass by step 6, 14 steps parked
while its offset multiplies by 1.22 a step (1 − 0.1·(−2.2), doubling every 3.5), off the pass at step
20, bottom at step 32, height 0.000 after 40. A second readout shows the distance from the saddle line,
and a maths card at the bottom of the frame carries each stage: the hill and the saddle test, the update
rule, the per-step multiplication with the live offset and its next value, the fall, and the
n-dimensional counting argument (one flat point in 2ⁿ is a bottom). Rendered by `valleys/hillgen.py` on
the tinted grid; 2,400 frames.

| Output | Where |
| --- | --- |
| Reel | `saddle/output/saddle_reel.mp4` (40.000 s, 1080×1920, 60 fps) |
| Cover | `saddle/output/cover.png` |
| Posting kit | [`saddle/POSTING.md`](saddle/POSTING.md) |

## Reel 31 — *A neural network, one neuron at a time.* (dark theme, 48 s)

The two-arm spiral of reel 2, and five networks trained from scratch on it with ReLU neurons and Adam
(`nnbuild/nnsim.py`, median of six seeds each): 1, 2, 3 and 8 neurons in one layer, then two layers
of 8: 62%, 62%, 67%, 81%, 96%. Each stage trains live on screen from its random start: the diagram grows
on the left (edges coloured and thickened by the live first-layer weights), the output field trains on
the right with the decision boundary and each first-layer neuron's fold line w·x + b = 0 dashed over
it, and the readouts count neurons, accuracy and steps (no formulas on screen, by request). The story: the activation makes the fold (one neuron, one straight crease), a layer adds folds
(straight creases), a second layer folds the folded field (bent creases; a ReLU net stays piecewise
linear, so the on-screen "curve" is many short straight pieces). 2,880 frames on the night-blue gradient of reels 1–22, by request.

| Output | Where |
| --- | --- |
| Reel | `nnbuild/output/nnbuild_reel.mp4` (48.000 s, 1080×1920, 60 fps) |
| Cover | `nnbuild/output/cover.png` |
| Posting kit | [`nnbuild/POSTING.md`](nnbuild/POSTING.md) |

## Reel 32 — *What is each neuron looking at?* (dark theme, no formulas on screen, 45 s)

The trained 8 + 8 network of reel 31 taken apart neuron by neuron: every tile is one neuron's actual
activation over the input plane from the saved weights. Layer 1: eight straight folds with their fold
lines, a highlight walking them. Layer 2: eight folds of weighted sums of those, the pieces curving.
The output: the layer-2 tiles with their output weights, and the big field assembled one field at
a time in the greedy order that helps the accuracy most (57, 65, 70, 72, 70, 69, 89, 96%), with the live
decision boundary. 2,700 frames on the night-blue gradient.

| Output | Where |
| --- | --- |
| Reel | `nnfields/output/nnfields_reel.mp4` (45.000 s, 1080×1920, 60 fps) |
| Cover | `nnfields/output/cover.png` |
| Posting kit | [`nnfields/POSTING.md`](nnfields/POSTING.md) |

## Reel 33 — *Backpropagation: how a network learns from one mistake.* (dark theme, no formulas on screen, 46 s)

One training step in slow motion on two networks from reel 31 (`backprop/bpsim.py`: same data, same
training, stopped at step 300 for the one-neuron net and step 150 for the 8 + 8 net). For each, one
vermilion point the network gets wrong goes forward (gold pulses, neurons lit by activation, a guess at
the output: 0.49 and 0.34), the error appears (guess − answer: −0.51, −0.66), the blame goes backward
(magenta pulses; every edge then glows by the real gradient on its weight, every neuron ringed by the
blame reaching it, and the neurons that were off get none), and a plain gradient step of stated size
(0.3, 0.02) moves every weight: the field, fold lines, guess and count of points right are recomputed
from the live weights every frame (0.49 → 0.62 and 249 → 255 of 400; 0.34 → 0.72 and 293 → 298).
2,760 frames on the night-blue gradient.

| Output | Where |
| --- | --- |
| Reel | `backprop/output/backprop_reel.mp4` (46.000 s, 1080×1920, 60 fps) |
| Cover | `backprop/output/cover.png` |
| Posting kit | [`backprop/POSTING.md`](backprop/POSTING.md) |

## Reel 34 — *Backpropagation, with nothing but the maths.* (dark theme, equations only, 60 s)

Reel 33's one-neuron step as pure maths: no diagram and no field, only typeset equations (italic
variables, stacked fractions, sub- and superscripts from a small typesetter in `backpropmath/scene.py`)
with every number recomputed at start-up from reel 33's `bpsim.py` (and asserted equal to its
derivatives). Nine pages: the network; the forward pass (z = 0.568, p = 0.490, L ≈ 0.714); the loss
derivative p − y = −0.510; the output neuron; through the fold (∂L/∂w₁ ≈ 0.460); a finite-difference
check (0.460); the update with η = 0.3; the forward pass again (L ≈ 0.484, 249 → 255 of 400 right); and
the four general rules for any depth. A table under the page fills in the five derivatives and new values.
Numbers shown to 3 decimals; "≈" marks the five lines where the rounded numbers miss the rounded
result. 3,600 frames on the night-blue gradient.

| Output | Where |
| --- | --- |
| Reel | `backpropmath/output/backpropmath_reel.mp4` (60.000 s, 1080×1920, 60 fps) |
| Cover | `backpropmath/output/cover.png` |
| Posting kit | [`backpropmath/POSTING.md`](backpropmath/POSTING.md) |

## Reel 35 — *The vanishing gradient, measured.* (dark theme, 55 s)

Two networks of 10 hidden layers × 16 neurons on the spiral (`vanishing/vgsim.py`), sigmoid against
ReLU, each with its standard start (median of 6 seeds). A ladder per network fills from the output down
with the measured gradient each of the 11 weight layers receives at the start (log-scale bars, the
sigmoid's shrink factor between rows): sigmoid 4.9 × 10⁻² down to 3.1 × 10⁻⁹, 16 million times less;
ReLU's layer 1 still gets 38% of layer 11's. Then why: σ′ never above 0.25 (measured average 0.24) against
ReLU′ = 1 when on. Then the forward half: how different the 400 points still are after each layer
(sigmoid 0.46 at the input down to 2 × 10⁻⁷ at layer 10, one guess for every point; ReLU about 0.1 all the
way up). Then both trained with plain gradient descent (step 0.1, 3,000 steps): the ladder
switches to how far each layer moved (sigmoid layer 1: 0.00014%; ReLU: 66%) and the fields train live
(sigmoid stuck at loss 0.693 with one guess, 0.500, for every point, labelled on its field, with a colour
key; ReLU 396 / 400). 3,300 frames on the night-blue gradient.

| Output | Where |
| --- | --- |
| Reel | `vanishing/output/vanishing_reel.mp4` (55.000 s, 1080×1920, 60 fps) |
| Cover | `vanishing/output/cover.png` |
| Posting kit | [`vanishing/POSTING.md`](vanishing/POSTING.md) |

## Reel 36 — *Every ReLU neuron is a hinge.* (dark theme, no formulas on screen, 50 s)

One hidden layer of 1, 2, 4, 8, 16 and 32 ReLU neurons fitted to one curve, sin(2πx)(0.55 + 0.45x) on
[−1, 1] (`hinges/hgsim.py`: Adam, 6,000 steps, median of 6 seeds), each width training live: the target
dashed, the network in gold with a dot at every hinge inside the plot, and underneath every neuron's own
bent line, whose sum is the gold curve. Errors 0.393, 0.393 (the second neuron is off everywhere), 0.325
(the median run gets stuck), 0.037, 0.019, 0.012, then a closing staircase of the six errors: the
universal approximation theorem, made visible. 3,000 frames on the night-blue gradient.

| Output | Where |
| --- | --- |
| Reel | `hinges/output/hinges_reel.mp4` (50.000 s, 1080×1920, 60 fps) |
| Cover | `hinges/output/cover.png` |
| Posting kit | [`hinges/POSTING.md`](hinges/POSTING.md) |

## Reel 37 — *Gradient descent vs saddle points: why second derivatives and the Hessian matter.* (white style, maths on screen, 50 s)

The two-valley hill of reel 30, f(u, v) = 2.2((u/2)² − 1)² + 1.8v², rendered by `valleys/hillgen.py`. First
the second derivative in 1D (curving up or down, slope zero at both turning points), then the Hessian as a
typeset matrix, with the curvature drawn on the hill: −2.2 along the pass and +3.6 across it (a saddle),
+4.4 and +3.6 at a valley bottom (a minimum). Then three balls from the same start, 0.01 off the line
through the pass, heading for the marked bottom in the open valley (`hessian/scene.py`): gradient descent (×1.22 per step along the pass: off at step 20,
bottom at 35), Newton's method (×0: onto the saddle in one step, stuck at height 2.2 forever) and
saddle-free Newton with |H| (×2: off at step 6, bottom at 10), with a live card deriving each factor.
Closes on the n² cost of the Hessian. 3,000 frames.

| Output | Where |
| --- | --- |
| Reel | `hessian/output/hessian_reel.mp4` (50.000 s, 1080×1920, 60 fps) |
| Cover | `hessian/output/cover.png` |
| Posting kit | [`hessian/POSTING.md`](hessian/POSTING.md) |

## Reel 38 — *The determinant is a volume.* (3D stage, maths card in Computer Modern, narrated, 59 s)

A new look for linear algebra: a black stage with a soft pool of light, a glossy cyan solid (flat-shaded
faces, glowing edges, dashed hidden edges, a soft shadow) under a slowly orbiting camera, and a maths card
set in Computer Modern via matplotlib mathtext (`determinant/scene.py`, a small projective renderer). A
3 × 3 matrix morphs live and reshapes the unit cube: its columns are the cube's three edges (coloured to
match the card), the card shows the matrix and its determinant every frame. Identity (1), stretch 2, 1, 1.5
(3), shear (still 3), a general matrix (2.233, checked by a million random points: 2.229), the third
column pushed into the plane of the other two (0: flat, no inverse) and through to its mirror image
(−2.233, drawn in magenta with the camera swinging round), and back. The first narrated reel: Kokoro
(open-source TTS, `am_michael` at 1.05×, `determinant/narration.py`), with the timeline sized to the lines.
3,517 frames.

| Output | Where |
| --- | --- |
| Reel | `determinant/output/determinant_reel.mp4` (58.617 s, 1080×1920, 60 fps) |
| Cover | `determinant/output/cover.png` |
| Posting kit | [`determinant/POSTING.md`](determinant/POSTING.md) |

## Reel 39 — *Eigenvectors: the directions a matrix only stretches.* (3D stage, narrated, 68 s)

On the 3D stage of reel 38 (now shared as `stage3d.py`), narrated by Kokoro through the shared `voicekit.py`. 80 unit
arrows in a glass ball; a symmetric matrix with eigenvalues 2, 1.2 and 0.5 on tilted axes turns the ball into an egg
and knocks every arrow off its line (5.5° to 36.3°, one followed live), except the three eigenvectors, which only
stretch (A v = λ v, stretches measured live); 2 × 1.2 × 0.5 = 1.2 = det A. Then power iteration with continuous
k: the rescaled arrows swing onto the top eigenvector (mean angle 57.4° → 0.53°, 72 of 80 within a degree after 12 steps).

| Output | Where |
| --- | --- |
| Reel | `eigen/output/eigen_reel.mp4` (68.0 s, 1080×1920, 60 fps) |
| Cover | `eigen/output/cover.png` |
| Posting kit | [`eigen/POSTING.md`](eigen/POSTING.md) |

## Reel 40 — *Every matrix is a turn, a stretch and a turn (the SVD).* (3D stage, narrated, 73 s)

A matrix M = U Σ Vᵀ (rotations U, V; Σ = diag(2.4, 1.25, 0.5)) acting on a ball with its right singular vectors
painted on: all at once (a tilted egg), then in slow motion: the turn Vᵀ (slerp; the arrows line up with the axes),
the stretch Σ, the turn U, landing exactly on the first egg's ghost. σ₁σ₂σ₃ = 1.5 = det M. Then the low-rank
versions: drop σ₃ (a flat disc, seen edge-on by a camera swing, 96.7% of the squared singular values kept) and keep
only σ₁ (a needle, 76.1%): the idea behind PCA and compression. `svd/scene.py`.

| Output | Where |
| --- | --- |
| Reel | `svd/output/svd_reel.mp4` (73.1 s, 1080×1920, 60 fps) |
| Cover | `svd/output/cover.png` |
| Posting kit | [`svd/POSTING.md`](svd/POSTING.md) |

## Reel 41 — *Why backpropagation is fast.* (dark theme, narrated, 63 s)

The 8 + 8 network of reels 31 and 33 (105 numbers): nudging each number by ±0.00001 (210 runs, the counter and a bar
chart of the slopes filling one by one) against backpropagation (one gold pass forward, one magenta pass back, all 105
at once); the same slopes to within 7 × 10⁻¹¹. Then a 1,005,001-weight network timed on this machine
(`bpspeed/speedsim.py`): forward 48.3 ms, backpropagation 94.3 ms, nudging 2,010,002 runs ≈ 27 hours (log-scale bars).

| Output | Where |
| --- | --- |
| Reel | `bpspeed/output/bpspeed_reel.mp4` (63.4 s, 1080×1920, 60 fps) |
| Cover | `bpspeed/output/cover.png` |
| Posting kit | [`bpspeed/POSTING.md`](bpspeed/POSTING.md) |

## Reel 42 — *Three classes and softmax.* (dark theme, narrated, 58 s)

A three-arm spiral (450 points) and a 2 → 16 → 16 → 3 network with softmax and cross-entropy, trained live
(`softmax/smsim.py`, median of 6 seeds: 96.7%). The map mixes the three probabilities by colour with the decision
boundaries; for one ringed point the card works softmax live (z, e^z, p) and then shows its blame p − y as three
numbers shrinking as it learns (blue on top from step 615; (0.01, −0.25, 0.24) at the end).

| Output | Where |
| --- | --- |
| Reel | `softmax/output/softmax_reel.mp4` (58.2 s, 1080×1920, 60 fps) |
| Cover | `softmax/output/cover.png` |
| Posting kit | [`softmax/POSTING.md`](softmax/POSTING.md) |

## Reel 43 — *PCA on real data: 150 flowers, squashed flat.* (3D stage, narrated, 68 s)

Fisher's iris data (`pca/iris.csv`): 150 flowers, three measurements each (sepal length, petal length, petal width),
one point per flower in 3D. The SVD of the centred data gives the principal directions (spreads 2.05, 0.39, 0.18 cm;
shares 95.8%, 3.4%, 0.8%). The cloud is flattened onto the best plane (99.2% kept, points move 0.14 cm on average),
turned to face the camera, squashed onto one line (95.8%), then coloured by species, which PCA never used: setosa sits
apart, a 1.19 cm gap along the first direction.

| Output | Where |
| --- | --- |
| Reel | `pca/output/pca_reel.mp4` (68.1 s, 1080×1920, 60 fps) |
| Cover | `pca/output/cover.png` |
| Posting kit | [`pca/POSTING.md`](pca/POSTING.md) |

## Reel 44 — *The line of best fit is a shadow (least squares in 3D).* (3D stage, narrated, 60 s)

Three points (−1, 1), (0, 3), (1, 2): their heights are one arrow y = (1, 3, 2) in 3D, and every line a + b x is a
point a (1, 1, 1) + b (−1, 0, 1) on a plane. A 2D panel shows the same line through the points with its misses; the
sum of squared misses (live) is the squared distance to y. Tried lines wander on the plane, then the best one is the
foot of the perpendicular, ŷ = (1.5, 2, 2.5), with the miss at a right angle (camera swing) and error² 1.5; the
normal equations [[3, 0], [0, 2]] (a, b) = (6, 1) give a = 2, b = 0.5.

| Output | Where |
| --- | --- |
| Reel | `lsq/output/lsq_reel.mp4` (60.2 s, 1080×1920, 60 fps) |
| Cover | `lsq/output/cover.png` |
| Posting kit | [`lsq/POSTING.md`](lsq/POSTING.md) |

## Reel 45 — *Why gradient descent zig-zags (the condition number).* (3D stage, narrated, 72 s)

A bowl ½(x²/κ + y²) morphing from round to stretched 10× and 100×, with real runs (`zigzag/zzsim.py`, each method's
best settings per bowl on a grid, steps until within 0.01 of the bottom): gradient descent 1, 29 and 284 steps,
bouncing across and crawling along; the curvatures 1 and 0.01 (the Hessian's eigenvalues) and their ratio, the
condition number; then momentum (23 steps) and Adam (29 on every bowl, the valley lying along an axis).

| Output | Where |
| --- | --- |
| Reel | `zigzag/output/zigzag_reel.mp4` (71.8 s, 1080×1920, 60 fps) |
| Cover | `zigzag/output/cover.png` |
| Posting kit | [`zigzag/POSTING.md`](zigzag/POSTING.md) |

## Reel 46 — *Overfitting, measured.* (dark theme, narrated, 66 s)

20 noisy points (evenly spaced) and polynomials of degree 0 to 19 (`polyfit/pfsim.py`, least squares in the Legendre
basis): the error on the 20 points falls to 0 at degree 19, which passes through all of them; on 5,000 new points
from the same source it falls to 0.29 (degrees 3–7) and then rises to 84 (a U on a log-scale chart). Two fixes,
measured: 200 points (0.26) and a ridge penalty with λ from leave-one-out cross-validation (0.31).

| Output | Where |
| --- | --- |
| Reel | `polyfit/output/polyfit_reel.mp4` (66.4 s, 1080×1920, 60 fps) |
| Cover | `polyfit/output/cover.png` |
| Posting kit | [`polyfit/POSTING.md`](polyfit/POSTING.md) |

## Reel 47 — *A million random networks vs gradient descent.* (dark theme, narrated, 73 s)

The 8 + 8 network of reels 31, 33 and 41 on the same spiral (`randsearch/rssim.py`): 1,000,000 networks with every
number drawn at random (the map shows the first 40 tries, then the best so far; 69.0% at try 59,327, nothing better
in the next 940,673) against gradient descent from one random start (the slope from backpropagation, Adam): past the
best of the million at step 94, 95.5% at step 1,500. Both on one chart, accuracy against runs of the network (log
scale): 3,000 against 1,000,000. Then 10¹⁰⁵ networks for ten values per number, against about 10⁸⁰ atoms.

| Output | Where |
| --- | --- |
| Reel | `randsearch/output/randsearch_reel.mp4` (72.5 s, 1080×1920, 60 fps) |
| Cover | `randsearch/output/cover.png` |
| Posting kit | [`randsearch/POSTING.md`](randsearch/POSTING.md) |

## Reel 48 — *Why AI runs on matrix multiplication.* (dark theme, narrated, 69 s)

One 512 × 512 matrix multiplication (134,217,728 multiply-adds) timed on this computer (`matmul/mmsim.py`): plain
Python fills the answer one entry at a time (13.4 s, shown at 2× speed), NumPy on one core (5.3 ms) and on all four
(2.3 ms) fill it at once: the same answer to 2.3 × 10⁻¹³, 5,861× faster. Then log-scale bars, and the million-weight
network of reel 41: one forward pass is three matrix multiplications, 10 ms in NumPy, about 40 s in plain Python
(estimated from the measured loop speed).

| Output | Where |
| --- | --- |
| Reel | `matmul/output/matmul_reel.mp4` (69.1 s, 1080×1920, 60 fps) |
| Cover | `matmul/output/cover.png` |
| Posting kit | [`matmul/POSTING.md`](matmul/POSTING.md) |

## Reel 49 — *A network learns to read handwriting.* (dark theme, narrated, 74 s)

The UCI handwritten digits (`digits/digits.csv`, 1,797 digits, 8 × 8 pixels) and a 64 → 32 → 10 network
(`digits/dgsim.py`), trained on 1,257 and tested on the 540 it never sees: 10.4% → 90% by step 31 → 96.7% at step 300
(522 right). The digit as its pixels, the 32 neurons as pictures of their weights, the ten probabilities, a strip of
unseen digits with live answers; then backpropagation carried to the pixels (the blame on each pixel of an unseen 3),
and one of the 18 mistakes, a 4 read as a 9.

| Output | Where |
| --- | --- |
| Reel | `digits/output/digits_reel.mp4` (73.5 s, 1080×1920, 60 fps) |
| Cover | `digits/output/cover.png` |
| Posting kit | [`digits/POSTING.md`](digits/POSTING.md) |

## Reel 50 — *How to fool a neural network.* (dark theme, narrated, 72 s)

The handwriting network of reel 49, retrained exactly (`adversarial/advsim.py` checks it is the same network). For its
unseen 3 (p = 0.999), backpropagation to the pixels gives the direction that makes "3" less likely; every pixel moved
one shade of ink (of 16) that way leaves p(3) = 0.63, two shades and it reads a 9 (p = 0.992). Nudged toward each
digit instead (2 to 4 shades), the same 3 reads as every digit 0–9. All 540 unseen digits, 3 shades along their own
gradients: 4 still right; 3 shades in random directions: 450.

| Output | Where |
| --- | --- |
| Reel | `adversarial/output/adversarial_reel.mp4` (72.3 s, 1080×1920, 60 fps) |
| Cover | `adversarial/output/cover.png` |
| Posting kit | [`adversarial/POSTING.md`](adversarial/POSTING.md) |

## Reel 51 — *Ten layers without ReLU are one neuron.* (dark theme, narrated, 62 s)

The spiral of reels 31–47 and a 2 → 8 × 10 → 1 network (681 numbers, `relu/relusim.py`). With no activation it reaches
62.0%, exactly what one neuron gets, and its layers merge one by one into a single neuron (681 → 3 numbers, the same
output to 14 decimal places, the same line the lone neuron learns). The same network from the same start with ReLU
after every neuron: 98.5% (median of 9 starts). On screen: the network with its weights, the map, the score, the count
of numbers, the two training curves.

| Output | Where |
| --- | --- |
| Reel | `relu/output/relu_reel.mp4` (62.2 s, 1080×1920, 60 fps) |
| Cover | `relu/output/cover.png` |
| Posting kit | [`relu/POSTING.md`](relu/POSTING.md) |

## Reel 52 — *Move the digit one pixel (why image networks use convolution).* (dark theme, narrated, 64 s)

The handwriting network of reel 49 reads 96.7% of its 540 unseen digits; moved one pixel right, 46.5%; one pixel
left, 39.1% (`convolution/cnnsim.py`). One neuron's weights under the moved ink show why. A small convolutional network
(32 filters of 3 × 3 slid over every position, the strongest response of each kept, 650 numbers), trained on the same
unmoved digits: 95.9%, and 96.5% / 95.9% after the same moves. On screen: the digit sliding, both networks'
probabilities, the accuracy by pixels moved, a filter sliding over the digit and its responses before and after.

| Output | Where |
| --- | --- |
| Reel | `convolution/output/convolution_reel.mp4` (63.7 s, 1080×1920, 60 fps) |
| Cover | `convolution/output/cover.png` |
| Posting kit | [`convolution/POSTING.md`](convolution/POSTING.md) |

## Reel 53 — *I showed a neural network pure static.* (dark theme, narrated, 77 s)

The handwriting network of reel 49 shown 1,000 images of random static (`static/staticsim.py`): 817 read with more
than 90% certainty (mostly as a 4 or a 1), and a letter A read as "a 4" at 99.96%, because softmax has no "none of
these". An 11th answer, "not a digit", trained on 600 static images, catches 999 of 1,000 new ones and keeps 96.5% on
real digits, but catches only 72.8% of a different noise it never saw.

| Output | Where |
| --- | --- |
| Reel | `static/output/static_reel.mp4` (77.5 s, 1080×1920, 60 fps) |
| Cover | `static/output/cover.png` |
| Posting kit | [`static/POSTING.md`](static/POSTING.md) |

## Reel 54 — *Every digit, squeezed to two numbers (an autoencoder).* (dark theme, narrated, 75 s)

The UCI digits of reel 49 and an autoencoder (64 → 64 → 2 → 64 → 64, `autoencoder/aesim.py`) trained only to rebuild
its input: the 540 unseen digits as points on its 2-number map, a blob at the start and sorted by digit after training
(colours added afterwards; it never sees a label). Against PCA's 2 numbers: 5-nearest-neighbour score 62.8% vs 72.2%
(the median of 9 starts; range 68–81%), rebuild error 24% lower. Then the map backwards: the decoder's digit at each
point of a 9 × 9 grid.

| Output | Where |
| --- | --- |
| Reel | `autoencoder/output/autoencoder_reel.mp4` (75.1 s, 1080×1920, 60 fps) |
| Cover | `autoencoder/output/cover.png` |
| Posting kit | [`autoencoder/POSTING.md`](autoencoder/POSTING.md) |

## Reel 55 — *I deleted 60% of a neural network (pruning).* (dark theme, narrated, 64 s)

The handwriting network of reel 49 (2,368 weights) with weights set to zero (`prune/prunesim.py`): half at random
leaves 40.4% (median of 20); the smallest half leaves 90.0%; the smallest deleted and the rest retrained for 100
steps, 60% deleted still reads 96.7%, 80% 93.1%, 90% only 77.6%. On screen: the 32 neurons' weights with the deleted
ones in black, the weights left, the three accuracy curves, and what it reads for ten unseen digits.

| Output | Where |
| --- | --- |
| Reel | `prune/output/prune_reel.mp4` (63.9 s, 1080×1920, 60 fps) |
| Cover | `prune/output/cover.png` |
| Posting kit | [`prune/POSTING.md`](prune/POSTING.md) |

### Build (reel 01)

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy numba pycairo opencv-python-headless pyloudnorm matplotlib kokoro-onnx soundfile   # reel 38: maths card, narration
cd pendulum && ../.venv/bin/python build.py /path/to/workdir
```

Needs `ffmpeg` with libx264/aac and the Inter fonts (fontconfig names
`Inter`, `Inter Display`, `Inter Display Black`). ~25 s to simulate,
~6 min to render on 4 cores.

### Files

| File | Role |
| --- | --- |
| `reelkit.py` | shared helpers: cairo layers, bloom, dither, parallel frame → ffmpeg pipe |
| `audiokit.py` | shared procedural-audio helpers |
| `pendulum/sim.py` | 10,000 frictionless double pendulums, RK4 at 1,200 Hz, float64, running-max gap and lens extent |
| `pendulum/timemap.py` | video time → signed sim time (forward, tape-stop, rewind), camera, counter display rule, lens schedule |
| `pendulum/precompute.py` | per-frame tips, HUD strings, lens magnification, CHAOS frame, claim assertions |
| `pendulum/raster.py` | numba splat rasteriser (rods, bobs, comet trails) into a linear-light buffer; spectrum that sums to white |
| `pendulum/scene.py` | the frame renderer: emissive pass, bloom, tone map, lens, HUD, kinetic headlines, VHS glitch |
| `pendulum/audio.py` | sonification: harp from tip heights, beat rate from the gap, ticks/dings, riser, impact, tape-stop, circular reverb, −14 LUFS |
| `pendulum/build.py` | runs everything, checks the loop seam, writes cover + QA stills |

### The math on screen (all verified by the build)

* Model: ideal double pendulum, m₁ = m₂ = 1, L₁ = L₂ = 1, g = 9.81, released at
  rest from θ₁ = 140°, θ₂ = 176° + (i/9999 − ½)·0.99999×10⁻⁶ °.
* Integrator: classic RK4, dt = 1/1200 s; max relative energy drift 5.9×10⁻⁹.
* "Largest gap so far" = running max of max(range θ₁, range θ₂) over all 10,000,
  using raw unwrapped angles. It passes 180° at sim time 8.2833 s.
* 180° / 10⁻⁶ ° ≥ 1.8×10⁸ → "180 million× bigger in under 8.3 seconds"; average
  doubling time 8.283 / log₂(1.8×10⁸) = 0.302 s.
* The rewind is the recorded forward states played backwards. Because every
  pendulum starts at rest, θ(−t) = θ(t), so the playback passes through the
  release pose and keeps falling into frame 0: the loop seam is one ordinary
  1/60 s step of physics.
