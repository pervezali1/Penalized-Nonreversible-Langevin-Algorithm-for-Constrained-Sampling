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

## Reel 28 — *Two valleys.* (white style, a new hill, twelve shaken copies)

A hill with two bottoms, rendered in the reel-23 style by a general-surface ray-marcher
(`valleys/hillgen.py`, numba, with the same camera, shading, contours, cut face and strokes; the rim is a
true level-set contour): a double well along the long axis, f = 2.2·((u/2)² − 1)² + 0.6u + c + 1.8v²,
ridge 3.52, shallow bottom 2.39, deep bottom 0. Two balls dropped 0.2 apart on the ridge (step size
0.1, 18 steps) settle in different valleys (0.09 and 2.46). B then becomes twelve copies, each shaken
for 80 fast steps with Gaussian noise of size 3.5 on each slope component (its own random stream), with
a live count of how many are over the ridge: the first at step 9, 5 of 12 by step 80; with the shaking
off they settle at 0. The closing cue states the population rate, 41% of 1,000 random copies; the
twelve shown are the first runs that stay inside the drawn hill, chosen to match it. Background: a
faint tint with a light grid, by request.

| Output | Where |
| --- | --- |
| Reel | `valleys/output/valleys_reel.mp4` (30.000 s, 1080×1920, 60 fps) |
| Cover | `valleys/output/cover.png` |
| Posting kit | [`valleys/POSTING.md`](valleys/POSTING.md) |

### Build (reel 01)

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy numba pycairo opencv-python-headless pyloudnorm
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
