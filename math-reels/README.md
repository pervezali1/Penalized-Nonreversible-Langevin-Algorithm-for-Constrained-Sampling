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
