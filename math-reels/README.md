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
