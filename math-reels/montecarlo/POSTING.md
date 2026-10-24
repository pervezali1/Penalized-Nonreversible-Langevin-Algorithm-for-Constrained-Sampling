# Posting kit — *Throw darts at random. That's π.*

## Files

* Reel: `output/montecarlo_reel.mp4` — 23.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.4 LUFS.
* Cover: `output/cover.png` — the full board at one million darts, "1,000,000 darts: π ≈ 3.1403 / Correct digits: 3."

## What's on screen (all of it is real)

**The board.** A square with its inscribed circle. Darts land uniformly at random (numpy, seed 1).
Circle area ÷ square area = π/4, so π ≈ 4 × inside ÷ total. The darts are shown on a log scale in
time (1 → 1,000,000 over 8.4 s) so the first ones can be seen landing one by one. Cyan = inside,
magenta = outside. The estimate is shown to six decimals with the correct leading digits in green
(compared with 3.14159265…), and the digit count is live: it locks in and slips back out dozens of
times on the way up.

| Darts | Inside | Estimate | Correct digits |
| --- | --- | --- | --- |
| 1,000 | 780 | 3.120000 | 2 |
| 10,000 | 7,810 | 3.124000 | 2 |
| 100,000 | 78,552 | 3.142080 | 3 |
| 1,000,000 | 785,077 | 3.140308 | 3 |

**The funnel.** 300 independent dart-throwers (numba, seed 101), each one's running estimate drawn
against the number of darts on a log axis from 10 to 1,000,000. The dashed band is the exact 95%
interval π ± 1.96 × 1.642/√N (1.642 = 4·√(p(1−p)) with p = π/4). At one million darts the measured
spread across the 300 runs is 0.0017, the formula says 0.00164, and 285 of 300 runs are inside the
95% band (expected 285).

**The claim.** Error ∝ 1/√N, so one more correct digit (10× smaller error) costs 100× more darts.
Ten correct digits (error below 5 × 10⁻¹⁰ at 95% confidence) need about 4 × 10¹⁹ darts: at a
billion darts per second that is 1,300 years. "A thousand years" on screen is the rounded-down
version.

**The last line.** Monte Carlo error is 1/√N regardless of dimension, while grid methods need
exponentially more points as dimensions grow. That is why it is a terrible way to get π and the
standard way to integrate in 1,000 dimensions (physics, finance, Bayesian statistics).

## Caption (paste as-is)

```
Throw darts at a square with a circle inside. Count how many land in the circle. Multiply by 4. That's π.

It works because circle ÷ square = π/4. I threw one million random darts and watched the digits: 3.1 after a few hundred, 3.14 somewhere past ten thousand, and at one million darts... still 3.14. Three correct digits. The fourth kept locking in and slipping away.

That's not bad luck. The error of this method shrinks like 1/√N. Every extra digit costs 100 times more darts. I ran 300 independent dart-throwers side by side and they form a funnel that narrows exactly at that rate: the dashed band on screen is the formula, and 285 of the 300 stayed inside it, which is what 95% looks like.

Ten digits of π this way: about 10¹⁹ darts. A billion a second, for over a thousand years. Archimedes got three digits with a pencil in 250 BC.

So why does anyone use it? Because 1/√N doesn't care how many dimensions you're in. Grids explode; darts don't. This is how physicists simulate nuclei, how banks price options, how statisticians sample a posterior. Hopeless for π, unbeatable in a thousand dimensions.

(numpy + numba, 1,000,000 darts, 300 runs for the funnel; band = π ± 1.96 × 1.642/√N.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #pi #montecarlo #probability #statistics #mathfacts
```

Short alternative:

```
Random darts can compute π. One million of them get you 3.14 and not much more: the error shrinks like 1/√N, so each digit costs 100× the darts. Ten digits would take a billion darts a second for a thousand years.

Follow for more.

#math #pi #montecarlo #probability #mathfacts
```

## Pinned comment

```
The maths: each dart is a coin flip with p = π/4 of landing inside. The estimate 4·(inside/N) has standard deviation 4·√(p(1−p)/N) = 1.642/√N. Want the error below 0.0005 (three decimals, 95% of the time)? N ≈ (1.96 × 1.642 / 0.0005)² ≈ 41 million darts. Ten correct digits: 4 × 10¹⁹. And yes, there are smarter random methods (quasi-random sequences get 1/N instead of 1/√N), but the real point is the other way round: this method is exactly as slow in 1,000 dimensions as in 2, and nothing else is. Next: Buffon's needle, computing π by dropping matchsticks on floorboards. Want it?
```

## Audio

Keep the built-in track: a tick per dart while they can be heard one by one (higher for inside,
lower for outside), then a rain whose density follows darts per frame; a coin each time a new digit
locks in and a dull knock when one slips back; a pad whose pitch climbs with the digits; a swell into
the funnel and a descending sweep as the band narrows; bells at the reveals.

## Replies to expect

* "Just use a formula for π" → yes, that's the point of the last ten seconds: the method is for problems that have no formula.
* "Your random numbers aren't random" → they're numpy's PCG64 generator; the funnel's 285/300 inside the 95% band shows they behave like random darts.
* "Why does the digit count go up and down?" → the estimate wobbles around π; a digit is "correct" only while the wobble is on the right side of a rounding boundary.
* "Does it ever reach 3.14159?" → only by luck (several of the 300 runs passed through 3.14159… on the way, one as early as 300,000 darts, and drifted off again); to be *sure* of six digits (95%) you need about 400 billion darts.
* "Quasi-Monte Carlo / Sobol?" → gives 1/N instead of 1/√N in low dimensions; the reel's claim is about plain random darts.
