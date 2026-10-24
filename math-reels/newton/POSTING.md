# Posting kit — *Guess. Slide down the tangent. Repeat.*

## Files

* Reel: `output/newton_reel.mp4` — 24.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.5 LUFS.
* Cover: `output/cover.png` — the Newton fractal at iteration 12, "Every step, the borders grow borders."

## What's on screen (all of it is real)

**Part 1, the real line.** f(x) = x² − 2, starting guess x₀ = 1. Each step: draw the tangent at
(xₙ, f(xₙ)), the next guess is where it crosses zero: xₙ₊₁ = xₙ − f(xₙ)/f′(xₙ) = (xₙ + 2/xₙ)/2.
The iterates are computed with 120-digit decimal arithmetic and compared digit by digit with √2.

| Step | xₙ | Correct digits | Error |
| --- | --- | --- | --- |
| 0 | 1 | 1 | 4.1 × 10⁻¹ |
| 1 | 1.5 | 1 | 8.6 × 10⁻² |
| 2 | 1.41666… | 3 | 2.5 × 10⁻³ |
| 3 | 1.41421568… | 6 | 2.1 × 10⁻⁶ |
| 4 | 1.41421356237468… | 12 | 1.6 × 10⁻¹² |
| 5 | 1.414213562373095048801689… | 24 | 9.0 × 10⁻²⁵ |
| 6 | … | 48 | 2.9 × 10⁻⁴⁹ |

The camera window shrinks with the error (±0.95, ±1.2 × 10⁻², ±10⁻⁵, ±8 × 10⁻¹²), so each new
guess appears in the same place the previous one did: that is quadratic convergence made visible.

**Part 2, the complex plane.** f(z) = z³ − 1, roots 1, −½ ± i√3/2 (amber, cyan, magenta). Every
pixel of the full frame (2 million starting points, re ∈ [−1.5, 1.5]) is iterated with
z ← z − (z³ − 1)/(3z²) and coloured by the root it is nearest to; bright once it has converged
(|z³ − 1| < 10⁻⁴), dim while it hasn't. Iterations 0 to 14 are shown one by one: iteration 0 is
three plain sectors, and each step adds a new generation of bulbs along every border. Then a zoom
of 100,000× onto z = −(½)^(1/3) ≈ −0.7937, the point Newton's step sends to 0, where all three
basins meet, shaded by iterations to converge.

Facts stated: every boundary point of one basin is a boundary point of all three (the Wada
property of Newton's method for z³ − 1); Newton described the method in 1669 (*De analysi*);
Cayley posed the cubic basin problem in 1879 and the picture was first seen on computers.

## Caption (paste as-is)

```
Newton's method, from 1669: guess a root, draw the tangent, go to where the tangent hits zero, repeat.

For √2, starting from the guess "1", the correct digits go 1, 3, 6, 12, 24, 48. They double every step, because each step squares the error. Six steps and you have more digits than any calculator shows.

Then I asked it a harder question: z³ = 1 in the complex plane, which has three roots. I started it from every one of the 2,000,000 pixels in the frame and coloured each one by the root it ended up at.

You'd expect three neat regions. Instead, iteration by iteration, the borders sprout borders, and the borders of those sprout borders. Zoom in 100,000× on any boundary point and you find all three colours touching, forever. Two starting guesses a hair apart can end at completely different roots.

This is the Newton fractal. Arthur Cayley asked what this picture looked like in 1879 and gave up; nobody saw it until computers could draw it. All of it comes out of one tangent line.

(Real line: 120-digit decimal arithmetic, digits checked against √2. Plane: numpy + numba, Newton for z³ − 1 on every pixel, 1080×1920, zoom to half-width 1.5 × 10⁻⁵ at z = −0.7937.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #newtonsmethod #fractal #calculus #chaos #mathfacts
```

Short alternative:

```
Guess, slide down the tangent, repeat. On √2 the correct digits double every step: 1, 3, 6, 12, 24, 48. On z³ = 1, the same method draws this. Newton's method, 1669. The picture, 300 years later.

Follow for more.

#math #newtonsmethod #fractal #calculus #mathfacts
```

## Pinned comment

```
Why the digits double: near a root, Newton's error obeys e(n+1) ≈ C·e(n)². Square a number with k leading zeros and you get one with 2k. (Starting from 1 the first step only "warms up", then 1 → 3 → 6 → 12 → 24 → 48.) Why the fractal: Newton's step for z³ − 1 is z − (z³ − 1)/(3z²), which blows up near z = 0, so a tiny neighbourhood of 0 is thrown over the whole plane and lands in all three basins. Every point that eventually maps to 0 inherits that, and those points are dense along every border. Hence the Wada property: no border separates just two colours; every border point touches all three. Next: why gradient descent is Newton's method with the curvature thrown away. Want it?
```

## Audio

Keep the built-in track: a tick when each tangent lands, a sparkle of coins, one per newly correct
digit (so the bursts double in length), a pad that climbs with the digits, a swell into the plane,
a click and a rising tone per iteration, a long filtered riser under the zoom, bells at the reveals.

## Replies to expect

* "Isn't this just the Mandelbrot set?" → no: Mandelbrot iterates z² + c and asks about escape; this iterates Newton's step for z³ − 1 and colours by which root you reach. Different map, different picture.
* "Why does step 1 only give 1 digit?" → 1.5 vs 1.414: the "1" is right, the "5" isn't; quadratic doubling kicks in once you're close (from step 2 on).
* "Does it always converge?" → no: start exactly on a border and you never settle, and some functions have cycles Newton falls into forever. That's the dark filigree.
* "Why those three colours?" → one per root of z³ = 1: 1, and the two complex cube roots of 1.
* "What's the zoom point?" → z = −(1/2)^(1/3): Newton's step sends it to 0, and 0 is the point whose neighbourhood touches every basin.
