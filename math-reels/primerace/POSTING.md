# Posting kit — *Primes split into two teams. One almost never loses.* (the prime race, dark theme, no narration, 30 s)

## Files

* Reel: `output/primerace_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — "26,861: Team 1 leads! For exactly 2 numbers.", with the ×300 zoom.
* Computation: `psim.py` (a segmented sieve of every prime up to 10^9, about 20 s; `python build.py <workdir>`).

## What's on screen (all of it is computed)

Chebyshev's bias. Every odd prime leaves remainder 1 or 3 when divided by 4: **Team 1** (5, 13, 17, 29, …) or
**Team 3** (3, 7, 11, 19, …). 2 is the only even prime and sits out. In the long run each team gets half the primes
(Dirichlet), so the race "should" be a tie.

| Up to | 100 | 1,000 | 10,000 | 1 million | 1 billion |
| --- | --- | --- | --- | --- | --- |
| Team 3 | 13 | 87 | 619 | 39,322 | 25,424,042 |
| Team 1 | 11 | 80 | 609 | 39,175 | 25,423,491 |

* **26,861:** the first number at which Team 1 is ahead (26,861 is a Team 1 prime; 26,863, a Team 3 prime, ties it
  again). Team 1 leads for exactly 2 numbers, 26,861 and 26,862. The zoom shows 26,800 to 26,900 with every prime.
* **Up to a billion, Team 1 is ahead in 4 bursts**, starting at 26,861, 616,841, 12,306,137 and 951,784,481 (each
  burst is made of many short stretches; 310 in all). That's 150,932 numbers. Team 3 is ahead for 999,837,520 of
  the numbers up to a billion (99.98%); they're tied for 11,549.
* Biggest leads: Team 3 by 2,615 (at 749,626,151); Team 1 by 48 (at 951,867,557).
* **The squares:** every odd square leaves remainder 1 (an odd number is 4k ± 1, and (4k ± 1)² = 16k² ± 8k + 1).
  The count that Riemann-style formulas treat fairly includes prime squares at half a point each. Giving Team 1 its
  3,400 odd prime squares up to a billion at half a point each (the white line) turns the race even: Team 3 ahead
  47.6% of the time, Team 1 52.4%.
* The charts show the lead column by column, each column covering everything inside it, so no lead is skipped. The
  billion-wide chart uses a logarithmic x-axis from 10,000 and rescales its height as it grows.

## Honesty notes

* "Its swings come from zeros, just like Riemann's": the race is governed by the zeros of a Dirichlet L-function,
  a cousin of the zeta function. Each zero adds a wave; the squares add a steady push toward Team 3.
* The famous figure for forever, 99.59% (Rubinstein and Sarnak, 1994), assumes those zeros all sit on the critical
  line (the generalised Riemann Hypothesis) and a second hypothesis (the zeros' heights are linearly independent), and
  it is measured on a logarithmic scale (each number x weighted by 1/x). The reel's own figure, 99.98%, counts every
  whole number up to a billion equally. The two aren't the same measurement.
* Littlewood proved in 1914 that the lead changes hands infinitely often: Team 1 will get ahead again, forever.

## Caption (paste as-is)

```
Primes split into two teams. One almost never loses.

Divide any odd prime by 4. The remainder is 1 (Team 1: 5, 13, 17, 29…) or 3 (Team 3: 3, 7, 11, 19…). In the long run each team gets exactly half the primes. So it should be a tie.

It isn't. Team 3 grabs the lead and keeps it. Team 1 gets ahead for the first time at 26,861, for exactly 2 numbers.

I raced every odd prime up to a billion: 50,847,533 of them. Team 3 leads for 99.98% of the numbers. Team 1 gets ahead in just 4 bursts.

Why? Squares. Every odd square (9, 25, 49…) leaves remainder 1. In the fair count mathematicians use, each prime square is worth half a point. Give Team 1 its squares back and the race is even: 48% to 52%.

And the swings? They come from the zeros of a cousin of Riemann's zeta function. If those zeros sit on Riemann's line too (plus one more assumption), Team 3 leads 99.59% of the time forever, measured on a log scale.

Which team would you bet on? 👇

(Every prime up to a billion, sieved and counted. Nothing estimated.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #primenumbers #numbertheory #riemannhypothesis #mathematics
```

Exactly five hashtags.

## YouTube

Title:

```
Two Teams of Primes Race to a Billion. One Almost Never Loses. #shorts
```

Description:

```
Chebyshev's bias, computed exactly. Odd primes leave remainder 1 or 3 when divided by 4, and each remainder gets half the primes in the long run. But up to a billion, primes with remainder 3 are ahead for 99.98% of the numbers (999,837,520 of them). Remainder 1 first leads at 26,861, for exactly 2 numbers, and gets ahead in only 4 bursts below a billion (from 26,861, 616,841, 12,306,137 and 951,784,481). Final score at a billion: 25,424,042 to 25,423,491.

The reason: every odd square leaves remainder 1, and the fair count includes prime squares at half a point each. Add them back and the race is even (47.6% to 52.4%). The swings come from the zeros of a Dirichlet L-function, a cousin of the Riemann zeta function.

#math #primes #numbertheory #shorts
```

## Pinned comment

```
Why squares? Counting primes is the "unfair" version. The formulas that make prime counts exact (Riemann's explicit formula and its cousins) count prime powers too: each prime square as half a point, each cube as a third. Every odd square leaves remainder 1 when divided by 4, so the fair count gives Team 1 a steady extra share that plain prime-counting leaves out. Take it away, and Team 3 starts every race with a head start of about √x / ln x. The zeros add waves on top, and only a rare pile-up of waves lets Team 1 through. Littlewood proved in 1914 that it happens infinitely often.
```

## Audio

Keep the built-in track: a plink for every prime as it drops into its lane (Team 3 low, Team 1 high), a rattle as
the race speeds up, a tone that follows Team 3's lead, a bright two-note chime each time Team 1 takes the lead, a bell
for 26,861, plinks for the squares, a soft pad for the fair count.

## Replies to expect

* "Doesn't Team 1 catch up eventually?" → it gets ahead infinitely often (Littlewood, 1914), but only rarely.
* "Is 99.98% the real answer?" → up to a billion, counting every number equally. Forever, on a log scale and
  assuming the generalised Riemann Hypothesis, it's 99.59% (Rubinstein and Sarnak).
* "Does this work with other numbers than 4?" → yes, the teams that are squares (like remainder 1) usually trail;
  mod 3, remainder 2 leads remainder 1 the same way.
