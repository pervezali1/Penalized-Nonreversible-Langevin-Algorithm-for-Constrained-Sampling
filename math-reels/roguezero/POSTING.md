# Posting kit — *What if ONE zero left the line? (The Riemann Hypothesis, part 2.)* (dark theme, no narration, 30 s)

## Files

* Reel: `output/roguezero_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — "Real primes stray by 959 at most. With one rogue zero: 95,000. And rising."
* Computation: `wsim.py` (needs the zeros from `../riemann/rsim.py`; `python build.py <workdir>` runs both).

## What's on screen (all of it is computed)

Sequel to reel 70. In Riemann's explicit formula every zeta zero adds one wave to the prime count, and a zero with
real part σ gives a wave that grows like x^σ. The reel takes the first zero (height 14.134725) and slides its real
part from 0.5 (on the line) to 0.8, leaving the other 562 zeros alone.

* **Top:** the true prime staircase up to 50 (amber) and Riemann's formula (cyan), rebuilt from 563 zeros in the
  recap. With the zero pushed to 0.8 the curve (magenta) bends away from where it was (cyan, the gap filled in):
  midway between whole numbers its largest error goes from 0.04 to 0.58, so rounding it no longer gives the right
  count.
* **Bottom left:** the critical strip (real part 0 to 1, heights up to 46) with the first eight zeros; zero 1 slides
  off the line.
* **Bottom right:** zero 1's own wave. First for x up to 50 (largest size 0.42 on the line, 0.82 at 0.8), then the
  wide view on a logarithmic x-axis, with the height rescaled to fit as the right edge runs out to a billion and on
  to 10^13:

| Up to | a thousand | a million | a billion | a trillion |
| --- | --- | --- | --- | --- |
| Wave on the line (0.5) | 0.6 | 9.9 | 199 | 4,698 |
| Wave off the line (0.8) | 4.8 | 581 | 95,362 | 17,691,139 |

  (Largest size over the last full oscillation before each point.) The number top right is the largest size so far.
* **"Real primes stray by 959 at most":** a sieve counts the real primes up to 10^9 with a checkpoint every 10,000;
  the largest gap between the count and Riemann's smooth estimate R(x) is 959 (at 905,040,000). At 10^9 itself:
  50,847,534 primes, estimate 50,847,455, gap 79.
* **"12 trillion zeros checked":** Platt and Trudgian (2021), the lowest 12,363,153,437,138 zeros, all on the line.

## Honesty notes

* **It is a what-if, not a possible world.** No zeta function has that zero, and a prime count cannot really bend:
  it is a whole number at every step. The reel shows the size and the growth of the wave an off-the-line zero
  would bring.
* A real zero off the line would come with a mirror twin at real part 0.2. Its wave grows like x^0.2 (0.4 at a
  billion) and is left out.
* A real rogue zero could only sit above height 3,000,175,332,800, and a wave's size is divided by its zero's
  height. At real part 0.8 it would stay smaller than the first zero's wave until about 10^38, and win after that.
* "It drowns out every other zero": at a billion the pushed wave (95,362) is about 100 times the whole real gap
  (959 at most), which is the sum of every real zero's wave.
* "As small as it can be": zeros come in mirror pairs (σ and 1 − σ), and the larger of the two real parts is at
  least 0.5. Both exactly on the line is the smallest the waves can be.
* The wave uses the n = 1, 2, 3, 5 terms of the formula up to 50 and n = 1, 2, 3, 5, 6, 7, 10 in the wide view.
* The first 2.4 s continue the end of the loop (the wide view beyond a trillion).

## Caption (paste as-is)

```
What if ONE zero left the line? (The Riemann Hypothesis, part 2.)

Last time: the zeros of Riemann's zeta function build the primes, and every zero ever found sits on one line.

So I pushed one off. The first zero, moved from 0.5 to 0.8. Nothing else changed.

Each zero is a wave in the prime count. Off the line, that wave swells. Up to 50 it bends the prime staircase. Then zoom out: by a billion that single wave is 95,000 primes tall, and it never stops growing.

The real primes? Up to a billion they never stray more than 959 from Riemann's estimate.

That's why the line matters. On it, every wave is as small as it can possibly be, so the primes are as orderly as they can be. 12 trillion zeros have been checked. None off the line.

(A what-if, computed with Riemann's own formula. No zero has ever been found off the line.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #riemannhypothesis #primenumbers #mathematics #science
```

Exactly five hashtags.

## YouTube

Title:

```
What If ONE Zero Left the Line? (Riemann Hypothesis, Part 2) #shorts
```

Description:

```
Why does it matter that the zeros of the zeta function sit on one line? A what-if, computed with Riemann's explicit formula: the first zero (height 14.13) is pushed from real part 0.5 to 0.8, and nothing else changes.

Its wave in the prime count swells: up to 50 it bends the prime staircase, and in the wide view it reaches 581 by a million, 95,362 by a billion and 17.7 million by a trillion, while the same wave on the line reaches 199 by a billion. The real primes, counted by sieve up to a billion, never stray more than 959 from Riemann's smooth estimate.

On the line every wave grows only like the square root of x: as small as it can be. No zero has ever been found off the line (12,363,153,437,138 checked, Platt and Trudgian 2021).

#math #riemannhypothesis #primes #shorts
```

## Pinned comment

```
Why "as small as it can be"? A zero with real part s adds a wave that grows like x to the power s. Zeros come in mirror pairs: one at s means another at 1 − s, and the bigger of the two is always at least 1/2. So the calmest the primes can possibly be is every zero exactly on the line. That is the Riemann Hypothesis. And to be fair to this what-if: a real rogue zero could only hide above height 3 trillion, so its wave would start out tiny. At 0.8 it wouldn't pass the first zero's wave until around 10^38. But it would win in the end.
```

## Audio

Keep the built-in track: rising ticks as the formula is rebuilt, a plink for each zero placed on the line, a steady
tone for zero 1's wave and a second tone that slides out of tune as the zero leaves the line, a growl and a rising
tone as the wave grows, a bell and impact for the two numbers, a chime for the last line.

## Replies to expect

* "So is there a zero off the line?" → none has ever been found; this shows what one would do.
* "Is this what the new proof rules out?" → yes: a proof of the hypothesis says no zero like this exists, at any
  height.
* "Why does 0.8 matter so much more than 0.5?" → the wave grows like x^0.8 instead of x^0.5; at a billion that is
  the difference between 199 and 95,362.
