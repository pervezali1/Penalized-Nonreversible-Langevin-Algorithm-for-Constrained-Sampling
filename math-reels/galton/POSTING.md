# Posting kit — *10,000 balls. Every peg: a coin flip.* (Galton board, dark theme, no narration, 22 s)

## Files

* Reel: `output/galton_reel.mp4` — 22.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the piles with the predicted bell curve drawn over them.
* Simulation: `gsim.py` (seed 68, chosen before running; `python gsim.py out.json` writes every number below).

## What's on screen (all of it is real)

A Galton board: 10,000 balls, 16 rows of pegs, 17 bins. At every peg a fair coin flip sends a ball half a peg left or
right, so a ball's bin is the number of rights out of 16: 160,000 coin flips in all. The exact chance of bin k is
C(16, k) / 2^16 = C(16, k) / 65,536.

| Bin (rights) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Landed | 0 | 5 | 12 | 81 | 288 | 649 | 1,249 | 1,716 | 1,914 | 1,811 | 1,217 | 658 | 269 | 100 | 27 | 4 | 0 |
| Predicted | 0.15 | 2.4 | 18.3 | 85.4 | 277.7 | 666.5 | 1,221.9 | 1,745.6 | 1,963.8 | 1,745.6 | 1,221.9 | 666.5 | 277.7 | 85.4 | 18.3 | 2.4 | 0.15 |

* **Middle bin:** predicted 1,964 (19.6%), landed 1,914 (19.1%), 1.25 standard deviations low (one standard
  deviation is 39.7 balls): ordinary luck, shown as it came out.
* **Edges:** 16 flips the same way, 1 in 65,536 each; expected 0.15 balls per edge; landed 0 and 0.
* Within two bins of the middle: 79.1% landed, 79.0% predicted. A chi-square test over bins 2–14 gives 15.1 on 12
  degrees of freedom: the piles match the prediction as well as random data should.
* The HUD counts the balls dropped and the share of landed balls in the middle bin, frame by frame.

## Honesty notes

* This is the ideal Galton board: balls don't collide with each other, and each bounce is exactly 50/50. Real wooden
  boards are close to this but not exact.
* The balls are released faster and faster (the first second of the stream starts at the end of the loop, so the
  first frame already shows balls falling); release timing doesn't affect where they land.
* The amber dots are the exact predicted heights of the 17 piles; the dashed line between them is a smooth curve
  through the dots, drawn as a guide for the eye.
* Colours: landed balls shade from cyan (middle) to magenta (edges) by bin.

## Caption (paste as-is)

```
10,000 balls. Every peg is a coin flip: left or right, 50/50.

Nobody can say where one ball will land. 16 flips per ball, 160,000 flips in all, pure chance.

But together they pile into a bell curve, and its shape was predicted before the first ball fell: the chance of each bin is C(16, k) / 65,536.

Middle bin: predicted 1,964. Landed 1,914.
The edges need 16 flips the same way: 1 in 65,536. Out of 10,000 balls: 0.

One ball is chaos. 10,000 are almost exactly predictable. That's why casinos and insurers sleep at night.

Which bin would you bet on? 👇

(A real simulation in Python: 10,000 balls, 16 rows, every bounce a fair coin flip. Seed 68, not cherry-picked.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #probability #statistics #datascience #bellcurve
```

Exactly five hashtags.

## YouTube

Title:

```
10,000 Random Balls. One Bell Curve. (Galton Board) #shorts
```

Description:

```
A Galton board, simulated: 10,000 balls, 16 rows of pegs, and every bounce a fair coin flip (160,000 flips in all). Nobody can predict one ball, but the piles follow C(16, k) / 65,536: middle bin predicted 1,964, landed 1,914; within two bins of the middle 79.1% vs 79.0% predicted; each edge needs 16 flips the same way (1 in 65,536, 0.15 balls expected) and got 0.

#math #probability #statistics #shorts
```

## Pinned comment

```
Why a bell? A ball lands in bin k if it goes right exactly k times out of 16. There are C(16, k) different paths to bin k, out of 2^16 = 65,536 equally likely paths: 1 path to each edge, 12,870 to the middle. Those path counts are a row of Pascal's triangle, and as the rows grow they look more and more like the normal curve. Add up lots of small random nudges and you get a bell: that's the central limit theorem, and it's why measurement errors and test scores so often look like this.
```

## Audio

Keep the built-in track: peg hits plink (pitched by the peg's position, left low, right high), a rain texture
follows the hits, a sand hiss follows the landings, a pad rises as the bins fill; a bell and chime when the
prediction is drawn, low blips for the empty edges, a rumble as the floor opens.

## Replies to expect

* "Real Galton boards aren't perfectly binomial" → true; this is the ideal one (honesty notes).
* "Why didn't the middle hit 1,964?" → randomness: one standard deviation is about 40 balls; 1,914 is 1.25 of them.
* "Do it with a million balls" → the relative wobble shrinks like 1/√n: ten times closer with 100× the balls.
