# Posting kit — *1693. A gambler asks Newton: which bet wins?* (v2)

## Files

* Reel: `output/pepys2_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the reveal ("Newton was right. A wins: 66.5%").

## What changed from v1

v1 raced three thin bars; the *reason* was a line of text. v2 shows the reason as a
picture: each bet gets a live histogram of "how many sixes came up", **magenta below the
target, cyan at or above it**, so the win rate is literally the cyan area and the
question becomes "which pile has the most cyan?". At the end an amber marker shows that
the target is exactly the average (6/6 = 1, 12/6 = 2, 18/6 = 3) and the exact binomial
outline is drawn over the simulated bars.

## What's on screen (all of it is real)

| Bet | Dice | Need | Exact | 10,000 simulated rolls (seed 6) |
| --- | --- | --- | --- | --- |
| A | 6 | ≥ 1 six | 66.51% | 66.51% |
| B | 12 | ≥ 2 sixes | 61.87% | 62.11% |
| C | 18 | ≥ 3 sixes | 59.73% | 59.30% |

Histograms are the real counts of sixes per roll; the white outline at the reveal is the
exact binomial distribution. Expected sixes = n/6 = the target in every bet.

## Caption (paste as-is)

```
In 1693 Samuel Pepys asked Isaac Newton which bet was most likely:
A) at least one six from 6 dice,
B) at least two sixes from 12 dice,
C) at least three sixes from 18 dice.

Pepys was sure it was C. More dice, more sixes.

Newton: A. 66.5% vs 61.9% vs 59.7%. I rolled each bet 10,000 times; every roll adds to a pile of "how many sixes", cyan if it clears the target, magenta if not. A has the most cyan.

The twist: Newton's arithmetic was perfect (he counted the cases by hand), but the explanation he gave Pepys was wrong. He argued by proportions, "one six per six dice is the same ratio as three per eighteen, so the smaller bet is easier", which happens to give the right answer here and fails in general.

The reason is in the piles: the target is exactly the average number of sixes (1 of 6, 2 of 12, 3 of 18). Roll more dice and the pile hugs the average more tightly, so there's less luck to lift you over the line.

Which did you pick before the reveal?

(Exact: P = Σ C(n,k)(1/6)^k(5/6)^(n−k). Simulation: numpy, fair dice, 10,000 rolls per bet.)

#probability #math #newton #dice #mathfacts
```

Short alternative:

```
1693: Pepys asks Newton which bet wins. 6 dice/one six? 12/two? 18/three? Pepys bet on 18. Newton proved 6. The piles show why: the target is exactly the average, and more dice means less luck.

#probability #math #newton #dice #mathfacts
```

## Pinned comment

```
The exact numbers: A = 1 − (5/6)^6 = 66.51%, B = 61.87%, C = 59.73%. Newton's arithmetic was right (31,031/46,656 for A, worked by hand). His explanation to Pepys was not: he reasoned by proportions, which doesn't hold up (Stephen Stigler, 2006, "Isaac Newton as a Probabilist"). The clean way to see it: in every bet the expected number of sixes equals the target (n/6), and a bigger sample sits closer to its expectation, so you're less likely to be lucky enough to clear it. Keep going (24 dice, 4 sixes; 30 dice, 5 sixes) and the odds slide toward 50%. Next: the gambler's ruin. Want it?
```

## Audio

Keep the built-in track, **"the sound of 10,000 dice"**: a clatter per roll, a ding per six
and a bell per win in the slow rolls, a rattle that thickens as the rolls speed up, three
tones whose pitches track the three running win rates (A ends highest), bells at the reveal.

## Replies to expect

* "Newton got it wrong?" → his numbers were right; his proportions argument was wrong (Stigler 2006). It's on screen at 16 s.

* "Isn't B 'exactly two'?" → the bets are "at least", as in Pepys' letter.
* "Why is the first magenta bar so tall for A?" → P(zero sixes in 6 dice) = (5/6)^6 = 33.5%; that single bar is A's entire losing side.
* "Different seed?" → the ranking A > B > C held for every seed tried; exact values are in the caption.
