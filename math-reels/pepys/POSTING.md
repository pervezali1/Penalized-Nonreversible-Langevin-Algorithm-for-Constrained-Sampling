# Posting kit — *1693. Pepys asks Newton: which bet wins?*

## Files

* Reel: `output/pepys_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the reveal ("Newton was right. A wins: 66.5%").

## What's on screen (all of it is real)

The Newton–Pepys problem (1693). Samuel Pepys asked Isaac Newton which bet was most
likely:

| Bet | Dice | Need | Exact probability | In 10,000 simulated rolls (seed 6) |
| --- | --- | --- | --- | --- |
| A | 6 | at least one 6 | 66.51% | 66.51% |
| B | 12 | at least two 6s | 61.87% | 62.11% |
| C | 18 | at least three 6s | 59.73% | 59.30% |

Pepys expected C. Newton showed A, and the reel settles it both ways: the exact
binomial formula P = Σ C(n,k)(1/6)^k(5/6)^(n−k) and 10,000 live rolls per bet. The
dice faces shown are the actual simulated rolls; sixes light up cyan; the bars are
the running win rates; the ticks at the reveal mark the exact values.

## Caption (paste as-is)

```
In 1693 Samuel Pepys wrote to Isaac Newton with a gambling question. Which is most likely:
A) at least one six when you roll 6 dice,
B) at least two sixes with 12 dice,
C) at least three sixes with 18 dice?

Pepys was sure it was C. More dice, more sixes.

Newton worked it out by hand: A wins. 66.5% vs 61.9% vs 59.7%. I rolled each bet 10,000 times to check, and the dice agree.

The catch: you also need more sixes. Rolling more dice makes the result more predictable, and the expected number of sixes (1, 2, 3) is exactly the bar you have to clear. Fewer surprises — and you need a surprise.

Which one did you pick?

(Exact: P = Σ C(n,k)(1/6)^k(5/6)^(n−k). Simulation: numpy, fair dice, 10,000 rolls per bet.)

#probability #math #newton #dice #mathfacts
```

Short alternative:

```
1693: Pepys asks Newton which bet wins. 6 dice/one six? 12/two? 18/three? Pepys bet on 18. Newton proved 6: 66.5% vs 61.9% vs 59.7%.

Which would you pick?

#probability #math #newton #dice #mathfacts
```

## Pinned comment

```
The exact numbers: A = 1 − (5/6)^6 = 66.51%. B = 61.87%. C = 59.73%. Newton got the arithmetic right but, as later historians noted, his explanation to Pepys was shaky; the clean way to see it is that the expected number of sixes equals the target in every bet (1 of 6, 2 of 12, 3 of 18), and bigger samples sit closer to their expectation, so you're less likely to be lucky enough to clear it. Keep going (24 dice, 4 sixes; 30 dice, 5 sixes) and the odds slide toward 50%. Next: the gambler's ruin — how long your money lasts at a fair table. Want it?
```

## Audio

Keep the built-in track, **"the sound of 10,000 dice"**: a clatter per roll, a ding per
six and a bell per win in the slow rolls, a rattle that thickens as the rolls speed up,
and three tones whose pitches track the three running win rates (A ends highest).

## Replies to expect

* "Isn't B 'exactly two'?" → the bets are "at least", as in Pepys' letter; "exactly" gives a different ranking.
* "Newton's reasoning was wrong" → his numbers were right; his verbal explanation is the disputed part (pinned comment).
* "Different seed, different result?" → the ranking A > B > C holds for every seed tried (10 of 10); the exact values are in the caption.
