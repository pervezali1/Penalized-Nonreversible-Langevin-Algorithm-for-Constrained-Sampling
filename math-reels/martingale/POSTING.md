# Posting kit — *Lose? Double your bet.*

## Files

* Reel: `output/martingale_reel.mp4` — 24.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −15 LUFS.
* Cover: `output/cover.png` — the solo gambler's cliff: "Next bet $1,024", bankroll $341, ten red L's.

## What's on screen (all of it is real)

Roulette red/black, win probability 18/38 = 47.37%. The Martingale: bet $1, double after every
loss, go back to $1 after a win, so every completed round nets +$1. Bankroll $1,023 = 1 + 2 + 4 +
… + 512, exactly enough for nine doublings. Ten losses in a row and the stake is gone.

| Part | What happens | Numbers |
| --- | --- | --- |
| One gambler (player 9 of the simulation, seed 3) | climbs $1 a round to a peak of $1,364, then loses ten spins in a row at spins 642–651 | bets 1, 2, 4, 8, 16, 32, 64, 128, 256, 512; next bet would be $1,024; left with $341 (the $1 wins banked earlier) |
| 1,000 players, 5,000 spins each | a tight climbing band that thins as players drop into a lower band of busts | **978 busted**, 22 survived (up about $2,370 each); all 1,000 together: **−$402,270** |
| The math card | per round | P(ten losses) = (20/38)¹⁰ = 0.163%; EV per round = 0.99837 × $1 − 0.00163 × $1,023 = **−$0.67** |

Why −67¢: the average amount wagered in a round is $12.74 (a geometric sum over the doublings), and
the house keeps its 2/38 = 5.26% of it: 12.74 × 0.0526 = 0.67. The system changes nothing about the
edge, it only concentrates the losses into rare, enormous rounds.

## Caption (paste as-is)

```
The Martingale: bet $1 on red. Lose? Bet $2. Lose again? $4. Keep doubling until you win, and every round ends with you $1 up. People have been "discovering" this system for 300 years.

I gave 1,000 simulated players $1,023 each (enough to double nine times) and let them play 5,000 spins of roulette.

One player's chart first: a beautiful staircase, 341 rounds won, bankroll $1,364. Then ten losses in a row. Bet 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. The next bet is $1,024 and there's $341 left in the pocket. The staircase took 641 spins to build. The cliff took 10.

The crowd: 978 of 1,000 busted. All 1,000 together finished −$402,270.

The math, per round: 99.84% chance of winning $1, 0.16% chance of losing $1,023. Average: −67 cents a round. That is exactly the house's 5.3% applied to the $12.74 you wager in an average round. Martingale doesn't touch the edge. It just lets you win tiny amounts hundreds of times and then pays all of it back, plus interest, in one round.

It doesn't change if you lose. Only when.

(Simulation: numpy, roulette p = 18/38, 1,000 players × 5,000 spins. Exact: P(10 losses) = (20/38)^10 = 0.00163.)

#probability #math #martingale #casino #gamblersruin #mathfacts
```

Short alternative:

```
Double after every loss and you "can't lose": 978 of 1,000 simulated Martingale players busted in 5,000 spins. Each round: 99.84% chance of +$1, 0.16% chance of −$1,023. Average: −67¢. It doesn't change if you lose. Only when.

#probability #math #martingale #casino #mathfacts
```

## Pinned comment

```
The maths: with q = 20/38 the chance of 10 losses in a row is q^10 = 0.163%. Expected profit per round = (1 − q^10)·1 − q^10·1,023 = −$0.67. Same thing from the other side: average money wagered per round is $12.74 and the house keeps 2/38 of everything wagered, 12.74 × 0.0526 = 0.67. No betting system can change that number, because the edge is per dollar wagered, not per decision (that's a theorem: for a sequence of unfavourable bets, every stopping/sizing strategy has negative expectation). With an INFINITE bankroll and no table limit Martingale would work, which is exactly why casinos have table limits and you don't have an infinite bankroll. Previous reel: why betting slowly is the worst way to try to double your money.
```

## Audio

Keep the built-in track: chip clicks per spin, a soft coin for each round the solo gambler wins,
ten rising drum hits for the losing streak (one per doubled bet) ending in an impact, a chip-click
rain and a thud per 25 busts for the crowd, bells at the reveals.

## Replies to expect

* "Just stop after you're up $100" → you still face the 0.163% per round; to bank 100 rounds you survive with (1 − q¹⁰)¹⁰⁰ = 85%, and the other 15% lose $1,023. EV is still −$0.67 × 100 rounds.
* "Use a bigger bankroll" → each extra doubling halves the bust probability per round and doubles the loss when it happens; the product, your expected loss, stays the same (it grows with the money wagered).
* "Table limits are the real reason" → they're one cap; your bankroll is the other. Either one turns the infinite-money argument into a finite one and the formula takes over.
* "This is gambler's ruin again" → yes, reel 13 was the slow version; this is the version that feels like winning until it doesn't.
* "Anti-Martingale / Paroli?" → doubling after wins changes the shape too (many small losses, rare big wins), not the −5.26% per dollar.
