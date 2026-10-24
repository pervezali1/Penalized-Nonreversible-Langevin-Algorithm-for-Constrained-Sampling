# Posting kit — *You have $100. Goal: $200.*

## Files

* Reel: `output/ruin_reel.mp4` — 22.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the river of 1,000 bankrolls sunk to the floor, "Doubled: 1 of 1,000. Broke: 998."

## What's on screen (all of it is real)

Roulette red/black: win probability 18/38 = 47.37%, house edge 5.26%. Start $100, goal $200.

| Strategy | What happens | Doubled (simulated, 1,000 gamblers) | Exact probability |
| --- | --- | --- | --- |
| A: $1 a bet until $0 or $200 | the walks fan out, then the edge drags them to the floor; one lucky walk touches $200 around bet 2,100, 998 go broke, one is still grinding at bet 7,000 | **1** (998 broke) | 1 in **37,650** (2.66 × 10⁻⁵) |
| B: everything on one spin | one jump: 496 up, 504 down | **496** | **47.37%** |

Exact formula (gambler's ruin, unit stakes): P(reach N from i) = (1 − (q/p)ⁱ) / (1 − (q/p)ᴺ) with
p = 18/38, q = 20/38, i = 100, N = 200. The chart is the real bankroll of each of the 1,000
simulated gamblers (seed 12); magenta dots are ruins, cyan would be doublings.

## Caption (paste as-is)

```
You walk in with $100 and want to leave with $200. Roulette, red or black, 47.4% to win each spin.

Strategy A, the "sensible" one: bet $1 at a time, grind it out. I simulated 1,000 people doing exactly that. 998 went broke. One got lucky. The exact odds of doubling this way: 1 in 37,650.

Strategy B: put the whole $100 on one spin. 496 of 1,000 walk out with $200.

Same money, same game, same edge. The difference is how many times you let the house take its 5%. Every bet is another slice; a thousand small bets are a thousand slices. That's gambler's ruin: against a house with more money and an edge, the slow careful player is the one who's guaranteed to lose.

(And no, this isn't gambling advice. The only strategy that beats a 5% edge is not playing.)

Which strategy would you have picked?

(Exact: P(double) = (1 − (q/p)^100) / (1 − (q/p)^200), p = 18/38. Simulation: numpy, 1,000 bankrolls, $1 bets to $0 or $200.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#probability #math #gamblersruin #casino #mathfacts
```

Short alternative:

```
$100 → $200 on roulette. $1 at a time: 1 in 37,650. All in once: 47%. Same game, same edge. Every bet pays the house. Gambler's ruin.

Follow for more.

#probability #math #gamblersruin #casino #mathfacts
```

## Pinned comment

```
The maths: with win probability p and loss probability q, betting $1 from i toward N succeeds with probability (1 − (q/p)^i)/(1 − (q/p)^N). For red/black q/p = 20/18, so from 100 to 200 that's about 1 in 37,650. With a FAIR coin it's exactly i/N = 50%, and against a house with unlimited money and unlimited time, even a fair coin ruins you with probability 1 (it just takes forever). Bold play (one big bet) is provably optimal for reaching a target in any game where the odds are against you (Dubins & Savage, 1965). Next: why the Martingale "double after every loss" system fails. Want it?
```

## Audio

Keep the built-in track, **"the sound of going broke"**: a chip-click rain whose density
follows bets per frame, a minor pad whose pitch sinks with the average bankroll, a thud for
every 25 ruins, a swoosh and one big flip for the all-in, bells at the reveals.

## Replies to expect

* "This promotes gambling" → the caption says the opposite in plain words; the maths says the only winning move is not to play.
* "What about Martingale?" → it changes when you lose, not whether; the next reel.
* "With a fair coin I'd be fine" → 50% to double, but infinite play against infinite money is certain ruin (pinned comment).
* "Simulation luck?" → the exact probability is in the caption; 1 of 1,000 is luckier than 1 in 37,650, which is exactly the point about small samples, and 496 of 1,000 is bang on 47.4%.
* "One of them doubled, so it works" → one in a thousand simulated, one in 37,650 for real; that one gambler is the lottery winner the casino puts on the poster.
