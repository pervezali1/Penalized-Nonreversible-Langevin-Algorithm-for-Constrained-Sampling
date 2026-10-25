# Posting kit — *Drop a needle on floorboards. That's π.*

## Files

* Reel: `output/buffon_reel.mp4` — 25.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.5 LUFS.
* Cover: `output/cover.png` — Lazzarini's histogram with the 1,808 bar lit, "1,000 repeats: 15 hit 1,808."

## What's on screen (all of it is real)

**The rain.** Floorboards with cracks every 100 px; needles exactly as long as the gap, centre
uniform on the floor, angle uniform in [0, π) (numpy, seed 3). A needle crosses a crack when half
its vertical extent, ½·|sin θ|·gap, reaches the nearest crack. Crossing needles are amber, misses
cyan; at high counts the cracks emerge as amber stripes by themselves. With needle = gap the
crossing probability is 2/π, so π ≈ 2 × needles ÷ crossings. The estimate is shown to six decimals,
correct leading digits in green.

| Needles | Crossing | Estimate | Correct digits |
| --- | --- | --- | --- |
| 100,000 | 63,802 | 3.134698 | 2 |

(The standard error of this estimate at 100,000 needles is 0.0075, so two or three digits is
exactly what to expect. That slowness is the setup for the fraud at the end.)

**Why π.** One needle rotates through 180° on a small floor while a graph fills with its crossing
chance at each angle, |sin θ|. The running average is shown live and lands on 2/π = 0.6366; the
dashed line on the graph is that average. The reason π appears at all is that the needle's angle
is uniform on a circle of directions, and the average of |sin θ| over that circle is 2/π.

**Lazzarini, 1901.** Mario Lazzarini reported 3,408 needle drops with needle = 5/6 of the gap and
1,808 crossings, giving π ≈ 5 × 3,408 ÷ (3 × 1,808) = 3.1415929, which is 355/113, correct to six
decimals. The reel shows what one crossing either way would have given: 1,807 → 3.1433, 1,809 →
3.1399. Only exactly 1,808 produces six digits. Then his experiment is simulated 1,000 times (numpy,
seed 7): the histogram of crossing counts, mean 1,808, standard deviation 29, and **15 of 1,000**
repeats land on 1,808 exactly. The accepted explanation (Gridgeman 1960, Badger 1994) is that he
chose the needle length and stopped at a count that reproduces 355/113: 3,408 = 16 × 213 and
1,808 = 16 × 113, so his numbers are that fraction in disguise.

## Caption (paste as-is)

```
In 1777 Buffon asked: drop a needle on a floor of parallel boards, how often does it cross a crack? If the needle is as long as the gap, the answer is 2/π. So you can measure π by dropping needles and counting.

I dropped 100,000 of them. Two correct digits. That's not a bug, it's the method: the error shrinks like 1/√N.

Why does π show up at all? Because of the needle's angle. A needle at angle θ crosses a crack with probability |sin θ|, and the average of |sin θ| over every direction is 2/π. The reel shows that average being built, angle by angle.

Now the fraud. In 1901 Mario Lazzarini published the result of 3,408 needle drops: π ≈ 3.1415929. Six correct digits. From 3,408 needles.

Look at what the neighbouring outcomes give. 1,807 crossings: 3.1433. 1,809 crossings: 3.1399. Exactly 1,808: 3.1415929, which happens to be 355/113, the famous fraction. I ran his experiment 1,000 times. Fifteen of them hit 1,808. He either got 1-in-70 lucky, or he kept dropping needles until the count said what he wanted, and then stopped.

Random can find π. Only if you don't peek at the answer.

(numpy, 100,000 needles with needle = gap; 1,000 × 3,408 drops with needle = 5/6 gap for Lazzarini; π = 2 × N ÷ crossings.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #pi #buffonsneedle #probability #statistics #mathfacts
```

Short alternative:

```
Drop needles on floorboards, count the crossings, get π. 100,000 needles: two digits. In 1901 a man claimed six digits from 3,408 needles. One crossing more or fewer and he'd have had two. He stopped counting at the right moment.

Follow for more.

#math #pi #buffonsneedle #probability #mathfacts
```

## Pinned comment

```
The maths: a needle of length l on boards d apart, with l ≤ d, crosses a crack with probability 2l/(πd). For l = d that's 2/π ≈ 0.6366. Where π comes from: the chance of crossing at angle θ is (l/d)·|sin θ|, and averaging |sin θ| over θ from 0 to π gives (1/π)·∫|sin θ| dθ = 2/π. Lazzarini used l/d = 5/6, so π ≈ (5/3)·N/C; with N = 3,408 you need C = 1,808 to get 355/113 and the step between neighbouring outcomes is 0.0017, a thousand times the "error" he reported. Gridgeman pointed out in 1960 that 3,408 = 16 × 213 and 1,808 = 16 × 113: the numbers are 355/113 in disguise. Previous reel: the same slowness with darts instead of needles.
```

## Audio

Keep the built-in track: a wooden clack per needle while they can be heard one by one (higher when
it crosses), then a rain following needles per frame; a coin when a digit locks in and a knock when
one slips; a pad whose pitch follows |sin θ| as the needle turns; typewriter ticks as Lazzarini's
experiment is repeated and a stamp for every exact hit; bells at the reveals.

## Replies to expect

* "Isn't this the same as the darts reel?" → same 1/√N slowness, different mechanism: darts measure an area, needles measure an angle average, which is why the reel spends four seconds on |sin θ|.
* "Maybe Lazzarini was just lucky" → 1.5% chance, and the choice of 5/6 and 3,408 is exactly what you'd pick if you wanted 355/113; the two together are the tell.
* "Why needle = gap?" → it makes the crossing probability exactly 2/π; any l ≤ d works with 2l/(πd).
* "What if the needle is longer than the gap?" → the formula changes (it can cross two cracks); the 2l/(πd) version needs l ≤ d.
* "Why 355/113?" → the best fraction for π with a small denominator, known to Zu Chongzhi in the 5th century; it was the obvious target to aim at.
