# Posting kit — *This test is 99% accurate. You test positive. Are you sick?* (dark theme, no narration, 22 s)

## Files

* Reel: `output/bayes_reel.mp4` — 22.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — everyone who tested positive in one block: one row of 10 sick, 114 false alarms,
  "Only 8% are sick."
* Simulation: `bsim.py` (seed 69, chosen before running; `python bsim.py out.json` writes every number below).

## What's on screen (all of it is real)

The medical test paradox (Bayes' theorem). 10,000 people, exactly 10 of them sick (1 in 1,000), placed at random.
The test is right 99% of the time for everyone: it catches 99% of sick people and clears 99% of healthy people.

| | Simulated (10,000 people) | Exact |
| --- | --- | --- |
| Test positive | 124 (10 sick + 114 false alarms) | 10 × 0.99 + 9,990 × 0.01 = 109.8 |
| Sick, given one positive test | **8.1%** (10 of 124) | **9.0%** |
| Positive on both tests | 11 (10 sick + 1 false alarm) | — |
| Sick, given two positive tests | **90.9%** (10 of 11) | **90.75%** |

* The crowd is a 100 × 100 grid, one dot per person. The 10 sick light up (magenta), a scan line tests every row,
  positives glow (amber), then all 124 positives fly into one block: the 10 sick in the top row, the 114 false alarms
  below. The second test clears 113 of the false alarms one by one; 1 tests positive again, and all 10 sick do.
* No sick person was missed by either test in this run (expected misses: 0.1 per test).
* The HUD counts are computed per frame; the exact answers are printed under the block.

## Honesty notes

* 114 false alarms is more than the 99.9 expected (1.4 standard deviations, ordinary luck), which is why the
  simulated 8.1% is below the exact 9.0%; the reel shows both.
* "99% accurate" here means both: 99% of sick people test positive and 99% of healthy people test negative. Real
  tests have different rates for the two.
* The answer depends on how rare the disease is: if 1 in 10 people had it, the same positive test would mean 91.7%
  sick. The paradox is about rare conditions.
* The second test is assumed to make its mistakes independently of the first. In practice a confirming test is
  often a different kind of test for exactly this reason.
* A teaching example with round numbers, not medical advice.

## Caption (paste as-is)

```
This test is 99% accurate. You test positive. What's the chance you're actually sick?

Most people say 99%. So I simulated 10,000 people.

1 in 1,000 has the disease: 10 people. Test everyone. 124 test positive.

Only 10 of them are sick. That's 8%. (The exact maths says 9%.)

Why? The test is wrong just 1% of the time, but 1% of 9,990 healthy people is about 100 false alarms: ten times more than all the sick people put together.

Test the 124 again: 11 are positive twice, and 10 of those are sick. 91%.

One positive test for a rare disease isn't a diagnosis. That's why a positive screening test is usually followed by a second test.

Did you guess 99%? 👇

(A real simulation: 10,000 people, a test that's right 99% of the time for sick and healthy people alike. Seed 69, not cherry-picked.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #probability #statistics #bayes #datascience
```

Exactly five hashtags.

## YouTube

Title:

```
This Test Is 99% Accurate. You Test Positive. Are You Sick? #shorts
```

Description:

```
The medical test paradox, simulated. 10,000 people, 10 of them sick (1 in 1,000). A test that is right 99% of the time, for sick and healthy people alike, flags 124 of them: all 10 sick people and 114 false alarms. So a positive test means only 8% sick here (exact: 9.0%). Test the 124 again: 11 are positive twice, 10 of them sick, 91% (exact: 90.75%). The rarer the disease, the more the false alarms outnumber the real cases.

#math #probability #bayes #shorts
```

## Pinned comment

```
The exact maths (Bayes' theorem): out of 1,000 people, 1 is sick and tests positive (99% of the time), and 999 are healthy, of whom 1% test positive anyway: 9.99 false alarms. So P(sick | positive) = 0.99 / (0.99 + 9.99) = 9.0%. After a second positive, the false alarms shrink by another factor of 100 while the sick barely change: 0.98 / (0.98 + 0.0999) = 90.75%. And if the disease were common (1 in 10), a single positive would already mean 91.7%. Rare disease + one positive test = test again.
```

## Audio

Keep the built-in track: ten low plucks as the sick light up, a rising tone during the sweep with a plink for every
positive (lower for the sick), a whoosh as the positives gather, a bell and impact on "only 8%", a falling tick for
each false alarm cleared by the second test, a bright bell on "positive twice".

## Replies to expect

* "Then why are tests called 99% accurate?" → they are: about 99% of results are right. Here 9,886 of 10,000 results were
  right; but almost all of those are healthy people testing negative.
* "Real tests aren't like that" → right, real rates differ (honesty notes); the effect holds whenever the condition
  is rare compared with the false-alarm rate.
* "What if 1 in 10 had it?" → 91.7% after one positive (pinned comment).
