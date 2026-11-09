# Posting kit — *10 digits that fooled 10 neural networks. Can you read them?* (dark theme, narrated, 62 s)

## Files

* Reel: `output/ensemble_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the challenge: the ten digits with question marks (no answers), and the scores of the
  ten networks and their vote.
* Runs: `ens_results.json` (from `python enssim.py`, about a minute).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10, 300 Adam steps on the same 1,257 training digits), trained ten times
from ten random starts (seeds 1–10; seed 3 is reel 49's network), tested on the same 540 unseen digits.

* Alone: 523, 522, 522, 519, 520, 521, 520, 523, 521, 521 right (median 521). Their majority vote: 523 (96.9%), the
  same as the best of them. (Averaging their probabilities instead of voting: 524.)
* 34 digits fool at least one network; 10 fool all ten. On 8 of those 10, all ten give the same wrong answer:

| Real digit | 1 | 6 | 8 | 8 | 4 | 3 | 3 | 6 | 5 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The networks' answer | 9 (10 of 10) | 1 (10) | 2 (10) | 1 (10) | 9 (10) | 5 (10) | 2 (10) | 5 (7) | 6 (10) | 8 (8) |
| Nearest neighbour | 1 | 6 | 8 | 8 | 4 | 3 | 3 | 6 | 6 | 3 |

* Nearest neighbour (reel 57: copy the label of the closest training picture) reads 8 of these 10 right.

## Honesty notes

* "The real answers" are the labels the writers gave in the data set (UCI handwritten digits); a couple of these
  digits are hard for people too.
* Nearest neighbour is chosen to show a method with different mistakes; overall it makes mistakes of its own (7 of
  the 540, reel 57). A vote that mixes different methods is not computed here.
* Ten random starts on the same pictures is the weakest kind of ensemble; ensembles that train on different data or
  different models disagree more, and usually gain more.

## Caption (paste as-is)

```
These 10 handwritten digits fooled 10 different neural networks. Every single one of them. Can you read them?

I trained the same network 10 times, from 10 random starts. Alone, each reads between 519 and 523 of the 540 unseen digits.

So I let them vote. The vote gets 523 right: no better than the best network on its own.

Because they don't make different mistakes. They all learned from the same pictures. On these 10 digits all ten networks were wrong, and on 8 of them, all ten gave the same wrong answer.

The real answers are in the video. How many did you get? 👇

The lazy nearest neighbor method, which just compares pixels, reads 8 of these 10 correctly. A vote only helps when the voters make different mistakes.

(Every network, answer and score on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #ai #deeplearning #ensemble #datascience #challenge
```

## YouTube

Title:

```
10 Digits That Fooled 10 Neural Networks. Can You Read Them? #shorts
```

Description:

```
An ensemble of ten networks, measured: why voting didn't help.

The handwriting network (64 → 32 → 10, 300 Adam steps) trained from 10 random starts on the same 1,257 UCI digits reads 519–523 of 540 held-out digits (median 521). Their majority vote reads 523, the same as the best single network. 34 test digits fool at least one network and 10 fool all ten, 8 of them with the same wrong answer from every network. 1-nearest neighbor reads 8 of those 10 correctly. (The answers are revealed in the video.)

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #ensemble #neuralnetwork #shorts
```

## Pinned comment

```
Why didn't the vote help? A vote fixes a mistake only when most voters get that digit right. Ten copies of the same network, trained on the same pictures, learn nearly the same rules, so the digits that fool one usually fool them all: here, 10 digits fooled every one of them. Voters that learned differently (other data, other methods) disagree on different digits, and that disagreement is what a vote turns into accuracy.
```
