# Posting kit — *A network that learned to read from 10 pictures.* (dark theme, narrated, 56 s)

## Files

* Reel: `output/fewshot_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — all 1,257 training pictures, 96.7%, the accuracy curve against the number of pictures with
  the two gains marked.
* Runs: `few_results.json` (from `python fewsim.py`, a few seconds).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10, the same 300 Adam steps every time), trained on k randomly picked
examples of each digit out of the 1,257 training digits, and tested on the same 540 unseen digits. Each size is trained
on 10 different random picks (each with its own random start); the median is reported, the spread shown as a bar, and
the training set and the strip come from the run just above the median.

| Pictures | Right on 540 unseen digits: median (range over 10 picks) | Right on its own training pictures |
| --- | --- | --- |
| 10 (1 of each) | 58.7% (52.6–67.4%) | 100% |
| 20 | 73.7% (67.4–81.9%) | 100% |
| 30 | 79.1% (72.8–81.9%) | 100% |
| 50 | 84.3% (80.6–89.1%) | 100% |
| 100 | 89.4% (85.9–90.9%) | 100% |
| 200 | 92.9% (90.6–94.3%) | 100% |
| 500 | 95.1% (93.7–96.1%) | 100% |
| 1,257 (all) | 96.7% | 100% |

From 10 to 100 pictures: +30.6 points. From 500 to 1,257: +1.6 points.

## Honesty notes

* The narration rounds: "fifty-nine" (58.7%), "eighty-nine" (89.4%), "ninety-five" (95.1%), "thirty-one points" (30.6).
* Every run trains for the same 300 steps, so the small sets are trained to memorization (100% on their own pictures);
  other training choices (fewer steps, regularization, data augmentation) would change the small-set numbers.
* The digits come from 13 writers, and the 540 test digits share those writers, so "new handwriting" means new digits,
  not new people.

## Caption (paste as-is)

```
This network learned to read from just 10 pictures, one of each digit. It still gets 59% of new handwriting right.

It's the network from the handwriting reel. Normally it learns from 1,257 digits. So what happens with fewer?

One example of each digit: 59%. It gets all 10 of its own pictures right, every time. It just has no idea how differently other people write.

10 of each, 100 pictures: 89%. 50 of each: 95%.

All 1,257: 96.7%. Going from 10 pictures to 100 bought 31 points. The last 757 bought less than 2.

More data always helps. But each new picture helps a little less than the one before.

(Every score is the median of 10 random picks of training pictures, tested on the same 540 unseen digits.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #datascience #data #learning
```

## YouTube

Title:

```
A Neural Network That Learned to Read From 10 Pictures #shorts
```

Description:

```
How accuracy grows with training data, measured.

The handwriting network (64 → 32 → 10, trained with Adam for 300 steps) on k examples per digit of the UCI handwritten digits, tested on 540 held-out digits (median of 10 random subsets each): 10 pictures 58.7%, 20 73.7%, 50 84.3%, 100 89.4%, 200 92.9%, 500 95.1%, all 1,257 96.7%. Every subset is learned perfectly (100% training accuracy); the gap is generalization. Going from 10 to 100 examples gains 30.6 points; from 500 to 1,257, 1.6: diminishing returns.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #datascience #neuralnetwork #shorts
```

## Pinned comment

```
Why the first pictures matter most: with one example per digit, the network only knows one way to write each digit, so any 4 that looks different is a guess. Each new picture shows it another way to write it, but after a few hundred, most new pictures look like ones it has already seen, so they teach it less. Large AI models show the same pattern: each further improvement takes far more data.
```
