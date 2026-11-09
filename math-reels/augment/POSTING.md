# Posting kit — *Same network, different pictures: moved digits 46% → 94%.* (dark theme, narrated, 64 s)

## Files

* Reel: `output/augment_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — a training 3 and its four shifted copies, the scores for every kind of move for both
  networks, and ten moved digits with what each network reads.
* Runs: `aug_results.json` (from `python augsim.py`, about a minute).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10, same random start, 300 Adam steps) trained two ways on the same
1,257 training digits: on the pictures as they are, and on five copies of each (as it is, and moved one pixel left,
right, up and down: 6,285 pictures; what slides in at the edge is blank). Both are tested on the same 540 unseen
digits, moved or not:

| Test digits | Trained on the pictures | Trained on the 5 copies |
| --- | --- | --- |
| unmoved | 96.7% | 96.5% |
| 1 pixel right | 46.5% | 94.3% |
| 1 pixel left | 39.1% | 93.7% |
| 1 pixel up | 40.9% | 92.8% |
| 1 pixel down | 41.7% | 95.2% |
| 2 pixels right | 18.5% | 52.6% |
| 2 pixels left | 13.7% | 42.8% |

* The ten moved digits: the network trained on the pictures reads 3 of 10 right; the one trained on the copies, 10.
* On six other random splits and starts, 1 pixel right goes from 44–52% to 94–97%.

## Honesty notes

* The copies cost a little on unmoved digits: 96.7% → 96.5% here, and lower on 4 of the 6 other splits (by 1.5 to
  2.4 points; slightly higher on the other 2); same network size and the same 300 steps on five times as many pictures.
* "Image models are routinely trained on shifted, flipped and cropped copies": standard practice in computer vision;
  this reel shows only shifts.
* A convolutional network (reel 52) gets the moves for free from its design; augmentation teaches them by example,
  and only the moves it was shown.

## Caption (paste as-is)

```
This network failed when a digit moved one pixel. I fixed it without changing a single line of the network. I just changed what it saw.

It's the handwriting network: 96.7% right on digits it has never seen. Move every digit one pixel to the right, and it reads only 46%.

So I gave it 5 copies of every training picture: as it is, and moved one pixel left, right, up and down. Same network, same 300 steps of training.

Now, moved one pixel right: 94%. Left: 94%. Up and down: 93% and 95%. Unmoved digits: still 96.5%.

But it only learns what it's shown. Move a digit two pixels, a move it never saw, and it drops to 53%.

That's data augmentation: when you can't get more data, make more from what you have. Image models are routinely trained on shifted, flipped and cropped copies.

(Same 540 unseen digits for both networks. Every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #dataaugmentation #computervision #datascience
```

## YouTube

Title:

```
Same Network, Different Pictures: 46% → 94% #shorts
```

Description:

```
Data augmentation, measured on a real handwriting network.

The network (64 → 32 → 10, same start, 300 Adam steps) trained on the 1,257 UCI training digits reads 96.7% of 540 held-out digits, but only 46.5% when they move one pixel right. Trained on five copies of each picture (unmoved, and shifted one pixel left, right, up and down), it reads 94.3% of the moved digits (left 93.7%, up 92.8%, down 95.2%) and 96.5% of the unmoved ones. Shifts of two pixels, never in the copies: 52.6% (from 18.5%). On six other random splits, one pixel right goes from 44–52% to 94–97%.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #dataaugmentation #neuralnetwork #shorts
```

## Pinned comment

```
Why did one pixel break it? This network gives every pixel position its own weights, so a 3 moved one pixel right puts its ink on weights that never saw a 3 there. The shifted copies show it the same digits in neighboring positions, so those weights learn too. A convolutional network gets this for free by sliding the same small filters over every position.
```
