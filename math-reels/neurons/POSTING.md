# Posting kit — *This AI has 2 neurons.*

## Files

* Reel: `output/neurons_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the moment the 64-neuron network hits 100%.

## What's on screen (all of it is real)

Three neural networks are trained live inside the render on the same 400-point
two-arm spiral, with Adam at learning rate 0.01, the same seed, and the weights
recorded at every video frame:

| Stage | Hidden layers | Neurons | Steps | Final accuracy |
| --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 1,500 | 65% |
| 2 | 8 | 8 | 2,000 | 82% |
| 3 | 32 + 32 | 64 | 1,500 | 100% (first hit at step 713) |

The background is the network's actual decision field p(x, y); hollow rings are
the points it currently gets wrong; the small diagram draws the real weights
(line brightness = |w|). The accuracy counter is computed from the predictions.

## Caption (paste as-is)

```
Same data. Same training. The only difference is the number of neurons.

2 neurons can't bend the line enough and get stuck at 65%. 8 neurons manage 82%. 64 neurons find the whole spiral and hit 100%.

Your brain has about 86 billion.

This is why AI models keep getting bigger: a network with too few neurons can't represent the pattern no matter how long it trains. (That's roughly what the universal approximation theorem says: with enough neurons, a network can approximate any continuous pattern.)

Which one did you expect to win?

(Trained live in Python: 400 points, two-arm spiral, Adam at lr 0.01, 1,500–2,000 steps each. The background is the real decision field, the hollow rings are the points it gets wrong, the diagram shows the real weights. Every sound is driven by the accuracy.)

#machinelearning #neuralnetwork #ai #deeplearning #datascience
```

Exactly five hashtags.

Shorter alternative:

```
2 neurons: stuck at 65%. 8 neurons: 82%. 64 neurons: 100%. Your brain: 86,000,000,000.

Same data, same training. Neurons are the only difference.

#machinelearning #neuralnetwork #ai #deeplearning #datascience
```

## Pinned comment

```
For the "it's overfitting" crowd: fair point, 100% on the training points doesn't mean it generalises. The video is about capacity, not generalisation: 2 and 8 neurons can't even fit what they're shown; 64 can. "86 billion" is the standard estimate for the human brain (Azevedo et al., 2009). Next: I train the same 64-neuron network on data with the labels shuffled, to show what memorising looks like. Want to see it?
```

## Audio

Keep the built-in track and name it **"the sound of a network learning"**: a
four-note cluster that glides into a major chord as accuracy rises, static that
clears as the loss falls, a tick every 50 training steps, bells at 90% and 100%.

## Before posting

Same checklist as the pendulum reel: Creator account, "Upload at highest quality",
phone check muted and with sound, native upload, cover from `cover.png`, pin the
comment immediately, reply to every comment in the first hour.

## Things to be careful about in replies

* "It's overfitting" → the pinned comment covers it; agree it's a training-set fit, the point is capacity.
* "Brain neurons aren't the same as artificial neurons" → correct; the comparison is about count, say so plainly.
* "Why not one wide layer of 64?" → a single 64-unit layer also gets there, just slower on this spiral; two layers of 32 was chosen so it finishes inside the reel.
* Never claim a parameter count for any commercial model; the only hard number used is 86 billion.

## Series ideas

1. "Same network, shuffled labels" — memorising vs learning (the pinned promise).
2. "Learning rate too high" — the network that explodes (chaos tie-in with reel 01).
3. "One neuron at a time" — add neurons one by one and watch the boundary bend.
4. "The network that learned your handwriting" — MNIST digit forming in the decision field.
