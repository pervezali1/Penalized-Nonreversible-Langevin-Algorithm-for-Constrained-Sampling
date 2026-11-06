# Posting kit — *Every weight: one bit.* (dark theme, narrated, 61 s)

## Files

* Reel: `output/quantize_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the 32 neurons with every weight just + or − (red or blue), the first layer's 2,048
  weights on two values, 296 bytes, 93.7%.
* Runs: `quant_results.json` (from `python quantsim.py`, under a minute).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10; 2,368 weights of 32 bits; 96.7% of the 540 unseen digits right),
retrained exactly. Every weight rounded, one scale s per layer; the 42 biases are left at 32 bits:

| Weights | Allowed values per layer | Right on 540 unseen digits | Weights take |
| --- | --- | --- | --- |
| 32 bits (as trained) | — | 96.7% | 9,472 bytes |
| 8 bits: s · round(w / s), s = max\|w\| / 127 | 255 (133 used in layer 1) | 96.7% | 2,368 bytes |
| 6 bits | 63 | 96.5% | 1,776 bytes |
| 4 bits: s = max\|w\| / 7 | 15 (14 used) | 93.3% | 1,184 bytes |
| 3 bits | 7 | 75.0% | 888 bytes |
| 1 bit: s · sign(w), s = mean\|w\| | 2 | 50.2% | 296 bytes |
| 1 bit + 100 steps of retraining | 2 | 93.7% | 296 bytes |

* "Retraining": 100 Adam steps (step 0.01) on the 1,257 training digits with the rounding inside the network (the
  gradient at the rounded weights moves hidden unrounded copies; the network that is scored is only the rounded
  weights). 163 of the 2,368 signs flip. The same retraining at 4 bits gives 96.3%.
* On 6 other random splits and starts: 1 bit straight away 38.0–75.7%; with the 100 steps 90.7–95.4%.
* The pictures: each neuron's 64 first-layer weights, red +, blue −, each neuron scaled to its own largest weight;
  the histogram: the 2,048 first-layer weights by value (square-root heights; the 24 beyond ±1.5 drawn at the ends).
* The strip: what the network reads for ten unseen digits at each stage (the retrained 1-bit network reads the 9 as a 2).

## Honesty notes

* "Thirty-two times smaller" is the weights: 9,472 bytes → 296 bytes. With the 42 biases (still 32 bits) and the two
  scales, the whole network goes from 9,640 to 472 bytes, about 20 times smaller.
* "At most two hundred and fifty-five values": 8 bits allow 255 values per layer here (−127 to 127 times s); the
  first layer actually uses 133 of them.
* Retraining is a second, short round of training on the same 1,257 digits.
* Real systems quantize more carefully (a scale per channel or per small group of weights, calibration data, and
  formats such as 8-bit and 4-bit integers or floats); this is the simplest version, on a small network.

## Caption (paste as-is)

```
Every weight in this network is now just one bit: plus or minus. It still reads 93.7% of handwriting.

It's the network from the handwriting reel: 2,368 weights, each stored in 32 bits, 96.7% right on digits it has never seen.

Round every weight to 8 bits (at most 255 values): still 96.7%. 4 bits (15 values): 93.3%.

Now 1 bit: each weight keeps only its sign, plus or minus the same size. Just rounding, it falls to 50%.

But train it 100 more steps with the rounding built in. 163 signs flip, and it's back to 93.7%, with weights 32 times smaller.

That's quantization: how big AI models squeeze onto laptops and phones, often at 8 or 4 bits per weight.

(Every weight, picture and score on screen is computed from the real network.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #quantization #datascience #modelcompression
```

## YouTube

Title:

```
Every Weight in This Neural Network Is 1 Bit. It Still Works. #shorts
```

Description:

```
Weight quantization, measured on a real handwriting network.

The network (64 → 32 → 10, 2,368 weights, 96.7% on 540 held-out UCI digits), rounded with one scale per layer: 8 bits 96.7%, 6 bits 96.5%, 4 bits 93.3%, 3 bits 75.0%. Binarized (sign × mean |w| per layer): 50.2%. With 100 steps of quantization-aware training (straight-through gradients, Adam): 1 bit 93.7% (163 signs flipped), 4 bits 96.3%. On 6 other random splits, the 1-bit network goes from 38–76% to 90.7–95.4% with the retraining.

Quantization, along with pruning and distillation, is one of the standard ways to make trained networks smaller and faster.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #neuralnetwork #quantization #shorts
```

## Pinned comment

```
Why can retraining rescue 1 bit? Rounding the trained weights all at once throws away which weights were big and which were tiny, and nothing makes up for it. Training with the rounding inside lets the network re-choose its signs knowing it only gets plus or minus (163 of them flipped), so the pattern of signs takes over the job the sizes used to do.
```
