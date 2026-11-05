# Posting kit — *Ten layers without ReLU are one neuron.* (dark theme, narrated, 62 s)

## Files

* Reel: `output/relu_reel.mp4` — 62.2 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the same 10-layer network with ReLU: 98.5%, the map following the spiral, the two
  training curves (with ReLU, and flat at 62.0% without).
* Runs: `relu_results.json` (from `python relusim.py`, under a minute; `relu_fields.npz` is rebuilt by the scene if
  missing).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).
  "ReLU" is voiced "RAY-loo" (a pronunciation fix in `voicekit.py`; espeak spells it out).

## What's on screen (all of it is real)

The two-arm spiral of reels 31–47 (400 points). The network: 2 inputs → 10 layers of 8 neurons → 1 output (681
numbers: 2·8 + 8, then 9 × (8·8 + 8), then 8 + 1), binary cross-entropy, backpropagation and Adam (step 0.01),
1,500 steps, both runs from the same random start (seed 1).

* **No activation** (every layer is just h W + b): 62.0%, 248 of the 400 points. A single neuron, p = σ(w₁x + w₂y + b),
  trained the same way: 62.0%, 248 points.
* **The collapse:** (x W₁ + b₁) W₂ + b₂ = x (W₁W₂) + (b₁W₂ + b₂). Merging the layers one at a time leaves 681, 609,
  537, 465, 393, 321, 249, 177, 105, 33 and finally 3 numbers: p = σ(−0.0342 x + 1.4497 y + 0.0020). At every stage the
  merged network's output matches the original on all 400 points and a 96 × 96 grid to within 4.4 × 10⁻¹⁵ (14 decimal
  places). The single neuron trained on its own lands on the same line: (−0.0342, 1.4497), 0.0020.
* **With ReLU** (ReLU(z) = max(0, z) after every hidden neuron), same start, same training: 98.5%, 394 of 400.
* The network diagram colours each connection by its weight (red +, blue −) from snapshots every 15 steps; while the
  layers merge, the connections drawn are the products of the merged weight matrices.

## Honesty notes

* Over 9 random starts (seeds 0–8), no activation gives 62.0% every time; ReLU gives 71.0–99.5%, median 98.5%. The run
  shown (seed 1) is the median. Deep, narrow ReLU networks sometimes get stuck (one start reached only 71%).
* The ReLU run's accuracy dips briefly at a few steps (visible in its curve): Adam overshooting, then recovering.
* Without an activation, training can still move the line; it just can never be anything but a line. 62% is what the
  line it learns gets on this spiral (the line of lowest cross-entropy, which the single neuron finds too; a line
  chosen for accuracy alone might do a little better, but it is still one line).

## Caption (paste as-is)

```
This network has 10 layers and 681 numbers. All it can ever draw is a straight line.

Train it on the spiral with 1,500 steps of backpropagation: 62%. Exactly what one single neuron gets.

The problem: there's no activation. Each layer just multiplies by a matrix and adds some numbers. And a matrix times a matrix is just one matrix.

So the 10 layers collapse, one by one, into a single neuron: 681 numbers → 3. Same answer for every point, to 14 decimal places. It even lands on the same line as a single neuron trained on its own.

Now add one tiny step after every neuron: if the number is negative, make it zero. That's ReLU.

Same 10 layers. Same starting numbers. Same training. 98.5%.

Those bends are what let a network learn a shape. Without them, all that depth does nothing.

(Every map, number and training curve on screen is a real run.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #relu #linearalgebra #ai #backpropagation
```

## YouTube

Title:

```
10 Layers Without ReLU = 1 Neuron (Why Networks Need Activations) #shorts
```

Description:

```
Why neural networks need activation functions, measured.

A 2 → 8 × 10 → 1 network (681 parameters) on a 400-point two-arm spiral, trained with backpropagation and Adam for 1,500 steps. With no activation function it reaches 62.0%, the same as a single neuron (logistic regression). The reason: composing affine maps gives an affine map, (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂), so the ten layers collapse exactly into one neuron with 3 parameters, p = σ(−0.0342x + 1.4497y + 0.0020), matching the full network to 14 decimal places.

The same network from the same initialization with ReLU after every hidden unit reaches 98.5% (median of 9 random starts: 98.5%; without activation: 62.0% every time).

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#relu #neuralnetwork #machinelearning #shorts
```

## Pinned comment

```
Why one ReLU is enough to break the collapse: max(0, z) is a straight line on each side of zero, but it bends at zero, and that bend is in a different place for every neuron (it depends on its weights and bias). Stack them and you get a function made of many flat pieces joined at many bends, which can follow a spiral as closely as you like. Without the bends, every layer is a straight-line map, and straight-line maps compose into one straight-line map, however many you stack.
```
