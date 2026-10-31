# Posting kit — *Backpropagation, with nothing but the maths.* (dark theme, equations only, 60 s)

## Files

* Reel: `output/backpropmath_reel.mp4` — 60.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — page 5, the chain rule through the fold: every derivative with its numbers.
* No archive: `scene.py` recomputes everything at start-up from reel 33's `../backprop/bpsim.py` (reel 31's
  training, one ReLU neuron, seed 1110, stopped at step 300; example #396).

## What's on screen (all of it is real)

No diagram and no field: typeset equations only, with italic variables, stacked fractions and every
number computed from the trained weights. A table under the page holds the five parameters (w₁, w₂, b in
the neuron; v, c at the output), and fills in their derivatives and their new values as the pages compute
them.

**Opening.** The whole chain for one weight:
∂L/∂w₁ = ∂L/∂s · ∂s/∂h · ∂h/∂z · ∂z/∂w₁ = (−0.510) · 1.032 · 1 · (−0.874) = 0.460.

1. **The network.** z = w₁x₁ + w₂x₂ + b; h = max(0, z); s = v·h + c; p = σ(s) = 1/(1 + e⁻ˢ); L = −log p.
   Example: x₁ = −0.874, x₂ = 0.051, answer y = 1. Parameters: w₁ = −0.004, w₂ = 1.768, b = 0.474,
   v = 1.032, c = −0.627.
2. **Forward.** z = (−0.004)·(−0.874) + 1.768·0.051 + 0.474 = 0.568; h = max(0, 0.568) = 0.568;
   s = 1.032·0.568 − 0.627 ≈ −0.040; p = σ(−0.040) = 0.490; L = −log 0.490 ≈ 0.714.
3. **Backward, from the loss.** ∂L/∂s = ∂L/∂p · ∂p/∂s = (−1/p) · p(1 − p) = p − 1 (in general p − y)
   = 0.490 − 1 = −0.510.
4. **Backward, the output neuron.** ∂L/∂c = ∂L/∂s · 1 = −0.510; ∂L/∂v = ∂L/∂s · h = (−0.510)·0.568 = −0.290;
   ∂L/∂h = ∂L/∂s · v = (−0.510)·1.032 ≈ −0.527.
5. **Backward, through the fold.** ∂h/∂z = ReLU′(z) = 1 because z = 0.568 > 0 (the neuron was on);
   ∂L/∂z = (−0.527)·1 = −0.527; ∂L/∂w₁ = ∂L/∂z · x₁ = (−0.527)·(−0.874) ≈ 0.460;
   ∂L/∂w₂ = ∂L/∂z · x₂ = (−0.527)·0.051 = −0.027; ∂L/∂b = ∂L/∂z · 1 = −0.527. Had z been negative,
   these three would all be 0.
6. **Check.** [L(w₁ + 0.001) − L(w₁ − 0.001)] / 0.002 = (0.714011 − 0.713091)/0.002 = 0.460, the same as
   backpropagation (full precision: 0.460233 both ways).
7. **Update.** θ ← θ − η·∂L/∂θ with η = 0.3: w₁ ← −0.004 − 0.3·0.460 = −0.142; w₂ ← 1.768 − 0.3·(−0.027) = 1.776;
   b ← 0.474 − 0.3·(−0.527) = 0.632; v ← 1.032 − 0.3·(−0.290) = 1.119; c ← −0.627 − 0.3·(−0.510) = −0.474.
8. **Forward again.** z = 0.847; s = 1.119·0.847 − 0.474 = 0.474; p = σ(0.474) = 0.616; L = −log 0.616 ≈ 0.484.
   Loss 0.714 → 0.484, guess 0.490 → 0.616. On all 400 points: 249 → 255 right.
9. **Any network, any depth.** δ_out = p − y; δⱼ = ReLU′(zⱼ) · Σₖ wⱼₖ δₖ; ∂L/∂wᵢⱼ = hᵢ δⱼ and ∂L/∂bⱼ = δⱼ;
   w ← w − η·∂L/∂w.

This is the same step as reel 33's one-neuron stage (same network, same example, same step size 0.3);
`scene.py` asserts that its derivatives match `bpsim.one_example` to 1e-12.

## Honesty notes

* Every number is shown to 3 decimals and computed in full precision. A relation is drawn as "=" when
  the rounded numbers on screen reproduce the rounded result, and as "≈" when rounding makes them miss in
  the last digit. The scene decides this line by line; five lines use "≈": s (−0.041 from the rounded
  numbers, −0.040 true), L (0.713 vs 0.714), ∂L/∂h (−0.526 vs −0.527), ∂L/∂w₁ (0.461 vs 0.460) and the
  second L (0.485 vs 0.484). The footer says so.
* One example and one plain gradient step, so the arithmetic can be followed. Real training averages
  the derivatives over all the points and, in reel 31, uses Adam.
* The one-neuron network is reel 31's typical run; it can only ever reach 62% on the spiral (a single
  straight fold). The point here is the arithmetic of one step, not the accuracy.

## Caption (paste as-is)

```
Backpropagation, with nothing but the maths.

One neuron. One example it gets wrong. Every number on screen comes from the real trained network.

Forward: z = w₁x₁ + w₂x₂ + b = 0.568. The fold: h = max(0, z) = 0.568. Out: s = v·h + c ≈ −0.040. The guess p = σ(s) = 0.490. The answer is 1, so the loss L = −log p ≈ 0.714.

Backward is the chain rule, one factor at a time. At the top, −log and σ cancel into something beautiful: ∂L/∂s = p − y = −0.510. Then each derivative is the one above times one local factor: ∂L/∂v = −0.510 × 0.568. ∂L/∂h = −0.510 × 1.032. Through the fold the factor is 1, because the neuron was on. And ∂L/∂w₁ = −0.527 × −0.874 ≈ 0.460.

Don't trust it? Nudge w₁ by ±0.001 and measure: (0.714011 − 0.713091) / 0.002 = 0.460. Same number.

Then every parameter steps against its derivative, w ← w − 0.3 × ∂L/∂w, and the loss falls from 0.714 to 0.484.

Any network, any depth, it is the same four lines: the error at the output, the blame passed back one layer (weights × blame, times 0 if the neuron was off), a weight's gradient = its input × the blame where it lands, and the step. Repeat a few thousand times. That's training.

(The same step as the last reel, now as pure maths. Numbers rounded to 3 decimals and computed in full; "≈" marks the lines where rounding shows.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #backpropagation #neuralnetworks #calculus #chainrule #deeplearning #math #ai
```

Short alternative:

```
Backpropagation as pure maths. Forward: z = 0.568, h = 0.568, s ≈ −0.040, p = 0.490, L ≈ 0.714. Backward with the chain rule: ∂L/∂s = p − y = −0.510, then one factor per step down to ∂L/∂w₁ ≈ 0.460. Checked by nudging w₁ ±0.001: 0.460. Step w ← w − 0.3·∂L/∂w and the loss falls to 0.484. Any depth, the same four lines.

Follow for more.

#machinelearning #backpropagation #calculus #ai
```

## YouTube description

```
Backpropagation with nothing but the maths: no diagrams, just the equations of one training step with every number computed from a real trained network.

The network: one ReLU neuron and a sigmoid output, trained on the two-arm spiral (reel 31, 300 steps). z = w₁x₁ + w₂x₂ + b, h = max(0, z), s = v·h + c, p = σ(s), loss L = −log p for an example whose answer is y = 1.

Forward: z = (−0.004)(−0.874) + 1.768·0.051 + 0.474 = 0.568, h = 0.568, s ≈ −0.040, p = 0.490, L ≈ 0.714.

Backward (the chain rule): ∂L/∂s = (−1/p)·p(1 − p) = p − 1 = −0.510. ∂L/∂c = −0.510. ∂L/∂v = −0.510·0.568 = −0.290. ∂L/∂h = −0.510·1.032 ≈ −0.527. ReLU′(z) = 1 since z > 0. ∂L/∂z = −0.527. ∂L/∂w₁ = −0.527·(−0.874) ≈ 0.460, ∂L/∂w₂ = −0.027, ∂L/∂b = −0.527.

Check by finite differences: [L(w₁ + 0.001) − L(w₁ − 0.001)]/0.002 = (0.714011 − 0.713091)/0.002 = 0.460.

Update with η = 0.3: w₁ −0.004 → −0.142, w₂ 1.768 → 1.776, b 0.474 → 0.632, v 1.032 → 1.119, c −0.627 → −0.474. Forward again: p = 0.616, L ≈ 0.484. On all 400 points, 249 → 255 classified correctly.

The general rules for any depth: δ_out = p − y; δⱼ = ReLU′(zⱼ) Σₖ wⱼₖ δₖ; ∂L/∂wᵢⱼ = hᵢ δⱼ, ∂L/∂bⱼ = δⱼ; w ← w − η ∂L/∂w.

Numbers are rounded to 3 decimals and computed in full precision; "≈" marks the five lines where the rounded numbers on screen miss the rounded result in the last digit. One example and one plain gradient step; real training averages over all the points.

#machinelearning #backpropagation #calculus #chainrule
```

## Pinned comment

```
Why does ∂L/∂s come out as just p − y? L = −log p and p = σ(s). The chain rule gives (−1/p) times σ′(s), and σ′(s) = p(1 − p). The p cancels and you are left with −(1 − p) = p − 1, which is p − y for the answer y = 1 (for y = 0 it is p − 0). That cancellation is the reason almost every classifier pairs a sigmoid (or softmax) with the log loss.

Why the "≈" on some lines: everything is computed in full precision and shown to 3 decimals. Where multiplying the rounded numbers on screen lands one unit off in the last digit, the line says ≈ instead of =. For example (−0.527)(−0.874) = 0.4606 from the rounded numbers, while the full-precision values give 0.4602.

The check on page 6 is how people test backprop code for real: compare every analytic derivative with a finite difference. Here both give 0.460233.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 60 s. A soft chord bed; a soft pluck as
each line of maths appears (rising through the forward and update pages, a darker falling voice through
the backward pages); a faint high note when a result is revealed; a bell at each new page; a three-note
chime when the finite-difference check agrees.
