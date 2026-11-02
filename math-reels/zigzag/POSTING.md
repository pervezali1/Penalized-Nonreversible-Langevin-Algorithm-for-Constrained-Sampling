# Posting kit — *Why gradient descent zig-zags (the condition number).* (3D stage, narrated, 72 s)

## Files

* Reel: `output/zigzag_reel.mp4` — 71.8 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the 100× stretched bowl with all three paths: gradient descent (284 steps), momentum (23), Adam (29).
* Runs: `zz_results.json` (from `python zzsim.py`: every path and the settings searched).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A bowl f(x, y) = ½(x²/κ + y²): its Hessian is diag(1/κ, 1), so its curvatures (eigenvalues) are 1/κ and 1 and its
condition number is κ. Every run starts at (−2.8, 1.6) and counts steps until it is within 0.01 of the bottom. Each
method gets its best settings for each bowl, searched on a grid (`zzsim.py`).

| Bowl | Method | Best settings (grid) | Steps |
| --- | --- | --- | --- |
| κ = 1 (round) | gradient descent | step 1.00 (of 0.01 … 1.99) | 1 |
| κ = 10 | gradient descent | step 1.77 | 29 |
| κ = 100 | gradient descent | step 1.97 | 284 |
| κ = 100 | heavy-ball momentum | step 2.60, momentum 0.76 (steps 0.05 … 3.95 × momentum 0 … 0.98) | 23 |
| κ = 1, 10, 100 | Adam (β₁ 0.9, β₂ 0.999) | step 0.62 (of 0.01 … 1.99) | 29, 29, 29 |

The explanation card: λmax = 1, λmin = 0.01, κ = 100; the step must stay below 2/λmax = 2 (or the bounce across the
valley grows), and along the floor each step closes only η·λmin ≈ 2% of the remaining distance.

## Honesty notes

* The valley lies along an axis on purpose. Adam rescales each coordinate separately, which is why it takes exactly
  the same 29 steps on every one of these bowls; on a valley turned at an angle that rescaling would not line up
  with it, and Adam would be slowed down too. (On the round bowl Adam is slower than plain gradient descent: 29 vs 1.)
* "Best step size" means the best on the grid named above, for this start and this tolerance; another start or
  tolerance gives different counts, but the same pattern (roughly proportional to κ for gradient descent, to √κ
  for momentum).
* On the 284-step run the animation speeds up after the first bounces (the step counter is exact throughout).

## Caption (paste as-is)

```
Same bowl, just stretched. Gradient descent goes from 1 step to 284.

A round bowl: with the right step size, gradient descent lands on the bottom in one step.

Stretch it 10× along one direction: even with its best step size, 29 steps, and it starts to zig-zag. Stretch it 100×: 284 steps. It bounces between the steep walls and crawls along the flat floor.

Why: the curvatures across and along the valley are the Hessian's eigenvalues, here 1 and 0.01. Their ratio, 100, is the condition number. The step size has to suit the steepest direction or it flies off, so along the floor it barely moves: 2% per step.

Momentum remembers its last step and builds up speed along the floor: 23 steps. Adam rescales each direction separately: 29 steps on every one of these bowls (it helps that this valley lines up with the axes).

Steepest over flattest curvature: the condition number. The bigger it is, the more plain gradient descent zig-zags.

(Every path and step count is a real run, each method with its best settings for each bowl.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #optimization #gradientdescent #deeplearning #linearalgebra #hessian #math #ai
```

## YouTube

Title:

```
Why Gradient Descent Zig-Zags (The Condition Number) #shorts
```

Description:

```
Gradient descent on a bowl f(x, y) = ½(x²/κ + y²), from the same start, until it is within 0.01 of the bottom, each time with its best step size (searched on a grid): κ = 1 (round): 1 step. κ = 10: 29 steps. κ = 100: 284 steps.

The Hessian's eigenvalues are the curvatures across and along the valley, 1 and 1/κ; their ratio κ is the condition number. Stability needs step < 2/λmax, and then each step shrinks the distance along the flat direction by only a factor 1 − η·λmin: about 2% per step here. The result: bouncing across the steep walls, crawling along the floor, a step count that grows like κ.

Momentum (heavy ball, best step 2.60 and momentum 0.76): 23 steps; its count grows like √κ. Adam (best step 0.62): 29 steps on all three bowls, because it rescales each coordinate separately and this valley lines up with the axes; on a rotated valley it would not be immune.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#gradientdescent #optimization #machinelearning #shorts
```

## Pinned comment

```
The fix that always works is changing the shape of the bowl: if you could rescale the space so both curvatures are equal (κ = 1), plain gradient descent would take one step. That's what Newton's method does with the full Hessian (the Hessian reel), what preconditioning does approximately, and what Adam does crudely, one coordinate at a time. Normalising your input features is the cheapest version of the same idea.
```
