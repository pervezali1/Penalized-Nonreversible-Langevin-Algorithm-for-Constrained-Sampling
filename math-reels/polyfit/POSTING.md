# Posting kit — *Overfitting, measured.* (dark theme, narrated, 66 s)

## Files

* Reel: `output/polyfit_reel.mp4` — 66.4 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — degree 19 through all 20 points, with the 5,000 new points and the errors: 0.00 on its own points, 84 on new ones.
* Fits: `pf_results.json` (from `python pfsim.py`).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

20 measurements at evenly spaced x from −0.95 to 0.95, y = sin(2.6x) + 0.3x + noise (standard deviation 0.25, seed
0). Polynomials of degree 0 to 19 fitted by least squares (in the Legendre basis: the same polynomials, better
conditioned). Errors are root-mean-square misses: on the 20 points, and on 5,000 new points from the same source
(x uniform over the same range; 400 of them drawn as faint dots).

| Degree | Error on its 20 points | Error on 5,000 new points |
| --- | --- | --- |
| 1 | 0.25 | 0.38 |
| 3 | 0.15 | 0.289 |
| 6 (lowest) | 0.15 | 0.289 |
| 9 | 0.14 | 0.29 |
| 15 | 0.09 | 0.57 |
| 17 | 0.07 | 4.8 |
| 19 | 0.00 (5 × 10⁻¹³) | 84 |

Degrees 3 to 7 all score 0.289–0.290 on new points (within 1% of the lowest); the true curve itself scores 0.25
(the noise). The chart plots both errors against degree on a log scale.

* **Fix 1, more data:** degree 19 fitted to 200 points from the same source: 0.26 on new points.
* **Fix 2, a penalty:** degree 19 on the same 20 points, minimising Σ(yᵢ − p(xᵢ))² + λ Σ c_k² (c_k: the Legendre
  coefficients) with λ = 0.18 chosen by leave-one-out cross-validation on the 20 points: 0.31 on new points.

## Honesty notes

* The 5,000 test points were never used to choose anything (degree, λ); they only measure.
* Evenly spaced x makes degree 19 swing mainly near the ends. With randomly placed x the same experiment is far
  worse: over 8 random draws of 20 points (x uniform over the same range), degree 19 scored between 252 and 25
  million on new points, while the best degree (3 to 6) always scored 0.25–0.30.
* Over 8 noise draws with evenly spaced x: the best degree was 3 to 6, degree 19 scored 12 to 208 on new points,
  200 points brought it to 0.26–0.65, and the penalty to 0.28–0.34.
* Degree 19 needs exact arithmetic to pass through 20 points; its "0.00" is 5 × 10⁻¹³.

## Caption (paste as-is)

```
This curve passes through every single point. That's the problem.

20 noisy points. Fit a polynomial and raise the degree: the curve bends to follow the points, and the error on those 20 points keeps falling. At degree 19 it passes through all of them. Error: 0.

Now test it on 5,000 new points from the same source. Error: 84.

Too simple misses the shape; too flexible memorises the noise. Degrees 3 to 7 all score about 0.29 on new points, close to the noise itself (0.25).

Fix 1: more data. Degree 19 fitted to 200 points: 0.26 on new points.
Fix 2: a small penalty on big coefficients. Back to 20 points, degree 19: 0.31.

A perfect score on your own data isn't the goal. The error on new data is.

(Every curve and error on screen is computed: least-squares fits, tested on 5,000 points they never saw.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #overfitting #datascience #statistics #regression #ai #math
```

## YouTube

Title:

```
Overfitting, Measured: Zero Error Is the Problem #shorts
```

Description:

```
Overfitting with real numbers. 20 noisy measurements of y = sin(2.6x) + 0.3x (noise sd 0.25), fitted with polynomials of degree 0 to 19 by least squares. The error on the 20 points keeps falling, to 0 at degree 19, which passes through every point. On 5,000 new points from the same source the error falls and then explodes: 0.38 at degree 1, 0.29 for degrees 3 to 7, 84 at degree 19 (root-mean-square misses; the true curve itself scores 0.25).

Two fixes, measured on the same 5,000 new points: 200 training points instead of 20 (degree 19: 0.26), or ridge regularisation, a penalty λ·Σc² on the coefficients with λ = 0.18 chosen by leave-one-out cross-validation (degree 19 on 20 points: 0.31). The test points are never used to choose anything.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#overfitting #machinelearning #datascience #shorts
```

## Pinned comment

```
Why does degree 19 go wild? 20 coefficients and 20 points: there's exactly one degree-19 polynomial through all of them, noise included, and to hit every noisy point it has to swing hard between them, worst near the ends (the Runge phenomenon). The penalty on big coefficients forbids the big swings; more data makes the swings cost too much. Same lesson as the label-noise reel: a model that can memorise everything, will.
```
