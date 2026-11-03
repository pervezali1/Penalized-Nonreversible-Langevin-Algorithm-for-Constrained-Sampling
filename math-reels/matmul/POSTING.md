# Posting kit — *Why AI runs on matrix multiplication.* (dark theme, narrated, 69 s)

## Files

* Reel: `output/matmul_reel.mp4` — 69.1 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the three times as bars on a log scale: Python 13.4 s, NumPy 5.3 ms (1 core) and 2.3 ms (4 cores).
* Timings: `mm_results.json` (from `python mmsim.py`, run on an otherwise idle machine).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

One multiplication of two 512 × 512 matrices of random numbers, C = A B: 512³ = 134,217,728 multiply-adds, timed
on this computer (4 cores; Python 3.13, NumPy 2.5 with OpenBLAS).

| How | Time | Multiply-adds per second |
| --- | --- | --- |
| Plain Python, three nested loops | 13.4 s | about 10 million |
| NumPy (`A @ B`), one core | 5.3 ms (median of 30) | about 25 billion |
| NumPy, all 4 cores | 2.3 ms (median of 60) | about 59 billion |

Same answer: the largest difference between the Python and NumPy results is 2.3 × 10⁻¹³ (rounding). Speed-up:
5,861× (13.4 s / 2.28 ms).

On screen the matrices are drawn as 16 × 16 blocks of 32 × 32 entries; the Python fill is shown at 2× speed with the
real elapsed time on the timer.

**The million-weight network** of reel 41 (2 → 1000 → 1000 → 1, 400 points): one forward pass is three matrix
multiplications, 401,200,000 multiply-adds. NumPy: 10 ms (median of 40). Plain Python: about 40 s, estimated as that
count divided by the measured Python speed (not run).

## Honesty notes

* Timings depend on the machine. On the machine of reel 41 the same forward pass took 48 ms; the ratios are what
  matter.
* The Python version is the straightforward triple loop (with the second matrix transposed so the inner loop reads a
  row), the way you would write it by hand; it was not tuned further.
* "GPUs take it to thousands of cores" is general background, not measured here.

## Caption (paste as-is)

```
Same multiplication. Python: 13.4 seconds. NumPy: 2.3 milliseconds.

Every layer of a neural network is one matrix multiplication: inputs times weights. Here, two 512 × 512 matrices: 134,217,728 multiply-adds.

Plain Python does them one at a time: 13.4 seconds, about 10 million multiply-adds a second.

NumPy hands the same job to compiled code that does many multiply-adds per instruction: 5.3 milliseconds on one core. On all 4 cores: 2.3 milliseconds. The same answer, 5,861 times faster.

The million-weight network from the backpropagation reel: one forward pass is three of these multiplications. NumPy: 10 milliseconds. Plain Python: about 40 seconds.

That's why AI runs on matrix multiplication: it's the one job computers have learned to do absurdly fast, and GPUs take it to thousands of cores.

(Timed on my computer: every time on screen is measured, and the 40 seconds is the measured Python speed applied to the network's 401 million multiply-adds.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #python #numpy #deeplearning #ai #linearalgebra #coding
```

## YouTube

Title:

```
Python vs NumPy: Why AI Runs on Matrix Multiplication #shorts
```

Description:

```
A neural-network layer is a matrix multiplication, and here is why that matters.

One 512 × 512 × 512 multiplication (134,217,728 multiply-adds), timed on my computer: plain Python triple loop 13.4 s (about 10 million multiply-adds per second); NumPy with OpenBLAS on one core 5.3 ms (about 25 billion per second); on all four cores 2.3 ms. Same result to 2.3 × 10⁻¹³, 5,861 times faster. Compiled, vectorized (SIMD), cache-friendly code and several cores do the same arithmetic in bulk.

The 1,005,001-weight network from the backpropagation video: one forward pass on 400 points is three matrix multiplications, 401.2 million multiply-adds: 10 ms in NumPy, about 40 s in plain Python (estimated from the measured loop speed). GPUs push the same idea to thousands of cores.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#numpy #python #machinelearning #shorts
```

## Pinned comment

```
Where the 5,861× comes from: compiled code instead of the Python interpreter (no type checks or object handling per number), SIMD instructions that do several multiply-adds at once, blocking the matrices so the numbers stay in the CPU's cache, and 4 cores instead of 1. Each is a factor; together they multiply. A GPU adds thousands of simpler cores built for exactly this, which is why training runs on them.
```
