# Posting kit — *89% of the numbers: gone. Still obviously a cat.* (dark theme, narrated, 71 s)

## Files

* Reel: `output/lowrank_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the cat rebuilt from 20 pieces (45,120 numbers, 11.1% of 405,900), the strengths of
  all 300 pieces, and the sum of pieces.
* Runs: `svd_results.json` (from `python svdsim.py`, a few seconds).
* Photo: "chelsea" from scikit-image's sample data (Chelsea the cat, by Stefan van der Walt; released with no
  copyright restrictions, CC0). Read with `skimage.data.chelsea()`; its SHA-256 is recorded in the results.
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

Each colour channel X of the 300 × 451 photo is split by numpy's SVD into X = Σ σᵢ uᵢ vᵢᵀ (σᵢ ≥ 0 the strengths,
strongest first; uᵢ a column of 300 numbers, vᵢᵀ a row of 451) and rebuilt from the first k pieces of all three
channels: 3 · k · (300 + 451 + 1) numbers.

| Pieces k | Numbers kept | Share of the photo's 405,900 | Relative error | Average error (shades of 255) |
| --- | --- | --- | --- | --- |
| 1 | 2,256 | 0.6% | 24.4% | 29.9 |
| 5 | 11,280 | 2.8% | 15.0% | 18.5 |
| 10 | 22,560 | 5.6% | 11.0% | 13.5 |
| 20 | 45,120 | 11.1% | 7.6% | 9.3 |
| 50 | 112,800 | 27.8% | 4.2% | 5.2 |
| 100 | 225,600 | 55.6% | 2.2% | 2.7 |

* The pieces picture: piece 1, then pieces 2 and 3 drawn at twice their size on gray (gray = 0, since they can be
  negative), then the first three added up.
* The curve: σ₁ … σ₃₀₀ of each channel, log scale (red channel: σ₁ = 54,514, σ₅₀ = 344). All 300 pieces rebuild the
  photo exactly (largest difference 1.2 × 10⁻¹¹).
* Past 180 pieces, the pieces take more numbers than the photo itself.

## Honesty notes

* "Threw away 89% of the numbers": counted in numbers, not bytes. The photo stores each number in 1 byte (0–255);
  the pieces are decimals, and stored as 32-bit floats the 20 pieces would take 180,480 bytes, about 44% of the
  photo's 405,900. Real formats such as JPEG compress photos much better than this; the reel shows the idea.
* "Average error" is the root-mean-square difference over all 405,900 numbers, before rounding back to whole shades
  (after rounding: the same to one decimal).
* "Still obviously a cat" is a judgement; the cover shows the 20-piece version.
* LoRA (low-rank adaptation; Hu et al., 2021) fine-tunes a large model by training a low-rank update B A added to a
  weight matrix, instead of the matrix itself.

## Caption (paste as-is)

```
I threw away 89% of the numbers in this photo. It's still obviously a cat.

To a computer, a photo is a grid of numbers: 300 rows, 451 columns, 3 colors. 405,900 numbers.

Linear algebra can split any grid into simple pieces, each just a column times a row, with a strength. That's the singular value decomposition (SVD), and the pieces come sorted, strongest first.

Keep only the strongest piece: plaid. 5 pieces: blurry shapes. 20 pieces: the cat, from 11% of the numbers.

50 pieces, 28% of the numbers: the average pixel is off by only 5 shades out of 255. The strengths drop fast, so the first pieces carry most of the photo.

That's a low-rank approximation. The same idea, called LoRA, fine-tunes giant AI models: instead of changing a huge matrix, it learns a few columns times rows.

(Counted in numbers, not bytes. The photo is Chelsea the cat, a public-domain sample image from scikit-image. Every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #svd #math #machinelearning #lora #datascience #imageprocessing
```

## YouTube

Title:

```
I Deleted 89% of This Photo's Numbers. Still a Cat. #shorts
```

Description:

```
Image compression with the singular value decomposition, computed.

A 300 × 451 color photo (405,900 numbers; scikit-image's public-domain "chelsea"), each color channel split by the SVD into rank-one pieces σ u vᵀ, strongest first, and rebuilt from the first k. k = 1: 0.6% of the numbers, average error 29.9 shades of 255; k = 5: 2.8%, 18.5; k = 20: 11.1%, 9.3; k = 50: 27.8%, 5.2; k = 100: 55.6%, 2.7. Past 180 pieces the pieces take more numbers than the photo. Counted in numbers, not bytes.

The same low-rank idea is LoRA (low-rank adaptation), which fine-tunes large models by training a product of two thin matrices instead of the full weight matrix.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#linearalgebra #svd #lora #shorts
```

## Pinned comment

```
Why do so few pieces work? Neighboring pixels in a photo are similar, so its rows look a lot like each other, and a few patterns (columns times rows) describe most of them. That's the plunging strength curve: σ₁ is 54,514 but σ₅₀ is only 344. A grid of pure static has no such patterns: rebuilt from 50 pieces, it's still off by 57 shades on average, against 5.2 for the cat.
```
