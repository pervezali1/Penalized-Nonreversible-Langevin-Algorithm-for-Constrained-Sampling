# Euler Sunday: weekly Facebook & Instagram carousels

One Project Euler problem a week, posted as a 5-slide photo carousel (1080 × 1350 JPEG)
on Facebook and Instagram every Sunday, the day the Threads "Euler Sunday" answer goes out.

| Sunday | Problem | Instagram | Facebook |
|---|---|---|---|
| Nov 15, 2026 | 1. Multiples of 3 or 5 | 2:00 PM ET | 6:00 PM ET |
| Nov 22 | 2. Even Fibonacci numbers | 2:00 PM | 6:00 PM |
| Nov 29 | 3. Largest prime factor | 2:00 PM | 6:00 PM |
| Dec 6 | 4. Largest palindrome product | 2:00 PM | 6:00 PM |
| Dec 13 | 5. Smallest multiple | 2:00 PM | 6:00 PM |
| Dec 20 | 6. Sum square difference | 2:00 PM | 6:00 PM |
| Dec 27 | 7. 10,001st prime | 2:00 PM | 6:00 PM |

Each carousel: the problem with a computed picture → the loop (Problems 1 and 2 use the
code from `project_euler_01.py` / `project_euler_02.py`) → the math shortcut → the answer and
a YOUR TURN challenge, whose answer goes in the post's first comment.

## Files

- `solutions.py`: every number on the slides, computed and checked against brute force
- `make_slides.py`: builds each slide as HTML/SVG from those numbers
- `render.js`: renders the slides to `images/pNN_sK.jpg` with Playwright (Chromium)
- `captions.py`: captions, first comments, alt text and the schedule → `posts.json`
- `images/`: the 35 finished slides (Metricool pulls them from here, so keep them)

## Regenerate

```bash
python3 make_slides.py && node render.js
python3 captions.py <commit-sha>
```
