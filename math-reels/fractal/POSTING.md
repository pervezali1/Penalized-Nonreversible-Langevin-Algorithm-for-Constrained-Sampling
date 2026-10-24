# Posting kit — *Roll a die. Move halfway to a corner.*

## Files

* Reel: `output/fractal_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the finished fern with "That's a fern."

## What's on screen (all of it is real)

The chaos game, run live with a seeded random generator:

* **Triangle (0–7 s):** three corners A, B, C. Roll a die: 1–2 → A, 3–4 → B, 5–6 → C.
  Move halfway from where you are to that corner, plot the point, repeat. The first
  12 moves are shown one at a time with the die; then 50,000 points. Each point is coloured
  by the corner it just moved toward, which is why the three sub-triangles come out in
  three colours. The result is the Sierpinski triangle.
* **Fern (7.3–19 s):** same process, four rules instead of three, chosen with
  probabilities 1% / 85% / 7% / 7% (Barnsley's 1988 fern: four affine maps). 200,000 points,
  coloured by which rule produced them (stem amber, main frond green, lower leaflets
  cyan/lime). Nothing is drawn by hand; every point is the previous point pushed through
  one random rule.

## Caption (paste as-is)

```
Roll a die. Move halfway to one of three corners. Put a dot. Repeat 50,000 times.

You'd expect a smear. You get a perfect Sierpinski triangle, every time, no matter where you start. The randomness decides nothing; the three rules decide everything.

Then change the rules slightly (four of them, picked with different odds), run the same random process, and 200,000 dots grow into a fern. Not a photo, not a drawing: four lines of arithmetic and a dice roll.

That's what a fractal is: simple rules, repeated forever. It's also the same idea that stores images as equations (fractal compression) and the reason a fern looks like a fern at every scale.

What would you expect the dots to make?

(The chaos game / Barnsley fern, run live in Python with a seeded RNG; every sound is a point landing.)

#fractal #math #chaostheory #mathart #satisfying
```

Short alternative:

```
Roll a die, move halfway to a corner, repeat 50,000 times: a perfect triangle. Change the four rules slightly: a fern. Nothing drawn. Randomness + rules = structure.

#fractal #math #chaostheory #mathart #satisfying
```

## Pinned comment

```
The four fern rules (x, y → new x, y): stem 0, 0.16y; main 0.85x+0.04y, −0.04x+0.85y+1.6; left 0.2x−0.26y, 0.23x+0.22y+1.6; right −0.15x+0.28y, 0.26x+0.24y+0.44, picked with probabilities 1 / 85 / 7 / 7 %. Why randomness draws a fixed shape: every rule is a contraction, so whatever point you start from, after a few steps you're on the one set that all four rules map into itself (the attractor). The dice just decide which part of it you visit next. Try it in 10 lines of Python; next reel: I turn the dials and watch the fern mutate. Want it?
```

## Audio

Keep the built-in track, **"the sound of a fern growing"**: a pluck per point while the
points are slow (note chosen by the rule rolled), a granular rain whose density follows how
many points land per frame once it speeds up, a chord that resolves as the fern fills in,
bells at both reveals.

## Replies to expect

* "Starting point matters?" → no: the first few points are off-pattern, then every start lands on the same attractor (that is the contraction argument in the pinned comment).
* "The colours are a cheat" → each point is coloured by the rule that made it; the colour is data, not paint.
* "Is this the same as the Mandelbrot set?" → different family: this is an iterated function system (random rules on points); Mandelbrot iterates one formula on complex numbers. Both are fractals.
