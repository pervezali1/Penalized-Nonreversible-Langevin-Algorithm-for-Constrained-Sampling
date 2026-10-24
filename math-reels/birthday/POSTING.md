# Posting kit — *How many people until two share a birthday?*

## Files

* Reel: `output/birthday_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the "23 people. 50% chance." reveal.

## What's on screen (all of it is real)

* **Act A** — one random room (seed 141) fills to 23 people on a 365-day ring. The
  counter shows the *exact* probability 1 − 365!/((365−n)!·365ⁿ): 50.7% at n = 23.
  In this particular room the first shared birthday happens at person 22.
* **Act B** — 10,000 simulated rooms of 23 light up a 100×100 grid (magenta = a
  match). The real count is **5,067 / 10,000 = 50.7%**.
* **Act C** — the 253 pairs among 23 people (23·22/2) drawn as chords.
* **Act D** — the room fills to 70 people: 99.9% (exact: 99.916%).

## Caption (paste as-is)

```
A room of 23 people has a 50.7% chance that two of them share a birthday. 70 people: 99.9%.

It feels wrong because you're thinking about YOUR birthday matching someone's (that really does take ~253 people for a coin-flip). But any two people can match, and 23 people make 253 pairs. 253 shots at a 1-in-365 target.

I simulated 10,000 rooms of 23 to check: 5,067 had a match.

What was your guess? Comment it before you read the replies.

(Exact formula: P = 1 − 365!/((365−n)! · 365ⁿ). Simulation: numpy, uniform birthdays, 10,000 rooms. Leap days and real-world birthday clustering only push the odds higher.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#probability #math #birthdayparadox #statistics #mathfacts
```

Exactly five hashtags.

Short alternative:

```
23 people → 50% chance two share a birthday. 70 people → 99.9%. Not a trick: 23 people are 253 pairs.

What was your guess?

Follow for more.

#probability #math #birthdayparadox #statistics #mathfacts
```

## Pinned comment

```
The exact numbers: 23 people → 50.73%, 30 → 70.6%, 50 → 97.0%, 70 → 99.92%. "But I've never met anyone with my birthday" is the other problem: for a match with YOU specifically you need 253 people for 50%. The paradox is that we instinctively answer the second question when asked the first. Next: how this same math breaks passwords (the birthday attack). Want it?
```

## Audio

Keep the built-in track, named **"the sound of 23 people"**: a pentatonic pluck per
person (note chosen by the birthday), a bell on every match, a ticking rain while
the 10,000 rooms fill with a tone that converges on the match rate, and a resolved
chord at 99.9%.

## Replies to expect

* "Leap years?" → Feb 29 and non-uniform birthdays both make a match *more* likely; 365-uniform is the conservative case.
* "That's not my experience" → people remember *their own* birthday matching, which is the 253-people problem.
* "Simulation proves nothing" → it's a check on the formula, not the proof; the formula is in the caption.

## Series ideas

1. "The birthday attack" — why 128-bit hashes only give 64 bits of collision resistance.
2. "How many people until THREE share a birthday?" (88).
3. "Same math, different question: shuffle a deck — has anyone ever had this order?"
