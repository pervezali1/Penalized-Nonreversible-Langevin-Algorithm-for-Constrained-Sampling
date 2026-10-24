# Posting kit — *3 doors. 1 car. 2 goats. Switch or stay?*

## Files

* Reel: `output/monty_reel.mp4` — 24.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — "Switch wins 66.3%" over the two-thirds-cyan grid.

## What's on screen (all of it is real)

* **0–9 s, the rules, for people who have never heard of it:** one big set of three doors.
  "Behind one door: a car. Behind the other two: goats." You pick door 1 (amber). "The host
  knows where the car is. He opens another door: a goat" (door 3). "Switch to the last door,
  or stay? Most people say: 50/50."
* 10,000 simulated games (seed 8): the car is placed at random, you pick at random, the host
  opens a goat door you didn't pick (choosing at random when he has two). STAY keeps your
  door; SWITCH takes the other closed one. The same game is shown in both columns.
* Every game fills one cell of a 100×100 grid: **cyan if switching won, magenta if staying
  won**. Exactly one of the two wins every game (the build asserts it), so the grid is a
  literal picture of the answer: two-thirds cyan.
* Result: stay **33.7%**, switch **66.3%** (exact: 1/3 and 2/3).
* 100 doors: you pick one (amber); the host opens 98 goats, one by one; the one door he
  skipped lights up cyan. 1% vs 99%.

## Caption (paste as-is)

```
Three doors. One car, two goats. You pick a door. The host, who knows where the car is, opens one of the others and shows you a goat. He offers you a switch.

Most people say it doesn't matter now: two doors, 50/50. I played 10,000 games. Staying won 33.7%. Switching won 66.3%.

Why: your first pick is wrong 2 times out of 3, and whenever it's wrong, the host's goat reveal hands you the car's door. Switching wins exactly when your first pick was wrong.

Still not convinced? 100 doors. Pick one, the host opens 98 goats, one door stays shut. Do you really think your 1-in-100 guess is as good as the door he carefully didn't open?

Switch or stay: be honest, what would you have done?

(Simulation: numpy, random car, random pick, host always opens a goat you didn't choose. The grid is every game: cyan = switch won, magenta = stay won.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#montyhall #probability #math #brainteaser #mathfacts
```

Short alternative:

```
3 doors, 1 car. You pick, the host shows a goat, offers a switch. 10,000 games: stay 33.7%, switch 66.3%. Your first pick is wrong 2 in 3 times, and that's exactly when switching wins.

Follow for more.

#montyhall #probability #math #brainteaser #mathfacts
```

## Pinned comment

```
The one assumption that makes it 2/3: the host KNOWS where the car is and always opens a goat you didn't pick. If he opened a random door (and just got lucky it was a goat), it really would be 50/50. That's the version most "it's 50/50" arguments are secretly about. Marilyn vos Savant published the 2/3 answer in 1990 and got ~10,000 letters, including from PhDs, telling her she was wrong. She wasn't. Next: the Two Envelopes paradox, where the switching argument goes wrong. Want it?
```

## Audio

Keep the built-in track, **"the sound of 10,000 doors"**: a knock when you pick, a click when
the host opens, a bell for the winning column and a low blip for the losing one, a rattle that
thickens as the games speed up, two tones that track the two win rates, 98 rising clicks for
the 100-door host, bells at the reveals.

## Replies to expect

* "It's 50/50, two doors" → the host's choice isn't random, so the two doors aren't symmetric: your door still carries its original 1/3; the other carries everything else.
* "What if the host doesn't know?" → then 50/50, and the game is different (pinned comment).
* "What if he only offers the switch when you're right?" → a different game again; this reel is the standard rules (always offers).
* "Depends on his choice when you picked the car" → it doesn't affect stay/switch win rates (the host's tie-break is random here; any rule gives 1/3 vs 2/3).
