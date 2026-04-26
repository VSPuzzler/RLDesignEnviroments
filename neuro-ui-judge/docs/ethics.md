# Ethics & disclaimers

NeuroUI Judge is a research prototype. **Do not deploy it in production
without the steps below.**

## Hard claims we refuse to make

1. **Not mind reading.** No part of this system measures any individual
   person's preferences, intentions, or mental state. The "neural" channel is
   a population-level encoder running on a *picture of a UI*, not on any
   user's brain.
2. **Not a preference oracle.** A high `overall_reward` is *evidence under our
   model*, not evidence about real users. The only place real preference
   signal enters the system is the pairwise human labels you collect.
3. **Not a substitute for accessibility testing.** Our deterministic audit
   covers a useful but narrow subset of WCAG. Passing the gate is a *necessary*
   condition for accessibility, not sufficient. Real accessibility testing
   includes screen-reader walkthroughs, keyboard-only tests, motion / vestibular
   review, and assistive-tech compatibility.
4. **Not anatomically faithful in visualisations.** The cortical heatmap on
   the candidate page is a stylised illustration. ROI patches are placed for
   legibility, not anatomical accuracy. Do not show it to a stakeholder
   without this caveat.

## What we do require before product use

- Human user testing with the target population, on the *deployed* UI, before
  any rollout decision.
- Accessibility review by someone qualified, including assistive-technology
  walkthroughs.
- Independent review of any "TRIBE-derived" claims by a domain expert, with
  documented validation against human preference data.
- A clear audit trail: every report logs the active weight version, the
  TRIBE backend mode, and the deterministic violations. Keep this.

## TRIBE v2 licensing

TRIBE v2 may be released under non-commercial / research-only terms. Before
running real TRIBE inference in any product context:

- Read the model's license carefully.
- Confirm whether your use case is permitted (typically: research, internal
  evaluation; *not* automatically: commercial UI optimisation pipelines).
- Document the license review in your codebase.
- If unsure, contact the model authors.

## Reward hacking and gaming

The hybrid reward is designed to be hard to game *visually* (you cannot raise
reward by, say, painting the page with a single salient pixel) but no reward
is impossible to game. Defences shipped in this MVP:

- Hard accessibility gate caps the score below 0.55 on WCAG failure.
- Defect penalty is monotone in violation severity.
- Aesthetic / valuation channels are weighted low because their confidence is low.
- Density-overlap and visual-entropy terms penalise pathological "all CTAs"
  layouts.

If you observe a reward exploit, file an issue and we will add a regression
test and a counter-term.

## Data handling

The dashboard stores candidates, screenshots, reports, and pairwise labels in
a local SQLite DB and the `data/` folder. Treat them as you would any other
user-provided content:

- Do not commit `data/` to source control by default.
- If you collect human preference labels, include explicit consent and a
  description of how the data will be used.
- The `rater_id` field is opaque to the system; if you populate it with PII,
  you are responsible for its handling.
