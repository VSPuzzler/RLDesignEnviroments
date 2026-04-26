# Scientific framing

NeuroUI Judge is a **decision-support prototype**, not a brain–computer interface,
not a preference oracle, and not a substitute for user research.

## What the neural component is

The "neural proxy" channel uses a population-level encoding model
(TRIBE v2 or a faithful mock) that maps a visual stimulus to predicted
cortical activations on a standardised cortical surface. We summarise those
predictions into a small set of region-of-interest (ROI) features:

| ROI proxy | Interpreted as |
|---|---|
| `visual` (occipital) | bottom-up visual stimulation, complexity |
| `dorsal_attention` | top-down spatial attention guidance |
| `salience` | salience / surprise (insular / ACC analogues) |
| `multiple_demand` | cognitive control / load |
| `language_vwfa` | text legibility / semantic recognition |
| `dmn` | mind-wandering / disengagement (lower = more focused) |
| `valuation_proxy` | weak affect / OFC-like signal — **low confidence by design** |

These are **priors over user behaviour**, not user behaviour itself. They are
*group-level* predictions; they do not reflect any individual person's response.

## What the neural component is NOT

- It is **not** a measurement of preference. Preference is collected separately
  via pairwise human labels.
- It is **not** a measurement of accessibility. Accessibility is enforced via
  WCAG-grounded deterministic checks (contrast ratios, alt text, focus order,
  interactive labelling) and acts as a **hard gate** on the reward.
- It is **not** anatomically precise in the dashboard's cortical heatmap. The
  heatmap is an **illustrative** visualisation; ROI patches are placed for
  legibility, not anatomical accuracy.
- It is **not** a substitute for user testing. It is a fast, cheap reward
  signal for *exploration* and *triage*; final decisions require humans.

## Confidence routing

Every neural channel ships a `[0, 1]` confidence value. The reward model
weights each sub-score by its confidence (`reward_model.py:_metric_confidence`).
The mock proxy fixes:

- `attention` and `load` confidence ≈ 0.55–0.9 (scales with how rich the page is)
- `aesthetic` confidence = 0.25 (intentionally low; aesthetics requires real
  subcortical/valuation validation against humans)
- `accessibility` confidence = 0.10 from the *neural* proxy — we explicitly
  refuse to let neural data dominate accessibility scoring

A real TRIBE backend may raise these only if it has been validated against
human preference data; the README documents how that's wired in.

## Mock vs. real mode

When `NEUROUI_TRIBE_BACKEND` is unset (the default), the system runs in **mock
mode**. Each ROI feature is a *deterministic* function of measurable visual
properties (element coverage, font dispersion, CTA prominence, text density,
colour harmony). This keeps the demo coherent — a busier UI raises predicted
load, a clearer hierarchy raises predicted dorsal-attention AUC, etc. — while
making no claim of fMRI fidelity.

When a real TRIBE backend is registered, the same `roi_features` keys are
populated by predicted cortical activations. The schema is identical so the
reward model and dashboard work unchanged; the only flip is the `mode` field.

## What "BOLD-like time series" means in the dashboard

The candidate page renders a time-series chart per ROI. In **mock mode** these
are smoothed gamma-shape responses derived from each ROI's `auc`, `peak`, and
`variance` summaries — a visualisation aid, not a fitted hemodynamic response.
In **TRIBE mode** the backend should provide actual predicted time courses and
the chart will show those directly.

## What we can and cannot conclude

We **can** say:

- "Under our hybrid reward and current weights, design A scores higher than B."
- "Design A's predicted cognitive load is lower."
- "Design B fails 3 WCAG contrast checks."

We **cannot** say:

- "Users will prefer A."
- "Design A is more beautiful."
- "Design A is accessible." (The deterministic audit answers a *necessary*
  condition, not a sufficient one — usability testing is still required.)

The agent's iteration loop optimises the reward, *which is itself a model*.
Improvements over iterations indicate the agent is learning to satisfy the
reward; whether that translates to better user outcomes is an empirical
question that requires measurement on humans.
