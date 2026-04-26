# Reward model

Notation:

- $u$ — UI candidate (rendered artifact)
- $d(u)$ — deterministic UI audit features
- $n(u)$ — neural proxy features (mock or real TRIBE)
- $h$ — learned weights from human pairwise preferences
- $A(u) \in \{0,1\}$ — accessibility hard gate
- $D(u) \in [0,1]$ — defect penalty
- $U(u) \in [0,1]$ — uncertainty penalty (mean confidence)

## Sub-scores

Each sub-score $s_m(u) \in [0,1]$ where 1 = best. Cognitive load is
**inverted at the sub-score level** so all reported sub-scores follow the
"higher = better" convention. The internal raw load is preserved in the
report's explanation.

We use simple parametric forms — products / sigmoids — chosen so each is
auditable line-by-line in `services/scorer/reward_model.py`.

```
usability         = 0.45·readability + 0.35·cta_clarity + 0.20·spacing
attention_guide   = σ(2·DA_auc + 1.5·cta_clarity + 1.0·hierarchy − 1.5·visual_var − 1)
visual_hierarchy  = 0.7·hierarchy_audit + 0.3·DA_peak
cognitive_load†   = 1 − σ(2.5·MD_auc + 1.5·sal_var + 1.5·(1−density) − hierarchy − readability)
readability       = 0.7·readability_audit + 0.3·VWFA_auc
aesthetic_quality = σ(1.5·balance + 1.2·spacing + 1.0·harmony + 0.6·val_auc − 1.5·(1−density) − 1)
accessibility     = (1 if WCAG pass else 0.6) · accessibility_audit
engagement_proxy  = σ(1.2·sal_auc + 1.0·val_auc + 1.0·DMN_supp + 0.8·cta_clarity − 1)
trust             = 0.30·balance + 0.25·spacing + 0.15·harmony + 0.20·WCAG + 0.10·readability − crit_penalty
```

† Reported as `1 − load_raw`; `load_raw` is exposed in the explanation.

## Confidence routing

Each sub-score carries a per-metric confidence $c_m(u)$:

- Pure deterministic metrics (`usability`, `accessibility`, `trust`,
  `visual_hierarchy`, `readability`) have $c_m \approx 0.85$–$1.0$.
- Neural-influenced metrics (`attention_guidance`, `cognitive_load`,
  `engagement_proxy`, `aesthetic_quality`) inherit from the proxy's
  confidence dictionary, with `aesthetic_quality` typically capped at $0.25$.

If `WCAG_pass = 0` we additionally damp every non-accessibility metric's
confidence by $\times 0.9$ — when a page is failing accessibility we trust
its other signals less.

## Aggregation

Confidence-weighted average over sub-scores:

$$
\bar R(u) = \frac{\sum_m w_m \, c_m(u)^\gamma \, s_m(u)}{\sum_m w_m \, c_m(u)^\gamma}, \quad \gamma = \text{CONFIDENCE\_GAIN} = 1
$$

Defect and uncertainty penalties:

$$
D(u) = \min\!\Big(1, \sum_v \omega_{\text{sev}(v)}\Big), \quad U(u) = \tfrac{1}{2}\big(1 - \tfrac{1}{4}\sum_k c_k\big)
$$

with $\omega_{\text{critical}} = 0.25$, $\omega_{\text{major}} = 0.10$,
$\omega_{\text{minor}} = 0.04$, $\omega_{\text{info}} = 0.01$.

Pre-gate composite (clipped to $[0,1]$):

$$
\tilde R(u) = \mathrm{clip}\big(\bar R(u) - 0.25 \, D(u) - 0.10 \, U(u), 0, 1\big)
$$

Hard accessibility gate:

$$
R(u) = \begin{cases}
\tilde R(u) & \text{if } A(u) = 1 \\
\min(\tilde R(u), 0.55) & \text{otherwise}
\end{cases}
$$

A failed gate caps the reported reward at 0.55 and the dashboard surfaces
this prominently.

## Letter grade

```
A: R ≥ 0.85
B: R ≥ 0.70
C: R ≥ 0.55
D: R ≥ 0.40
F: otherwise
```

## Default weights

```
usability          0.13
attention_guidance 0.12
visual_hierarchy   0.12
cognitive_load     0.13
readability        0.10
aesthetic_quality  0.07   # low because confidence is low
accessibility      0.18   # weighted high
engagement_proxy   0.07
trust              0.08
```

These sum to 1. They are intentionally aesthetic-light and accessibility-heavy
in the absence of preference data. Once enough pairwise preferences are
collected the calibration step (`preference_model.fit_preference_weights`)
overrides these per-metric weights.

## Calibration

The Bradley–Terry / logistic calibration solves:

$$
\min_w \; \frac{1}{N}\sum_{i=1}^N \mathrm{BCE}\!\Big(\sigma\!\Big(\frac{(\phi(A_i) - \phi(B_i))^\top w}{\tau}\Big), \; y_i\Big) + \lambda \|w - w_0\|_2^2
$$

where $\phi(u)$ is the vector of sub-scores in canonical order, $\tau$ is a
temperature, $w_0$ are the default weights (acting as a prior), and $\lambda$
is the L2 regularisation strength. We project weights to be non-negative and
L1-normalise so they form a convex combination, keeping interpretation simple.

Reported metrics: pairwise accuracy on a held-out split, training BCE,
validation BCE, calibration curve (`preference_model.calibration_curve`).
Spearman / Kendall hooks are wired but populate only when scalar ratings
exist — pairwise-only datasets leave them as `None`.
