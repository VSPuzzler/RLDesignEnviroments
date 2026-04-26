# NeuroUI Judge

A research prototype that scores UI designs with a hybrid reward model:
**deterministic standards-based audit × population-level neural priors × human pairwise calibration**.
A small offline agent then iteratively improves the design against that reward.

> **Scientific framing.** TRIBE-derived features are *neural priors* for attention,
> hierarchy, cognitive load, semantic coherence, and weak affect — not measurements
> of preference. Accessibility is enforced via WCAG-grounded deterministic checks,
> never neural inference. Aesthetic / valuation channels are intentionally
> low-confidence. See `docs/scientific_framing.md` and `docs/ethics.md`.

---

## What it does

1. **Ingest** UI candidates as HTML/CSS (paste/upload), screenshots, or generation prompts.
2. **Render** them with Playwright at a fixed viewport, capturing a screenshot,
   optional frame sequence, DOM tree with bboxes / fonts / colors / interactivity, and accessibility tree.
3. **Audit** deterministically — WCAG contrast, alt text, focus order, hierarchy,
   density, balance, CTA clarity, color harmony.
4. **Predict** TRIBE-style ROI features through a pluggable adapter
   (mocked by default; real TRIBE v2 drops in via `tribe_adapter.set_backend`).
5. **Score** with a confidence-weighted hybrid reward function with a hard
   accessibility gate, defect penalty, and uncertainty penalty.
6. **Visualise** in a dark-themed dashboard: side-by-side, radar, sub-score bars,
   bbox overlays with attention/load heatmaps, ROI time series, and a stylised cortical heatmap.
7. **Calibrate** with human pairwise preferences via a Bradley-Terry / logistic model.
8. **Improve** via an evolutionary agent loop: deterministic CSS mutations
   targeting weak sub-scores, optionally augmented with an LLM redesign step.

---

## Quickstart

```bash
cd neuro-ui-judge

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# (optional) enable LLM redesign in the agent
export OPENROUTER_API_KEY="sk-or-..."

# Run the end-to-end demo: render 3 example pages, simulate preferences,
# retrain weights, and run the agent on the worst variant.
python -m examples.run_demo

# Serve the dashboard
uvicorn services.scorer.api:app --reload
# → http://localhost:8000
```

The demo populates `data/neuroui.db` with candidates, artifacts, reports,
preferences, weight versions, and one agent run. Open the dashboard and
navigate to **/agent** to see the iteration-by-iteration reward curve and
the side-by-side variants.

---

## Architecture

```
neuro-ui-judge/
├── apps/web/                  # server-rendered dashboard (Tailwind/Jinja)
│   ├── templates/             # home, upload, candidate, compare, preferences, agent
│   └── static/                # SVG/canvas charts (radar, ROI time series,
│                              # cortical heatmap, agent curve, bbox overlays)
├── renderer/
│   ├── playwright_render.py   # Python renderer (default)
│   └── playwright_render.ts   # parallel TS renderer (for a future Next.js front-end)
├── services/scorer/
│   ├── schemas.py             # Pydantic types: candidate, audit, neural, reward, prefs, agent
│   ├── deterministic_audit.py # WCAG / hierarchy / balance / density / CTA / harmony
│   ├── neural_proxy_mock.py   # mock TRIBE-like ROI features (transparent functions)
│   ├── tribe_adapter.py       # pluggable TRIBE backend (Protocol)
│   ├── reward_model.py        # hybrid reward + sub-scores + grade + recommendations
│   ├── preference_model.py    # Bradley-Terry weight calibration (numpy, no torch)
│   ├── agent.py               # evolutionary agent + 7 mutation operators
│   ├── storage.py             # SQLite persistence
│   └── api.py                 # FastAPI service + dashboard routes
├── examples/
│   ├── landing_clean.html     # high-quality baseline
│   ├── landing_busy.html      # high-density / low-hierarchy negative
│   ├── landing_low_contrast.html  # WCAG failure case
│   └── run_demo.py            # end-to-end pipeline demo
├── tests/
│   ├── test_deterministic_audit.py
│   ├── test_reward_model.py
│   └── test_preference_model.py
├── docs/
│   ├── scientific_framing.md
│   ├── reward_model.md
│   └── ethics.md
└── data/                       # SQLite DB + screenshots/frames written here at runtime
```

---

## API surface

JSON endpoints (FastAPI):

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/candidates` | upload an HTML candidate |
| POST | `/api/render` | render a stored candidate |
| POST | `/api/score` | render → audit → neural → reward |
| POST | `/api/compare` | rank a set of candidates |
| POST | `/api/preferences` | record a pairwise preference |
| POST | `/api/train-reward-model` | refit weights from preferences |
| POST | `/api/generate-variants` | run the design agent |
| GET | `/api/candidates` | list candidates |
| GET | `/api/candidates/{id}/report` | get a candidate's report |
| GET | `/api/experiments` | list agent runs |
| GET | `/api/experiments/{run_id}` | full run detail (with reports) |
| GET | `/api/screenshot/{id}` | rendered screenshot |
| GET | `/api/timeseries/{id}` | synthesized BOLD-like ROI traces |
| GET | `/api/artifact/{id}` | element bboxes (for overlays) |
| GET | `/api/weights` | active + historical weight versions |
| GET | `/api/health` | server + tribe-mode status |

---

## Plugging in real TRIBE v2

`services/scorer/tribe_adapter.py` defines a `TribeBackend` Protocol:

```python
class TribeBackend(Protocol):
    def predict(self, rendered: dict, audit: dict | None = None) -> dict: ...
```

Two ways to register a real backend:

1. **Programmatic**

    ```python
    from services.scorer import tribe_adapter
    from my_module import MyTribeBackend
    tribe_adapter.set_backend(MyTribeBackend())
    ```

2. **Environment variable**

    ```bash
    export NEUROUI_TRIBE_BACKEND="my_module:MyTribeBackend"
    ```

The contract: input is a `RenderedArtifact` dict (with `screenshot_path` and an
optional `frames_dir`); output is a `NeuralProxyFeatures` dict with
`mode == "tribe_v2"` and the same `roi_features` keys as the mock. A real
backend should also justify any upward adjustment of the `aesthetic` or
`valuation_proxy` confidence by reference to validation data.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The unit tests cover deterministic audit invariants, reward-model shape and
behaviour (accessibility gate, load inversion, monotonicity), and
preference-weight recovery on a synthetic single-feature dataset.

---

## License & ethics

This is a research prototype. **Read `docs/ethics.md` before use:**

- This is **not mind reading**. Neural proxies do not measure user preferences.
- Accessibility uses standards-based checks, not neural inference.
- Aesthetic / valuation channels are intentionally low-confidence and require
  validated subcortical TRIBE outputs to raise.
- Human testing is required before any product decision.
- TRIBE v2 may have non-commercial license terms. Real deployment requires a license review.
