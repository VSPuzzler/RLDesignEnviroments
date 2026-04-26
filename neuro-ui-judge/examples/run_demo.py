"""
End-to-end demo for NeuroUI Judge.

Loads three example HTML candidates, scores them, simulates a small set of
pairwise human preferences, retrains the reward weights, then runs the
design agent on the worst candidate and prints the final ranking.

Run from the repo root:
    cd neuro-ui-judge
    python -m examples.run_demo

The script never requires the FastAPI server to be running — it imports the
service modules directly.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=os.getenv("NEUROUI_LOG", "INFO"),
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

from renderer.playwright_render import render_html  # noqa: E402
from services.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    Storage,
    fit_preference_weights,
    run_audit,
    score_candidate,
    tribe_adapter,
)
from services.scorer.agent import AgentDeps, run_agent  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
RENDER_DIR = os.path.join(DATA_DIR, "renders")
DB_PATH = os.path.join(DATA_DIR, "neuroui.db")
os.makedirs(RENDER_DIR, exist_ok=True)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _load_examples() -> list[tuple[str, str, str]]:
    """Return [(candidate_id, label, html), ...]"""
    out = []
    for name in ("landing_clean.html", "landing_busy.html", "landing_low_contrast.html"):
        path = os.path.join(THIS_DIR, name)
        with open(path, encoding="utf-8") as f:
            html = f.read()
        cid = name.replace(".html", "")
        out.append((cid, cid, html))
    return out


def main() -> None:
    storage = Storage(DB_PATH)

    examples = _load_examples()
    log.info("Step 1/5  Render + score %d examples", len(examples))
    reports: dict[str, dict] = {}
    for cid, label, html in examples:
        storage.upsert_candidate({
            "candidate_id": cid, "label": label, "source": "html",
            "html": html, "task": "Sign up for free trial",
            "created_at": _now(), "metadata": {"example": True},
        })
        artifact = render_html(cid, html, RENDER_DIR)
        storage.upsert_artifact(artifact)
        audit = run_audit(artifact)
        neural = tribe_adapter.predict(artifact, audit)
        report = score_candidate(cid, audit, neural, weights=DEFAULT_WEIGHTS)
        storage.upsert_report(report)
        reports[cid] = report
        log.info(
            "  %-26s reward=%.3f grade=%s wcag=%s viols=%d",
            cid, report["overall_reward"], report["grade"],
            audit["wcag_pass"], len(audit["violations"]),
        )

    log.info("Step 2/5  Simulate pairwise preferences (clean > busy, clean > low-contrast)")
    sim_prefs = [
        ("landing_clean", "landing_busy", "a"),
        ("landing_clean", "landing_low_contrast", "a"),
        ("landing_busy", "landing_low_contrast", "a"),  # busy beats unreadable
        ("landing_clean", "landing_busy", "a"),
        ("landing_clean", "landing_low_contrast", "a"),
    ]
    for a, b, w in sim_prefs:
        storage.add_preference({
            "pref_id": uuid.uuid4().hex[:10],
            "ui_a_id": a, "ui_b_id": b, "winner": w,
            "task": "Sign up for free trial", "rater_id": "demo",
            "created_at": _now(),
        })

    log.info("Step 3/5  Fit reward weights from preferences")
    fit = fit_preference_weights(
        storage.list_preferences(), reports, n_steps=400
    )
    log.info("  pairwise_acc=%.3f n_train=%d n_val=%d",
             fit["metrics"]["pairwise_accuracy"],
             fit["metrics"]["n_train"], fit["metrics"]["n_val"])
    storage.add_weight_version({
        "version_id": fit["weights_version"],
        "weights": fit["weights"], "metrics": fit["metrics"],
        "notes": "demo run", "created_at": _now(),
    })

    log.info("Step 4/5  Re-score with calibrated weights")
    new_reports: dict[str, dict] = {}
    for cid, _, _ in examples:
        artifact = storage.get_artifact(cid)
        audit = run_audit(artifact)
        neural = tribe_adapter.predict(artifact, audit)
        rep = score_candidate(
            cid, audit, neural,
            weights=fit["weights"], weights_version=fit["weights_version"],
        )
        new_reports[cid] = rep
        log.info("  %-26s reward=%.3f (was %.3f)",
                 cid, rep["overall_reward"], reports[cid]["overall_reward"])
        storage.upsert_report(rep)

    log.info("Step 5/5  Run agent on the weakest candidate")
    weakest = min(new_reports.items(), key=lambda kv: kv[1]["overall_reward"])
    log.info("  starting from: %s (reward=%.3f)", weakest[0], weakest[1]["overall_reward"])
    seed_html = storage.get_candidate(weakest[0])["html"]

    def render_fn(cid, html, out_dir):
        art = render_html(cid, html, RENDER_DIR)
        storage.upsert_artifact(art)
        return art

    def score_fn(cid, art):
        audit = run_audit(art)
        neural = tribe_adapter.predict(art, audit)
        rep = score_candidate(
            cid, audit, neural,
            weights=fit["weights"], weights_version=fit["weights_version"],
        )
        storage.upsert_report(rep)
        return rep

    deps = AgentDeps(
        render_html=render_fn, score_artifact=score_fn,
        save_candidate=storage.upsert_candidate, save_report=storage.upsert_report,
    )
    run = run_agent(
        seed_html=seed_html,
        brief="Productivity SaaS landing page",
        task="Sign up for free trial",
        output_dir=RENDER_DIR,
        deps=deps,
        max_iterations=3,
        population_size=3,
        weights_version=fit["weights_version"],
    )
    storage.save_agent_run({**run, "started_at": _now(), "finished_at": _now()})

    seed_reward = run["iterations"][0]["best_reward"]
    final_reward = run["iterations"][-1]["best_reward"]
    log.info("  agent run %s: %.3f → %.3f (Δ=%+0.3f)",
             run["run_id"], seed_reward, final_reward,
             final_reward - seed_reward)

    print()
    print("=" * 78)
    print("  Demo finished. Visit http://localhost:8000 after running:")
    print("    uvicorn services.scorer.api:app --reload")
    print("  Recent agent run id:", run["run_id"])
    print("=" * 78)


if __name__ == "__main__":
    main()
