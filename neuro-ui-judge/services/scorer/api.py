"""
FastAPI service for NeuroUI Judge.

Endpoints (per spec):
    POST /api/candidates              upload an HTML candidate
    POST /api/render                  render a stored candidate
    POST /api/score                   render + audit + neural + reward
    POST /api/compare                 score two candidates side-by-side
    POST /api/preferences             record a pairwise human preference
    POST /api/train-reward-model      fit weights from preferences
    POST /api/generate-variants       run the design agent
    GET  /api/experiments             list agent runs
    GET  /api/experiments/{run_id}    fetch one agent run
    GET  /api/candidates/{id}/report  fetch a candidate's report

Plus a server-rendered dashboard at /, /candidate/{id}, /compare,
/preferences, /agent.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import (
    deterministic_audit,
    preference_model,
    reward_model,
    tribe_adapter,
    tribe_v2_backend,
)
from .agent import AgentDeps, run_agent
from .neural_proxy_mock import synthesize_roi_timeseries
from .storage import Storage

logger = logging.getLogger("neuroui.api")

# ── Paths ──────────────────────────────────────────────────────────────────

APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_db_path() -> str:
    """
    Resolve the SQLite DB path from env. Honors `DATABASE_URL=file:./data/x.db`
    (Prisma-style) and `NEUROUI_DB_PATH=/abs/path.db`. Falls back to
    `<APP_ROOT>/data/neuroui.db`.
    """
    explicit = os.environ.get("NEUROUI_DB_PATH", "").strip()
    if explicit:
        return explicit
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url.startswith("file:"):
        rel = db_url[len("file:") :]
        return rel if os.path.isabs(rel) else os.path.normpath(
            os.path.join(APP_ROOT, rel)
        )
    return os.path.join(APP_ROOT, "data", "neuroui.db")


DB_PATH = _resolve_db_path()
DATA_DIR = os.environ.get("NEUROUI_DATA_DIR", os.path.dirname(DB_PATH) or os.path.join(APP_ROOT, "data"))
RENDER_DIR = os.path.join(DATA_DIR, "renders")
TEMPLATE_DIR = os.path.join(APP_ROOT, "apps", "web", "templates")
STATIC_DIR = os.path.join(APP_ROOT, "apps", "web", "static")

os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

storage = Storage(DB_PATH)
# cache_size=0 avoids a Jinja2 LRU cache key bug seen on Python 3.14
# where unhashable globals dicts cause `TypeError: cannot use 'tuple' as a dict key`.
templates = Jinja2Templates(directory=TEMPLATE_DIR)
templates.env.cache = None


# ── Pydantic request bodies ────────────────────────────────────────────────


class CandidateIn(BaseModel):
    label: str | None = None
    html: str
    task: str | None = None
    candidate_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderIn(BaseModel):
    candidate_id: str
    viewport_width: int = 1440
    viewport_height: int = 900
    capture_frames: int = 0


class ScoreIn(BaseModel):
    candidate_id: str
    viewport_width: int = 1440
    viewport_height: int = 900


class CompareIn(BaseModel):
    candidate_ids: list[str]
    task: str | None = None


class PreferenceIn(BaseModel):
    ui_a_id: str
    ui_b_id: str
    winner: str
    task: str | None = None
    notes: str | None = None
    rater_id: str | None = None


class TrainIn(BaseModel):
    tau: float = 1.0
    n_steps: int = 600
    learning_rate: float = 0.5
    l2: float = 0.05


class GenerateIn(BaseModel):
    seed_html: str
    brief: str
    task: str
    max_iterations: int = 4
    population_size: int = 3
    use_llm: bool = False


# ── App + middleware ───────────────────────────────────────────────────────


app = FastAPI(
    title="NeuroUI Judge",
    description="Hybrid neural + deterministic UI scoring with preference calibration.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.utcnow().isoformat()


def _active_weights() -> tuple[dict[str, float], str]:
    aw = storage.get_active_weights()
    if aw and aw.get("weights"):
        return aw["weights"], aw["version_id"]
    return reward_model.DEFAULT_WEIGHTS, "default-v1"


def _render_candidate(candidate_id: str, html: str, **kwargs) -> dict[str, Any]:
    # Local import keeps the top-level import light if playwright is missing.
    from renderer.playwright_render import render_html  # type: ignore

    artifact = render_html(candidate_id, html, RENDER_DIR, **kwargs)
    storage.upsert_artifact(artifact)
    return artifact


def _score_artifact(candidate_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    audit = deterministic_audit.run_audit(artifact)
    neural = tribe_adapter.predict(artifact, audit)
    weights, version = _active_weights()
    report = reward_model.score_candidate(
        candidate_id, audit, neural, weights=weights, weights_version=version
    )
    storage.upsert_report(report)
    return report


def _ensure_scored(candidate_id: str) -> dict[str, Any]:
    rep = storage.get_report(candidate_id)
    if rep:
        return rep
    cand = storage.get_candidate(candidate_id)
    if not cand or not cand.get("html"):
        raise HTTPException(404, f"Candidate {candidate_id} has no HTML to score.")
    art = storage.get_artifact(candidate_id) or _render_candidate(
        candidate_id, cand["html"]
    )
    return _score_artifact(candidate_id, art)


# ── JSON API ───────────────────────────────────────────────────────────────


@app.post("/api/candidates")
def api_create_candidate(body: CandidateIn) -> dict[str, Any]:
    cid = body.candidate_id or uuid.uuid4().hex[:10]
    storage.upsert_candidate(
        {
            "candidate_id": cid,
            "label": body.label or cid[:6],
            "source": "html",
            "html": body.html,
            "task": body.task,
            "created_at": _now(),
            "metadata": body.metadata,
        }
    )
    return {"candidate_id": cid}


@app.post("/api/render")
def api_render(body: RenderIn) -> dict[str, Any]:
    cand = storage.get_candidate(body.candidate_id)
    if not cand or not cand.get("html"):
        raise HTTPException(404, "Candidate not found.")
    art = _render_candidate(
        body.candidate_id,
        cand["html"],
        viewport_width=body.viewport_width,
        viewport_height=body.viewport_height,
        capture_frames=body.capture_frames,
    )
    return {"candidate_id": body.candidate_id, "screenshot_path": art["screenshot_path"]}


@app.post("/api/score")
def api_score(body: ScoreIn) -> dict[str, Any]:
    cand = storage.get_candidate(body.candidate_id)
    if not cand or not cand.get("html"):
        raise HTTPException(404, "Candidate not found.")
    art = _render_candidate(
        body.candidate_id,
        cand["html"],
        viewport_width=body.viewport_width,
        viewport_height=body.viewport_height,
    )
    return _score_artifact(body.candidate_id, art)


@app.post("/api/compare")
def api_compare(body: CompareIn) -> dict[str, Any]:
    if len(body.candidate_ids) < 2:
        raise HTTPException(400, "Need at least two candidates.")
    reports = [_ensure_scored(cid) for cid in body.candidate_ids]
    weights, _ = _active_weights()
    ranking = sorted(reports, key=lambda r: r["overall_reward"], reverse=True)
    pairwise = []
    if len(reports) == 2:
        p = preference_model.predict_pairwise_probability(
            reports[0], reports[1], weights
        )
        pairwise = [{"a": reports[0]["candidate_id"], "b": reports[1]["candidate_id"], "p_a_over_b": p}]
    return {"reports": reports, "ranking": [r["candidate_id"] for r in ranking], "pairwise": pairwise}


@app.post("/api/preferences")
def api_add_preference(body: PreferenceIn) -> dict[str, Any]:
    pid = uuid.uuid4().hex[:10]
    if body.winner not in ("a", "b", "tie"):
        raise HTTPException(400, "winner must be 'a', 'b', or 'tie'")
    storage.add_preference(
        {
            "pref_id": pid,
            "ui_a_id": body.ui_a_id,
            "ui_b_id": body.ui_b_id,
            "winner": body.winner,
            "task": body.task,
            "notes": body.notes,
            "rater_id": body.rater_id,
            "created_at": _now(),
        }
    )
    return {"pref_id": pid}


@app.get("/api/preferences")
def api_list_preferences() -> list[dict[str, Any]]:
    return storage.list_preferences()


@app.post("/api/train-reward-model")
def api_train(body: TrainIn) -> dict[str, Any]:
    prefs = storage.list_preferences()
    if not prefs:
        raise HTTPException(400, "No preferences stored yet.")
    ids: set[str] = set()
    for p in prefs:
        ids.add(p["ui_a_id"])
        ids.add(p["ui_b_id"])
    reports = {r["candidate_id"]: r for r in storage.list_reports(ids)}
    # Score on the fly for any candidate that hasn't been scored yet.
    for cid in ids - set(reports):
        try:
            reports[cid] = _ensure_scored(cid)
        except HTTPException:
            continue
    out = preference_model.fit_preference_weights(
        prefs,
        reports,
        tau=body.tau,
        learning_rate=body.learning_rate,
        n_steps=body.n_steps,
        l2=body.l2,
    )
    out["created_at"] = _now()
    out["notes"] = f"Fit on {len(prefs)} preferences."
    storage.add_weight_version(
        {
            "version_id": out["weights_version"],
            "created_at": out["created_at"],
            "weights": out["weights"],
            "metrics": out["metrics"],
            "notes": out["notes"],
        }
    )
    out["calibration"] = preference_model.calibration_curve(prefs, reports, out["weights"], out["tau"])
    return out


@app.post("/api/generate-variants")
def api_generate(body: GenerateIn) -> dict[str, Any]:
    weights, version = _active_weights()

    def render_fn(cid: str, html: str, out_dir: str) -> dict[str, Any]:
        return _render_candidate(cid, html)

    def score_fn(cid: str, artifact: dict[str, Any]) -> dict[str, Any]:
        return _score_artifact(cid, artifact)

    deps = AgentDeps(
        render_html=render_fn,
        score_artifact=score_fn,
        save_candidate=storage.upsert_candidate,
        save_report=storage.upsert_report,
    )
    run = run_agent(
        seed_html=body.seed_html,
        brief=body.brief,
        task=body.task,
        output_dir=RENDER_DIR,
        deps=deps,
        max_iterations=body.max_iterations,
        population_size=body.population_size,
        use_llm=body.use_llm,
        weights_version=version,
    )
    run["started_at"] = run.get("started_at") or _now()
    run["finished_at"] = _now()
    storage.save_agent_run(run)
    return run


@app.get("/api/candidates")
def api_list_candidates() -> list[dict[str, Any]]:
    return storage.list_candidates()


@app.get("/api/candidates/{candidate_id}/report")
def api_candidate_report(candidate_id: str) -> dict[str, Any]:
    return _ensure_scored(candidate_id)


@app.get("/api/experiments")
def api_list_runs() -> list[dict[str, Any]]:
    return storage.list_agent_runs()


@app.get("/api/experiments/{run_id}")
def api_get_run(run_id: str) -> dict[str, Any]:
    run = storage.get_agent_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found.")
    # Attach reports for every candidate referenced in the run.
    cand_ids: list[str] = []
    for it in run.get("iterations", []):
        cand_ids.extend(it.get("candidate_ids", []))
    cand_ids = list(dict.fromkeys(cand_ids))
    reports = {r["candidate_id"]: r for r in storage.list_reports(cand_ids)}
    run["reports"] = reports
    return run


@app.get("/api/weights")
def api_list_weights() -> dict[str, Any]:
    versions = storage.list_weight_versions()
    return {
        "active": storage.get_active_weights() or {
            "version_id": "default-v1",
            "weights": reward_model.DEFAULT_WEIGHTS,
        },
        "versions": versions,
    }


@app.get("/api/screenshot/{candidate_id}")
def api_screenshot(candidate_id: str):
    art = storage.get_artifact(candidate_id)
    if not art or not art.get("screenshot_path"):
        raise HTTPException(404, "No screenshot available.")
    p = art["screenshot_path"]
    if not os.path.exists(p):
        raise HTTPException(404, "Screenshot file missing on disk.")
    return FileResponse(p, media_type="image/png")


@app.get("/api/timeseries/{candidate_id}")
def api_timeseries(candidate_id: str) -> dict[str, Any]:
    rep = _ensure_scored(candidate_id)
    series = synthesize_roi_timeseries(rep["neural_proxy"], n_steps=24, seed=hash(candidate_id) & 0xffff)
    return {"candidate_id": candidate_id, "series": series}


@app.get("/api/artifact/{candidate_id}")
def api_artifact(candidate_id: str) -> dict[str, Any]:
    art = storage.get_artifact(candidate_id)
    if not art:
        raise HTTPException(404, "No artifact for candidate.")
    # Strip large fields the dashboard doesn't need so payloads stay small.
    return {
        "candidate_id": candidate_id,
        "viewport_width": art.get("viewport_width"),
        "viewport_height": art.get("viewport_height"),
        "elements": art.get("elements", []),
        "page_metrics": art.get("page_metrics", {}),
    }


@app.get("/api/vertex-activation/{candidate_id}")
def api_vertex_activation(candidate_id: str) -> dict[str, Any]:
    """
    Per-candidate length-20484 cortical activation array for the 3D brain.

    Returns the array stored in the candidate's report. If the report
    doesn't include a vertex activation (older mock reports, real backend
    failed), we synthesise one on the fly from the ROI summary so the
    dashboard's brain still renders.
    """
    rep = _ensure_scored(candidate_id)
    neural = rep.get("neural_proxy") or {}
    arr = neural.get("vertex_activation")
    synthesised = False
    if not arr:
        from .neural_proxy_mock import synthesize_vertex_activation

        arr = synthesize_vertex_activation(
            neural, seed=hash(candidate_id) & 0xFFFF
        )
        synthesised = True
    return {
        "candidate_id": candidate_id,
        "mode": neural.get("mode", "mock"),
        "n_vertices": len(arr),
        "n_segments": neural.get("n_segments"),
        "synthesised": synthesised,
        "vertex_activation": arr,
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {"ok": True, "tribe_mode": tribe_adapter.active_mode()}


@app.get("/api/tribe-status")
def api_tribe_status() -> dict[str, Any]:
    """
    Where TRIBE is sourced right now — `mock` (deterministic synthesis),
    or `tribe_v2` (real sidecar). Includes a fresh /health probe of the
    sidecar so the dashboard can show whether weights are loaded yet.
    """
    sidecar_url = os.getenv(
        "TRIBE_V2_SERVICE_URL", tribe_v2_backend.DEFAULT_SIDECAR_URL
    )
    probe = tribe_v2_backend._ping(sidecar_url) if sidecar_url else None
    return {
        "active_mode": tribe_adapter.active_mode(),
        "sidecar_url": sidecar_url,
        "sidecar_alive": probe is not None,
        "sidecar_health": probe,
    }


@app.post("/api/tribe-reconnect")
def api_tribe_reconnect() -> dict[str, Any]:
    """
    Re-attempt to register the TRIBE HTTP backend. Useful after the user
    has booted the sidecar in a separate terminal.
    """
    registered = tribe_v2_backend.auto_register(tribe_adapter)
    return {
        "registered": registered,
        "active_mode": tribe_adapter.active_mode(),
    }


@app.on_event("startup")
def _startup_register_tribe() -> None:
    """Auto-wire the real TRIBE backend if the sidecar is reachable."""
    try:
        tribe_v2_backend.auto_register(tribe_adapter)
    except RuntimeError as exc:
        # Only raised when TRIBE_V2_REQUIRED=1; let uvicorn fail the boot.
        logger.error("Required TRIBE sidecar unavailable: %s", exc)
        raise


# ── Server-rendered dashboard ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def page_home(request: Request):
    cands = storage.list_candidates(limit=50)
    runs = storage.list_agent_runs()
    aw = storage.get_active_weights() or {
        "version_id": "default-v1",
        "weights": reward_model.DEFAULT_WEIGHTS,
        "metrics": {},
    }
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "candidates": cands,
            "runs": runs,
            "active_weights": aw,
            "tribe_mode": tribe_adapter.active_mode(),
            "preference_count": len(storage.list_preferences()),
        },
    )


@app.get("/candidate/{candidate_id}", response_class=HTMLResponse)
def page_candidate(request: Request, candidate_id: str):
    rep = _ensure_scored(candidate_id)
    cand = storage.get_candidate(candidate_id)
    return templates.TemplateResponse(
        request,
        "candidate.html",
        {
            "candidate": cand,
            "report": rep,
            "tribe_mode": tribe_adapter.active_mode(),
        },
    )


@app.get("/compare", response_class=HTMLResponse)
def page_compare(request: Request, a: str | None = None, b: str | None = None):
    cands = storage.list_candidates(limit=200)
    report_a = _ensure_scored(a) if a else None
    report_b = _ensure_scored(b) if b else None
    pairwise_p = None
    if report_a and report_b:
        weights, _ = _active_weights()
        pairwise_p = preference_model.predict_pairwise_probability(
            report_a, report_b, weights
        )
    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "candidates": cands,
            "selected_a": a,
            "selected_b": b,
            "report_a": report_a,
            "report_b": report_b,
            "pairwise_p": pairwise_p,
            "tribe_mode": tribe_adapter.active_mode(),
        },
    )


@app.get("/preferences", response_class=HTMLResponse)
def page_preferences(request: Request):
    prefs = storage.list_preferences()
    versions = storage.list_weight_versions()
    return templates.TemplateResponse(
        request,
        "preferences.html",
        {
            "preferences": prefs,
            "versions": versions,
            "tribe_mode": tribe_adapter.active_mode(),
        },
    )


@app.get("/agent", response_class=HTMLResponse)
def page_agent(request: Request, run_id: str | None = None):
    run = api_get_run(run_id) if run_id else None
    runs = storage.list_agent_runs()
    return templates.TemplateResponse(
        request,
        "agent.html",
        {
            "run": run,
            "runs": runs,
            "tribe_mode": tribe_adapter.active_mode(),
        },
    )


@app.get("/upload", response_class=HTMLResponse)
def page_upload(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {"tribe_mode": tribe_adapter.active_mode()},
    )
