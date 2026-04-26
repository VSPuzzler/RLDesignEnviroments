"""
Pydantic schemas for NeuroUI Judge.

These types are the contract between the renderer, the deterministic audit,
the neural proxy, the reward model, the preference model, and the agent loop.
They are intentionally explicit so a JS/TS frontend can be code-gen'd from
the same source of truth later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Candidates and rendered artifacts ───────────────────────────────────────


class UICandidate(BaseModel):
    """A single UI candidate: HTML/CSS or a screenshot reference."""

    candidate_id: str
    label: str | None = None
    source: Literal["html", "screenshot", "prompt"] = "html"
    html: str | None = None
    screenshot_path: str | None = None
    prompt: str | None = None
    task: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: str | None = None  # for mutated/derived variants
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class RenderedElement(BaseModel):
    """A single visible element captured from the DOM/accessibility tree."""

    tag: str
    role: str | None = None
    text: str | None = None
    name: str | None = None  # accessible name
    bbox: BoundingBox
    font_size_px: float | None = None
    color: str | None = None  # rgb/rgba
    background_color: str | None = None
    is_interactive: bool = False
    is_cta: bool = False  # heuristic: prominent button/link
    has_alt_or_label: bool = True
    tab_index: int | None = None


class RenderedArtifact(BaseModel):
    """Everything captured by the renderer for one candidate."""

    candidate_id: str
    viewport_width: int
    viewport_height: int
    screenshot_path: str
    frames_dir: str | None = None  # optional frame sequence
    dom_tree_path: str | None = None
    elements: list[RenderedElement] = Field(default_factory=list)
    visible_text: str = ""
    accessibility_tree: dict[str, Any] | None = None
    page_metrics: dict[str, float] = Field(default_factory=dict)


# ── Deterministic audit ─────────────────────────────────────────────────────


class AuditViolation(BaseModel):
    rule: str
    severity: Literal["info", "minor", "major", "critical"]
    message: str
    element_index: int | None = None  # index into RenderedArtifact.elements


class DeterministicAudit(BaseModel):
    """
    Standards-based UI audit.  Every score is in [0, 1] where 1 = best.
    `density_penalty` is also [0, 1] where 1 = no penalty (sparse, clean).
    """

    accessibility: float
    readability: float
    visual_hierarchy: float
    layout_balance: float
    cta_clarity: float
    density_penalty: float
    spacing_consistency: float = 0.5
    color_harmony: float = 0.5
    wcag_pass: bool = False
    violations: list[AuditViolation] = Field(default_factory=list)
    raw_features: dict[str, float] = Field(default_factory=dict)


# ── Neural proxy (TRIBE-like) ───────────────────────────────────────────────


class ROIFeatures(BaseModel):
    """
    Per-ROI summary of a (predicted) BOLD-like time series.

    These are *not* measurements of preference. They are population-level
    encoder predictions interpreted as priors over attention / load /
    semantic processing / weak affect.
    """

    auc: float                     # area under predicted activation
    peak: float | None = None      # peak activation
    variance: float | None = None  # temporal variance
    suppression: float | None = None  # only meaningful for DMN


class NeuralProxyConfidence(BaseModel):
    attention: float
    load: float
    aesthetic: float  # weak by design
    accessibility: float  # we explicitly say neural ≠ accessibility


class NeuralProxyFeatures(BaseModel):
    """
    Mocked or real TRIBE-derived ROI features.

    Always include `mode` so downstream consumers know whether to trust the
    aesthetic / valuation channels. When `mode == "tribe_v2"`,
    ``vertex_activation`` carries the length-20484 fsaverage5 cortical
    activation map (normalised to [0, 1]) used by the dashboard's 3D brain.
    Mock mode also synthesises a plausible vertex array so the visual works
    without the real model.
    """

    mode: Literal["mock", "tribe_v2"] = "mock"
    roi_features: dict[str, ROIFeatures]
    confidence: NeuralProxyConfidence
    vertex_activation: list[float] | None = None
    n_segments: int | None = None
    describer: dict[str, Any] | None = None  # debug from describer.build_tribe_text
    notes: str = (
        "Population-level neural priors; not preference. "
        "Aesthetic/valuation channel is low-confidence unless validated."
    )


# ── Reward and report ───────────────────────────────────────────────────────


class Subscores(BaseModel):
    usability: float
    attention_guidance: float
    visual_hierarchy: float
    cognitive_load: float           # already inverted: 1 = low load
    readability: float
    aesthetic_quality: float
    accessibility: float
    engagement_proxy: float
    trust: float


class CandidateReport(BaseModel):
    """The full scoring report — what the dashboard renders."""

    candidate_id: str
    overall_reward: float
    grade: Literal["A", "B", "C", "D", "F"]
    subscores: Subscores
    deterministic_audit: DeterministicAudit
    neural_proxy: NeuralProxyFeatures
    confidence: NeuralProxyConfidence
    violations: list[AuditViolation] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    explanation: str
    weights_version: str = "default-v1"
    accessibility_gate_passed: bool = True
    defect_penalty: float = 0.0
    uncertainty_penalty: float = 0.0


# ── Preference data ─────────────────────────────────────────────────────────


class PairwisePreference(BaseModel):
    pref_id: str
    ui_a_id: str
    ui_b_id: str
    winner: Literal["a", "b", "tie"]
    task: str | None = None
    notes: str | None = None
    rater_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PreferenceModelMetrics(BaseModel):
    pairwise_accuracy: float
    train_loss: float
    val_loss: float | None = None
    n_train: int
    n_val: int
    spearman: float | None = None
    kendall: float | None = None


class WeightVersion(BaseModel):
    version_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    weights: dict[str, float]
    metrics: PreferenceModelMetrics | None = None
    notes: str = ""


# ── Agent runs ──────────────────────────────────────────────────────────────


class AgentIteration(BaseModel):
    iteration: int
    candidate_ids: list[str]
    best_candidate_id: str
    best_reward: float
    mutation_plan: list[str]
    explanation: str


class AgentRun(BaseModel):
    run_id: str
    brief: str
    task: str
    constraints: list[str] = Field(default_factory=list)
    max_iterations: int
    population_size: int
    iterations: list[AgentIteration] = Field(default_factory=list)
    final_top_k: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    weights_version: str = "default-v1"
