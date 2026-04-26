"""
Offline UI design agent / reranker.

The agent runs an evolutionary search over HTML candidates:
  for iteration in range(max_iters):
      generate / mutate candidates
      render + audit + neural-proxy + score
      keep top-k
      pick a mutation plan based on the report's recommendations

We deliberately avoid online RL: every mutation is deterministic-or-LLM and
every selection is offline. The interface is structured so PPO / DPO can be
slotted in later by replacing `mutate_candidate`.

Mutation operators are pure HTML/CSS transforms. They are intentionally
simple and inspectable. Each one targets one of the spec's mutation goals:
  - improve contrast
  - simplify layout
  - make CTA more prominent
  - reduce text density
  - improve spacing
  - improve semantic labels
  - reorganize visual hierarchy

If `OPENROUTER_API_KEY` is set, we additionally allow an "llm_redesign"
operator that asks a model on OpenRouter (default: anthropic/claude-sonnet-4.5)
to produce a fresh variant guided by the previous report's recommendations.
All LLM traffic in NeuroUI Judge goes through `llm_client.chat`.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from . import llm_client

logger = logging.getLogger(__name__)


# ── Tiny HTML helpers (regex-based, intentionally non-fancy) ────────────────


def _ensure_style_block(html: str) -> tuple[str, int]:
    """Return (html_with_style_block, insertion_index_for_extra_css)."""
    m = re.search(r"<style[^>]*>", html, re.IGNORECASE)
    if m:
        return html, m.end()
    head = re.search(r"<head[^>]*>", html, re.IGNORECASE)
    if head:
        new = (
            html[: head.end()]
            + "<style>\n/* neuroui-judge mutations */\n</style>"
            + html[head.end():]
        )
        m2 = re.search(r"<style[^>]*>", new, re.IGNORECASE)
        return new, m2.end() if m2 else len(new)
    # No <head>: prepend.
    new = "<style>\n/* neuroui-judge mutations */\n</style>" + html
    m2 = re.search(r"<style[^>]*>", new, re.IGNORECASE)
    return new, m2.end() if m2 else 0


def _inject_css(html: str, css: str) -> str:
    new, idx = _ensure_style_block(html)
    return new[:idx] + "\n" + css + "\n" + new[idx:]


# ── Mutation operators ──────────────────────────────────────────────────────


def op_improve_contrast(html: str) -> str:
    css = """
:root, body { color: #111 !important; }
body { background: #ffffff !important; }
button, .cta, [role="button"], a.cta {
  background: #1a1a1a !important;
  color: #ffffff !important;
  border: 1px solid #1a1a1a !important;
}
.muted, .secondary { color: #444 !important; }
"""
    return _inject_css(html, css)


def op_simplify_layout(html: str) -> str:
    css = """
* { box-sizing: border-box; }
body { max-width: 1100px; margin: 0 auto !important; padding: 32px !important; }
section, .section { padding: 32px 0 !important; }
.grid, .columns { display: block !important; }
img, video { max-width: 100% !important; height: auto !important; }
.aside, .sidebar { display: none !important; }
"""
    return _inject_css(html, css)


def op_emphasise_cta(html: str) -> str:
    css = """
button, a.cta, [role="button"], .btn-primary {
  font-size: 18px !important;
  padding: 14px 24px !important;
  border-radius: 10px !important;
  background: #ff5630 !important;
  color: #fff !important;
  font-weight: 700 !important;
  box-shadow: 0 6px 20px rgba(255,86,48,.25) !important;
}
button + button, a.cta + a.cta { margin-left: 12px !important; opacity: .7; }
"""
    return _inject_css(html, css)


def op_reduce_density(html: str) -> str:
    css = """
p, li { line-height: 1.65 !important; max-width: 60ch !important; }
section, .section { padding-top: 48px !important; padding-bottom: 48px !important; }
h1 { margin-bottom: 16px !important; }
h2 { margin-top: 32px !important; margin-bottom: 12px !important; }
.dense, .compact { letter-spacing: .01em !important; }
small, .small { display: none !important; }
"""
    return _inject_css(html, css)


def op_improve_spacing(html: str) -> str:
    css = """
body { line-height: 1.55 !important; }
* { margin-block: 0; }
h1, h2, h3 { margin-top: 28px !important; margin-bottom: 12px !important; }
p { margin-block: 12px !important; }
section, .section { margin-block: 40px !important; }
button { margin: 8px !important; }
"""
    return _inject_css(html, css)


def op_semantic_labels(html: str) -> str:
    """Add aria-label="<text>" to bare <button>/<a> with no aria-label yet."""
    def repl(m: re.Match[str]) -> str:
        tag_open, body, tag_close = m.group(1), m.group(2), m.group(3)
        if "aria-label" in tag_open.lower():
            return m.group(0)
        text = re.sub(r"<[^>]+>", "", body).strip().replace('"', "'")[:60]
        if not text:
            return m.group(0)
        new_open = tag_open[:-1] + f' aria-label="{text}">'
        return new_open + body + tag_close

    pat = re.compile(r"(<(?:button|a)\b[^>]*>)([\s\S]*?)(</(?:button|a)>)", re.IGNORECASE)
    return pat.sub(repl, html)


def op_reorganise_hierarchy(html: str) -> str:
    css = """
h1 { font-size: 56px !important; line-height: 1.1 !important; font-weight: 800 !important; }
h2 { font-size: 32px !important; font-weight: 700 !important; }
h3 { font-size: 22px !important; font-weight: 600 !important; }
p, li { font-size: 17px !important; }
.eyebrow, .kicker { font-size: 13px !important; letter-spacing: .12em !important;
  text-transform: uppercase !important; opacity: .8 !important; }
"""
    return _inject_css(html, css)


MUTATION_OPS: dict[str, Callable[[str], str]] = {
    "improve_contrast": op_improve_contrast,
    "simplify_layout": op_simplify_layout,
    "emphasise_cta": op_emphasise_cta,
    "reduce_density": op_reduce_density,
    "improve_spacing": op_improve_spacing,
    "semantic_labels": op_semantic_labels,
    "reorganise_hierarchy": op_reorganise_hierarchy,
}


# ── Mutation-plan selector ──────────────────────────────────────────────────


def plan_from_report(report: dict[str, Any]) -> list[str]:
    """
    Translate a report's weak sub-scores into an ordered mutation plan.

    We pick at most three operators per iteration so changes are visible
    rather than washing out.
    """
    subs = report["subscores"]
    audit = report["deterministic_audit"]
    plan: list[str] = []

    if not audit["wcag_pass"] or subs["accessibility"] < 0.6:
        plan.append("improve_contrast")
        plan.append("semantic_labels")
    if subs["attention_guidance"] < 0.6:
        plan.append("emphasise_cta")
    if subs["visual_hierarchy"] < 0.6:
        plan.append("reorganise_hierarchy")
    if subs["cognitive_load"] < 0.6:
        plan.append("reduce_density")
    if subs["readability"] < 0.6:
        plan.append("reorganise_hierarchy")
    if subs["aesthetic_quality"] < 0.6 or audit["spacing_consistency"] < 0.6:
        plan.append("improve_spacing")

    # Deduplicate, preserve order, cap at 3.
    seen: set[str] = set()
    out: list[str] = []
    for op in plan:
        if op not in seen:
            out.append(op)
            seen.add(op)
        if len(out) >= 3:
            break
    if not out:
        out = ["improve_spacing"]  # gentle default
    return out


def mutate_html(html: str, plan: list[str]) -> str:
    """Apply each operator in `plan` in order."""
    out = html
    for op in plan:
        fn = MUTATION_OPS.get(op)
        if fn is None:
            logger.debug("Unknown mutation op: %s", op)
            continue
        out = fn(out)
    return out


# ── Optional LLM redesign ───────────────────────────────────────────────────


def llm_redesign(html: str, brief: str, recommendations: list[str]) -> str | None:
    """
    Ask an OpenRouter model to produce a fresh variant guided by the
    recommendations. Returns None if OpenRouter isn't configured or the
    call fails — the agent then falls back to operator-only mutations.
    """
    if not llm_client.is_configured():
        return None
    sys_prompt = (
        "You are a senior product designer. You will receive an HTML file "
        "and a list of issues. Return a single self-contained HTML file "
        "with inline <style> that addresses every issue. Output HTML only "
        "— no markdown fences, no commentary."
    )
    user = (
        f"Brief:\n{brief}\n\n"
        f"Issues to address:\n- " + "\n- ".join(recommendations) + "\n\n"
        f"Original HTML:\n{html[:12000]}"
    )
    raw = llm_client.chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
    )
    if raw is None:
        return None
    m = re.search(r"```(?:html)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    return (m.group(1) if m else raw).strip()


# ── Agent run ───────────────────────────────────────────────────────────────


@dataclass
class AgentDeps:
    """Callbacks injected by the orchestrator so the agent stays decoupled."""

    render_html: Callable[[str, str, str], dict[str, Any]]
    """(candidate_id, html, output_dir) -> RenderedArtifact dict."""

    score_artifact: Callable[[str, dict[str, Any]], dict[str, Any]]
    """(candidate_id, rendered) -> CandidateReport dict."""

    save_candidate: Callable[[dict[str, Any]], None]
    """Persist a candidate (dict matching `UICandidate`)."""

    save_report: Callable[[dict[str, Any]], None]
    """Persist a report (dict matching `CandidateReport`)."""


def run_agent(
    *,
    seed_html: str,
    brief: str,
    task: str,
    output_dir: str,
    deps: AgentDeps,
    max_iterations: int = 4,
    population_size: int = 3,
    use_llm: bool = False,
    weights_version: str = "default-v1",
) -> dict[str, Any]:
    """
    Run an offline evolutionary search starting from `seed_html`.

    Returns an `AgentRun`-shaped dict (with iterations and a final top-K).
    """
    run_id = uuid.uuid4().hex[:10]
    iterations: list[dict[str, Any]] = []

    # Iteration 0: score the seed.
    seed_id = f"{run_id}-seed"
    deps.save_candidate(
        {
            "candidate_id": seed_id,
            "label": "seed",
            "source": "html",
            "html": seed_html,
            "task": task,
            "parent_id": None,
            "metadata": {"iteration": 0, "operators": []},
        }
    )
    seed_rendered = deps.render_html(seed_id, seed_html, output_dir)
    seed_report = deps.score_artifact(seed_id, seed_rendered)
    deps.save_report(seed_report)

    population: list[tuple[str, str, dict[str, Any]]] = [
        (seed_id, seed_html, seed_report)
    ]

    iterations.append(
        {
            "iteration": 0,
            "candidate_ids": [seed_id],
            "best_candidate_id": seed_id,
            "best_reward": float(seed_report["overall_reward"]),
            "mutation_plan": [],
            "explanation": "Seeded with user-supplied baseline.",
        }
    )

    for it in range(1, max_iterations + 1):
        # Pick the best parent from the previous population.
        parent_id, parent_html, parent_report = max(
            population, key=lambda triple: triple[2]["overall_reward"]
        )
        plan = plan_from_report(parent_report)

        children: list[tuple[str, str, dict[str, Any]]] = []
        # Operator-mutated children.
        for k in range(min(population_size, len(plan)) or 1):
            child_plan = plan[: k + 1] if k < len(plan) else plan
            child_html = mutate_html(parent_html, child_plan)
            child_id = f"{run_id}-it{it}-m{k}"
            deps.save_candidate(
                {
                    "candidate_id": child_id,
                    "label": "+".join(child_plan),
                    "source": "html",
                    "html": child_html,
                    "task": task,
                    "parent_id": parent_id,
                    "metadata": {"iteration": it, "operators": child_plan},
                }
            )
            r = deps.render_html(child_id, child_html, output_dir)
            rep = deps.score_artifact(child_id, r)
            deps.save_report(rep)
            children.append((child_id, child_html, rep))

        # Optional LLM-driven redesign as one extra child.
        if use_llm:
            redesign = llm_redesign(parent_html, brief, parent_report["recommendations"])
            if redesign:
                child_id = f"{run_id}-it{it}-llm"
                deps.save_candidate(
                    {
                        "candidate_id": child_id,
                        "label": "llm_redesign",
                        "source": "html",
                        "html": redesign,
                        "task": task,
                        "parent_id": parent_id,
                        "metadata": {"iteration": it, "operators": ["llm_redesign"]},
                    }
                )
                r = deps.render_html(child_id, redesign, output_dir)
                rep = deps.score_artifact(child_id, r)
                deps.save_report(rep)
                children.append((child_id, redesign, rep))

        population = sorted(
            population + children,
            key=lambda triple: triple[2]["overall_reward"],
            reverse=True,
        )[: max(population_size, 3)]

        best = population[0]
        iterations.append(
            {
                "iteration": it,
                "candidate_ids": [c[0] for c in children],
                "best_candidate_id": best[0],
                "best_reward": float(best[2]["overall_reward"]),
                "mutation_plan": plan,
                "explanation": (
                    f"Applied {plan} to parent {parent_id[-8:]}; "
                    f"best child reward {max(c[2]['overall_reward'] for c in children):.3f}, "
                    f"running best {best[2]['overall_reward']:.3f}."
                ),
            }
        )

    final_top_k = [c[0] for c in population]

    return {
        "run_id": run_id,
        "brief": brief,
        "task": task,
        "constraints": [],
        "max_iterations": max_iterations,
        "population_size": population_size,
        "iterations": iterations,
        "final_top_k": final_top_k,
        "weights_version": weights_version,
    }
