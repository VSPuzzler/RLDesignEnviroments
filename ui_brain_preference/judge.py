"""LLM judge agent: evaluate two UI variants based on brain ROI activations."""

from __future__ import annotations

import json
import logging
import re
from typing import TypedDict

from openai import OpenAI

from config import JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)


class JudgmentResult(TypedDict):
    winner: str          # "A" | "B" | "tie"
    explanation: str
    preference_score: float  # -1 (strongly prefer A) … +1 (strongly prefer B)


_SYSTEM_PROMPT = """\
You are a cognitive neuroscience expert evaluating UI designs based on predicted
brain activation patterns from a neural encoding model (Tribe V2).

The activation values come from key functional ROIs (regions of interest):
- visual_cortex_v1/v4: early/mid visual processing – higher = richer visual stimulus
- fusiform_face_area:  object/face recognition – higher = more recognisable structure
- reward_pathway:      reward/motivation signal – higher = more engaging / desirable
- prefrontal_cortex:   executive processing – higher = more cognitive engagement
- default_mode_network: mind-wandering / self-referential – lower is often better for
                        focused UX
- attention_network:   directed attention – higher = design draws focus effectively

Given ROI data for two designs, you must:
1. Identify which design better achieves the stated goal.
2. Cite specific ROI differences to justify your choice.
3. Output a structured JSON object – nothing else.

JSON schema:
{
  "winner": "<A|B|tie>",
  "explanation": "<2-4 sentence neuroscience-grounded rationale>",
  "preference_score": <float between -1.0 and 1.0>
}

preference_score convention:
  -1.0 = Design A is strongly preferred
   0.0 = tie / no meaningful difference
  +1.0 = Design B is strongly preferred\
"""

_USER_TEMPLATE = """\
Design goal: {design_goal}

Candidate A – ROI activations:
{roi_a}

Candidate B – ROI activations:
{roi_b}

Evaluate the two designs and return only the JSON object.\
"""


def _parse_judgment(raw: str) -> JudgmentResult:
    """Extract the JSON judgment from the model response."""
    # Strip markdown fences if present
    match = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", raw, re.IGNORECASE)
    json_str = match.group(1) if match else raw.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt to find the first {...} block as a fallback
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            data = json.loads(brace_match.group())
        else:
            raise ValueError(f"Could not parse JSON from judge response:\n{raw}")

    winner = str(data.get("winner", "tie")).upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "tie"

    score = float(data.get("preference_score", 0.0))
    score = max(-1.0, min(1.0, score))

    return JudgmentResult(
        winner=winner.capitalize() if winner != "TIE" else "tie",
        explanation=str(data.get("explanation", "")),
        preference_score=score,
    )


def judge_preference(
    roi_a: dict[str, float],
    roi_b: dict[str, float],
    design_goal: str,
) -> JudgmentResult:
    """
    Ask the LLM judge to compare two designs based on their brain ROI data.

    Args:
        roi_a:        ROI activations for design A.
        roi_b:        ROI activations for design B.
        design_goal:  Natural-language statement of what the design should achieve.

    Returns:
        JudgmentResult with winner, explanation, and preference_score.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Export it before running."
        )

    def _fmt(roi: dict[str, float]) -> str:
        return "\n".join(f"  {k}: {v:.4f}" for k, v in roi.items())

    user_msg = _USER_TEMPLATE.format(
        design_goal=design_goal,
        roi_a=_fmt(roi_a),
        roi_b=_fmt(roi_b),
    )

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    logger.info("Requesting judgment from %s …", JUDGE_MODEL)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/vspuzzler/rldesignenviroments",
            "X-Title": "UI Brain Preference",
        },
    )

    raw = response.choices[0].message.content or ""
    logger.debug("Raw judge response:\n%s", raw)

    result = _parse_judgment(raw)
    logger.info(
        "Judgment: winner=%s  score=%.3f", result["winner"], result["preference_score"]
    )
    return result
