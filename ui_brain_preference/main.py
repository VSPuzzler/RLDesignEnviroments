"""
Orchestration: run one full UI preference comparison end-to-end.

Usage:
    python main.py "Create a landing page for a productivity app"

The pipeline:
  1. Generate two HTML/CSS variants via OpenRouter.
  2. Screenshot both at 1920×1080 using Playwright.
  3. Predict brain responses (Tribe V2 or simulation).
  4. LLM judge picks a winner based on ROI activations.
  5. Append the full comparison record to outputs/comparisons.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from brain_scorer import predict_brain_response
from config import (
    COMPARISONS_FILE,
    SCREENSHOTS_DIR,
    UI_VARIANTS_DIR,
)
from judge import judge_preference
from renderer import screenshot_html
from ui_generator import generate_ui_variant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _ensure_dirs() -> None:
    for d in (UI_VARIANTS_DIR, SCREENSHOTS_DIR, os.path.dirname(COMPARISONS_FILE)):
        os.makedirs(d, exist_ok=True)


def _load_comparisons() -> list[dict]:
    if os.path.exists(COMPARISONS_FILE):
        with open(COMPARISONS_FILE, encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_comparisons(records: list[dict]) -> None:
    with open(COMPARISONS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def run_comparison(spec: str, design_goal: str | None = None) -> dict:
    """
    Execute one full A/B comparison for the given UI spec.

    Args:
        spec:         Natural-language UI specification (drives generation).
        design_goal:  What the design should achieve (drives judgment).
                      Defaults to spec if not provided.

    Returns:
        The comparison record that was appended to comparisons.json.
    """
    _ensure_dirs()

    if design_goal is None:
        design_goal = spec

    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("=== Run %s ===  spec: %s", run_id, spec)

    # ── Step 1: Generate UI variants ─────────────────────────────────────────
    logger.info("Step 1/5 – Generating UI variants …")
    t0 = time.monotonic()

    ui_a = generate_ui_variant(spec, variant_seed=1)
    ui_b = generate_ui_variant(spec, variant_seed=2)

    html_a_path = os.path.join(UI_VARIANTS_DIR, f"{run_id}_variant_a.html")
    html_b_path = os.path.join(UI_VARIANTS_DIR, f"{run_id}_variant_b.html")
    with open(html_a_path, "w", encoding="utf-8") as f:
        f.write(ui_a)
    with open(html_b_path, "w", encoding="utf-8") as f:
        f.write(ui_b)

    logger.info("Variants saved (%.1fs).", time.monotonic() - t0)

    # ── Step 2: Screenshot ───────────────────────────────────────────────────
    logger.info("Step 2/5 – Rendering screenshots …")
    t0 = time.monotonic()

    img_a = screenshot_html(
        ui_a,
        os.path.join(SCREENSHOTS_DIR, f"{run_id}_variant_a.png"),
    )
    img_b = screenshot_html(
        ui_b,
        os.path.join(SCREENSHOTS_DIR, f"{run_id}_variant_b.png"),
    )

    logger.info("Screenshots done (%.1fs).", time.monotonic() - t0)

    # ── Step 3: Brain predictions ────────────────────────────────────────────
    logger.info("Step 3/5 – Predicting brain responses …")
    t0 = time.monotonic()

    roi_a = predict_brain_response(img_a)
    roi_b = predict_brain_response(img_b)

    logger.info("Brain scoring done (%.1fs).", time.monotonic() - t0)

    # ── Step 4: Judge ────────────────────────────────────────────────────────
    logger.info("Step 4/5 – Running judge agent …")
    t0 = time.monotonic()

    judgment = judge_preference(roi_a, roi_b, design_goal)

    logger.info("Judgment done (%.1fs).", time.monotonic() - t0)

    # ── Step 5: Log ──────────────────────────────────────────────────────────
    logger.info("Step 5/5 – Persisting comparison record …")

    record: dict = {
        "run_id": run_id,
        "timestamp": timestamp,
        "spec": spec,
        "design_goal": design_goal,
        "ui_a": ui_a,
        "ui_b": ui_b,
        "ui_a_path": html_a_path,
        "ui_b_path": html_b_path,
        "screenshot_a": img_a,
        "screenshot_b": img_b,
        "brain_a": roi_a,
        "brain_b": roi_b,
        "winner": judgment["winner"],
        "explanation": judgment["explanation"],
        "preference_score": judgment["preference_score"],
    }

    records = _load_comparisons()
    records.append(record)
    _save_comparisons(records)

    logger.info(
        "Done. Winner=%s  Score=%.3f  Saved → %s",
        judgment["winner"],
        judgment["preference_score"],
        COMPARISONS_FILE,
    )
    return record


def _print_summary(record: dict) -> None:
    print("\n" + "═" * 60)
    print(f"  Run ID   : {record['run_id']}")
    print(f"  Winner   : {record['winner']}")
    print(f"  Score    : {record['preference_score']:.3f}  (-1=A, +1=B)")
    print(f"\n  Brain A  : {record['brain_a']}")
    print(f"  Brain B  : {record['brain_b']}")
    print(f"\n  Reasoning: {record['explanation']}")
    print(f"\n  Log      : {COMPARISONS_FILE}")
    print("═" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single UI brain-preference comparison."
    )
    parser.add_argument(
        "spec",
        nargs="?",
        default="Create a landing page for a productivity app",
        help="Natural-language description of the desired UI.",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Design goal for the judge (defaults to spec).",
    )
    args = parser.parse_args()

    try:
        record = run_comparison(args.spec, args.goal)
        _print_summary(record)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during comparison run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
