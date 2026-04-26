"""
Lightweight SQLite-backed storage for NeuroUI Judge.

We keep this intentionally simple: one table per top-level entity, JSON
blobs for the variable-shape parts, and no ORM. The schema matches the
pydantic models in `schemas.py`.

This module is import-safe: opening a Storage instance creates the DB and
tables on first use.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id   TEXT PRIMARY KEY,
        label          TEXT,
        source         TEXT,
        html           TEXT,
        screenshot     TEXT,
        prompt         TEXT,
        task           TEXT,
        parent_id      TEXT,
        created_at     TEXT,
        metadata_json  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        candidate_id   TEXT PRIMARY KEY,
        viewport_w     INTEGER,
        viewport_h     INTEGER,
        screenshot     TEXT,
        frames_dir     TEXT,
        dom_tree_path  TEXT,
        artifact_json  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reports (
        candidate_id   TEXT PRIMARY KEY,
        overall        REAL,
        grade          TEXT,
        weights_ver    TEXT,
        report_json    TEXT,
        created_at     TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS preferences (
        pref_id        TEXT PRIMARY KEY,
        ui_a_id        TEXT,
        ui_b_id        TEXT,
        winner         TEXT,
        task           TEXT,
        notes          TEXT,
        rater_id       TEXT,
        created_at     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weight_versions (
        version_id     TEXT PRIMARY KEY,
        created_at     TEXT,
        weights_json   TEXT,
        metrics_json   TEXT,
        notes          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        run_id         TEXT PRIMARY KEY,
        brief          TEXT,
        task           TEXT,
        started_at     TEXT,
        finished_at    TEXT,
        run_json       TEXT
    )
    """,
]


class Storage:
    """Thin SQLite wrapper. Thread-safe via a single lock."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def _init(self) -> None:
        with self._connect() as con:
            for ddl in _SCHEMA:
                con.execute(ddl)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ── Candidates ─────────────────────────────────────────────────────────

    def upsert_candidate(self, c: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO candidates
                  (candidate_id, label, source, html, screenshot, prompt, task,
                   parent_id, created_at, metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                  label=excluded.label, source=excluded.source, html=excluded.html,
                  screenshot=excluded.screenshot, prompt=excluded.prompt,
                  task=excluded.task, parent_id=excluded.parent_id,
                  metadata_json=excluded.metadata_json
                """,
                (
                    c["candidate_id"],
                    c.get("label"),
                    c.get("source", "html"),
                    c.get("html"),
                    c.get("screenshot_path"),
                    c.get("prompt"),
                    c.get("task"),
                    c.get("parent_id"),
                    str(c.get("created_at") or ""),
                    json.dumps(c.get("metadata") or {}),
                ),
            )

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        return d

    def list_candidates(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT candidate_id, label, source, task, parent_id, created_at "
                "FROM candidates ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Artifacts ──────────────────────────────────────────────────────────

    def upsert_artifact(self, art: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO artifacts
                  (candidate_id, viewport_w, viewport_h, screenshot,
                   frames_dir, dom_tree_path, artifact_json)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                  viewport_w=excluded.viewport_w, viewport_h=excluded.viewport_h,
                  screenshot=excluded.screenshot, frames_dir=excluded.frames_dir,
                  dom_tree_path=excluded.dom_tree_path,
                  artifact_json=excluded.artifact_json
                """,
                (
                    art["candidate_id"],
                    int(art.get("viewport_width") or 0),
                    int(art.get("viewport_height") or 0),
                    art.get("screenshot_path"),
                    art.get("frames_dir"),
                    art.get("dom_tree_path"),
                    json.dumps(art),
                ),
            )

    def get_artifact(self, candidate_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT artifact_json FROM artifacts WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return json.loads(row["artifact_json"]) if row else None

    # ── Reports ────────────────────────────────────────────────────────────

    def upsert_report(self, report: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO reports (candidate_id, overall, grade, weights_ver, report_json)
                VALUES (?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                  overall=excluded.overall, grade=excluded.grade,
                  weights_ver=excluded.weights_ver, report_json=excluded.report_json
                """,
                (
                    report["candidate_id"],
                    float(report["overall_reward"]),
                    report["grade"],
                    report.get("weights_version", "default-v1"),
                    json.dumps(report),
                ),
            )

    def get_report(self, candidate_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT report_json FROM reports WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return json.loads(row["report_json"]) if row else None

    def list_reports(self, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            if ids:
                ids = list(ids)
                placeholders = ",".join(["?"] * len(ids))
                rows = con.execute(
                    f"SELECT report_json FROM reports WHERE candidate_id IN ({placeholders})",
                    ids,
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT report_json FROM reports ORDER BY rowid DESC LIMIT 500"
                ).fetchall()
        return [json.loads(r["report_json"]) for r in rows]

    # ── Preferences ────────────────────────────────────────────────────────

    def add_preference(self, p: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO preferences
                  (pref_id, ui_a_id, ui_b_id, winner, task, notes, rater_id, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    p["pref_id"],
                    p["ui_a_id"],
                    p["ui_b_id"],
                    p["winner"],
                    p.get("task"),
                    p.get("notes"),
                    p.get("rater_id"),
                    str(p.get("created_at") or ""),
                ),
            )

    def list_preferences(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT * FROM preferences ORDER BY rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Weight versions ────────────────────────────────────────────────────

    def add_weight_version(self, w: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO weight_versions
                  (version_id, created_at, weights_json, metrics_json, notes)
                VALUES (?,?,?,?,?)
                """,
                (
                    w["version_id"],
                    str(w.get("created_at") or ""),
                    json.dumps(w["weights"]),
                    json.dumps(w.get("metrics") or {}),
                    w.get("notes", ""),
                ),
            )

    def get_active_weights(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM weight_versions ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "version_id": row["version_id"],
            "weights": json.loads(row["weights_json"]),
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "notes": row["notes"] or "",
            "created_at": row["created_at"],
        }

    def list_weight_versions(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT version_id, created_at, weights_json, metrics_json, notes "
                "FROM weight_versions ORDER BY rowid DESC"
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "version_id": r["version_id"],
                    "created_at": r["created_at"],
                    "weights": json.loads(r["weights_json"]),
                    "metrics": json.loads(r["metrics_json"] or "{}"),
                    "notes": r["notes"] or "",
                }
            )
        return out

    # ── Agent runs ─────────────────────────────────────────────────────────

    def save_agent_run(self, run: dict[str, Any]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO agent_runs
                  (run_id, brief, task, started_at, finished_at, run_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    run["run_id"],
                    run.get("brief", ""),
                    run.get("task", ""),
                    str(run.get("started_at") or ""),
                    str(run.get("finished_at") or ""),
                    json.dumps(run),
                ),
            )

    def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT run_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["run_json"]) if row else None

    def list_agent_runs(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT run_id, brief, task, started_at, finished_at "
                "FROM agent_runs ORDER BY rowid DESC LIMIT 200"
            ).fetchall()
        return [dict(r) for r in rows]
