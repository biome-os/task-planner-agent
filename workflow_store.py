"""
workflow_store.py — SQLite persistence for the task-planner-agent.

Stores workflow plans and per-step execution state. Maps step
correlation_ids so the planner knows which workflow/step to advance
when an agent sends back a task_response callback.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_FILE = Path("planner_workflows.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class WorkflowStore:
    """SQLite-backed store for workflow plans and execution state."""

    def __init__(self, db_path: Path = _DB_FILE) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        logger.info("WorkflowStore opened: %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _migrate(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                task_id      TEXT PRIMARY KEY,
                goal         TEXT NOT NULL,
                title        TEXT,
                description  TEXT,
                requester_id TEXT,
                status       TEXT DEFAULT 'planning',
                total_steps  INTEGER DEFAULT 0,
                current_step INTEGER DEFAULT 0,
                steps_json   TEXT DEFAULT '[]',
                outputs_json TEXT DEFAULT '[]',
                error        TEXT,
                created_at   TEXT,
                updated_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS step_correlations (
                correlation_id TEXT PRIMARY KEY,
                task_id        TEXT NOT NULL,
                step_index     INTEGER NOT NULL,
                created_at     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sc_task
                ON step_correlations (task_id);
        """)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def create_workflow(
        self,
        task_id: str,
        goal: str,
        title: str,
        description: str,
        requester_id: str,
        steps: list[dict],
    ) -> None:
        now = _now_iso()
        self._conn.execute(
            """INSERT INTO workflows
               (task_id, goal, title, description, requester_id,
                status, total_steps, current_step,
                steps_json, outputs_json, created_at, updated_at)
               VALUES (?,?,?,?,?, 'planning',?,0, ?,?, ?,?)""",
            (
                task_id, goal, title, description, requester_id,
                len(steps),
                json.dumps(steps),
                json.dumps([None] * len(steps)),
                now, now,
            ),
        )
        self._conn.commit()

    def set_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "UPDATE workflows SET status=?, error=?, updated_at=? WHERE task_id=?",
            (status, error, _now_iso(), task_id),
        )
        self._conn.commit()

    def advance_step(
        self, task_id: str, step_index: int, output: Optional[dict]
    ) -> None:
        """Record *output* for *step_index* and increment current_step."""
        row = self._conn.execute(
            "SELECT outputs_json FROM workflows WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return
        outputs = json.loads(row["outputs_json"])
        if step_index < len(outputs):
            outputs[step_index] = output
        self._conn.execute(
            """UPDATE workflows
               SET outputs_json=?, current_step=?, updated_at=?
               WHERE task_id=?""",
            (json.dumps(outputs), step_index + 1, _now_iso(), task_id),
        )
        self._conn.commit()

    def save_correlation(
        self, correlation_id: str, task_id: str, step_index: int
    ) -> None:
        self._conn.execute(
            """INSERT INTO step_correlations
               (correlation_id, task_id, step_index, created_at)
               VALUES (?,?,?,?)""",
            (correlation_id, task_id, step_index, _now_iso()),
        )
        self._conn.commit()

    def pop_correlation(
        self, correlation_id: str
    ) -> Optional[tuple[str, int]]:
        """Return (task_id, step_index) and delete the row. None if not found."""
        row = self._conn.execute(
            "SELECT task_id, step_index FROM step_correlations WHERE correlation_id=?",
            (correlation_id,),
        ).fetchone()
        if not row:
            return None
        self._conn.execute(
            "DELETE FROM step_correlations WHERE correlation_id=?",
            (correlation_id,),
        )
        self._conn.commit()
        return row["task_id"], row["step_index"]

    def clear_stale_correlations(self, task_id: str, step_index: int) -> None:
        """Remove any pending correlations for a step (used before re-dispatch)."""
        self._conn.execute(
            "DELETE FROM step_correlations WHERE task_id=? AND step_index=?",
            (task_id, step_index),
        )
        self._conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_workflow(self, task_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE task_id=?", (task_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_running(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM workflows WHERE status='running' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_workflows(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM workflows ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["steps"]   = json.loads(d.pop("steps_json",   "[]"))
        d["outputs"] = json.loads(d.pop("outputs_json", "[]"))
        return d
