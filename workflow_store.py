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
                channel_id   TEXT DEFAULT '',
                thread_id    TEXT DEFAULT '',
                user_id      TEXT DEFAULT '',
                delivery_channel TEXT DEFAULT '',
                persona      TEXT DEFAULT '',
                summary_format TEXT DEFAULT '',
                source       TEXT DEFAULT '',
                replan_count INTEGER DEFAULT 0,
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
            CREATE TABLE IF NOT EXISTS pending_clarifications (
                id                   TEXT PRIMARY KEY,
                thread_id            TEXT NOT NULL,
                channel_id           TEXT NOT NULL,
                requester_id         TEXT NOT NULL,
                user_id              TEXT NOT NULL DEFAULT '',
                goal                 TEXT NOT NULL,
                questions            TEXT NOT NULL,
                clarification_message TEXT NOT NULL DEFAULT '',
                created_at           TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pc_thread
                ON pending_clarifications (thread_id, channel_id);
            CREATE TABLE IF NOT EXISTS pending_replan_approvals (
                id                    TEXT PRIMARY KEY,
                task_id               TEXT NOT NULL,
                thread_id             TEXT NOT NULL,
                channel_id            TEXT NOT NULL,
                requester_id          TEXT NOT NULL,
                user_id               TEXT NOT NULL DEFAULT '',
                replan_count          INTEGER NOT NULL DEFAULT 0,
                prompt_message        TEXT NOT NULL DEFAULT '',
                replanned_goal        TEXT NOT NULL DEFAULT '',
                replanned_title       TEXT NOT NULL DEFAULT '',
                replanned_description TEXT NOT NULL DEFAULT '',
                replanned_steps_json  TEXT NOT NULL DEFAULT '[]',
                created_at            TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pr_thread
                ON pending_replan_approvals (thread_id, channel_id);
            CREATE TABLE IF NOT EXISTS pending_followups (
                id             TEXT PRIMARY KEY,
                task_id        TEXT NOT NULL,
                step_index     INTEGER NOT NULL,
                step_id        TEXT NOT NULL,
                capability     TEXT NOT NULL,
                question_id    TEXT NOT NULL,
                question       TEXT NOT NULL,
                field_name     TEXT NOT NULL DEFAULT '',
                answer_format  TEXT NOT NULL DEFAULT 'text',
                choices_json   TEXT NOT NULL DEFAULT '[]',
                thread_id      TEXT NOT NULL,
                channel_id     TEXT NOT NULL,
                requester_id   TEXT NOT NULL,
                user_id        TEXT NOT NULL DEFAULT '',
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pf_thread
                ON pending_followups (thread_id, channel_id);
            CREATE TABLE IF NOT EXISTS pending_memory_consents (
                id            TEXT PRIMARY KEY,
                thread_id     TEXT NOT NULL,
                channel_id    TEXT NOT NULL,
                requester_id  TEXT NOT NULL,
                user_id       TEXT NOT NULL DEFAULT '',
                entries_json  TEXT NOT NULL,
                prompt_message TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pmc_thread
                ON pending_memory_consents (thread_id, channel_id);
        """)
        self._conn.commit()
        # Migrate existing tables created before clarification_message was added
        try:
            self._conn.execute(
                "ALTER TABLE pending_clarifications "
                "ADD COLUMN clarification_message TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        for ddl in (
            "ALTER TABLE workflows ADD COLUMN channel_id TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN thread_id TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN user_id TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN delivery_channel TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN persona TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN summary_format TEXT DEFAULT ''",
            "ALTER TABLE workflows ADD COLUMN replan_count INTEGER DEFAULT 0",
            "ALTER TABLE workflows ADD COLUMN source TEXT DEFAULT ''",
        ):
            try:
                self._conn.execute(ddl)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

    # ── Write ────────────────────────────────────────────────────────────────

    def create_workflow(
        self,
        task_id: str,
        goal: str,
        title: str,
        description: str,
        requester_id: str,
        steps: list[dict],
        channel_id: str = "",
        thread_id: str = "",
        user_id: str = "",
        delivery_channel: str = "",
        persona: str = "",
        summary_format: str = "",
        source: str = "",
    ) -> None:
        now = _now_iso()
        self._conn.execute(
            """INSERT INTO workflows
               (task_id, goal, title, description, requester_id,
                channel_id, thread_id, user_id, delivery_channel, persona, summary_format,
                source, status, total_steps, current_step,
                steps_json, outputs_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'planning',?,0, ?,?, ?,?)""",
            (
                task_id, goal, title, description, requester_id,
                channel_id, thread_id, user_id, delivery_channel, persona, summary_format,
                source,
                len(steps),
                json.dumps(steps),
                json.dumps([None] * len(steps)),
                now, now,
            ),
        )
        self._conn.commit()

    def replace_workflow_plan(
        self,
        task_id: str,
        goal: str,
        title: str,
        description: str,
        steps: list[dict],
        replan_count: int,
    ) -> None:
        """Replace plan content and reset execution cursor for a re-plan."""
        self._conn.execute(
            """UPDATE workflows
               SET goal=?, title=?, description=?, status='running',
                   total_steps=?, current_step=0,
                   steps_json=?, outputs_json=?, error=NULL,
                   replan_count=?, updated_at=?
               WHERE task_id=?""",
            (
                goal,
                title,
                description,
                len(steps),
                json.dumps(steps),
                json.dumps([None] * len(steps)),
                replan_count,
                _now_iso(),
                task_id,
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

    def reset_replan_count(self, task_id: str) -> None:
        """Reset the replan counter to zero (called when user approves retrying after limit)."""
        self._conn.execute(
            "UPDATE workflows SET replan_count=0, updated_at=? WHERE task_id=?",
            (_now_iso(), task_id),
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

    def update_step_input(
        self, task_id: str, step_index: int, input_data: dict
    ) -> None:
        """Patch input_data for a single step without advancing the workflow cursor."""
        row = self._conn.execute(
            "SELECT steps_json FROM workflows WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return
        steps = json.loads(row["steps_json"] or "[]")
        if not isinstance(steps, list) or step_index < 0 or step_index >= len(steps):
            return
        step = steps[step_index]
        if not isinstance(step, dict):
            return
        step["input_data"] = input_data
        self._conn.execute(
            "UPDATE workflows SET steps_json=?, updated_at=? WHERE task_id=?",
            (json.dumps(steps), _now_iso(), task_id),
        )
        self._conn.commit()

    def patch_step(self, task_id: str, step_index: int, fields: dict) -> None:
        """Merge *fields* into a single step without advancing the workflow cursor."""
        row = self._conn.execute(
            "SELECT steps_json FROM workflows WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return
        steps = json.loads(row["steps_json"] or "[]")
        if not isinstance(steps, list) or step_index < 0 or step_index >= len(steps):
            return
        if not isinstance(steps[step_index], dict):
            return
        steps[step_index].update(fields)
        self._conn.execute(
            "UPDATE workflows SET steps_json=?, updated_at=? WHERE task_id=?",
            (json.dumps(steps), _now_iso(), task_id),
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

    def get_stale_correlations(self, timeout_s: float) -> list[dict]:
        """
        Return correlations older than *timeout_s* whose workflow is still running.
        Used by the step-timeout watchdog to detect orphaned steps.
        """
        rows = self._conn.execute(
            """
            SELECT sc.correlation_id, sc.task_id, sc.step_index, sc.created_at
            FROM   step_correlations sc
            JOIN   workflows w ON w.task_id = sc.task_id
            WHERE  w.status = 'running'
              AND  CAST(strftime('%s','now') AS INTEGER)
                   - CAST(strftime('%s', sc.created_at) AS INTEGER) > ?
            """,
            (int(timeout_s),),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_correlation(self, correlation_id: str) -> None:
        """Hard-delete a single correlation row (used by the timeout watchdog)."""
        self._conn.execute(
            "DELETE FROM step_correlations WHERE correlation_id=?",
            (correlation_id,),
        )
        self._conn.commit()

    # ── Pending clarifications ────────────────────────────────────────────────

    def save_pending_clarification(
        self,
        id: str,
        thread_id: str,
        channel_id: str,
        requester_id: str,
        user_id: str,
        goal: str,
        questions_json: str,
        clarification_message: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pending_clarifications
               (id, thread_id, channel_id, requester_id, user_id, goal, questions,
                clarification_message, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (id, thread_id, channel_id, requester_id, user_id, goal, questions_json,
             clarification_message, _now_iso()),
        )
        self._conn.commit()

    def get_pending_clarification(self, thread_id: str, channel_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """SELECT * FROM pending_clarifications
               WHERE thread_id=? AND channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id, channel_id),
        ).fetchone()
        return dict(row) if row else None

    def get_latest_pending_clarification(self, channel_id: str) -> Optional[dict]:
        """Channel-only lookup — used by connectors that don't thread replies (e.g. dashboard chat)."""
        row = self._conn.execute(
            """SELECT * FROM pending_clarifications
               WHERE channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (channel_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_pending_clarification(self, id: str) -> None:
        self._conn.execute(
            "DELETE FROM pending_clarifications WHERE id=?", (id,)
        )
        self._conn.commit()

    def cleanup_stale_clarifications(self, older_than_hours: int = 24) -> int:
        """Delete clarifications older than *older_than_hours*; return count deleted."""
        threshold_s = older_than_hours * 3600
        cur = self._conn.execute(
            """DELETE FROM pending_clarifications
               WHERE CAST(strftime('%s','now') AS INTEGER)
                   - CAST(strftime('%s', created_at) AS INTEGER) > ?""",
            (threshold_s,),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Pending replan approvals ─────────────────────────────────────────────

    def save_pending_replan_approval(
        self,
        id: str,
        task_id: str,
        thread_id: str,
        channel_id: str,
        requester_id: str,
        user_id: str,
        replan_count: int,
        prompt_message: str,
        replanned_goal: str,
        replanned_title: str,
        replanned_description: str,
        replanned_steps_json: str,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pending_replan_approvals
               (id, task_id, thread_id, channel_id, requester_id, user_id, replan_count,
                prompt_message, replanned_goal, replanned_title, replanned_description,
                replanned_steps_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                id,
                task_id,
                thread_id,
                channel_id,
                requester_id,
                user_id,
                replan_count,
                prompt_message,
                replanned_goal,
                replanned_title,
                replanned_description,
                replanned_steps_json,
                _now_iso(),
            ),
        )
        self._conn.commit()

    def get_pending_replan_approval(self, thread_id: str, channel_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """SELECT * FROM pending_replan_approvals
               WHERE thread_id=? AND channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id, channel_id),
        ).fetchone()
        return dict(row) if row else None

    def delete_pending_replan_approval(self, id: str) -> None:
        self._conn.execute(
            "DELETE FROM pending_replan_approvals WHERE id=?", (id,)
        )
        self._conn.commit()

    def cleanup_stale_replan_approvals(self, older_than_hours: int = 24) -> int:
        threshold_s = older_than_hours * 3600
        cur = self._conn.execute(
            """DELETE FROM pending_replan_approvals
               WHERE CAST(strftime('%s','now') AS INTEGER)
                   - CAST(strftime('%s', created_at) AS INTEGER) > ?""",
            (threshold_s,),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Pending followups ────────────────────────────────────────────────────

    def save_pending_followup(
        self,
        id: str,
        task_id: str,
        step_index: int,
        step_id: str,
        capability: str,
        question_id: str,
        question: str,
        field_name: str,
        answer_format: str,
        choices_json: str,
        thread_id: str,
        channel_id: str,
        requester_id: str,
        user_id: str,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pending_followups
               (id, task_id, step_index, step_id, capability,
                question_id, question, field_name, answer_format, choices_json,
                thread_id, channel_id, requester_id, user_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                id, task_id, step_index, step_id, capability,
                question_id, question, field_name, answer_format, choices_json,
                thread_id, channel_id, requester_id, user_id, _now_iso(),
            ),
        )
        self._conn.commit()

    def get_pending_followup(self, thread_id: str, channel_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """SELECT * FROM pending_followups
               WHERE thread_id=? AND channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id, channel_id),
        ).fetchone()
        return dict(row) if row else None

    def get_pending_followup_by_user(self, user_id: str, channel_id: str) -> Optional[dict]:
        """
        Fallback lookup for when the user replied as a new message (no thread context).
        Returns the most recent pending followup for this user+channel regardless of thread.
        """
        row = self._conn.execute(
            """SELECT * FROM pending_followups
               WHERE user_id=? AND channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (user_id, channel_id),
        ).fetchone()
        return dict(row) if row else None

    def delete_pending_followup(self, id: str) -> None:
        self._conn.execute("DELETE FROM pending_followups WHERE id=?", (id,))
        self._conn.commit()

    def cleanup_stale_followups(self, older_than_hours: int = 24) -> int:
        threshold_s = older_than_hours * 3600
        cur = self._conn.execute(
            """DELETE FROM pending_followups
               WHERE CAST(strftime('%s','now') AS INTEGER)
                   - CAST(strftime('%s', created_at) AS INTEGER) > ?""",
            (threshold_s,),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Pending memory consent ───────────────────────────────────────────────

    def save_pending_memory_consent(
        self,
        id: str,
        thread_id: str,
        channel_id: str,
        requester_id: str,
        user_id: str,
        entries_json: str,
        prompt_message: str,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pending_memory_consents
               (id, thread_id, channel_id, requester_id, user_id, entries_json,
                prompt_message, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (id, thread_id, channel_id, requester_id, user_id, entries_json, prompt_message, _now_iso()),
        )
        self._conn.commit()

    def get_pending_memory_consent(self, thread_id: str, channel_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """SELECT * FROM pending_memory_consents
               WHERE thread_id=? AND channel_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id, channel_id),
        ).fetchone()
        return dict(row) if row else None

    def delete_pending_memory_consent(self, id: str) -> None:
        self._conn.execute("DELETE FROM pending_memory_consents WHERE id=?", (id,))
        self._conn.commit()

    def cleanup_stale_memory_consents(self, older_than_hours: int = 24) -> int:
        threshold_s = older_than_hours * 3600
        cur = self._conn.execute(
            """DELETE FROM pending_memory_consents
               WHERE CAST(strftime('%s','now') AS INTEGER)
                   - CAST(strftime('%s', created_at) AS INTEGER) > ?""",
            (threshold_s,),
        )
        self._conn.commit()
        return cur.rowcount

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
