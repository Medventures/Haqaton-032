"""SQLite persistence for a single API worker; no transcript/audio content is logged."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(directory / "sessions.sqlite3", check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                snapshot TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS audio (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                stream_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                sample_offset INTEGER NOT NULL,
                pcm BLOB NOT NULL,
                PRIMARY KEY(session_id, stream_id, seq)
            );
            """
        )
        self.connection.commit()
        # A new process cannot restore the provider's live recognition state. Expose that fact.
        rows = self.connection.execute("SELECT id, snapshot FROM sessions").fetchall()
        for session_id, raw in rows:
            snapshot = json.loads(raw)
            if snapshot.get("clinicalStatus") == "processing":
                snapshot["clinicalStatus"] = "error"
                snapshot["clinicalError"] = "Сервер перезапущен во время клинического анализа. Повторите анализ."
                snapshot["clinicalRevision"] = snapshot.get("clinicalRevision", 0) + 1
                self.save(snapshot)
            if snapshot["status"] in {"recording", "processing"}:
                snapshot["status"] = "error"
                snapshot["error"] = "Сервер перезапущен во время записи. Сохранённый черновик доступен; начните новую сессию для продолжения записи."
                self.save(snapshot)

    def save(self, snapshot: dict[str, Any]) -> None:
        raw = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        self.connection.execute(
            "INSERT INTO sessions(id, snapshot) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot, updated_at=CURRENT_TIMESTAMP",
            (snapshot["id"], raw),
        )
        self.connection.commit()

    def get(self, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT snapshot FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def append_audio(self, session_id: str, stream_id: str, seq: int, sample_offset: int, pcm: bytes) -> None:
        self.connection.execute(
            "INSERT INTO audio(session_id, stream_id, seq, sample_offset, pcm) VALUES (?, ?, ?, ?, ?)",
            (session_id, stream_id, seq, sample_offset, pcm),
        )
        self.connection.commit()

    def matches_audio(self, session_id: str, stream_id: str, seq: int, sample_offset: int, pcm: bytes) -> bool:
        row = self.connection.execute(
            "SELECT sample_offset, pcm FROM audio WHERE session_id=? AND stream_id=? AND seq=?",
            (session_id, stream_id, seq),
        ).fetchone()
        return row is not None and row[0] == sample_offset and row[1] == pcm

    def close(self) -> None:
        self.connection.close()
