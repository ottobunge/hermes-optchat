"""OptChat as the ROOT consumer of the host's durable event outbox.

The host (``SessionDB`` with ``read_durable_events``) calls ``project`` inside the
transcript transaction with the freshly inserted rows; the events commit with the rows or
not at all. ``project`` is pure: no files, no network, no tool calls. Later, ``Consumer``
drains the stream in global commit order (``seq``) into ROOT.

On hosts that pass ``raw_content``, the event carries the original multimodal parts
(without model thoughts). The consumer applies the same media/sidecar policy as live
messages; older hosts without raw parts retain their ``host-text`` fidelity label.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from .events import UnsupportedContent, events_from_message
from .store import Store

logger = logging.getLogger(__name__)

STREAM = "optchat.root"
VERSION = 1
_ROLES = ("user", "assistant", "tool")
_THOUGHT = frozenset(("thinking", "redacted_thinking", "reasoning"))


def project(session_id: str, rows, *, parent_session_id=None, subagent: bool = False) -> list[dict]:
    """``[{stream, key, event}]`` for each committed user/assistant/tool row.

    A delegated child contributes nothing (spec §9): its report reaches ROOT as a row of
    the parent session. Parent-linked branches and continuations are user-visible and do
    contribute their fresh events. Thought blocks are excluded even on older hosts.
    """
    if subagent:
        return []
    out = []
    for row in rows:
        role = row.get("role")
        if role not in _ROLES:
            continue
        uid = row.get("message_uid") or f"{session_id}:row:{row.get('_row_id')}"
        raw = row.get("raw_content")
        if "raw_content" in row and not isinstance(raw, list):
            raise ValueError("raw_content must be a list")
        content = ([part for part in raw if not (isinstance(part, dict) and part.get("type") in _THOUGHT)]
                   if isinstance(raw, list) else row.get("content"))
        if isinstance(raw, list):
            try:
                encoded = json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")
                lossless = json.loads(encoded) == content
            except (TypeError, ValueError, UnicodeError) as exc:
                raise ValueError("raw_content is not lossless JSON") from exc
            if not lossless:
                raise ValueError("raw_content is not lossless JSON")
        event = {"v": VERSION, "session_id": session_id, "role": role, "content": content,
                 "message_uid": uid, "timestamp": row.get("timestamp"),
                 "content_fidelity": "raw" if isinstance(raw, list) else "host-text"}
        if "authorized_media_sha256" in row:
            hashes = row["authorized_media_sha256"]
            if (not isinstance(hashes, list) or not hashes
                    or any(not isinstance(h, str) or len(h) != 64
                           or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
                raise ValueError("invalid authorized media snapshot manifest")
            event["authorized_media_sha256"] = hashes
        for field, name in (("tool_calls", "tool_calls"), ("_tool_call_uids", "tool_call_uids"),
                            ("tool_call_id", "tool_call_id"), ("_tool_call_uid", "tool_call_uid"),
                            ("tool_name", "tool_name"), ("display_kind", "display_kind")):
            if row.get(field) is not None:
                event[name] = row[field]
        if "tool_name" not in event and isinstance(row.get("name"), str):
            event["tool_name"] = row["name"]
        out.append({"stream": STREAM, "key": uid, "event": event})
    return out


PAGE = 1000  # the host's read_durable_events default limit


class OutboxMismatch(RuntimeError):
    """The chat's cursor belongs to another host database (another profile)."""


def _db_identity(db) -> str:
    path = getattr(db, "db_path", None)
    return str(Path(path).expanduser().resolve()) if path else f"{type(db).__name__}"


def _date(ts):
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    return None


class Consumer:
    """Drains ``STREAM`` into one chat's ROOT. One per ``Memory`` (it uses its writer lock).

    Turn admission (spec §7): before a user event is logged, the view must settle; its
    parts (node ids, never the text) are then frozen in ``outbox/admit.jsonl``, keyed by
    the event, and only then the event is logged. Whoever drains (the commit nudge, a
    request, another session of this process) admits it the same way, so ROOT always
    follows the outbox order. ``select_context`` renders the frozen parts of its user
    event, even after a restart or on a provider retry.

    ``outbox/cursor.jsonl`` gets ``{seq, db}`` only after every ROOT line of that outbox
    event is durably logged; a crash in between replays the event, and ROOT's ``src``
    dedupe keeps it logged once. Both files are append-only and fsynced.
    """

    def __init__(self, mem):
        self.mem = mem
        self.dir = mem.path / "outbox"
        self.cursor = 0
        self.db_id = None
        self._lock = threading.RLock()
        self.admitted: dict[str, list] = {}
        for rec in self._load("admit.jsonl"):
            if isinstance(rec.get("key"), str) and isinstance(rec.get("parts"), list):
                self.admitted.setdefault(rec["key"], rec["parts"])  # the first freeze wins
        for rec in self._load("cursor.jsonl"):
            if type(rec.get("seq")) is int and rec["seq"] >= self.cursor:
                self.cursor, self.db_id = rec["seq"], rec.get("db")

    # -- files -----------------------------------------------------------------------
    def _load(self, name: str) -> list[dict]:
        path = self.dir / name
        if not path.exists():
            return []
        raw = path.read_bytes()
        out = []
        for line in raw.split(b"\n"):
            try:
                rec = json.loads(line.decode("utf-8")) if line.strip() else None
            except (UnicodeDecodeError, ValueError):
                rec = None
            if isinstance(rec, dict):
                out.append(rec)
            elif line.strip():
                logger.warning("optchat: skipping torn/invalid line in %s", path)
        if raw and not raw.endswith(b"\n"):
            Store._append_bytes(path, b"\n")
        return out

    def _append(self, name: str, rec: dict) -> None:
        if self.mem.store._lock_fd is None:
            raise RuntimeError("optchat store is not open for writing")
        if not self.dir.exists():
            self.dir.mkdir(mode=0o700)
            _fsync_dir(self.dir.parent)
        path = self.dir / name
        new = not path.exists()
        Store._append_bytes(path, (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        if new:
            _fsync_dir(self.dir)

    # -- draining --------------------------------------------------------------------
    def drain(self, db, settle_timeout: float) -> bool:
        """Apply every committed event after the cursor, oldest first. False if it stopped
        at a user event because the view did not settle (that event stays in the outbox)."""
        with self._lock:
            ident = _db_identity(db)
            if self.db_id is not None and self.db_id != ident:
                # seq numbers of another database mean nothing here: never mix two outboxes
                raise OutboxMismatch(f"optchat chat {self.mem.path} follows the outbox of {self.db_id}, "
                                     f"not {ident}")
            while True:
                page = db.read_durable_events(STREAM, after_seq=self.cursor, limit=PAGE)
                for row in page:
                    if not self._admit(row, settle_timeout):
                        return False
                    self._apply(row["key"], row["event"])
                    self._append("cursor.jsonl", {"seq": row["seq"], "db": ident})
                    self.cursor, self.db_id = row["seq"], ident
                if len(page) < PAGE:
                    return True

    def _admit(self, row, settle_timeout: float) -> bool:
        key = row["key"]
        if row["event"].get("role") != "user" or key in self.admitted:
            return True
        frozen = self.mem.freeze() if self.mem.settle(settle_timeout) else None
        if frozen is None:
            return False
        parts, n = frozen
        self._append("admit.jsonl", {"key": key, "seq": row["seq"], "parts": parts, "messages": n})
        self.admitted[key] = parts
        return True

    def _apply(self, key: str, event: dict) -> None:
        msg = {"role": event.get("role"), "content": event.get("content"),
               "tool_calls": event.get("tool_calls"), "message_uid": key}
        date = _date(event.get("timestamp"))
        host_notice = event.get("display_kind") in {"failed_turn", "model_switch"}  # host-authored markers
        parsed = events_from_message(msg, fallback=key)
        if "authorized_media_sha256" in event:
            expected = Counter(event["authorized_media_sha256"])
            actual = Counter(hashlib.sha256(b.data).hexdigest() for ev in parsed
                             for b in (ev.archive.blobs if ev.archive else ())
                             if b.mime.startswith("image/"))
            if any(actual[digest] < count for digest, count in expected.items()):
                raise UnsupportedContent("authorized media snapshot missing or corrupt in the durable outbox")
        for ev in parsed:
            kind = "note" if host_notice and ev.kind in {"talk", "user"} else ev.kind
            self.mem.log(kind, ev.text, src=ev.src, archive=ev.archive, date=date)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
