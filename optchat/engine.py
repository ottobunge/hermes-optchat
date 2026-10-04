"""OptChat as a Hermes context engine (``context.engine: optchat``).

How the spec's turn loop (§7) maps onto the Hermes hooks:

* ``select_context`` runs before EVERY provider request of a turn. On the first request of
  a new turn it (1) logs whatever the previous turn left unlogged, (2) waits until the view
  is all summaries (settle), (3) renders the view, then (4) logs the new user message. The
  rendered view is frozen for the rest of the turn. Every request of the turn is
  ``[system + OptChat doc] [user: view, message] [this turn's assistant/tool tail]``;
  messages before the turn are never sent.
* ``on_turn_complete`` logs the finished turn's tail (best effort; the next turn reconciles).
* Every logged line carries a dedupe key (``message_uid`` or a fallback), so replays of
  the same host message are logged once.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from agent.context_engine import ContextEngine

try:  # hosts that can refuse a provider request (never fall back to full history)
    from agent.context_engine import ContextSelectionBlocked
except ImportError:
    ContextSelectionBlocked = None

from . import outbox as _outbox
from . import summarizer as _summarizer
from .events import _TEXT, _THOUGHT, UnsupportedContent, cap_echo, events_from_message, sources_from_message
from .memory import RETRY, VIEW, Memory
from .store import StoreLocked
from .prompts import system_doc
from .tools import DATE_DESCRIPTION, ZOOM_DESCRIPTION

logger = logging.getLogger(__name__)

_DEFAULT = object()
RECONCILE_WINDOW = 5000
_INT = {"type": "integer", "minimum": 0}
TOOL_SCHEMAS = [
    {"name": "zoom", "description": ZOOM_DESCRIPTION,
     "parameters": {"type": "object", "properties": {"id": dict(_INT, description="first message id of the line"),
                                                     "n": dict(_INT, description="messages the line covers")},
                    "required": ["id", "n"]}},
    {"name": "date", "description": DATE_DESCRIPTION,
     "parameters": {"type": "object", "properties": {"id": dict(_INT, description="message id")},
                    "required": ["id"]}},
]


def _as_int(x):
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return x  # how far back to look for the last logged message
_REGISTRY: dict[Path, Memory] = {}
_REGISTRY_LOCK = threading.Lock()


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def shared_memory(chat_dir: Path, **kw) -> Memory:
    """One Memory (one writer lock, one compactor) per chat directory per process."""
    key = chat_dir.resolve()
    with _REGISTRY_LOCK:
        mem = _REGISTRY.get(key)
        if mem is None or mem.closed:
            mem = Memory(key, **kw)
            mem.open()  # may raise StoreLocked / StoreCorrupt
            _REGISTRY[key] = mem
        return mem


def shutdown_all() -> None:
    with _REGISTRY_LOCK:
        for mem in _REGISTRY.values():
            mem.close()
        _REGISTRY.clear()


atexit.register(shutdown_all)


def _digest(*parts) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


class OptChatEngine(ContextEngine):
    requires_generic_turn_loop = True  # app-server bypasses select_context entirely
    optchat_view_cache = True  # preserve the selected view and its cache breakpoints

    def __init__(self, *, home=None, summarizer=_DEFAULT, agent: str | None = None,
                 settle_timeout: float | None = None, budget: int | None = None,
                 retry: float = RETRY):
        # Construction must be side-effect free: Hermes instantiates engines just to probe them.
        self._home = Path(home) if home else None
        self._summarizer = summarizer
        self.agent = agent or os.environ.get("OPTCHAT_AGENT_NAME") or "Hermes"
        self.settle_timeout = float(settle_timeout if settle_timeout is not None
                                    else os.environ.get("OPTCHAT_SETTLE_TIMEOUT") or 120)
        self.budget = int(budget or os.environ.get("OPTCHAT_VIEW_BYTES") or VIEW)
        self.retry = retry
        self.session_id = ""
        self.platform = ""
        self.session_source = ""
        self.memory: Memory | None = None
        self._db = _DEFAULT  # never bound: an old host
        self._detached = False
        self._lock = threading.RLock()
        self._turn = None  # (key, frozen view text)
        self.last_settle = None
        self.degraded = None

    @property
    def name(self) -> str:
        return "optchat"

    # -- host bookkeeping ------------------------------------------------------------
    def update_from_response(self, usage):
        usage = usage or {}
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def should_compress(self, prompt_tokens=None) -> bool:
        return False  # the view is bounded; history is never sent, so never compact it

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages

    def on_session_start(self, session_id, **kwargs):
        self.session_id = str(session_id or "")
        self.platform = str(kwargs.get("platform") or self.platform or "")
        self.session_source = str(kwargs.get("session_source") or "")
        if kwargs.get("hermes_home") and self._home is None:
            self._home = Path(kwargs["hermes_home"]) / "optchat"

    # -- durable outbox (hosts with SessionDB.read_durable_events) ----------------------
    def bind_session_state(self, session_db=None, session_id: str = "") -> None:
        self._db = session_db
        # A background-review fork severs its binding (no DB, no session): read the view only.
        self._detached = session_db is None and not session_id
        if session_id:
            self.session_id = str(session_id)

    @property
    def outbox_bound(self) -> bool:
        return self._db is not _DEFAULT and callable(getattr(self._db, "read_durable_events", None))

    def _consumer(self) -> _outbox.Consumer:
        mem = self._memory()
        with _REGISTRY_LOCK:
            if getattr(mem, "consumer", None) is None:
                mem.consumer = _outbox.Consumer(mem)
            return mem.consumer

    def on_durable_events_committed(self, session_id) -> None:
        if self.outbox_bound and self.logs:
            self._consumer().drain(self._db, self.settle_timeout)

    def project_durable_events(self, session_id, rows, *, parent_session_id=None):
        # Runs inside the host's transcript transaction: pure, never touches the chat.
        return _outbox.project(str(session_id or ""), rows, parent_session_id=parent_session_id,
                               subagent=self.is_subagent)

    @property
    def is_subagent(self) -> bool:
        # The live platform and stored session source both identify delegated children.
        # Parent linkage alone also describes user-visible branches/continuations.
        return self.platform == "subagent" or self.session_source == "subagent"

    @property
    def logs(self) -> bool:
        """Whether this engine writes the chat log at all (subagents and detached forks only read)."""
        return not (self.is_subagent or self._detached)

    def clone_for_agent(self):
        """A fresh per-agent engine sharing the process-wide chat backend (no turn state)."""
        clone = type(self)(home=self._home, summarizer=self._summarizer, agent=self.agent,
                           settle_timeout=self.settle_timeout, budget=self.budget, retry=self.retry)
        clone.threshold_percent = self.threshold_percent
        clone.context_length = self.context_length
        clone.threshold_tokens = self.threshold_tokens
        return clone

    def __deepcopy__(self, memo):
        return self.clone_for_agent()

    # -- memory ----------------------------------------------------------------------
    def chat_dir(self) -> Path:
        home = self._home or (Path(os.environ["OPTCHAT_HOME"]) if os.environ.get("OPTCHAT_HOME")
                              else _hermes_home() / "optchat")
        return home / "chat"

    def _memory(self, *, read: bool = False) -> Memory:
        """The process-wide writer; a reader (``read``) falls back to a read-only copy,
        reloaded from disk on every call, when another process owns the chat (spec §9)."""
        if self.memory is not None and self.memory.readonly:
            self.memory = None  # try for the writer again
        if self.memory is None:
            summ = self._summarizer
            if summ is _DEFAULT:
                summ = _summarizer.from_env()
            if os.environ.get("OPTCHAT_SUMMARIZER_CHILD"):
                summ = None  # we run inside a summarizer command: never compact recursively
            try:
                self.memory = shared_memory(self.chat_dir(), summarizer=summ, agent=self.agent,
                                            budget=self.budget, retry=self.retry)
            except StoreLocked:
                if not read:
                    raise
                reader = Memory(self.chat_dir(), agent=self.agent, budget=self.budget)
                reader.open_readonly()
                self.memory = reader
        return self.memory

    def _fallback(self, conv, j: int) -> str:
        msg = conv[j]
        return f"s:{self.session_id}:{j}:{_digest(msg.get('role'), msg.get('content'), msg.get('tool_calls'))}"

    def _ingest(self, mem: Memory, conv, j: int) -> None:
        sources = sources_from_message(conv[j], fallback=self._fallback(conv, j))
        if sources and all(mem.has_src(s) for s in sources):
            return  # logged already: do not decode or snapshot its payloads again
        for ev in events_from_message(conv[j], fallback=self._fallback(conv, j)):
            mem.log(ev.kind, ev.text, src=ev.src, archive=ev.archive)

    def _reconcile(self, mem: Memory, conv, u: int) -> None:
        """Log what earlier turns left unlogged (a missed on_turn_complete, a crash).

        Only messages after the latest one already in the log are taken: with no such
        anchor (a session that predates OptChat) nothing is imported.
        """
        for j in range(u - 1, max(-1, u - 1 - RECONCILE_WINDOW), -1):
            if any(mem.has_src(src) for src in sources_from_message(conv[j], fallback=self._fallback(conv, j))):
                for k in range(j, u):
                    self._ingest(mem, conv, k)
                return

    # -- per request -----------------------------------------------------------------
    def select_context(self, request_messages, *, conversation_messages=None, incoming_message=None,
                       budget_tokens=0):
        conv = list(conversation_messages or [])
        try:
            out = self._select(list(request_messages or []), conv, incoming_message)
            self.degraded = None
            return out
        except Exception as exc:
            # Fail closed: returning None (or raising anything but the host's sentinel) would
            # make the host send the whole session history.
            reason = f"{type(exc).__name__}: {exc}"
            if self.degraded != reason:
                logger.warning("optchat: memory unavailable, sending this turn only: %s", reason)
            self.degraded = reason
            if ContextSelectionBlocked is not None:
                # No provider call at all; the turn's message stays in the host transcript
                # (and its outbox), so it is logged once the memory recovers.
                raise ContextSelectionBlocked(f"OptChat memory is unavailable ({reason}); "
                                              f"no request was sent") from exc
            # An older host cannot abort the request: send this turn with a notice only.
            return self._fail_closed(list(request_messages or []), conv, incoming_message, reason,
                                     discard_input=isinstance(exc, UnsupportedContent))

    def _align(self, request, conv, incoming):
        u = self._find_turn(conv, incoming)
        r = len(request) - (len(conv) - u)
        if not (0 <= r < len(request)) or request[r].get("role") != "user":
            raise LookupError("cannot align the request with the conversation")
        return u, r

    def _select_outbox(self, request, conv, incoming):
        """The outbox is the only source of ROOT lines; this request only reads the view
        frozen when its user event was admitted (see ``outbox.Consumer``)."""
        u, r = self._align(request, conv, incoming)
        key = conv[u].get("message_uid")
        if not key:
            raise LookupError("the current user message is not durable yet (no message_uid)")
        mem = self._memory()
        with self._lock:
            consumer = self._consumer()
            consumer.drain(self._db, self.settle_timeout)
            if self._turn is None or self._turn[0] != key:
                parts = consumer.admitted.get(key)
                if parts is None:
                    raise RuntimeError("the view did not settle before this turn's message was admitted")
                self._turn = (key, mem.render_parts(parts))
            view = self._turn[1]
        system = self._system(request, doc=True)
        return [system, self._user_blocks(view, request[r])] + self._tail_for_send(request[r + 1:])

    def _select(self, request, conv, incoming):
        if self.outbox_bound and self.logs:
            return self._select_outbox(request, conv, incoming)
        u, r = self._align(request, conv, incoming)
        mem = self._memory(read=not self.logs)
        with self._lock:
            key = self._turn_key(conv, u)
            if self._turn is None or self._turn[0] != key:
                if self.logs:
                    self._reconcile(mem, conv, u)
                self.last_settle = mem.settle(self.settle_timeout)
                if not self.last_settle:
                    # Never put an unsummarized placeholder on the wire: block the request
                    # (or, on an older host, send a memory-unavailable notice instead).
                    raise RuntimeError("view did not settle before the request")
                view = mem.render()
                if self.logs:
                    self._ingest(mem, conv, u)
                self._turn = (key, view)
            view = self._turn[1]
            if self.logs:
                for j in range(u + 1, len(conv)):
                    self._ingest(mem, conv, j)
        system = self._system(request, doc=True)
        return [system, self._user_blocks(view, request[r])] + self._tail_for_send(request[r + 1:])

    @staticmethod
    def _tail_for_send(tail):
        out = []
        for message in tail:
            item = dict(message)
            if item.get("role") == "tool":
                content = item.get("content")
                if isinstance(content, str):
                    item["content"] = cap_echo(content)
                elif isinstance(content, list):
                    texts = [p if isinstance(p, str) else p["text"] for p in content
                             if isinstance(p, str) or (isinstance(p, dict) and p.get("type") in _TEXT
                                                        and isinstance(p.get("text"), str))]
                    media = [p for p in content if isinstance(p, dict) and p.get("type") not in _TEXT
                             and p.get("type") not in _THOUGHT]
                    capped = cap_echo("\n".join(texts))
                    # Cap the text the provider resends; forward media parts as the host gave them.
                    item["content"] = [{"type": "text", "text": capped}] + media if media else capped
            out.append(item)
        return out

    def _system(self, request, *, doc: bool) -> dict:
        system = dict(request[0]) if request and request[0].get("role") == "system" \
            else {"role": "system", "content": ""}
        if doc:
            content = system.get("content") or ""
            extra = system_doc(self.agent)
            if isinstance(content, list):
                system["content"] = list(content) + [{"type": "text", "text": extra}]
            else:
                system["content"] = (content + "\n\n" + extra) if content else extra
        return system

    @staticmethod
    def _user_blocks(first: str, message: dict) -> dict:
        incoming = message.get("content")
        blocks = [{"type": "text", "text": first}]
        if isinstance(incoming, list):
            blocks += incoming
        else:
            blocks.append({"type": "text", "text": incoming or ""})
        return {"role": "user", "content": blocks}

    def _fail_closed(self, request, conv, incoming, reason: str, *, discard_input: bool = False):
        notice = (f"[OptChat memory is unavailable ({reason}). Earlier chat is not shown and this "
                  f"turn is not recorded; tell the user if it matters.]")
        if discard_input:
            # Content the archive cannot keep losslessly (e.g. non-JSON values): never
            # forward it for action while recording a lossy stand-in for it.
            return [self._system(request, doc=False), {"role": "user", "content": notice}]
        r = None
        try:
            u = self._find_turn(conv, incoming)
            cand = len(request) - (len(conv) - u)
            if 0 <= cand < len(request) and request[cand].get("role") == "user":
                r = cand
        except Exception:
            pass
        if r is None:  # fall back to the request's last user message
            r = next((k for k in range(len(request) - 1, -1, -1) if request[k].get("role") == "user"), None)
        system = self._system(request, doc=False)
        if r is None:
            return [system, {"role": "user", "content": notice}]
        return [system, self._user_blocks(notice, request[r])] + self._tail_for_send(request[r + 1:])

    # -- tools -----------------------------------------------------------------------
    def get_tool_schemas(self):
        return json.loads(json.dumps(TOOL_SCHEMAS))  # constant; callers may mutate their copy

    def handle_tool_call(self, name, args, **kwargs):
        args = args if isinstance(args, dict) else {}
        try:
            mem = self._memory(read=True)
            if name == "zoom":
                return json.dumps({"result": mem.zoom(_as_int(args.get("id")), _as_int(args.get("n")))},
                                  ensure_ascii=False)
            if name == "date":
                return json.dumps({"result": mem.date(_as_int(args.get("id")))}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"OptChat memory unavailable: {type(exc).__name__}: {exc}"})
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self):
        status = super().get_status()
        info = {"last_settle": self.last_settle, "degraded": self.degraded,
                "chat_dir": str(self.chat_dir())}
        if self.memory is not None:
            info.update(self.memory.stats())
        status["optchat"] = info
        return status

    def on_turn_complete(self, messages, usage=None, **kwargs):
        if self.outbox_bound or not self.logs:
            # Only committed rows are logged, from the outbox (never the live list).
            self.on_durable_events_committed(self.session_id)
            return
        conv = list(messages or [])
        with self._lock:
            if self._turn is None or self.memory is None or not self.logs:
                return
            for u in range(len(conv) - 1, -1, -1):
                if conv[u].get("role") == "user" and self._turn_key(conv, u) == self._turn[0]:
                    for j in range(u + 1, len(conv)):
                        self._ingest(self.memory, conv, j)
                    return

    def _find_turn(self, conv, incoming) -> int:
        for j in range(len(conv) - 1, -1, -1):
            if conv[j] == incoming:
                return j
        raise LookupError("current turn's user message not found")

    def _turn_key(self, conv, u: int) -> str:
        m = conv[u]
        return m.get("message_uid") or f"s:{self.session_id}:{u}:{_digest(m.get('role'), m.get('content'))}"
