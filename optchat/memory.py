"""The chat memory: log + tree + live view + background compactor (spec §2-6).

One ``Memory`` per chat directory per process. It is thread-safe: every public method
takes ``self._lock``; summarizer calls run on worker threads outside the lock.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from pathlib import Path

from . import compact
from .prompts import NODE, compact_prompt
from . import tools
from .store import Store
from .view import View

logger = logging.getLogger(__name__)

VIEW = 128_000
JOBS = 8
RETRY = 10.0


class Memory:
    def __init__(self, path, *, summarizer=None, agent: str = "OptChat", budget: int = VIEW,
                 jobs: int = JOBS, retry: float = RETRY, node: int = NODE):
        self.path = Path(path)
        self.store = Store(self.path)
        self.summarizer = summarizer
        self.system = compact_prompt(agent)
        self.budget, self.jobs, self.retry, self.node = budget, jobs, retry, node
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self.view = View(budget)
        self.busy: set[tuple[int, int]] = set()
        self.failed: set[tuple[int, int]] = set()
        self.pending: set[tuple[int, int]] = set()  # unbuilt nodes whose sources exist
        self._timers: set[threading.Timer] = set()
        self.closed = True

    # -- lifecycle -------------------------------------------------------------------
    def open(self) -> None:
        with self._lock:
            self.store.open()
            self.view = View.replay(self.store.events(), self.store, self.budget)
            self.pending = {(0, i) for i in range(len(self.store.messages)) if self.store.node(0, i) is None}
            for (l, i) in list(self.store.nodes):
                self._note_built(l, i)
            self.closed = False
            self._pump()

    def open_readonly(self) -> None:
        """A reader of a chat another process writes: view and zoom only, no compactor."""
        with self._lock:
            self.store.open_readonly()
            self.view = View.replay(self.store.events(), self.store, self.budget)
            self.summarizer = None
            self.closed = False

    @property
    def readonly(self) -> bool:
        return self.store.readonly

    def close(self) -> None:
        with self._lock:
            self.closed = True
            for t in self._timers:
                t.cancel()
            self._timers.clear()
            close = getattr(self.summarizer, "close", None)
            if callable(close):
                close()
            self.store.close()
            self._cond.notify_all()

    # -- writing ---------------------------------------------------------------------
    def log(self, kind: str, text: str, src: str | None = None, archive=None, date: str | None = None) -> int:
        with self._lock:
            if src is not None and self.store.has_src(src):
                return self.store.src_index(src)
            i = self.store.append_message(kind, text, src=src, archive=archive, date=date)
            self.pending.add((0, i))
            self.view.append(i, self.store)
            self._fit()
            self._pump()
            return i

    def has_src(self, src: str) -> bool:
        with self._lock:
            return self.store.has_src(src)

    # -- reading ---------------------------------------------------------------------
    def render(self) -> str:
        with self._lock:
            return self.view.render(self.store)

    def freeze(self):
        """``(parts, messages)`` of the view if it is all summaries, else None."""
        with self._lock:
            if not self.view.settled(self.store):
                return None
            return [list(p) for p in self.view.parts], len(self.store.messages)

    def render_parts(self, parts) -> str:
        """Render a frozen view: its nodes are immutable once built."""
        with self._lock:
            view = View(self.budget)
            view.parts = [tuple(p) for p in parts]
            if not view.settled(self.store):
                raise LookupError("a frozen view refers to a node that is not in the tree")
            return view.render(self.store)

    def zoom(self, id, n) -> str:
        with self._lock:
            return tools.zoom(self.store, id, n)

    def date(self, id) -> str:
        with self._lock:
            return tools.date(self.store, id)

    def stats(self) -> dict:
        with self._lock:
            return {"messages": len(self.store.messages), "nodes": len(self.store.nodes),
                    "view_lines": len(self.view.parts), "view_bytes": self.view.size(self.store),
                    "busy": len(self.busy), "failing": len(self.failed),
                    "summarizer": self.summarizer is not None}

    def settle(self, timeout: float) -> bool:
        """Wait until every line of the view is a summary; False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self.view.settled(self.store):
                if self.summarizer is None:
                    return False  # nothing can ever summarize it: fail now, not at the timeout
                left = deadline - time.monotonic()
                if left <= 0 or self.closed:
                    return False
                self._cond.wait(left)
            return True

    # -- compactor -------------------------------------------------------------------
    def _fit(self) -> None:
        self.view.fit(len(self.store.messages), self.store)
        self._cond.notify_all()

    def _note_built(self, l: int, i: int) -> None:
        """Keep ``pending`` = unbuilt nodes whose sources exist (spec §4.1 rule 2)."""
        self.pending.discard((l, i))
        if self.store.node(l, i ^ 1) is not None and self.store.node(l + 1, i // 2) is None:
            self.pending.add((l + 1, i // 2))

    def _pump(self) -> None:
        """Start every pending node whose view context is all summaries (spec §4.1), lowest
        level first, up to ``jobs`` at once. Free nodes are committed inline."""
        if self.closed:
            return
        progress = True
        while progress:
            progress = False
            first = self.view.first_unbuilt(len(self.store.messages), self.store)
            for l, i in sorted(self.pending):
                if len(self.busy) >= self.jobs:
                    return
                end = i if l == 0 else (i + 1) << l
                if (l, i) in self.busy or end > first:
                    continue
                free = self._free(l, i)
                if free is not None:
                    self._commit(l, i, free)
                    progress = True
                    break
                if self.summarizer is not None:
                    self._start(l, i)

    def _free(self, l: int, i: int):
        if l == 0:
            m = self.store.messages[i]
            return compact.free_leaf(m["kind"], m["text"], self.node)
        return compact.free_merge(self.store.node(l - 1, 2 * i), self.store.node(l - 1, 2 * i + 1), self.node)

    def _request(self, l: int, i: int) -> dict:
        if l == 0:
            m = self.store.messages[i]
            ctx = [t for s, n, t in self.view.lines(self.store) if s + n <= i]
            return compact.leaf_request(ctx, m["kind"], m["text"], self.node)
        end = (i + 1) << l
        ctx = [t for s, n, t in self.view.lines(self.store) if s < end]
        return compact.merge_request(ctx, self.store.node(l - 1, 2 * i), self.store.node(l - 1, 2 * i + 1), self.node)

    def _start(self, l: int, i: int) -> None:
        self.busy.add((l, i))
        request = self._request(l, i)
        threading.Thread(target=contextvars.copy_context().run, args=(self._job, l, i, request), daemon=True,
                         name=f"optchat-node-{l}-{i}").start()

    def _job(self, l: int, i: int, request: dict) -> None:
        try:
            text = compact.summarize(self.summarizer, self.system, request, self.node)
        except compact.SummaryFailed as exc:
            with self._lock:
                if (l, i) not in self.failed:  # report only a node's first failure
                    self.failed.add((l, i))
                    logger.warning("optchat: node %d+%d failed (retrying every %ss): %s",
                                   i << l, 1 << l, self.retry, exc)
                if self.closed:
                    return
                # Stay busy for RETRY, then try again, forever. No exponential backoff:
                # the next turn waits for these summaries.
                timer = threading.Timer(self.retry, contextvars.copy_context().run,
                                        args=(self._retry, l, i))
                timer.daemon = True
                self._timers.add(timer)
                timer.start()
            return
        with self._lock:
            self.busy.discard((l, i))
            self.failed.discard((l, i))
            if self.closed:
                return
            self._commit(l, i, text)
            self._pump()

    def _retry(self, l: int, i: int) -> None:
        with self._lock:
            self._timers = {t for t in self._timers if t.is_alive() and t is not threading.current_thread()}
            self.busy.discard((l, i))
            self._pump()

    def _commit(self, l: int, i: int, text: str) -> None:
        old = self.view.text((l, i), self.store)
        self.store.append_node(l, i, text)
        self._note_built(l, i)
        self.view.node_built(l, i, old, self.store)
        self._fit()
