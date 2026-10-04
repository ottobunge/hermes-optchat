"""Deterministic test summarizers (no model calls, no token cost)."""

import threading

from optchat.view import PLACEHOLDER


def _texts(messages):
    first = messages[0]["content"]
    return first[0]["text"], first[1]["text"]


class FakeSummarizer:
    """Returns a short deterministic line per step; records every call.

    Also asserts the spec's input invariants on every call: no placeholder in the
    context (rule 3) and no ``id+n|`` markers anywhere (no ids in compactor input).
    """

    def __init__(self, line=None, delay=0.0, fail_first=0):
        self.calls = []
        self.lock = threading.Lock()
        self.line = line
        self.delay = delay
        self.fail_first = fail_first
        self.active = 0
        self.max_active = 0
        self.violations = []

    def __call__(self, system, messages):
        import re
        import time

        ctx, step = _texts(messages)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((ctx, step))
            n = len(self.calls)
            if PLACEHOLDER in ctx or PLACEHOLDER in step:
                self.violations.append(("placeholder", n))
            if re.search(r"^\d+\+\d+\|", ctx + "\n" + step, re.M):
                self.violations.append(("id", n))
        try:
            if self.delay:
                time.sleep(self.delay)
            if n <= self.fail_first:
                raise RuntimeError("transient failure")
            if self.line is not None:
                return self.line
            body = step.split("bytes:\n", 2)[-1]
            return "S[" + " ".join(body.split())[:60] + "]"
        finally:
            with self.lock:
                self.active -= 1


class GateSummarizer(FakeSummarizer):
    """Blocks every call until ``release()``; lets tests observe in-flight jobs."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()

    def __call__(self, system, messages):
        self.gate.wait(10)
        return super().__call__(system, messages)

    def release(self):
        self.gate.set()
