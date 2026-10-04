"""One compactor step: build a tree node from a message or from two child lines (spec §3-4)."""

from __future__ import annotations

from .prompts import NODE, SCALE
from .store import nbytes
from .view import flat

TRIES = 5


class SummaryFailed(RuntimeError):
    """The summarizer gave no usable line; the node stays unbuilt (never invent one)."""


def free_leaf(kind: str, text: str, node: int = NODE):
    line = f"{kind}: {text}"
    return line if nbytes(line) <= node else None


def free_merge(a: str, b: str, node: int = NODE):
    line = f"{a}\n{b}"
    return line if nbytes(line) <= node else None


def _context(lines) -> str:
    body = "\n".join(flat(t) for t in lines)
    return f"<chat>\n{body}\n</chat>" if body else "<chat>\n</chat>"


def _scale(node: int) -> str:
    return f"For scale, this line is exactly {node} bytes:\n{SCALE}\n\n"


def leaf_request(context_lines, kind: str, text: str, node: int = NODE) -> dict:
    step = (_scale(node) + f"Compress this message into one line, in at most {node} bytes:\n"
            f"{kind}: {text}")
    return {"role": "user", "content": [{"type": "text", "text": _context(context_lines)},
                                        {"type": "text", "text": step}]}


def merge_request(context_lines, a: str, b: str, node: int = NODE) -> dict:
    step = (_scale(node) + f"Merge these two lines into one, in at most {node} bytes:\n"
            f"{flat(a)}\n{flat(b)}")
    return {"role": "user", "content": [{"type": "text", "text": _context(context_lines)},
                                        {"type": "text", "text": step}]}


def cut_bytes(text: str, limit: int) -> str:
    """The first ``limit`` bytes of ``text``, never splitting a UTF-8 character."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def summarize(summarizer, system: str, request: dict, node: int = NODE, tries: int = TRIES) -> str:
    """Ask for a line; while it is over ``node`` bytes, show it cut at the limit and ask again
    in the same conversation. Returns the shortest try. Raises SummaryFailed on any error or
    empty reply, so a node is never built from made-up or partial text."""
    convo = [request]
    got: list[str] = []
    while True:
        try:
            reply = summarizer(system, list(convo))
        except Exception as exc:  # the summarizer is an external process/model
            raise SummaryFailed(f"summarizer failed: {exc}") from exc
        line = reply.strip() if isinstance(reply, str) else ""
        if not line:
            raise SummaryFailed("summarizer returned an empty line")
        got.append(line)
        size = nbytes(line)
        if size <= node or len(got) >= tries:
            return min(got, key=nbytes)
        convo.append({"role": "assistant", "content": line})
        convo.append({"role": "user", "content": (
            f"That line is {size} bytes; the limit is {node}. It must end where it is cut here:\n"
            f"{cut_bytes(line, node)}| ← LIMIT")})
