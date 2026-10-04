"""The agent's memory tools: ``zoom`` and ``date`` (spec §7.1)."""

from __future__ import annotations

from .view import flat

ZOOM_DESCRIPTION = ("Open the line id+n of the view into the two lines of n/2 under it; "
                    "n = 1 gives the message whole.")
DATE_DESCRIPTION = "The date and time of message id."


def _int(x):
    return x if type(x) is int else None


def zoom(store, id, n) -> str:
    i, k = _int(id), _int(n)
    T = len(store.messages)
    if i is None or k is None or k < 1 or k & (k - 1) or i < 0 or i % k or i + k > T:
        return f"No line {id}+{n}."
    if k == 1:
        m = store.messages[i]
        return f"{i}+0|{m['kind']}: {m['text']}" + _archived(store, m)
    l = k.bit_length() - 2  # level of the children
    a, b = (2 * i) // k, (2 * i) // k + 1
    ta, tb = store.node(l, a), store.node(l, b)
    if ta is None or tb is None:
        return f"No line {id}+{n}."
    half = k // 2
    return f"{i}+{half}|{flat(ta)}\n{i + half}+{half}|{flat(tb)}"


def _archived(store, m) -> str:
    """Where a message's original payloads are: private local files, never inline base64."""
    if not m.get("assets"):
        return ""
    lines = [f"\nArchived originals of message {m['i']} (private local files; open one with a "
             f"file, image or audio tool to inspect it):"]
    for a in m["assets"]:
        line = f"- {a.get('type', 'application/octet-stream')}, {a['bytes']} bytes, sha256 {a['sha256']}: " \
               f"{store.asset_path(a)}"
        if a.get("source"):
            line += f" (snapshot of {a['source']})"
        if not store.asset_ok(a):
            line += " (MISSING or altered: the original cannot be recovered from this file)"
        lines.append(line)
    return "\n".join(lines)


def date(store, id) -> str:
    i = _int(id)
    if i is None or not 0 <= i < len(store.messages):
        return f"No message {id}."
    return store.messages[i]["date"]
