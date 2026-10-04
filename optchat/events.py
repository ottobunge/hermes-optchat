"""Hermes (OpenAI-format) messages -> OptChat log events ``(kind, text, src, archive)``.

``src`` is the durable dedupe key stored with the logged line: the host's ``message_uid``
(plus a suffix for each tool call), or the caller's ``fallback`` key when the row has no
uid yet. Model reasoning is never turned into an event (spec §2).

``text`` is what the view, the compactor and ``zoom`` show: text only, never base64. A
message with media (or a tool result over ``CAP``) also gets an ``archive``: its original
content with every payload moved to a content-addressed sidecar file, so it can be rebuilt
exactly. Its text says, per attachment, what is not shown and where the original is; it
never stands in for the media as if it were the content.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import re
from typing import NamedTuple
from urllib.parse import unquote, urlparse

from .store import Archive, Blob, asset_markers, media_type

CAP = 30_000  # characters kept of one tool result (head + tail)


_SURROGATE = re.compile("[\ud800-\udfff]")
_TEXT = ("text", "input_text", "output_text")
_THOUGHT = ("thinking", "redacted_thinking", "reasoning")
_DATA_URL = re.compile(r"^data:([^,;]*)((?:;[^,;]*)*?);base64,", re.I | re.S)


def _file_mime(path: str) -> str | None:
    # The system MIME database varies across Python/OS releases for .wav.
    return "audio/x-wav" if path.lower().endswith(".wav") else mimetypes.guess_type(path)[0]


class UnsupportedContent(ValueError):
    """Content that cannot be archived losslessly; never log a lossy stand-in for it."""


class Event(NamedTuple):
    kind: str
    text: str
    src: str
    archive: Archive | None = None


def clean(text: str) -> str:
    return _SURROGATE.sub("�", text)


def cap_echo(text: str, cap: int = CAP) -> str:
    if len(text) <= cap:
        return text
    cut = len(text)
    while True:  # the note's length depends on the number it reports
        note = f"\n[... {cut:,} characters cut ...]\n"
        keep = cap - len(note)
        new_cut = len(text) - keep
        if new_cut == cut:
            break
        cut = new_cut
    head = keep - keep // 2
    return text[:head] + note + text[len(text) - keep // 2:]


def _canonical_b64(s: str):
    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None
    return data if base64.b64encode(data).decode("ascii") == s else None


class _Media:
    """Collects payload blobs and per-attachment descriptions for one message."""

    def __init__(self):
        self.blobs: list[Blob] = []
        self.count = 0

    def _keep(self, data: bytes, mime: str, source=None) -> str:
        self.blobs.append(Blob(data, media_type(mime), source))
        return hashlib.sha256(data).hexdigest()

    def _label(self, what: str) -> str:
        self.count += 1
        return f"[attachment {self.count} not shown: {what}]"

    def _archived(self, what: str, mime: str, data: bytes, digest: str) -> str:
        return self._label(f"{what}{media_type(mime)}, {len(data)} bytes, sha256 {digest[:16]}; "
                           f"original archived, zoom this message (n = 1) for the file")

    def text(self, s: str):
        """Keep a long or non-UTF-8 string out of the ROOT line; returns a marker."""
        digest = self._keep(s.encode("utf-8", "surrogatepass"), "text/plain")
        return {"$optchat": "asset", "sha256": digest, "enc": "utf8"}

    def b64(self, s: str, mime: str):
        """A bare base64 payload -> (marker, description)."""
        data = _canonical_b64(s)
        if data is None:
            return self._raw(s, mime)
        digest = self._keep(data, mime)
        return {"$optchat": "asset", "sha256": digest, "enc": "base64"}, self._archived("", mime, data, digest)

    def _raw(self, s: str, mime: str):
        data = s.encode("utf-8", "surrogatepass")
        digest = self._keep(data, "text/plain")
        what = (f"{media_type(mime)} payload kept verbatim as text (not canonical base64), "
                f"{len(data)} bytes, sha256 {digest[:16]}; zoom this message (n = 1) for the file")
        return {"$optchat": "asset", "sha256": digest, "enc": "utf8"}, self._label(what)

    def url(self, url, mime: str):
        """A URL-valued payload -> (replacement value, description)."""
        if not isinstance(url, str):
            return None
        m = _DATA_URL.match(url)
        if m:
            data = _canonical_b64(url[m.end():])
            mime = m.group(1) or mime
            if data is None:
                return self._raw(url, mime)
            digest = self._keep(data, mime)
            marker = {"$optchat": "asset", "sha256": digest, "enc": "data-url", "prefix": url[:m.end()]}
            return marker, self._archived("", mime, data, digest)
        if url.startswith("data:"):
            return self._raw(url, mime)
        path = _local_path(url)
        if path is not None:
            # A message is not a capability: arbitrary user/model/tool parts can name
            # any local path. The host must capture an authorized attachment's bytes
            # at ingress as a data URL; the consumer never opens paths from content.
            return url, self._label(f"local file {path} not host-authorized; only the path is kept, "
                                    "its bytes are NOT archived")
        return url, self._label(f"remote {mime.split('/')[0]} at {url}; only the URL is kept, "
                                f"not its content, which may change or disappear")


    def json(self, part, why: str):
        try:
            raw = json.dumps(part, ensure_ascii=True)
            ok = json.loads(raw) == part
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise UnsupportedContent(f"content part {why} cannot be archived losslessly")
        data = raw.encode("ascii")
        digest = self._keep(data, "application/json")
        what = f"{why}, archived as JSON ({len(data)} bytes, sha256 {digest[:16]}); zoom this message (n = 1)"
        return {"$optchat": "asset", "sha256": digest, "enc": "json"}, self._label(what)

    def part(self, part: dict):
        """A non-text part -> (skeleton part, description)."""
        kind = part.get("type")
        if any(asset_markers(part)):  # would be ambiguous with our markers: keep it whole
            return self.json(part, "with reserved keys")
        out = dict(part)
        got = None
        if kind in ("image_url", "input_image", "video_url", "audio_url"):
            field = "image_url" if kind == "input_image" else kind
            default = {"video_url": "video/*", "audio_url": "audio/*"}.get(kind, "image/*")
            inner = part.get(field)
            if isinstance(inner, dict) and "url" in inner:
                got = self.url(inner["url"], default)
                if got:
                    out[field] = dict(inner, url=got[0])
            else:
                got = self.url(inner, default)
                if got:
                    out[field] = got[0]
            if got is None and isinstance(part.get("file_id"), str):
                return out, self._label(f"provider file id {part['file_id']}; held by the provider, not archived")
        elif kind == "input_audio" and isinstance(part.get("input_audio"), dict) \
                and isinstance(part["input_audio"].get("data"), str):
            inner = part["input_audio"]
            got = self.b64(inner["data"], f"audio/{inner.get('format') or '*'}")
            out["input_audio"] = dict(inner, data=got[0])
        elif kind in ("image", "document", "audio") and isinstance(part.get("source"), dict):
            src = part["source"]
            mime = src.get("media_type") or f"{kind}/*"
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                got = self.b64(src["data"], mime)
                out["source"] = dict(src, data=got[0])
            elif src.get("type") == "url":
                got = self.url(src.get("url"), mime)
                if got:
                    out["source"] = dict(src, url=got[0])
        elif kind in ("file", "input_file"):
            inner = part.get("file") if isinstance(part.get("file"), dict) else part
            name = inner.get("filename")
            mime = (_file_mime(name) if isinstance(name, str) else None) or "application/octet-stream"
            data = inner.get("file_data")
            if isinstance(data, str):
                got = self.url(data, mime) if data.startswith("data:") else self.b64(data, mime)
                new = dict(inner, file_data=got[0])
            elif isinstance(inner.get("file_url"), str):
                got = self.url(inner["file_url"], mime)
                new = dict(inner, file_url=got[0])
            elif isinstance(inner.get("file_id"), str):
                return out, self._label(f"provider file id {inner['file_id']}; held by the provider, not archived")
            if got:
                if inner is part:
                    out = new
                else:
                    out["file"] = new
        if got is None:
            return self.json(part, f"of type {str(kind)[:64]!r}")
        return out, got[1]


def _local_path(url: str):
    if url.startswith("file://"):
        p = urlparse(url)
        return unquote(p.path) if p.netloc in ("", "localhost") and p.path.startswith("/") else None
    return url if url.startswith("/") and "\n" not in url else None




def _content(content, *, echo: bool = False):
    """``(text, archive or None)`` for a message content value."""
    if content is None:
        return "", None
    if isinstance(content, str):
        if echo and len(content) > CAP:
            media = _Media()
            return cap_echo(clean(content)), Archive(media.text(content), tuple(media.blobs))
        return content, None
    if not isinstance(content, list):
        raise UnsupportedContent(f"unsupported content value {type(content).__name__}")
    media = _Media()
    texts, lines, skeleton = [], [], []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            lines.append(part)
            skeleton.append(part)
        elif isinstance(part, dict) and part.get("type") in _TEXT and isinstance(part.get("text"), str):
            texts.append(part["text"])
            lines.append(part["text"])
            long = (echo and len(part["text"]) > CAP) or _SURROGATE.search(part["text"])
            skeleton.append(dict(part, text=media.text(part["text"])) if long else part)
        elif isinstance(part, dict) and part.get("type") in _THOUGHT:
            continue  # never log model thoughts, not even in the archive
        else:
            if not isinstance(part, dict):
                new, line = media.json(part, f"of type {type(part).__name__}")
            else:
                new, line = media.part(part)
            lines.append(line)
            skeleton.append(new)
    if echo and len("\n".join(lines)) > CAP:
        # A cap on the combined echo also cuts parts that were individually short.
        # Keep their original positions and fields, with text in sidecars instead of ROOT.
        for i, part in enumerate(skeleton):
            if isinstance(part, str):
                skeleton[i] = media.text(part)
            elif isinstance(part, dict) and part.get("type") in _TEXT and isinstance(part.get("text"), str):
                skeleton[i] = dict(part, text=media.text(part["text"]))
    if not media.count and not media.blobs:
        return "\n".join(texts), None
    text = "\n".join(lines)
    if echo and len(text) > CAP:  # cap the result's text, never the attachment notes
        notes = [line for line in lines if line.startswith("[attachment ")]
        text = cap_echo("\n".join(texts)) + ("\n" + "\n".join(notes) if notes else "")
    try:
        json.dumps(skeleton, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise UnsupportedContent("content cannot be archived losslessly")
    return clean(text), Archive(skeleton, tuple(media.blobs))


def _call_text(call) -> str:
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict):
        return json.dumps(call, ensure_ascii=False, default=str)
    args = fn.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False, default=str)
    return f"{fn.get('name', '?')} {args}".rstrip()


def sources_from_message(msg: dict, *, fallback: str) -> list[str]:
    """The dedupe keys ``events_from_message`` would use, without reading any payload."""
    key = msg.get("message_uid") or fallback
    if msg.get("role") in ("user", "tool"):
        return [key]
    if msg.get("role") == "assistant":
        return [key] + [f"{key}#tool:{k}" for k in range(len(msg.get("tool_calls") or []))]
    return []


def events_from_message(msg: dict, *, fallback: str) -> list[Event]:
    role = msg.get("role")
    key = msg.get("message_uid") or fallback
    if role == "user":
        text, archive = _content(msg.get("content"))
        return [Event("user", clean(text), key, archive)] if text.strip() else []
    if role == "assistant":
        out = []
        text, archive = _content(msg.get("content"))
        if text.strip():
            out.append(Event("talk", clean(text), key, archive))
        for k, call in enumerate(msg.get("tool_calls") or []):
            out.append(Event("tool", clean(_call_text(call)), f"{key}#tool:{k}"))
        return out
    if role == "tool":
        text, archive = _content(msg.get("content"), echo=True)
        return [Event("echo", cap_echo(clean(text)) if archive is None else text, key, archive)]
    return []
