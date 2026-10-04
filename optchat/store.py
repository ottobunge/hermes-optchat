"""Append-only storage of the chat log (ROOT) and the summary tree.

Layout (spec §2)::

    <dir>/main/YYYY-MM-DD.jsonl   one message per line: {i, kind, text, size, date, seq[, src][, orig, assets]}
    <dir>/tree/YYYY-MM-DD.jsonl   one node per line:    {l, i, text, size, seq}
    <dir>/assets/ab/<sha256>.<ext> original media / uncapped tool payloads (sidecar)

Every line is written with a single ``write`` followed by ``fsync`` before the call
returns. Nothing is ever edited or deleted. A message with an archive has ``orig`` (its
original content, payload strings replaced by asset markers) and ``assets`` (the files);
the files are fsynced before the line is written. ``seq`` is one counter shared by both
streams: the order the writer committed messages and nodes, which the live view
depends on. Lines written before ``seq`` existed have none.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import stat
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

KINDS = ("user", "talk", "tool", "echo", "note")


class StoreLocked(RuntimeError):
    """Another process (or another Store in this process) owns the chat."""


class StoreCorrupt(RuntimeError):
    """The log is not a contiguous sequence of message ids; refuse to guess."""


class AssetCorrupt(RuntimeError):
    """A media sidecar file is missing, not a private regular file, or has the wrong digest."""


# Media sidecar: <dir>/assets/<sha256[:2]>/<sha256>.<ext>. Names come only from the digest
# and this table, never from message content.
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp",
        "image/heic": "heic", "image/svg+xml": "svg", "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/ogg": "ogg", "audio/webm": "weba",
        "audio/flac": "flac", "audio/mp4": "m4a", "video/mp4": "mp4", "video/webm": "webm",
        "application/pdf": "pdf", "application/json": "json", "text/plain": "txt"}
_MIME = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ASSET_FILE = re.compile(r"^([0-9a-f]{64})\.([a-z0-9]{1,8})$")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def media_type(mime) -> str:
    mime = mime.strip().lower() if isinstance(mime, str) else ""
    return mime if _MIME.match(mime) else "application/octet-stream"


def valid_asset_ref(ref) -> bool:
    if not isinstance(ref, dict) or not isinstance(ref.get("sha256"), str):
        return False
    m = _ASSET_FILE.match(ref.get("file") or "")
    return (bool(m) and _DIGEST.match(ref["sha256"]) is not None and m.group(1) == ref["sha256"]
            and type(ref.get("bytes")) is int and ref["bytes"] >= 0)


class Blob(NamedTuple):
    """Original payload bytes to keep in the sidecar; ``source`` names a local file it was read from."""
    data: bytes
    mime: str
    source: str | None = None


class Archive(NamedTuple):
    """The original content of a message: ``orig`` is its JSON value with each payload string
    replaced by an asset marker ``{"$optchat": "asset", "sha256", "enc"[, "prefix"]}``."""
    orig: object
    blobs: tuple = ()


ENCODINGS = ("utf8", "base64", "data-url", "json")


def asset_markers(node):
    """Every asset marker in a skeleton, depth first."""
    if isinstance(node, dict):
        if node.get("$optchat") == "asset":
            yield node
            return
        for v in node.values():
            yield from asset_markers(v)
    elif isinstance(node, list):
        for v in node:
            yield from asset_markers(v)


def _valid_marker(m, digests) -> bool:
    return (m.get("sha256") in digests and m.get("enc") in ENCODINGS
            and (m["enc"] != "data-url" or isinstance(m.get("prefix"), str)))


def _valid_message(r) -> bool:
    if not (isinstance(r, dict) and type(r.get("i")) is int and r["i"] >= 0
            and r.get("kind") in KINDS and isinstance(r.get("text"), str)):
        return False
    assets = r.get("assets", [])
    if not isinstance(assets, list) or not all(valid_asset_ref(a) for a in assets):
        return False
    digests = {a["sha256"] for a in assets}
    return all(_valid_marker(m, digests) for m in asset_markers(r.get("orig")))


def _valid_node(r) -> bool:
    return (isinstance(r, dict) and type(r.get("l")) is int and type(r.get("i")) is int
            and r["l"] >= 0 and r["i"] >= 0 and isinstance(r.get("text"), str))


def nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.messages: list[dict] = []
        self.nodes: dict[tuple[int, int], dict] = {}
        self.srcs: dict[str, int] = {}
        self._seq = 0  # next sequence number
        self.main_dir = self.path / "main"
        self.tree_dir = self.path / "tree"
        self._lock_fd = None
        self.readonly = False

    def open_readonly(self) -> None:
        """Load the chat without its writer lock (another process owns it); never writes.
        A line the owner is writing right now may be skipped as torn until the next load."""
        if not self.main_dir.is_dir() or not self.tree_dir.is_dir():
            raise FileNotFoundError(f"no optchat chat at {self.path}")
        self.readonly = True
        self._open_locked()

    def open(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path, 0o700)
        fd = os.open(self.path / "lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Held for the life of the owner; the OS releases it when the owner dies.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise StoreLocked(f"optchat chat at {self.path} is in use by another writer")
        self._lock_fd = fd
        try:
            self._open_locked()
        except BaseException:
            self.close()
            raise

    def _open_locked(self) -> None:
        if not self.readonly:
            self.main_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.tree_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.main_dir, 0o700)
            os.chmod(self.tree_dir, 0o700)
        self.messages = self._load(self.main_dir, _valid_message)
        for n, rec in enumerate(self.messages):
            if rec["i"] != n:
                raise StoreCorrupt(f"optchat log {self.main_dir}: expected message {n}, found {rec['i']}")
        self.srcs = {}
        for rec in self.messages:
            if isinstance(rec.get("src"), str):
                self.srcs.setdefault(rec["src"], rec["i"])
        self.nodes = {(r["l"], r["i"]): r for r in self._load(self.tree_dir, _valid_node)}
        seqs = [r["seq"] for r in (*self.messages, *self.nodes.values()) if type(r.get("seq")) is int]
        self._seq = max(seqs) + 1 if seqs else 0
        if not self.readonly:
            self._clean_partial_assets()

    def _clean_partial_assets(self) -> None:
        """Drop temporary files of asset writes a crash interrupted (never linked into place).

        A complete asset whose ROOT line never came is kept: it is content-addressed, so a
        retry of the same message reuses it, and nothing in the archive is ever deleted.
        """
        if not (self.path / "assets").exists():
            return
        tmp = self._assets_fd(".tmp")
        try:
            for name in os.listdir(tmp):
                logger.warning("optchat: removing interrupted asset write %s", name)
                os.unlink(name, dir_fd=tmp)
        finally:
            os.close(tmp)

    def orphan_assets(self) -> list[str]:
        """Sidecar files no logged message refers to (a crash between asset and ROOT write)."""
        used = {a["file"] for m in self.messages for a in m.get("assets") or ()}
        return sorted(f.name for f in (self.path / "assets").glob("??/*")
                      if _ASSET_FILE.match(f.name) and f.name not in used)

    def close(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # closing the descriptor releases the flock
            self._lock_fd = None

    def _load(self, folder: Path, valid) -> list[dict]:
        out = []
        for f in sorted(folder.glob("*.jsonl")):
            raw = f.read_bytes()
            for n, line in enumerate(raw.split(b"\n"), 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    rec = None
                if not valid(rec):
                    logger.warning("optchat: skipping torn/invalid line %s:%d", f, n)
                    continue
                out.append(rec)
            if raw and not raw.endswith(b"\n") and not self.readonly:
                # A crash mid-write: start the next record on its own line.
                self._append_bytes(f, b"\n")
        # Stable sort, then keep the first record written for each id.
        out.sort(key=lambda r: (r.get("l", 0), r["i"]))
        kept, seen = [], set()
        for r in out:
            key = (r.get("l", 0), r["i"])
            if key in seen:
                logger.warning("optchat: ignoring duplicate record %s in %s", key, folder)
                continue
            seen.add(key)
            kept.append(r)
        return kept

    @staticmethod
    def _append_bytes(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _write_line(self, folder: Path, record: dict) -> None:
        if self._lock_fd is None:
            raise RuntimeError("optchat store is not open for writing")
        record["seq"] = self._seq
        self._seq += 1  # consumed even if the write fails: never reuse a number
        now = datetime.now().astimezone()
        data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        target = folder / f"{now:%Y-%m-%d}.jsonl"
        new = not target.exists()
        self._append_bytes(target, data)
        if new:  # make the new file's directory entry durable too
            dfd = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    def append_message(self, kind: str, text: str, src: str | None = None,
                       archive: Archive | None = None, date: str | None = None) -> int:
        """Log one message (``date``: when it happened, if not now). With an ``archive``, its payloads are fsynced into the sidecar
        before the ROOT line is written, so a committed line never points to a missing file;
        any failure raises before the line exists."""
        if kind not in KINDS:
            raise ValueError(f"unknown message kind {kind!r}")
        assets = []
        if archive is not None:
            digests = {hashlib.sha256(b.data).hexdigest() for b in archive.blobs}
            if not all(_valid_marker(m, digests) for m in asset_markers(archive.orig)):
                raise ValueError("optchat archive refers to a payload it does not carry")
            for blob in archive.blobs:
                ref = self.put_asset(blob.data, blob.mime)
                if blob.source is not None:
                    ref["source"] = blob.source
                assets.append(ref)
        i = len(self.messages)
        rec = {"i": i, "kind": kind, "text": text, "size": nbytes(f"{kind}: {text}"),
               "date": date or datetime.now().astimezone().isoformat(timespec="seconds")}
        if src is not None:
            rec["src"] = src
        if archive is not None:
            rec["orig"] = archive.orig
            rec["assets"] = assets
        self._write_line(self.main_dir, rec)
        self.messages.append(rec)
        if src is not None:
            self.srcs.setdefault(src, i)
        return i

    # -- media sidecar ---------------------------------------------------------------
    @staticmethod
    def _subdir(parent: int, name: str, *, create: bool = True) -> int:
        """A descriptor for private folder ``name`` under ``parent``; never follows a symlink."""
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
        try:
            fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            raise
        except OSError as exc:  # ELOOP/ENOTDIR: a symlink or a file where a folder should be
            raise AssetCorrupt(f"optchat asset folder {name!r} is not a private directory: {exc}") from exc
        if os.fstat(fd).st_uid != os.getuid():
            os.close(fd)
            raise AssetCorrupt(f"optchat asset folder {name!r} is owned by another user")
        os.fchmod(fd, 0o700)
        return fd

    def _assets_fd(self, *names: str, create: bool = True) -> int:
        fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        for name in ("assets", *names):
            try:
                nxt = self._subdir(fd, name, create=create)
            finally:
                os.close(fd)
            fd = nxt
        return fd

    @staticmethod
    def _read_at(folder: int, ref: dict, *, missing_ok: bool = False):
        try:
            fd = os.open(ref["file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise AssetCorrupt(f"optchat asset {ref['file']} is missing") from None
        except OSError as exc:
            raise AssetCorrupt(f"optchat asset {ref['file']} cannot be opened safely: {exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size != ref["bytes"]:
                raise AssetCorrupt(f"optchat asset {ref['file']} is not the archived file")
            chunks = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
        finally:
            os.close(fd)
        data = b"".join(chunks)
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise AssetCorrupt(f"optchat asset {ref['file']} does not match its sha256")
        return data

    def put_asset(self, data: bytes, mime) -> dict:
        """Durably store ``data`` (fsynced file and folder) and return its reference.

        Content-addressed and append-only: a file is created once, under a temporary name,
        then hard-linked into place, so an existing file is never overwritten.
        """
        if self._lock_fd is None:
            raise RuntimeError("optchat store is not open for writing")
        data = bytes(data)
        mime = media_type(mime)
        digest = hashlib.sha256(data).hexdigest()
        ref = {"sha256": digest, "bytes": len(data), "type": mime,
               "file": f"{digest}.{_EXT.get(mime, 'bin')}"}
        shard = self._assets_fd(digest[:2])
        try:
            if self._read_at(shard, ref, missing_ok=True) is not None:
                return ref
            tmp = self._assets_fd(".tmp")
            try:
                name = f"{digest}.{secrets.token_hex(8)}.part"
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=tmp)
                try:
                    try:
                        view = memoryview(data)
                        while view:
                            view = view[os.write(fd, view):]
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    os.link(name, ref["file"], src_dir_fd=tmp, dst_dir_fd=shard, follow_symlinks=False)
                finally:
                    os.unlink(name, dir_fd=tmp)
                os.fsync(shard)
            finally:
                os.close(tmp)
            self._read_at(shard, ref)
            return ref
        finally:
            os.close(shard)

    def asset_path(self, ref) -> Path:
        if not valid_asset_ref(ref):
            raise AssetCorrupt(f"invalid optchat asset reference {ref!r}")
        return self.path / "assets" / ref["sha256"][:2] / ref["file"]

    def asset_ok(self, ref) -> bool:
        """Cheap check (no hashing): the archived file is a regular file of the right size."""
        try:
            st = os.lstat(self.asset_path(ref))
        except (OSError, AssetCorrupt):
            return False
        return stat.S_ISREG(st.st_mode) and st.st_size == ref["bytes"]

    def read_asset(self, ref) -> bytes:
        if not valid_asset_ref(ref):
            raise AssetCorrupt(f"invalid optchat asset reference {ref!r}")
        try:
            shard = self._assets_fd(ref["sha256"][:2], create=False)
        except FileNotFoundError:
            raise AssetCorrupt(f"optchat asset {ref['file']} is missing") from None
        try:
            return self._read_at(shard, ref)
        finally:
            os.close(shard)

    def original(self, i: int):
        """Message ``i``'s original content, rebuilt byte for byte from ROOT and the sidecar."""
        rec = self.messages[i]
        if "orig" not in rec:
            return rec["text"]
        refs = {a["sha256"]: a for a in rec.get("assets", ())}
        return self._resolve(rec["orig"], refs)

    def _resolve(self, node, refs):
        if isinstance(node, list):
            return [self._resolve(v, refs) for v in node]
        if not isinstance(node, dict):
            return node
        if node.get("$optchat") != "asset":
            return {k: self._resolve(v, refs) for k, v in node.items()}
        data = self.read_asset(refs[node["sha256"]])
        enc = node["enc"]
        if enc == "utf8":
            return data.decode("utf-8", "surrogatepass")
        if enc == "json":
            return json.loads(data.decode("utf-8"))
        b64 = base64.b64encode(data).decode("ascii")
        return node["prefix"] + b64 if enc == "data-url" else b64

    def has_src(self, src: str) -> bool:
        return src in self.srcs

    def src_index(self, src: str):
        return self.srcs.get(src)

    def append_node(self, l: int, i: int, text: str) -> None:
        rec = {"l": l, "i": i, "text": text, "size": nbytes(text)}
        self._write_line(self.tree_dir, rec)
        self.nodes[(l, i)] = rec

    def node(self, l: int, i: int):
        rec = self.nodes.get((l, i))
        return rec["text"] if rec else None

    def events(self) -> list[tuple]:
        """Every message ``("message", i)`` and node ``("node", l, i)`` in commit order.

        Lines without ``seq`` predate it and come first: legacy nodes (all known before
        any message, as the old fold assumed), then legacy messages by id.
        """
        keyed = []
        for k, ((l, i), r) in enumerate(self.nodes.items()):
            seq = r.get("seq")
            keyed.append(((1, seq, 0) if type(seq) is int else (0, 0, k), ("node", l, i)))
        for r in self.messages:
            seq = r.get("seq")
            keyed.append(((1, seq, 1) if type(seq) is int else (0, 1, r["i"]), ("message", r["i"])))
        keyed.sort(key=lambda e: e[0])
        return [e for _, e in keyed]
