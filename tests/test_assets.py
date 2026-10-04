"""Media sidecar: private, append-only, content-addressed original payloads."""

import hashlib
import json
import os

import pytest

from optchat.store import AssetCorrupt, Store

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


@pytest.fixture
def store(chat_dir):
    s = Store(chat_dir)
    s.open()
    yield s
    s.close()


def test_put_asset_is_content_addressed_private_and_verified(store, chat_dir):
    ref = store.put_asset(PNG, "image/png")
    digest = hashlib.sha256(PNG).hexdigest()
    assert ref == {"sha256": digest, "bytes": len(PNG), "type": "image/png", "file": f"{digest}.png"}
    path = store.asset_path(ref)
    assert path == chat_dir / "assets" / digest[:2] / f"{digest}.png"
    assert path.read_bytes() == PNG
    assert path.stat().st_mode & 0o777 == 0o600
    for d in (chat_dir / "assets", path.parent):
        assert d.stat().st_mode & 0o777 == 0o700
    assert store.read_asset(ref) == PNG
    # Same bytes again: same file, nothing rewritten.
    ino = path.stat().st_ino
    assert store.put_asset(PNG, "image/png") == ref
    assert path.stat().st_ino == ino


def test_unknown_or_hostile_media_type_never_reaches_the_file_name(store):
    ref = store.put_asset(b"x", "../../etc/passwd")
    assert ref["type"] == "application/octet-stream"
    assert ref["file"] == hashlib.sha256(b"x").hexdigest() + ".bin"


def test_tampered_asset_is_detected(store):
    ref = store.put_asset(PNG, "image/png")
    path = store.asset_path(ref)
    os.chmod(path, 0o600)
    path.write_bytes(b"evil")
    with pytest.raises(AssetCorrupt):
        store.read_asset(ref)
    with pytest.raises(AssetCorrupt):
        store.put_asset(PNG, "image/png")  # never "dedupe" against a wrong file


@pytest.mark.parametrize("ref", [
    {"sha256": "../" + "a" * 61, "bytes": 1, "file": "../" + "a" * 61 + ".png"},
    {"sha256": "a" * 64, "bytes": 1, "file": "b" * 64 + ".png"},          # name != digest
    {"sha256": "a" * 64, "bytes": 1, "file": "a" * 64 + ".png/../../x"},
    {"sha256": "A" * 64, "bytes": 1, "file": "A" * 64 + ".png"},
    {"sha256": "a" * 64, "bytes": -1, "file": "a" * 64 + ".png"},
    "a" * 64,
])
def test_invalid_references_are_refused(store, ref):
    with pytest.raises(AssetCorrupt):
        store.asset_path(ref)
    with pytest.raises(AssetCorrupt):
        store.read_asset(ref)


def test_symlinked_asset_folder_is_refused(store, chat_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, chat_dir / "assets")
    with pytest.raises(AssetCorrupt):
        store.put_asset(PNG, "image/png")
    assert list(outside.iterdir()) == []


def test_symlinked_shard_and_asset_file_are_refused(store, chat_dir, tmp_path):
    digest = hashlib.sha256(PNG).hexdigest()
    secret = tmp_path / "secret"
    secret.write_bytes(PNG)
    (chat_dir / "assets" / digest[:2]).mkdir(parents=True)
    os.symlink(secret, chat_dir / "assets" / digest[:2] / f"{digest}.png")
    with pytest.raises(AssetCorrupt):
        store.put_asset(PNG, "image/png")  # a planted link is not "our" file
    other = b"other"
    od = hashlib.sha256(other).hexdigest()
    if od[:2] != digest[:2]:
        os.symlink(tmp_path, chat_dir / "assets" / od[:2])
        with pytest.raises(AssetCorrupt):
            store.put_asset(other, "text/plain")


def test_crash_leftovers_are_cleaned_and_orphans_kept_on_reopen(chat_dir):
    s = Store(chat_dir)
    s.open()
    ref = s.put_asset(PNG, "image/png")  # committed sidecar whose ROOT line never came
    part = chat_dir / "assets" / ".tmp" / "deadbeef.0.part"
    part.write_bytes(b"half a write")
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    try:
        assert not part.exists()
        assert s2.read_asset(ref) == PNG  # append-only: never deleted, reused on retry
        assert s2.orphan_assets() == [ref["file"]]
        assert s2.messages == []
    finally:
        s2.close()


def _image_archive():
    import base64
    from optchat.store import Archive, Blob
    digest = hashlib.sha256(PNG).hexdigest()
    original = [{"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}]
    skeleton = [{"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": {"$optchat": "asset", "sha256": digest,
                                                            "enc": "data-url", "prefix": "data:image/png;base64,"}}}]
    return original, Archive(skeleton, (Blob(PNG, "image/png"),))


def test_archived_message_round_trips_the_exact_original(chat_dir):
    original, archive = _image_archive()
    s = Store(chat_dir)
    s.open()
    i = s.append_message("user", "look\n[attachment 1: image/png ...]", src="u1", archive=archive)
    rec = s.messages[i]
    payload = original[1]["image_url"]["url"].split(",", 1)[1]
    assert payload[:40] not in json.dumps(rec)  # the payload lives only in the sidecar
    assert rec["assets"][0]["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert s.original(i) == original
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    try:
        assert s2.original(i) == original
        assert s2.orphan_assets() == []
    finally:
        s2.close()


def test_text_only_records_are_unchanged(chat_dir):
    s = Store(chat_dir)
    s.open()
    try:
        s.append_message("user", "hello")
        assert set(s.messages[0]) == {"i", "kind", "text", "size", "date", "seq"}
        assert s.original(0) == "hello"
        assert not (chat_dir / "assets").exists()
    finally:
        s.close()


def test_failed_asset_write_commits_no_root_line(chat_dir, monkeypatch):
    _, archive = _image_archive()
    s = Store(chat_dir)
    s.open()
    monkeypatch.setattr(s, "put_asset", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left")))
    with pytest.raises(OSError):
        s.append_message("user", "look", src="u1", archive=archive)
    assert s.messages == [] and not s.has_src("u1")
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    try:
        assert s2.messages == []
        monkeypatch.undo()
        assert s2.append_message("user", "look", src="u1", archive=archive) == 0  # retry succeeds
    finally:
        s2.close()


def test_skeleton_must_only_reference_supplied_blobs(store):
    from optchat.store import Archive
    bad = Archive({"$optchat": "asset", "sha256": "a" * 64, "enc": "utf8"}, ())
    with pytest.raises(ValueError):
        store.append_message("echo", "x", archive=bad)
    assert store.messages == []


def test_record_with_malformed_asset_reference_is_not_trusted(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "a")
    s.close()
    f = next((chat_dir / "main").glob("*.jsonl"))
    with open(f, "a") as fh:
        fh.write(json.dumps({"i": 1, "kind": "user", "text": "b", "size": 7, "date": "x", "seq": 9,
                             "assets": [{"sha256": "../x", "bytes": 1, "file": "../../x"}], "orig": "b"}) + "\n")
    s2 = Store(chat_dir)
    s2.open()
    try:
        assert len(s2.messages) == 1  # skipped like any invalid line
    finally:
        s2.close()
