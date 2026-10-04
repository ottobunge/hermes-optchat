import pytest

from optchat.store import Store
from optchat.tools import date, zoom


@pytest.fixture
def store(chat_dir):
    s = Store(chat_dir)
    s.open()
    for k in range(4):
        s.append_message("user" if k % 2 == 0 else "talk", f"message {k}\nline two")
    for k in range(4):
        s.append_node(0, k, f"sum{k}")
    s.append_node(1, 0, "sum01")
    s.append_node(1, 1, "sum23")
    s.append_node(2, 0, "sum0123")
    yield s
    s.close()


def test_zoom_one_gives_the_whole_original_message(store):
    assert zoom(store, 1, 1) == "1+0|talk: message 1\nline two"


def test_zoom_opens_a_line_into_its_two_children(store):
    assert zoom(store, 0, 4) == "0+2|sum01\n2+2|sum23"
    assert zoom(store, 2, 2) == "2+1|sum2\n3+1|sum3"


@pytest.mark.parametrize("args", [(1, 2), (0, 3), (0, 8), (4, 1), (-1, 1), (0, 0), ("0", 1), (True, 1), (1.0, 1)])
def test_zoom_validates(store, args):
    assert zoom(store, *args) == f"No line {args[0]}+{args[1]}."


def test_zoom_on_children_not_built(store, chat_dir):
    store.append_message("user", "x")
    store.append_message("user", "y")
    # 4..5: level-0 nodes not built -> the line cannot be opened
    assert zoom(store, 4, 2) == "No line 4+2."


def test_date_returns_local_time(store):
    out = date(store, 2)
    assert out == store.messages[2]["date"]
    assert date(store, 99) == "No message 99."
    assert date(store, "x") == "No message x."


def test_zoom_one_on_media_names_the_archived_files(chat_dir):
    import base64
    import hashlib
    import re
    from optchat.events import events_from_message
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
    s = Store(chat_dir)
    s.open()
    try:
        s.append_message("user", "plain")
        content = [{"type": "text", "text": "look"},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
        [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
        i = s.append_message(ev.kind, ev.text, archive=ev.archive)
        out = zoom(s, i, 1)
        assert out.startswith(f"{i}+0|user: {ev.text}\n")
        digest = hashlib.sha256(png).hexdigest()
        m = re.search(rf"^- image/png, {len(png)} bytes, sha256 {digest}: (/\S+)$", out, re.M)
        assert m, out
        with open(m.group(1), "rb") as fh:
            assert fh.read() == png  # a concrete file a vision tool can open
        assert base64.b64encode(png).decode()[:16] not in out
        assert zoom(s, 0, 1) == "0+0|user: plain"  # text-only output unchanged
    finally:
        s.close()


def test_zoom_one_on_capped_tool_result_points_to_the_full_text(chat_dir):
    import re
    from optchat.events import events_from_message
    s = Store(chat_dir)
    s.open()
    try:
        huge = "R" * 70_000
        [ev] = events_from_message({"role": "tool", "content": huge}, fallback="f")
        s.append_message(ev.kind, ev.text, archive=ev.archive)
        out = zoom(s, 0, 1)
        m = re.search(r"^- text/plain, 70000 bytes, sha256 [0-9a-f]{64}: (/\S+)$", out, re.M)
        assert m and open(m.group(1)).read() == huge
        assert len(out) < 31_000
    finally:
        s.close()


def test_zoom_reports_a_missing_archive_file_instead_of_pretending(chat_dir):
    from optchat.events import events_from_message
    s = Store(chat_dir)
    s.open()
    try:
        [ev] = events_from_message({"role": "tool", "content": "Q" * 40_000}, fallback="f")
        s.append_message(ev.kind, ev.text, archive=ev.archive)
        s.asset_path(s.messages[0]["assets"][0]).unlink()
        assert "MISSING" in zoom(s, 0, 1)
    finally:
        s.close()
