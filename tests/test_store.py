from optchat.store import Store


def test_private_permissions_for_new_log_directory(chat_dir):
    s = Store(chat_dir)
    s.open()
    try:
        s.append_message("user", "a private instruction")
        for path in (chat_dir, chat_dir / "main", chat_dir / "tree"):
            assert path.stat().st_mode & 0o777 == 0o700
        assert (chat_dir / "lock").stat().st_mode & 0o777 == 0o600
    finally:
        s.close()


def test_append_message_persists_and_reloads(chat_dir):
    s = Store(chat_dir)
    s.open()
    assert s.append_message("user", "hello") == 0
    assert s.append_message("talk", "hi there") == 1
    s.close()

    s2 = Store(chat_dir)
    s2.open()
    assert len(s2.messages) == 2
    m = s2.messages[1]
    assert (m["i"], m["kind"], m["text"]) == (1, "talk", "hi there")
    assert m["size"] == len("talk: hi there".encode())
    assert "date" in m
    s2.close()


def test_append_node_persists_and_reloads(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "a")
    s.append_node(0, 0, "user: a")
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    assert s2.node(0, 0) == "user: a"
    assert s2.node(1, 0) is None
    s2.close()


def test_torn_line_is_skipped_reported_and_newline_repaired(chat_dir, caplog):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "first")
    s.close()
    day = next((chat_dir / "main").glob("*.jsonl"))
    with open(day, "ab") as f:
        f.write(b'{"i": 1, "kind": "user", "te')  # crash mid-write
    s2 = Store(chat_dir)
    with caplog.at_level("WARNING"):
        s2.open()
    assert [m["text"] for m in s2.messages] == ["first"]
    assert any("torn" in r.message.lower() for r in caplog.records)
    assert s2.append_message("user", "second") == 1
    s2.close()
    s3 = Store(chat_dir)
    s3.open()
    assert [m["text"] for m in s3.messages] == ["first", "second"]
    s3.close()


def test_records_with_wrong_shape_are_skipped(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "ok")
    s.close()
    day = next((chat_dir / "main").glob("*.jsonl"))
    with open(day, "a") as f:
        f.write('[1, 2]\n{"i": "x"}\n')
    s2 = Store(chat_dir)
    s2.open()
    assert len(s2.messages) == 1
    s2.close()


def test_second_writer_is_refused_until_first_closes(chat_dir):
    import pytest
    from optchat.store import StoreLocked

    s = Store(chat_dir)
    s.open()
    other = Store(chat_dir)
    with pytest.raises(StoreLocked):
        other.open()
    s.close()
    other.open()  # stale lock released by the OS when the owner closes/dies
    other.close()


def test_lock_held_by_another_process(chat_dir):
    import subprocess
    import sys

    s = Store(chat_dir)
    s.open()
    code = ("import sys; sys.path.insert(0, %r); from optchat.store import Store, StoreLocked\n"
            "try:\n Store(%r).open(); print('opened')\nexcept StoreLocked: print('locked')"
            % (str(__import__('pathlib').Path(__file__).parents[1]), str(chat_dir)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "locked"
    s.close()


def test_gap_in_message_ids_fails_closed(chat_dir):
    import pytest
    from optchat.store import StoreCorrupt

    s = Store(chat_dir)
    s.open()
    s.append_message("user", "a")
    s.close()
    day = next((chat_dir / "main").glob("*.jsonl"))
    with open(day, "a") as f:
        f.write('{"i": 5, "kind": "user", "text": "x", "size": 7, "date": "d"}\n')
    with pytest.raises(StoreCorrupt):
        Store(chat_dir).open()


def test_duplicate_ids_keep_first(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "a")
    s.append_node(0, 0, "user: a")
    s.close()
    day = next((chat_dir / "main").glob("*.jsonl"))
    with open(day, "a") as f:
        f.write('{"i": 0, "kind": "user", "text": "dup", "size": 9, "date": "d"}\n')
    tday = next((chat_dir / "tree").glob("*.jsonl"))
    with open(tday, "a") as f:
        f.write('{"l": 0, "i": 0, "text": "dup", "size": 3}\n')
    s2 = Store(chat_dir)
    s2.open()
    assert [m["text"] for m in s2.messages] == ["a"]
    assert s2.node(0, 0) == "user: a"
    s2.close()


def test_writes_require_the_lock(chat_dir):
    import pytest

    s = Store(chat_dir)
    with pytest.raises(RuntimeError):
        s.append_message("user", "no lock")
    s.open()
    s.close()
    with pytest.raises(RuntimeError):
        s.append_node(0, 0, "x")


def test_unknown_kind_rejected(chat_dir):
    import pytest

    s = Store(chat_dir)
    s.open()
    with pytest.raises(ValueError):
        s.append_message("thought", "never logged")
    s.close()


def test_source_keys_are_durable_for_dedupe(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "hi", src="uid:abc")
    assert s.has_src("uid:abc")
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    assert s2.has_src("uid:abc") and not s2.has_src("uid:zzz")
    assert s2.src_index("uid:abc") == 0
    s2.close()


def test_messages_and_nodes_share_one_append_only_sequence(chat_dir):
    s = Store(chat_dir)
    s.open()
    s.append_message("user", "a")
    s.append_message("user", "b")
    s.append_node(0, 1, "user: b")
    s.append_node(0, 0, "user: a")
    s.close()
    s2 = Store(chat_dir)
    s2.open()
    try:
        assert [m["seq"] for m in s2.messages] == [0, 1]
        assert s2.nodes[(0, 1)]["seq"] == 2 and s2.nodes[(0, 0)]["seq"] == 3
        s2.append_message("user", "c")  # continues after the highest persisted seq
        assert s2.messages[2]["seq"] == 4
        assert s2.events() == [("message", 0), ("message", 1), ("node", 0, 1),
                               ("node", 0, 0), ("message", 2)]
    finally:
        s2.close()


def test_legacy_lines_without_seq_replay_before_sequenced_ones(chat_dir):
    import json
    (chat_dir / "main").mkdir(parents=True)
    (chat_dir / "tree").mkdir(parents=True)
    (chat_dir / "main" / "2020-01-01.jsonl").write_text(
        json.dumps({"i": 0, "kind": "user", "text": "a"}) + "\n"
        + json.dumps({"i": 1, "kind": "user", "text": "b"}) + "\n")
    (chat_dir / "tree" / "2020-01-01.jsonl").write_text(
        json.dumps({"l": 0, "i": 1, "text": "user: b"}) + "\n")
    s = Store(chat_dir)
    s.open()
    try:
        s.append_node(0, 0, "user: a")
        # legacy nodes first (the old fold saw them all), then legacy messages in order
        assert s.events() == [("node", 0, 1), ("message", 0), ("message", 1), ("node", 0, 0)]
        assert s.nodes[(0, 0)]["seq"] == 0
    finally:
        s.close()
