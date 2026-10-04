import json
import os

import pytest

from optchat.events import CAP, cap_echo, events_from_message


def test_cap_keeps_short_results_whole():
    assert cap_echo("abc") == "abc"
    assert cap_echo("x" * CAP) == "x" * CAP


def test_cap_keeps_head_and_tail_with_indicator():
    text = "H" * 20_000 + "M" * 20_000 + "T" * 20_000
    out = cap_echo(text)
    assert len(out) <= CAP
    assert out.startswith("H" * 10_000)
    assert out.endswith("T" * 10_000)
    assert "characters cut" in out
    cut = int(out.split("[... ")[1].split(" ")[0].replace(",", ""))
    head = len(out.split("\n[... ")[0])
    tail = len(out.split(" characters cut ...]\n")[1])
    assert head + tail + cut == len(text)


def test_user_message_with_uid():
    ev = events_from_message({"role": "user", "content": "hello", "message_uid": "u1"}, fallback="f")
    assert ev == [("user", "hello", "u1", None)]


def test_content_that_cannot_be_archived_losslessly_is_still_rejected():
    import pytest
    from optchat.events import UnsupportedContent
    msg = {"role": "user", "content": [{"type": "text", "text": "look"},
                                       {"type": "mystery", "blob": b"raw bytes object"}]}
    with pytest.raises(UnsupportedContent, match="mystery"):
        events_from_message(msg, fallback="f")


def test_assistant_reply_and_tool_calls_never_reasoning():
    msg = {"role": "assistant", "content": "Let me check.", "message_uid": "a1",
           "reasoning": "secret thoughts", "reasoning_content": "more thoughts",
           "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "read_file", "arguments": "{\"path\": \"x.py\"}"}}]}
    ev = events_from_message(msg, fallback="f")
    assert ev == [("talk", "Let me check.", "a1", None),
                  ("tool", 'read_file {"path": "x.py"}', "a1#tool:0", None)]
    assert all("thought" not in e.text for e in ev)


def test_assistant_list_content_drops_thinking_blocks():
    msg = {"role": "assistant", "message_uid": "a2",
           "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Done."}]}
    assert events_from_message(msg, fallback="f") == [("talk", "Done.", "a2", None)]


def test_thinking_only_assistant_logs_nothing():
    msg = {"role": "assistant", "content": "", "reasoning": "only thoughts", "message_uid": "a3"}
    assert events_from_message(msg, fallback="f") == []


def test_tool_result_is_capped_echo():
    msg = {"role": "tool", "tool_call_id": "c1", "content": "R" * 40_000, "message_uid": "t1"}
    [(kind, text, src, _)] = events_from_message(msg, fallback="f")
    assert kind == "echo" and len(text) <= CAP and src == "t1"


def test_system_messages_are_not_logged():
    assert events_from_message({"role": "system", "content": "sys"}, fallback="f") == []


def test_lone_surrogates_are_replaced():
    [(_, text, _, _)] = events_from_message({"role": "user", "content": "a\ud800b"}, fallback="f")
    assert text == "a�b"
    text.encode("utf-8")


PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


def _b64(data):
    import base64
    return base64.b64encode(data).decode()


def _round_trip(chat_dir, ev):
    from optchat.store import Store
    s = Store(chat_dir)
    s.open()
    try:
        i = s.append_message(ev.kind, ev.text, src=ev.src, archive=ev.archive)
        return s.original(i), s.messages[i]
    finally:
        s.close()


def test_text_only_events_carry_no_archive():
    [ev] = events_from_message({"role": "user", "content": [{"type": "text", "text": "a"},
                                                            {"type": "text", "text": "b"}]}, fallback="f")
    assert ev == ("user", "a\nb", "f", None)


def test_image_data_url_is_archived_and_described_never_faked(chat_dir):
    import hashlib
    from optchat.store import Blob
    content = [{"type": "text", "text": "look"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(PNG), "detail": "high"}}]
    [ev] = events_from_message({"role": "user", "content": content, "message_uid": "u1"}, fallback="f")
    assert (ev.kind, ev.src) == ("user", "u1")
    digest = hashlib.sha256(PNG).hexdigest()
    assert ev.text == (f"look\n[attachment 1 not shown: image/png, {len(PNG)} bytes, sha256 {digest[:16]}; "
                       f"original archived, zoom this message (n = 1) for the file]")
    assert "[image]" not in ev.text and _b64(PNG)[:16] not in ev.text
    assert ev.archive.blobs == (Blob(PNG, "image/png"),)
    original, rec = _round_trip(chat_dir, ev)
    assert original == content
    assert _b64(PNG)[:16] not in json.dumps(rec)


def test_remote_url_is_kept_verbatim_and_disclosed_as_mutable(chat_dir):
    url = "https://example.com/cat.png?sig=abc&x=1"
    content = [{"type": "image_url", "image_url": {"url": url}}]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert url in ev.text and "only the URL is kept" in ev.text and "may change or disappear" in ev.text
    assert ev.archive.blobs == ()
    assert _round_trip(chat_dir, ev)[0] == content


def test_non_canonical_base64_is_kept_verbatim_as_text(chat_dir):
    url = "data:image/png;base64," + _b64(PNG)[:40] + "\n" + _b64(PNG)[40:]
    content = [{"type": "image_url", "image_url": {"url": url}}]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert "kept verbatim as text" in ev.text and _b64(PNG)[:16] not in ev.text
    assert _round_trip(chat_dir, ev)[0] == content


@pytest.mark.parametrize("part, mime, payload", [
    ({"type": "input_audio", "input_audio": {"data": _b64(b"RIFFwav"), "format": "wav"}}, "audio/wav", b"RIFFwav"),
    ({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(b"\xff\xd8jpg")},
      "cache_control": {"type": "ephemeral"}}, "image/jpeg", b"\xff\xd8jpg"),
    ({"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64," + _b64(b"%PDF-1")}},
     "application/pdf", b"%PDF-1"),
    ({"type": "input_file", "filename": "notes.pdf", "file_data": _b64(b"%PDF-2")}, "application/pdf", b"%PDF-2"),
    ({"type": "input_image", "image_url": "data:image/webp;base64," + _b64(b"RIFFwebp")}, "image/webp", b"RIFFwebp"),
])
def test_inline_payload_shapes_round_trip_exactly(chat_dir, part, mime, payload):
    content = [{"type": "text", "text": "see"}, part]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert [(b.data, b.mime) for b in ev.archive.blobs] == [(payload, mime)]
    assert _b64(payload) not in ev.text and mime in ev.text
    original, rec = _round_trip(chat_dir, ev)
    assert original == content
    assert _b64(payload) not in json.dumps(rec)


def test_unknown_part_is_archived_whole_as_json(chat_dir):
    part = {"type": "hologram", "data": "A" * 5000, "meta": {"k": [1, 2.5, None, True]}}
    [ev] = events_from_message({"role": "user", "content": [part]}, fallback="f")
    assert "'hologram'" in ev.text and "archived as JSON" in ev.text and "A" * 100 not in ev.text
    original, rec = _round_trip(chat_dir, ev)
    assert original == [part] and "A" * 100 not in json.dumps(rec)


def test_provider_file_id_is_disclosed_not_invented():
    [ev] = events_from_message({"role": "user", "content": [{"type": "input_image", "file_id": "file-123"}]},
                               fallback="f")
    assert "provider file id file-123" in ev.text and "not archived" in ev.text
    assert ev.archive.blobs == ()


def test_thoughts_are_dropped_even_from_the_archive(chat_dir):
    content = [{"type": "thinking", "thinking": "secret"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(PNG)}}]
    [ev] = events_from_message({"role": "assistant", "content": content}, fallback="f")
    assert ev.kind == "talk"
    original, rec = _round_trip(chat_dir, ev)
    assert original == content[1:] and "secret" not in json.dumps(rec)


def test_non_canonical_bare_base64_audio_is_kept_verbatim(chat_dir):
    content = [{"type": "input_audio", "input_audio": {"data": "not base64!", "format": "mp3"}}]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert "audio/mp3 payload kept verbatim as text" in ev.text
    assert _round_trip(chat_dir, ev)[0] == content


@pytest.mark.parametrize("uri", [False, True])
def test_message_local_path_is_not_permission_to_read(chat_dir, tmp_path, monkeypatch, uri):
    private = tmp_path / "hosts.png"
    private.write_bytes(b"private hosts-like file")
    url = private.as_uri() if uri else str(private)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("untrusted message requested a local file read")
    content = [{"type": "image_url", "image_url": {"url": url}}]
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", forbidden)
        [ev] = events_from_message({"role": "assistant", "content": content}, fallback="f")
    assert ev.archive.blobs == ()
    assert "not host-authorized" in ev.text
    assert _round_trip(chat_dir, ev)[0] == content


def test_untrusted_file_url_does_not_archive_local_audio(tmp_path):
    wav = tmp_path / "note.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    part = {"type": "file", "file": {"filename": "note.wav", "file_url": wav.as_uri()}}
    [ev] = events_from_message({"role": "user", "content": [part]}, fallback="f")
    assert ev.archive.blobs == ()
    assert "not host-authorized" in ev.text


def test_inline_wav_file_uses_the_same_canonical_mime():
    content = [{"type": "input_file", "filename": "note.wav", "file_data": _b64(b"RIFF....WAVEfmt ")}]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert [(b.data, b.mime) for b in ev.archive.blobs] == [(b"RIFF....WAVEfmt ", "audio/x-wav")]


def test_missing_or_nonregular_local_message_path_is_never_opened(tmp_path):
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    content = [{"type": "image_url", "image_url": {"url": str(tmp_path / "gone.png")}},
               {"type": "image_url", "image_url": {"url": str(fifo)}}]
    [ev] = events_from_message({"role": "user", "content": content}, fallback="f")
    assert ev.archive.blobs == ()
    assert ev.text.count("not host-authorized") == 2


def test_long_tool_result_is_capped_for_the_model_but_archived_whole(chat_dir):
    huge = "H" * 20_000 + "M" * 20_000 + "T" * 20_000
    [ev] = events_from_message({"role": "tool", "content": huge, "message_uid": "t1"}, fallback="f")
    assert ev.text == cap_echo(huge) and len(ev.text) <= CAP
    original, rec = _round_trip(chat_dir, ev)
    assert original == huge
    assert "M" * 1000 not in json.dumps(rec)  # the cut middle lives only in the sidecar
    [short] = events_from_message({"role": "tool", "content": "x" * CAP}, fallback="f")
    assert short.archive is None  # results within the cap are logged exactly as before


def test_multipart_tool_echo_capped_in_aggregate_round_trips_after_restart(chat_dir):
    from optchat.store import Store

    content = [{"type": "text", "text": "A" * 20_000, "metadata": {"path": ["a", "b"]}},
               {"type": "output_text", "text": "B" * 20_000}]
    [ev] = events_from_message({"role": "tool", "content": content}, fallback="f")
    assert len(ev.text) <= CAP and "characters cut" in ev.text
    assert ev.archive is not None

    store = Store(chat_dir)
    store.open()
    try:
        store.append_message(ev.kind, ev.text, src=ev.src, archive=ev.archive)
    finally:
        store.close()
    restarted = Store(chat_dir)
    restarted.open()
    try:
        assert restarted.original(0) == content
        assert "B" * 1000 not in json.dumps(restarted.messages[0]["orig"])
    finally:
        restarted.close()


def test_multipart_tool_echo_with_media_preserves_raw_and_nested_parts(chat_dir):
    from optchat.store import Store

    payload = PNG * 16
    data_url = "data:image/png;base64," + _b64(payload)
    content = ["H" * 20_000,
               {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
               {"type": "output_text", "text": "T" * 20_000, "metadata": {"nested": [1, {"ok": True}]}}]
    [ev] = events_from_message({"role": "tool", "content": content}, fallback="f")
    note = ev.text[ev.text.index("\n[attachment 1 not shown:") + 1:]
    assert len(ev.text) <= CAP + len(note) + 1
    assert "characters cut" in ev.text and "image/png" in note
    assert _b64(payload) not in ev.text

    store = Store(chat_dir)
    store.open()
    try:
        store.append_message(ev.kind, ev.text, archive=ev.archive)
    finally:
        store.close()
    restarted = Store(chat_dir)
    restarted.open()
    try:
        assert restarted.original(0) == content
        root = restarted.messages[0]
        assert _b64(payload) not in json.dumps(root)
        assert "T" * 1000 not in json.dumps(root["orig"])
    finally:
        restarted.close()


def test_tool_result_with_screenshot_keeps_the_note_when_text_is_capped(chat_dir):
    content = [{"type": "text", "text": "A" * 50_000},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(PNG)}},
               {"type": "text", "text": "done"}]
    [ev] = events_from_message({"role": "tool", "content": content}, fallback="f")
    assert ev.kind == "echo"
    assert "characters cut" in ev.text and "[attachment 1 not shown: image/png" in ev.text
    assert len(ev.text) <= CAP + 300
    original, rec = _round_trip(chat_dir, ev)
    assert original == content and "A" * 1000 not in json.dumps(rec["orig"])
