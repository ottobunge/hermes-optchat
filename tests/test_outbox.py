"""OptChat as the authoritative ROOT consumer of the host's durable event outbox.

Uses the real core ``SessionDB`` (``hermes_state``) with temporary databases and chat
directories; summaries come from fakes. No user sessions, no model calls.
"""

import json
import base64

import pytest

from fakes import FakeSummarizer
from optchat.events import CAP

hermes_state = pytest.importorskip("hermes_state")
if not hasattr(hermes_state.SessionDB, "read_durable_events"):
    pytest.skip("host without the durable event outbox", allow_module_level=True)

from optchat import engine as engine_mod  # noqa: E402
from optchat.engine import OptChatEngine, shutdown_all  # noqa: E402
from optchat.outbox import STREAM  # noqa: E402

SessionDB = hermes_state.SessionDB


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    shutdown_all()


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli")
    yield d
    d.close()


def make_engine(tmp_path, db=None, sid="s1", platform="cli", summarizer=None, **kw):
    e = OptChatEngine(home=tmp_path / "optchat", summarizer=summarizer, settle_timeout=kw.pop("settle", 2), **kw)
    if db is not None:
        e.bind_session_state(session_db=db, session_id=sid)
    e.on_session_start(sid, platform=platform)
    return e


def tool_turn():
    return [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "checking", "reasoning": "private thought",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "a.txt"},
    ]


def flush(db, engine, messages, sid="s1", parent=None, notify=True):
    """What the host flush does: one txn with the projector, then the post-commit nudge."""
    db.append_messages_batch(session_id=sid, messages=messages,
                             project_events=lambda rows: engine.project_durable_events(
                                 sid, rows, parent_session_id=parent))
    if notify:
        engine.on_durable_events_committed(sid)
    return messages


def root(engine):
    return [(m["kind"], m["text"]) for m in engine._memory().store.messages]


# -- projector -----------------------------------------------------------------------

def test_projector_is_pure_and_maps_each_row_without_reasoning(tmp_path, db):
    e = make_engine(tmp_path, db)
    flush(db, e, tool_turn(), notify=False)
    assert not (tmp_path / "optchat").exists()  # the projector wrote nothing
    events = db.read_durable_events(STREAM)
    assert [ev["event"]["role"] for ev in events] == ["user", "assistant", "tool"]
    assert "private thought" not in json.dumps(events)
    user, asst, tool = (ev["event"] for ev in events)
    assert user["content"] == "list files" and user["session_id"] == "s1"
    assert [ev["key"] for ev in events] == [user["message_uid"], asst["message_uid"], tool["message_uid"]]
    assert asst["tool_calls"][0]["function"]["name"] == "terminal"
    assert asst["tool_call_uids"]["call_1"] == tool["tool_call_uid"]
    assert tool["tool_call_id"] == "call_1" and tool["tool_name"] == "terminal"
    assert isinstance(user["timestamp"], float)
    assert user["content_fidelity"] == "host-text"  # media is not in the outbox yet


def test_projector_is_deterministic(tmp_path):
    e = make_engine(tmp_path)
    rows = [{"role": "user", "content": "hi", "message_uid": "u1", "timestamp": 1.0, "_row_id": 1}]
    assert e.project_durable_events("s1", rows) == e.project_durable_events("s1", rows)


def test_real_host_multimodal_flush_archives_original_media_without_thoughts(tmp_path, db):
    from run_agent import AIAgent

    e = make_engine(tmp_path, db)
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, session_db=db, session_id="s1")
    agent._session_db_created = True
    agent.context_compressor = e
    payload = b"\x89PNG\r\n\x1a\nimage-bytes"
    data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    content = [{"type": "text", "text": "look"},
               {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}]
    messages = [{"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "private thought"},
                                                   {"type": "text", "text": "Seen."}]}]
    assert agent._flush_messages_to_session_db(messages, None) is True

    events = db.read_durable_events(STREAM)
    assert events[0]["event"]["content"] == content
    assert events[0]["event"]["content_fidelity"] == "raw"
    assert "private thought" not in json.dumps(events)
    store = e._memory().store
    assert store.original(0) == content
    assert store.messages[0]["kind"] == "user"
    assert "attachment 1 not shown: image/png" in store.messages[0]["text"]
    assert data_url not in json.dumps(store.messages)
    assert store.read_asset(store.messages[0]["assets"][0]) == payload
    assert root(e)[-1] == ("talk", "Seen.")


def test_projector_excludes_thought_parts_even_if_raw_row_contains_them(tmp_path):
    e = make_engine(tmp_path)
    raw = [{"type": "reasoning", "text": "private thought"},
           {"type": "redacted_thinking", "data": "private thought"},
           {"type": "thinking", "thinking": "private thought"},
           {"type": "text", "text": "public"}]
    row = {"role": "assistant", "content": "public", "raw_content": raw, "message_uid": "a1"}
    [projected] = e.project_durable_events("s1", [row])
    assert projected["event"]["content"] == raw[-1:]
    assert "private thought" not in json.dumps(projected)
    assert row["raw_content"] == raw  # pure projection, including on retries


def test_projector_does_not_copy_unused_provider_api_content(tmp_path):
    e = make_engine(tmp_path)
    rows = [{"role": "assistant", "content": "public", "api_content": "private thought",
             "message_uid": "a1"}]
    [projected] = e.project_durable_events("s1", rows)
    assert projected["event"]["content"] == "public"
    assert "private thought" not in json.dumps(projected)


@pytest.mark.parametrize("bad_part", [
    {"type": "mystery", "data": b"not JSON"},
    {"type": "mystery", "data": float("nan")},
    {"type": "mystery", "data": (1, 2)},
])
def test_projector_rejects_raw_parts_that_cannot_round_trip_through_json(tmp_path, bad_part):
    e = make_engine(tmp_path)
    row = {"role": "user", "content": "[unavailable]", "raw_content": [bad_part], "message_uid": "u1"}
    with pytest.raises(ValueError, match="JSON"):
        e.project_durable_events("s1", [row])


@pytest.mark.parametrize("raw", [None, {"type": "image_url"}, b"binary"])
def test_projector_does_not_silently_fall_back_when_raw_content_is_malformed(tmp_path, raw):
    e = make_engine(tmp_path)
    row = {"role": "user", "content": "[screenshot]", "raw_content": raw, "message_uid": "u1"}
    with pytest.raises(ValueError, match="raw_content"):
        e.project_durable_events("s1", [row])


def test_delegated_child_projects_no_internal_trace(tmp_path, db):
    db.create_session(session_id="child", source="subagent", parent_session_id="s1")
    child = make_engine(tmp_path, db, sid="child", platform="subagent")
    flush(db, child, tool_turn(), sid="child", parent="s1")
    assert db.read_durable_events(STREAM) == []
    other = make_engine(tmp_path, db, sid="child2")
    rows = [{"role": "user", "content": "x", "message_uid": "c1", "timestamp": 1.0}]
    # A parent link alone also belongs to user-visible branches and continuations.
    assert [event["event"]["content"] for event in
            other.project_durable_events("child2", rows, parent_session_id="s1")] == ["x"]

def test_branched_session_with_parent_contributes_only_new_turns(tmp_path, db):
    parent = make_engine(tmp_path, db)
    flush(db, parent, [U("p1", "original")])
    db.create_session("branch", source="desktop", parent_session_id="s1",
                      model_config={"_branched_from": "s1"})
    from run_agent import AIAgent
    branch = make_engine(tmp_path, db, sid="branch", platform="desktop")
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, session_db=db, session_id="branch",
                    parent_session_id="s1", platform="desktop")
    agent.context_compressor = branch
    agent._session_db_created = True
    # Copied history is a seed, not a fresh branch event. The host flush adds only its new turn.
    assert agent._flush_messages_to_session_db([U("b1", "new branch turn")], None) is True
    assert root(parent) == [("user", "original"), ("user", "new branch turn")]

def test_source_subagent_excludes_child_even_with_non_subagent_platform(tmp_path):
    child = make_engine(tmp_path, sid="child", platform="desktop")
    child.on_session_start("child", platform="desktop", session_source="subagent")
    assert child.project_durable_events("child", [U("c1", "internal task")],
                                        parent_session_id="s1") == []


# -- draining into ROOT ----------------------------------------------------------------

def test_committed_events_are_drained_into_root_in_order(tmp_path, db):
    e = make_engine(tmp_path, db)
    flush(db, e, tool_turn())
    assert root(e) == [("user", "list files"), ("talk", "checking"), ("tool", "terminal {}"), ("echo", "a.txt")]
    events = db.read_durable_events(STREAM)
    msgs = e._memory().store.messages
    assert [m["src"] for m in msgs] == [events[0]["key"], events[1]["key"], events[1]["key"] + "#tool:0",
                                        events[2]["key"]]
    assert e._consumer().cursor == events[-1]["seq"]
    e.on_durable_events_committed("s1")  # a duplicate nudge logs nothing twice
    assert len(e._memory().store.messages) == 4


def test_outbox_tool_audio_and_long_text_follow_media_and_echo_cap_policy(tmp_path, db):
    e = make_engine(tmp_path, db)
    audio = b"RIFF....WAVEfmt "
    raw = [{"type": "text", "text": "A" * 40_000},
           {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode("ascii"),
                                                    "format": "wav"}}]
    flush(db, e, [{"role": "tool", "content": "[audio]", "raw_content": raw,
                   "tool_call_id": "c1", "message_uid": "t1"}])

    [event] = db.read_durable_events(STREAM)
    assert event["event"]["content"] == raw and event["event"]["content_fidelity"] == "raw"
    store = e._memory().store
    [rec] = store.messages
    assert rec["kind"] == "echo" and rec["src"] == "t1"
    assert "characters cut" in rec["text"] and len(rec["text"]) <= CAP + 300
    assert "attachment 1 not shown: audio/wav" in rec["text"]
    assert store.original(0) == raw
    assert [ref["type"] for ref in rec["assets"]] == ["text/plain", "audio/wav"]
    assert store.read_asset(rec["assets"][1]) == audio
    assert "A" * 1000 not in json.dumps(rec["orig"])


def test_root_date_is_the_commit_timestamp_not_the_drain_time(tmp_path, db):
    from datetime import datetime

    e = make_engine(tmp_path, db)
    flush(db, e, [{"role": "user", "content": "old", "timestamp": 1_000_000_000.0}])
    assert e._memory().store.messages[0]["date"] == \
        datetime.fromtimestamp(1_000_000_000.0).astimezone().isoformat(timespec="seconds")


def _restart(tmp_path, db, **kw):
    shutdown_all()  # the process died: its writer lock and in-memory state are gone
    return make_engine(tmp_path, db, **kw)


def test_crash_between_root_line_and_checkpoint_replays_once(tmp_path, db, monkeypatch):
    from optchat.outbox import Consumer

    e = make_engine(tmp_path, db)
    real = Consumer._append
    monkeypatch.setattr(Consumer, "_append", lambda self, name, rec: (_ for _ in ()).throw(OSError("crash"))
                        if name == "cursor.jsonl" else real(self, name, rec))
    with pytest.raises(OSError):
        flush(db, e, tool_turn())
    assert root(e) == [("user", "list files")]  # logged, but its checkpoint is not
    monkeypatch.setattr(Consumer, "_append", real)

    e2 = _restart(tmp_path, db)
    e2.on_durable_events_committed("s1")
    assert root(e2) == [("user", "list files"), ("talk", "checking"), ("tool", "terminal {}"), ("echo", "a.txt")]
    assert e2._consumer().cursor == db.read_durable_events(STREAM)[-1]["seq"]


def test_missing_authorized_media_snapshot_never_advances_cursor(tmp_path, db):
    """A host-tagged attachment cannot become a logged-but-missing stand-in."""
    import hashlib

    e = make_engine(tmp_path, db)
    payload = b"original image payload"
    digest = hashlib.sha256(payload).hexdigest()
    content = [{"type": "text", "text": "look"},
               {"type": "image_url", "image_url": {"url": "file:///already-gone/captured.png"}}]
    def corrupt_projector(rows):
        enriched = [dict(row, raw_content=content, authorized_media_sha256=[digest]) for row in rows]
        return e.project_durable_events("s1", enriched)
    db.append_message("s1", "user", content="look", project_events=corrupt_projector)
    (committed,) = db.read_durable_events(STREAM)
    assert committed["event"]["authorized_media_sha256"] == [digest]
    with pytest.raises(Exception, match="authorized media snapshot"):
        e.on_durable_events_committed("s1")
    assert e._consumer().cursor == 0
    assert not e._memory().store.messages
    e2 = _restart(tmp_path, db)
    with pytest.raises(Exception, match="authorized media snapshot"):
        e2.on_durable_events_committed("s1")
    assert e2._consumer().cursor == 0


def test_partially_applied_outbox_event_is_completed_after_restart(tmp_path, db, monkeypatch):
    from optchat.memory import Memory

    e = make_engine(tmp_path, db)
    real = Memory.log

    def dying(self, kind, *a, **k):
        if kind == "tool":  # the assistant row's talk line landed, its tool-call line did not
            raise OSError("crash")
        return real(self, kind, *a, **k)

    monkeypatch.setattr(Memory, "log", dying)
    with pytest.raises(OSError):
        flush(db, e, tool_turn())
    monkeypatch.setattr(Memory, "log", real)
    e2 = _restart(tmp_path, db)
    e2.on_durable_events_committed("s1")
    assert root(e2) == [("user", "list files"), ("talk", "checking"), ("tool", "terminal {}"), ("echo", "a.txt")]


# -- turn admission ------------------------------------------------------------------

def U(uid, text):
    return {"role": "user", "content": text, "message_uid": uid}


def A(uid, text, **kw):
    return dict({"role": "assistant", "content": text, "message_uid": uid}, **kw)


def request_of(conv):
    return [{"role": "system", "content": "SYS"}] + [{k: v for k, v in m.items() if k != "message_uid"}
                                                    for m in conv]


def select(e, conv, incoming=None):
    incoming = incoming if incoming is not None else next(m for m in reversed(conv) if m["role"] == "user")
    return e.select_context(request_of(conv), conversation_messages=conv, incoming_message=incoming)


@pytest.mark.parametrize("notify", [True, False])
def test_view_is_rendered_before_the_current_user_event_is_logged(tmp_path, db, notify):
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", "deploy on fridays is banned"), A("a1", "Noted.")])
    conv += flush(db, e, [U("u2", "what is banned?")], notify=notify)  # the host persists it first
    out = select(e, conv)
    assert out[1]["content"][0]["text"] == ("<chat>\n0+1|user: deploy on fridays is banned\n"
                                            "1+1|talk: Noted.\n</chat>")
    assert out[1]["content"][1]["text"] == "what is banned?"
    assert len(out) == 2
    assert root(e)[-1] == ("user", "what is banned?")  # logged after the view was frozen


class Blocked(Exception):
    """Stands in for the host's ``agent.context_engine.ContextSelectionBlocked``."""


LONG = "x" * 600  # over the free-node size: needs a summarizer


def test_unsettled_view_blocks_the_request_and_keeps_the_user_event_recoverable(tmp_path, db, monkeypatch):
    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", Blocked)
    e = make_engine(tmp_path, db)  # no summarizer: the long message can never be summarized
    conv = flush(db, e, [U("u1", LONG), A("a1", "ok")])
    conv += flush(db, e, [U("u2", "next")])
    with pytest.raises(Blocked):
        select(e, conv)
    assert [k for k, _ in root(e)] == ["user", "talk"]  # u2 was neither shown nor logged

    e2 = _restart(tmp_path, db, summarizer=FakeSummarizer())
    out = select(e2, conv)  # e.g. the host retries the turn once a summarizer is configured
    assert LONG not in out[1]["content"][0]["text"]
    assert root(e2)[-1] == ("user", "next")


def test_without_the_host_sentinel_an_unsettled_view_sends_only_a_notice(tmp_path, db, monkeypatch):
    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", None)  # an older host: cannot abort
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", LONG), A("a1", "ok")])
    conv += flush(db, e, [U("u2", "next")])
    out = select(e, conv)
    assert LONG not in json.dumps(out) and "ok" not in json.dumps(out[1:])
    assert "memory is unavailable" in out[1]["content"][0]["text"]
    assert [k for k, _ in root(e)] == ["user", "talk"]


def _tool_round(n):
    call = {"id": f"call_{n}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
    return [A(f"t{n}", "", tool_calls=[call]),
            {"role": "tool", "tool_call_id": f"call_{n}", "name": "terminal", "content": f"out {n}",
             "message_uid": f"r{n}"}]


def test_tool_loop_keeps_the_frozen_view_and_logs_each_round(tmp_path, db):
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", "hi"), A("a1", "hello")])
    conv += flush(db, e, [U("u2", "run it")])
    first = select(e, conv)
    conv += flush(db, e, _tool_round(1))
    out = select(e, conv, incoming=conv[2])
    assert out[1] == first[1]
    assert [m["role"] for m in out[2:]] == ["assistant", "tool"]
    assert root(e)[-2:] == [("tool", "terminal {}"), ("echo", "out 1")]


def test_restart_and_provider_retry_render_the_persisted_frozen_view(tmp_path, db):
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", "hi"), A("a1", "hello")])
    conv += flush(db, e, [U("u2", "run it")])
    first = select(e, conv)
    conv += flush(db, e, _tool_round(1))

    e2 = _restart(tmp_path, db)  # crash mid-turn; the host resumes the same turn
    out = select(e2, conv, incoming=conv[2])
    assert out[1] == first[1]
    assert "run it" not in out[1]["content"][0]["text"] and "out 1" not in out[1]["content"][0]["text"]
    assert len(e2._memory().store.messages) == 5  # hi, hello, run it, tool call, echo


def test_old_live_paths_log_nothing_while_bound_to_the_outbox(tmp_path, db, monkeypatch):
    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", Blocked)
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", "hi"), A("a1", "hello")])
    conv += flush(db, e, [U("u2", "go")])
    select(e, conv)
    conv.append(A("a2", "never committed"))  # live only: not in the transcript, not in the outbox
    e.on_turn_complete(conv)
    conv.append(U("u3", "also not committed"))
    with pytest.raises(Blocked):
        select(e, conv)  # its event is not in the outbox: nothing to admit
    assert [t for _, t in root(e)] == ["hi", "hello", "go"]


# -- several agents, one profile -----------------------------------------------------

def test_sessions_of_one_process_share_one_writer_and_follow_the_global_order(tmp_path, db):
    db.create_session(session_id="s2", source="telegram")
    e1, e2 = make_engine(tmp_path, db, sid="s1"), make_engine(tmp_path, db, sid="s2")
    assert e1._memory() is e2._memory() and e1._consumer() is e2._consumer()
    c1 = flush(db, e1, [U("a1", "from s1")], sid="s1")
    c2 = flush(db, e2, [U("b1", "from s2")], sid="s2")
    v1 = select(e1, c1)[1]["content"][0]["text"]
    v2 = select(e2, c2)[1]["content"][0]["text"]
    assert v1 == "<chat>\n</chat>"
    assert v2 == "<chat>\n0+1|user: from s1\n</chat>"
    assert [t for _, t in root(e1)] == ["from s1", "from s2"]


def test_concurrent_flushes_are_logged_once_in_commit_order(tmp_path, db):
    import threading

    db.create_session(session_id="s2", source="cli")
    engines = {sid: make_engine(tmp_path, db, sid=sid) for sid in ("s1", "s2")}

    def run(sid):
        for k in range(40):
            flush(db, engines[sid], [A(f"{sid}-{k}", f"{sid} says {k}")], sid=sid)

    threads = [threading.Thread(target=run, args=(sid,)) for sid in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    expected = [ev["event"]["content"] for ev in db.read_durable_events(STREAM, limit=10_000)]
    assert [t for _, t in root(engines["s1"])] == expected and len(expected) == 80


def test_drain_pages_past_the_host_read_limit(tmp_path, db):
    e = make_engine(tmp_path, db)
    flush(db, e, [A(f"m{k}", f"line {k}") for k in range(1005)], notify=False)
    e.on_durable_events_committed("s1")
    texts = [t for _, t in root(e)]
    assert len(texts) == 1005 and texts[-1] == "line 1004"


def test_profiles_are_isolated_and_a_chat_never_follows_another_profiles_outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", Blocked)
    dbs = {}
    for p in ("work", "home"):
        dbs[p] = SessionDB(db_path=tmp_path / p / "state.db")
        dbs[p].create_session(session_id="s1", source="cli")
    try:
        work = make_engine(tmp_path / "work", dbs["work"])
        home = make_engine(tmp_path / "home", dbs["home"])
        flush(dbs["work"], work, [U("w1", "work secret")])
        flush(dbs["home"], home, [U("h1", "home note")])
        assert [t for _, t in root(work)] == ["work secret"]
        assert [t for _, t in root(home)] == ["home note"]

        wrong = make_engine(tmp_path / "work", dbs["home"])  # work's chat, home's database
        conv = flush(dbs["home"], home, [U("h2", "again")])
        with pytest.raises(Blocked):
            select(wrong, conv)
        assert [t for _, t in root(work)] == ["work secret"]
    finally:
        for d in dbs.values():
            d.close()


# -- subagents (spec §9) ---------------------------------------------------------------

def _child(tmp_path, db):
    db.create_session(session_id="child", source="subagent", parent_session_id="s1")
    return make_engine(tmp_path, db, sid="child", platform="subagent")


def test_in_process_child_reads_the_view_and_its_report_is_logged_by_the_parent(tmp_path, db):
    parent = make_engine(tmp_path, db)
    flush(db, parent, [U("u1", "delegate the audit")])
    child = _child(tmp_path, db)
    cconv = flush(db, child, [U("c1", "audit task")], sid="child", parent="s1")
    out = select(child, cconv)
    assert out[1]["content"][0]["text"] == "<chat>\n0+1|user: delegate the audit\n</chat>"
    cconv += flush(db, child, _tool_round(9), sid="child", parent="s1")
    child.on_turn_complete(cconv)
    flush(db, parent, [{"role": "tool", "tool_call_id": "d1", "name": "delegate_task",
                        "content": "REPORT: all clear", "message_uid": "p-report"}])
    assert [t for _, t in root(parent)] == ["delegate the audit", "REPORT: all clear"]


def test_child_process_without_the_writer_lock_gets_a_read_only_view_and_zoom(tmp_path, db, monkeypatch):
    from optchat.store import Store

    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", Blocked)
    parent = make_engine(tmp_path, db)
    flush(db, parent, [U("u1", "parent fact")])
    shutdown_all()
    other = Store(tmp_path / "optchat" / "chat")
    other.open()  # another process owns the chat
    try:
        before = sorted(p.name for p in (tmp_path / "optchat" / "chat").rglob("*"))
        child = _child(tmp_path, db)
        cconv = flush(db, child, [U("c1", "task")], sid="child", parent="s1")
        out = select(child, cconv)
        assert out[1]["content"][0]["text"] == "<chat>\n0+1|user: parent fact\n</chat>"
        assert json.loads(child.handle_tool_call("zoom", {"id": 0, "n": 1}))["result"].endswith("parent fact")
        assert sorted(p.name for p in (tmp_path / "optchat" / "chat").rglob("*")) == before

        root_engine = make_engine(tmp_path, db)  # a root session cannot log: no request at all
        rconv = flush(db, root_engine, [U("u2", "root turn")], notify=False)
        with pytest.raises(Blocked):
            select(root_engine, rconv)
    finally:
        other.close()


# -- other bindings ------------------------------------------------------------------

def test_detached_fork_reads_the_view_but_logs_nothing(tmp_path, db):
    parent = make_engine(tmp_path, db)
    flush(db, parent, [U("u1", "real chat")])
    fork = make_engine(tmp_path, db)
    fork.bind_session_state(session_db=None, session_id="")  # background review severs the binding
    conv = [U("u1", "real chat"), A("x1", "review notes"), U("x2", "review prompt")]
    out = select(fork, conv)
    assert out[1]["content"][0]["text"] == "<chat>\n0+1|user: real chat\n</chat>"
    fork.on_turn_complete(conv + [A("x3", "review result")])
    assert [t for _, t in root(parent)] == ["real chat"]


def test_host_without_an_outbox_keeps_the_live_reconciling_path(tmp_path):
    e = make_engine(tmp_path)
    e.bind_session_state(session_db=object(), session_id="s1")  # an older SessionDB
    conv = [U("u1", "hi")]
    select(e, conv)
    e.on_turn_complete(conv + [A("a1", "hello")])
    assert [t for _, t in root(e)] == ["hi", "hello"]


def test_host_failed_turn_notice_is_a_note_not_the_agent_talking(tmp_path, db):
    e = make_engine(tmp_path, db)
    flush(db, e, [U("u1", "hi"), A("f1", "This turn failed before a reply.", display_kind="failed_turn")])
    assert root(e) == [("user", "hi"), ("note", "This turn failed before a reply.")]

def test_model_switch_marker_is_note_not_user_instruction(tmp_path, db):
    e = make_engine(tmp_path, db)
    flush(db, e, [U("u1", "hi"), dict(U("switch", "Switched model to m2"), display_kind="model_switch")])
    assert root(e) == [("user", "hi"), ("note", "Switched model to m2")]


# -- the real host seams ---------------------------------------------------------------

def test_real_host_flush_and_selection_route_through_the_outbox(tmp_path, db):
    import logging

    from agent.conversation_loop import _apply_context_engine_selection
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, session_db=db, session_id="s1")
    agent._session_db_created = True
    e = make_engine(tmp_path, db)
    agent.context_compressor = e
    messages = [{"role": "user", "content": "deploy on fridays is banned"},
                {"role": "assistant", "content": "Noted.", "reasoning": "private thought"}]
    assert agent._flush_messages_to_session_db(messages, None) is True
    messages.append({"role": "user", "content": "what is banned?"})
    assert agent._flush_messages_to_session_db(messages, None) is True  # turn-start persistence
    # The commit nudge admitted it: the view was frozen first, then the line was logged.
    assert root(e)[-1] == ("user", "what is banned?") and len(root(e)) == 3

    api = [{"role": "system", "content": "SYS"}] + [{"role": m["role"], "content": m["content"]} for m in messages]
    out = _apply_context_engine_selection(agent, api, messages, messages[2], logger=logging.getLogger("t"))
    assert out[1]["content"][0]["text"] == ("<chat>\n0+1|user: deploy on fridays is banned\n"
                                            "1+1|talk: Noted.\n</chat>")
    assert len(root(e)) == 3
    assert all("private thought" not in p.read_text() for p in (tmp_path / "optchat").rglob("*.jsonl"))


def test_mid_turn_steer_is_logged_in_order_without_changing_the_frozen_view(tmp_path, db):
    e = make_engine(tmp_path, db)
    conv = flush(db, e, [U("u1", "hi"), A("a1", "hello")])
    conv += flush(db, e, [U("u2", "run it")])
    first = select(e, conv)
    conv += flush(db, e, _tool_round(1) + [U("st", "also check logs")])
    out = select(e, conv, incoming=conv[2])
    assert out[1] == first[1]
    assert [t for _, t in root(e)][-3:] == ["terminal {}", "out 1", "also check logs"]
