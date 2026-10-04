import json

import pytest

from agent.context_engine import ContextEngine
from fakes import FakeSummarizer
from optchat import OptChatEngine
from optchat import engine as engine_mod
from optchat.engine import shutdown_all


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # These tests pin the older-host contract (no request-blocking sentinel): fail closed
    # with a notice. test_host_sentinel_blocks_instead_of_sending_a_notice covers newer hosts.
    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", None)
    for k in ("OPTCHAT_SUMMARIZER", "OPTCHAT_SUMMARIZER_CMD", "OPTCHAT_HOME", "OPTCHAT_SUMMARIZER_CHILD"):
        monkeypatch.delenv(k, raising=False)
    yield
    shutdown_all()


def test_engine_satisfies_abc_and_constructs_without_side_effects(tmp_path):
    e = OptChatEngine()
    assert isinstance(e, ContextEngine)
    assert e.name == "optchat"
    assert not (tmp_path / "hermes").exists()  # probing availability must not create state
    assert e.should_compress() is False
    msgs = [{"role": "user", "content": "hi"}]
    assert e.compress(msgs) == msgs


SYSTEM = {"role": "system", "content": "You are Hermes."}


def make_engine(tmp_path, summarizer=None, **kw):
    kw.setdefault("settle_timeout", 5)
    kw.setdefault("retry", 0.05)
    e = OptChatEngine(summarizer=summarizer, **kw)
    e.on_session_start("sess-1", hermes_home=str(tmp_path / "hermes"), platform="cli")
    return e


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_") and k != "message_uid"}


def request_for(conv):
    return [dict(SYSTEM)] + [strip(m) for m in conv]


def select(e, conv, incoming_idx):
    return e.select_context(request_for(conv), conversation_messages=[dict(m) for m in conv],
                            incoming_message=dict(conv[incoming_idx]), budget_tokens=200_000)


def logged(e):
    return [(m["kind"], m["text"]) for m in e.memory.store.messages]


def user(text, uid):
    return {"role": "user", "content": text, "message_uid": uid}


def test_first_turn_sends_view_then_message_and_logs_after_render(tmp_path):
    e = make_engine(tmp_path)
    conv = [user("hello", "u1")]
    out = select(e, conv, 0)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("You are Hermes.")
    assert "zoom(id, n)" in out[0]["content"]
    assert out[1]["role"] == "user"
    assert out[1]["content"] == [{"type": "text", "text": "<chat>\n</chat>"},
                                 {"type": "text", "text": "hello"}]
    assert len(out) == 2
    assert logged(e) == [("user", "hello")]
    assert (tmp_path / "hermes" / "optchat" / "chat" / "main").is_dir()


def assistant_call(uid, name="read_file", args='{"path": "a.py"}', content=""):
    return {"role": "assistant", "content": content, "message_uid": uid,
            "tool_calls": [{"id": "c-" + uid, "type": "function", "function": {"name": name, "arguments": args}}]}


def tool_result(uid, call_uid, text):
    return {"role": "tool", "tool_call_id": "c-" + call_uid, "content": text, "message_uid": uid}


def test_in_turn_tail_is_preserved_and_view_frozen(tmp_path):
    fake = FakeSummarizer()
    e = make_engine(tmp_path, fake)
    conv = [user("hello", "u1")]
    first = select(e, conv, 0)
    conv += [assistant_call("a1", content="Checking."), tool_result("t1", "a1", "file body " * 200)]
    second = select(e, conv, 0)
    assert second[:2] == first[:2]  # same system, same frozen view + message
    assert second[2:] == request_for(conv)[2:]  # the whole in-turn tail, verbatim
    assert logged(e) == [("user", "hello"), ("talk", "Checking."),
                         ("tool", 'read_file {"path": "a.py"}'), ("echo", ("file body " * 200))]
    # nodes built mid-turn do not change this turn's view
    assert e.memory.settle(5)
    third = select(e, conv, 0)
    assert third[1] == first[1]


def test_in_turn_tool_result_is_capped_only_on_provider_copy(tmp_path):
    from optchat.events import CAP
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("hello", "u1")]
    select(e, conv, 0)
    huge = "H" * 20_000 + "M" * 20_000 + "T" * 20_000
    conv += [assistant_call("a1"), tool_result("t1", "a1", huge)]
    sent = select(e, conv, 0)
    assert len(sent[-1]["content"]) <= CAP
    assert sent[-1]["content"].startswith("H" * 1000)
    assert sent[-1]["content"].endswith("T" * 1000)
    assert conv[-1]["content"] == huge  # request-local rewrite, never mutate Hermes history
    assert e.memory.store.messages[-1]["kind"] == "echo"
    assert len(e.memory.store.messages[-1]["text"]) <= CAP


def test_in_turn_multiblock_tool_result_is_capped(tmp_path):
    from optchat.events import CAP
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("hello", "u1")]
    select(e, conv, 0)
    result = tool_result("t1", "a1", "")
    result["content"] = [{"type": "text", "text": "X" * 60000}]
    conv += [assistant_call("a1"), result]
    out = select(e, conv, 0)
    assert isinstance(out[-1]["content"], str)
    assert len(out[-1]["content"]) <= CAP
    assert len(conv[-1]["content"][0]["text"]) == 60000


def test_turn_complete_logs_final_reply_and_next_turn_sees_it(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("hello", "u1")]
    select(e, conv, 0)
    conv += [{"role": "assistant", "content": "Hi! How can I help?", "message_uid": "a1"}]
    e.on_turn_complete([dict(m) for m in conv], usage=None)
    assert logged(e)[-1] == ("talk", "Hi! How can I help?")
    conv += [user("next question", "u2")]
    out = select(e, conv, 2)
    assert out[1]["content"][0]["text"] == "<chat>\n0+1|user: hello\n1+1|talk: Hi! How can I help?\n</chat>"
    assert out[1]["content"][1]["text"] == "next question"
    assert len(out) == 2  # earlier turns are never sent as history
    assert logged(e)[-1] == ("user", "next question")


def test_missed_turn_complete_is_reconciled_at_next_turn_before_render(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("q1", "u1")]
    select(e, conv, 0)
    conv += [{"role": "assistant", "content": "answer one", "message_uid": "a1"}]
    # host aborted: no on_turn_complete
    conv += [user("q2", "u2")]
    out = select(e, conv, 2)
    assert logged(e) == [("user", "q1"), ("talk", "answer one"), ("user", "q2")]
    assert "1+1|talk: answer one" in out[1]["content"][0]["text"]


def test_both_paths_and_retries_never_duplicate(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("q1", "u1")]
    select(e, conv, 0)
    select(e, conv, 0)  # provider retry of the same request
    conv += [assistant_call("a1"), tool_result("t1", "a1", "ok")]
    select(e, conv, 0)
    conv += [{"role": "assistant", "content": "done", "message_uid": "a2"}]
    e.on_turn_complete([dict(m) for m in conv])
    e.on_turn_complete([dict(m) for m in conv])
    conv += [user("q2", "u2")]
    select(e, conv, 4)
    assert [k for k, _ in logged(e)] == ["user", "tool", "echo", "talk", "user"]


def test_existing_session_history_is_not_migrated(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    old = [user("old secret plan", "o1"), {"role": "assistant", "content": "old reply", "message_uid": "o2"}]
    conv = old + [user("first optchat turn", "u1")]
    out = select(e, conv, 2)
    assert logged(e) == [("user", "first optchat turn")]
    sent = json.dumps(out)
    assert "old secret plan" not in sent and "old reply" not in sent


def test_messages_without_uid_dedupe_by_session_position(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [{"role": "user", "content": "no uid"}]
    select(e, conv, 0)
    conv += [{"role": "assistant", "content": "reply"}]
    e.on_turn_complete([dict(m) for m in conv])
    conv += [{"role": "user", "content": "again"}]
    select(e, conv, 2)
    assert logged(e) == [("user", "no uid"), ("talk", "reply"), ("user", "again")]


def test_next_turn_waits_for_summaries_before_the_request(tmp_path):
    import threading
    import time
    from fakes import GateSummarizer

    gate = GateSummarizer()
    e = make_engine(tmp_path, gate)
    big = "B" * 5000
    conv = [user(big, "u1")]
    select(e, conv, 0)
    conv += [{"role": "assistant", "content": "ok", "message_uid": "a1"}]
    e.on_turn_complete([dict(m) for m in conv])
    conv += [user("q2", "u2")]
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("out", select(e, conv, 2)))
    t.start()
    time.sleep(0.2)
    assert t.is_alive()  # waiting for the compactor, not sending placeholders
    gate.release()
    t.join(10)
    view = box["out"][1]["content"][0]["text"]
    assert "not summarized yet" not in view
    assert view.startswith("<chat>\n0+1|S[")


def test_settle_timeout_never_sends_unsummarized_view_or_raw_history(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer(fail_first=10**6), settle_timeout=0.2, retry=0.01)
    big = "SECRET-HEAD " + "B" * 5000 + " SECRET-TAIL"
    conv = [user(big, "u1"), {"role": "assistant", "content": "ok", "message_uid": "a1"}, user("q2", "u2")]
    select(e, conv[:1], 0)
    out = select(e, conv, 2)
    sent = json.dumps(out)
    assert "not summarized yet" not in out[1]["content"][0]["text"]
    assert "OptChat memory is unavailable" in sent
    assert "SECRET" not in sent
    assert e.get_status()["optchat"]["last_settle"] is False


def test_host_sentinel_blocks_instead_of_sending_a_notice(tmp_path, monkeypatch):
    class Blocked(Exception):
        pass

    monkeypatch.setattr(engine_mod, "ContextSelectionBlocked", Blocked)
    e = make_engine(tmp_path, FakeSummarizer(fail_first=10**6), settle_timeout=0.2, retry=0.01)
    conv = [user("B" * 5000, "u1"), {"role": "assistant", "content": "ok", "message_uid": "a1"}, user("q2", "u2")]
    select(e, conv[:1], 0)
    with pytest.raises(Blocked, match="memory is unavailable"):
        select(e, conv, 2)
    assert [m["text"] for m in e.memory.store.messages][-1] == "ok"  # q2 not logged; reconciled later


def _history_with_secret():
    return [user("OLD-HISTORY-SECRET", "o1"), {"role": "assistant", "content": "OLD-REPLY", "message_uid": "o2"}]


def test_lock_held_elsewhere_fails_closed_without_history(tmp_path):
    from optchat.store import Store
    other = Store(tmp_path / "hermes" / "optchat" / "chat")
    other.open()  # another process/gateway owns the chat
    try:
        e = make_engine(tmp_path, FakeSummarizer())
        conv = _history_with_secret() + [user("now", "u1"), assistant_call("a1"), tool_result("t1", "a1", "r")]
        out = select(e, conv, 2)
        sent = json.dumps(out)
        assert "OLD-HISTORY-SECRET" not in sent and "OLD-REPLY" not in sent
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert "OptChat memory is unavailable" in out[1]["content"][0]["text"]
        assert out[1]["content"][1]["text"] == "now"
        assert out[2:] == request_for(conv)[4:]  # in-turn tail kept so tool calls still pair up
        assert "in use by another writer" in e.get_status()["optchat"]["degraded"]
    finally:
        other.close()


def test_failed_memory_does_not_forward_uncapped_tool_result(tmp_path):
    from optchat.events import CAP
    from optchat.store import Store
    other = Store(tmp_path / "hermes" / "optchat" / "chat")
    other.open()
    try:
        e = make_engine(tmp_path, FakeSummarizer())
        conv = [user("now", "u1"), assistant_call("a1"), tool_result("t1", "a1", "X" * 60000)]
        out = select(e, conv, 0)
        assert len(out[-1]["content"]) <= CAP
        assert conv[-1]["content"] == "X" * 60000
    finally:
        other.close()


def test_unlocatable_turn_fails_closed(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = _history_with_secret() + [user("now", "u1")]
    out = e.select_context(request_for(conv), conversation_messages=[dict(m) for m in conv],
                           incoming_message={"role": "user", "content": "something else"})
    sent = json.dumps(out)
    assert "OLD-HISTORY-SECRET" not in sent and "OLD-REPLY" not in sent
    assert isinstance(out, list) and out and all(isinstance(m, dict) for m in out)


def test_internal_error_fails_closed(tmp_path, monkeypatch):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = _history_with_secret() + [user("now", "u1")]
    select(e, conv, 2)
    monkeypatch.setattr(e.memory, "render", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    conv2 = conv + [{"role": "assistant", "content": "x", "message_uid": "a9"}, user("again", "u2")]
    out = select(e, conv2, 4)
    sent = json.dumps(out)
    assert "OLD-HISTORY-SECRET" not in sent
    assert out[-1]["content"][-1]["text"] == "again"


PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(200))


def _png_url():
    import base64
    return "data:image/png;base64," + base64.b64encode(PNG).decode()


def test_multimodal_user_turn_is_archived_losslessly_and_still_sent(tmp_path):
    import re
    fake = FakeSummarizer()
    e = make_engine(tmp_path, fake)
    content = [{"type": "text", "text": "What is in this image? " + "context " * 80},
               {"type": "image_url", "image_url": {"url": _png_url()}}]
    conv = [{"role": "user", "message_uid": "img1", "content": content}]
    out = select(e, conv, 0)
    assert out[1]["content"][1:] == content  # the model still gets the real image
    assert "OptChat memory is unavailable" not in json.dumps(out)
    [rec] = e.memory.store.messages
    assert rec["kind"] == "user" and "[attachment 1 not shown: image/png" in rec["text"]
    assert e.memory.store.original(0) == content
    payload = _png_url().split(",", 1)[1][:24]
    conv += [{"role": "assistant", "content": "A test pattern.", "message_uid": "a1"}, user("thanks", "u2")]
    second = select(e, conv, 2)
    assert e.memory.settle(5)
    assert fake.calls and all(payload not in c + s for c, s in fake.calls)  # compactor: text only
    assert payload not in second[1]["content"][0]["text"]  # the view: text only
    assert "[image]" not in json.dumps(second)
    got = json.loads(e.handle_tool_call("zoom", {"id": 0, "n": 1}))["result"]
    path = re.search(r"sha256 [0-9a-f]{64}: (/\S+)$", got, re.M).group(1)
    assert open(path, "rb").read() == PNG


def test_in_turn_tool_screenshot_is_forwarded_not_replaced_by_a_placeholder(tmp_path):
    from optchat.events import CAP
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [user("take a screenshot", "u1")]
    select(e, conv, 0)
    result = tool_result("t1", "a1", "")
    result["content"] = [{"type": "text", "text": "X" * 60_000},
                         {"type": "image_url", "image_url": {"url": _png_url()}}]
    conv += [assistant_call("a1"), result]
    out = select(e, conv, 0)
    sent = out[-1]["content"]
    assert "unsupported multimodal" not in json.dumps(out)
    assert sent[-1] == {"type": "image_url", "image_url": {"url": _png_url()}}
    assert sent[0]["type"] == "text" and len(sent[0]["text"]) <= CAP
    assert e.memory.store.original(len(e.memory.store.messages) - 1) == result["content"]


def _media_conv():
    return [user("hello", "u0"), {"role": "assistant", "content": "hi", "message_uid": "a0"},
            {"role": "user", "message_uid": "img1", "content": [
                {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": _png_url()}}]}]


def test_failed_asset_write_fails_closed_and_next_turn_recovers_after_restart(tmp_path, monkeypatch):
    from optchat.store import Store
    e = make_engine(tmp_path, FakeSummarizer())
    conv = _media_conv()
    select(e, conv[:1], 0)
    e.on_turn_complete(conv[:2])
    real = Store.put_asset
    monkeypatch.setattr(Store, "put_asset", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left")))
    out = select(e, conv, 2)
    sent = json.dumps(out)
    assert "OptChat memory is unavailable" in sent and "not recorded" in sent and "No space left" in sent
    assert "[image]" not in sent
    assert logged(e) == [("user", "hello"), ("talk", "hi")]  # no ROOT line without its asset
    monkeypatch.setattr(Store, "put_asset", real)
    shutdown_all()  # restart
    e2 = make_engine(tmp_path, FakeSummarizer())
    conv += [{"role": "assistant", "content": "(unanswered)", "message_uid": "a1"}, user("again", "u2")]
    select(e2, conv, 4)
    kinds = [k for k, _ in logged(e2)]
    assert kinds == ["user", "talk", "user", "talk", "user"]
    assert e2.memory.store.original(2) == conv[2]["content"]
    assert e2.memory.store.orphan_assets() == []


def test_crash_between_asset_and_root_leaves_a_reused_orphan(tmp_path, monkeypatch):
    from optchat.store import Store
    e = make_engine(tmp_path, FakeSummarizer())
    conv = _media_conv()
    select(e, conv[:1], 0)
    e.on_turn_complete(conv[:2])
    real = Store._write_line
    def crash(self, folder, record):
        if record.get("assets"):
            raise OSError(5, "simulated crash before the ROOT line")
        return real(self, folder, record)
    monkeypatch.setattr(Store, "_write_line", crash)
    select(e, conv, 2)
    assert len(e.memory.store.messages) == 2
    assert len(e.memory.store.orphan_assets()) == 1
    monkeypatch.setattr(Store, "_write_line", real)
    shutdown_all()
    e2 = make_engine(tmp_path, FakeSummarizer())
    conv += [{"role": "assistant", "content": "(unanswered)", "message_uid": "a1"}, user("again", "u2")]
    select(e2, conv, 4)
    assert e2.memory.store.original(2) == conv[2]["content"]
    assert e2.memory.store.orphan_assets() == []  # the same sidecar file, now referenced
    files = list((tmp_path / "hermes" / "optchat" / "chat" / "assets").glob("??/*"))
    assert len(files) == 1


def test_unarchivable_input_is_still_declined_not_logged_lossily(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    conv = [{"role": "user", "message_uid": "x1", "content": [
        {"type": "text", "text": "what is this?"}, {"type": "mystery", "blob": object()}]}]
    out = select(e, conv, 0)
    assert "OptChat memory is unavailable" in json.dumps(out, default=str)
    assert e.memory.store.messages == []


def test_tool_schemas_and_calls(tmp_path):
    e = make_engine(tmp_path, FakeSummarizer())
    names = [t["name"] for t in e.get_tool_schemas()]
    assert names == ["zoom", "date"]
    zs = e.get_tool_schemas()[0]
    assert zs["description"].startswith("Open the line id+n of the view")
    assert zs["parameters"]["required"] == ["id", "n"]
    conv = [user("hello\nworld", "u1")]
    select(e, conv, 0)
    assert json.loads(e.handle_tool_call("zoom", {"id": 0, "n": 1})) == {"result": "0+0|user: hello\nworld"}
    assert json.loads(e.handle_tool_call("zoom", {"id": 0, "n": 2})) == {"result": "No line 0+2."}
    assert json.loads(e.handle_tool_call("date", {"id": 0}))["result"][:4].isdigit()
    assert "error" in json.loads(e.handle_tool_call("nope", {}))
    assert json.loads(e.handle_tool_call("zoom", {"id": "0", "n": 1})) == {"result": "0+0|user: hello\nworld"}


def test_tool_schemas_are_constant(tmp_path):
    a, b = make_engine(tmp_path), OptChatEngine()
    assert json.dumps(a.get_tool_schemas()) == json.dumps(b.get_tool_schemas())


def test_subagent_sees_view_but_logs_nothing(tmp_path):
    main = make_engine(tmp_path, FakeSummarizer())
    conv = [user("main question", "u1")]
    select(main, conv, 0)
    sub = OptChatEngine(summarizer=FakeSummarizer(), settle_timeout=5)
    sub.on_session_start("child-1", hermes_home=str(tmp_path / "hermes"), platform="subagent")
    task = [user("TASK: list files", "c1")]
    out = select(sub, task, 0)
    assert out[1]["content"][0]["text"] == "<chat>\n0+1|user: main question\n</chat>"
    assert out[1]["content"][1]["text"] == "TASK: list files"
    task += [assistant_call("ca1"), tool_result("ct1", "ca1", "files")]
    select(sub, task, 0)
    sub.on_turn_complete([dict(m) for m in task])
    assert logged(main) == [("user", "main question")]
    assert sub.memory is main.memory  # one backend per chat per process


def test_clone_and_deepcopy_share_backend_but_not_turn_state(tmp_path):
    import copy
    e = make_engine(tmp_path, FakeSummarizer())
    select(e, [user("hi", "u1")], 0)
    for c in (e.clone_for_agent(), copy.deepcopy(e)):
        assert isinstance(c, OptChatEngine) and c is not e
        assert c._turn is None
        assert c.chat_dir() == e.chat_dir()
        c.on_session_start("other", hermes_home=str(tmp_path / "hermes"), platform="cli")
        select(c, [user("from clone", "x-" + str(id(c)))], 0)
        assert c.memory is e.memory
    assert [t for _, t in logged(e)] == ["hi", "from clone", "from clone"]


def test_concurrent_sessions_share_one_consistent_log(tmp_path):
    import threading
    proto = make_engine(tmp_path, FakeSummarizer(delay=0.001))
    errors = []

    def session(n):
        try:
            e = proto.clone_for_agent()
            e.on_session_start(f"s{n}", hermes_home=str(tmp_path / "hermes"), platform="telegram")
            conv = []
            for t in range(5):
                conv.append(user(f"s{n} turn {t} " + "p" * (50 if t % 2 else 900), f"s{n}-u{t}"))
                u = len(conv) - 1
                select(e, conv, u)
                conv += [assistant_call(f"s{n}-a{t}"), tool_result(f"s{n}-t{t}", f"s{n}-a{t}", "r" * 700)]
                select(e, conv, u)
                conv.append({"role": "assistant", "content": f"s{n} done {t}", "message_uid": f"s{n}-f{t}"})
                e.on_turn_complete([dict(m) for m in conv])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=session, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    msgs = proto._memory().store.messages
    assert [m["i"] for m in msgs] == list(range(len(msgs)))
    assert len(msgs) == 4 * 5 * 4
    srcs = [m["src"] for m in msgs]
    assert len(srcs) == len(set(srcs))
    assert proto.memory.settle(20)


def test_profiles_are_isolated(tmp_path):
    a = OptChatEngine(summarizer=None)
    a.on_session_start("x", hermes_home=str(tmp_path / "profileA"), platform="cli")
    b = OptChatEngine(summarizer=None)
    b.on_session_start("y", hermes_home=str(tmp_path / "profileB"), platform="cli")
    select(a, [user("in A", "a1")], 0)
    select(b, [user("in B", "b1")], 0)
    assert logged(a) == [("user", "in A")] and logged(b) == [("user", "in B")]
    assert a.chat_dir() != b.chat_dir()


def test_default_home_follows_hermes_home_env(tmp_path):
    e = OptChatEngine()
    assert e.chat_dir() == tmp_path / "hermes" / "optchat" / "chat"


def test_summarizer_child_never_compacts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTCHAT_SUMMARIZER_CHILD", "1")
    monkeypatch.setenv("OPTCHAT_SUMMARIZER", "claude-code")
    e = OptChatEngine()
    e.on_session_start("z", hermes_home=str(tmp_path / "hermes"), platform="cli")
    select(e, [user("hi", "u1")], 0)
    assert e.memory.summarizer is None
