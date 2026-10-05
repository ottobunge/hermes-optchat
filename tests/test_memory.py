import pytest

from fakes import FakeSummarizer
from optchat.memory import Memory


@pytest.fixture
def make(chat_dir):
    made = []

    def _make(summarizer=None, **kw):
        kw.setdefault("retry", 0.05)
        m = Memory(chat_dir, summarizer=summarizer, agent="Hermes", **kw)
        m.open()
        made.append(m)
        return m

    yield _make
    for m in made:
        m.close()


def test_close_stops_the_compactor_backend(chat_dir):
    import threading
    class ClosableSummarizer:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.closed = False
        def __call__(self, system, messages):
            self.started.set()
            self.stopped.wait(5)
            return "summary"
        def close(self):
            self.closed = True
            self.stopped.set()
    backend = ClosableSummarizer()
    m = Memory(chat_dir, summarizer=backend)
    m.open()
    m.log("user", "x" * 1000)
    assert backend.started.wait(2)
    m.close()
    assert backend.closed


def test_compactor_worker_inherits_profile_scope(make, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "isolated-profile"
    observed = []

    def summarize(system, messages):
        observed.append(get_hermes_home())
        return "summary"

    token = set_hermes_home_override(home)
    try:
        memory = make(summarize)
        memory.log("user", "x" * 1000)
        assert memory.settle(5)
    finally:
        reset_hermes_home_override(token)
    assert observed == [home]


def test_retry_worker_retains_profile_scope(make, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "isolated-profile"
    observed = []

    def summarize(system, messages):
        observed.append(get_hermes_home())
        if len(observed) == 1:
            raise RuntimeError("retry once")
        return "summary"

    token = set_hermes_home_override(home)
    try:
        memory = make(summarize, retry=0.01)
        memory.log("user", "x" * 1000)
        assert memory.settle(5)
    finally:
        reset_hermes_home_override(token)
    assert observed == [home, home]


def test_short_messages_are_free_nodes_and_view_is_verbatim(make):
    m = make()
    m.log("user", "hello")
    m.log("talk", "hi\nthere")
    assert m.settle(timeout=1)
    assert m.render() == "<chat>\n0+1|user: hello\n1+1|talk: hi there\n</chat>"


def test_long_message_is_summarized_with_context_and_no_ids(make):
    fake = FakeSummarizer()
    m = make(fake)
    m.log("user", "earlier short")
    m.log("echo", "x" * 2000)
    assert m.settle(timeout=5)
    assert len(fake.calls) == 1
    ctx, step = fake.calls[0]
    assert ctx == "<chat>\nuser: earlier short\n</chat>"
    assert step.endswith("echo: " + "x" * 2000)
    assert m.store.node(0, 1).startswith("S[")
    assert not fake.violations


def _built_order(m):
    # tree file order == commit order (one writer, append-only)
    import json
    out = []
    for f in sorted((m.path / "tree").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out.append((r["l"], r["i"]))
    return out


def test_compactor_order_leaves_in_sequence_merges_after_sources_and_context(make):
    fake = FakeSummarizer(delay=0.01)
    m = make(fake)
    for k in range(8):
        m.log("user" if k % 2 == 0 else "echo", f"{k} " + "y" * 900)
    assert m.settle(timeout=10)
    deadline = __import__("time").time() + 10
    while m.store.node(3, 0) is None and __import__("time").time() < deadline:
        __import__("time").sleep(0.01)
    order = _built_order(m)
    leaves = [i for l, i in order if l == 0]
    assert leaves == list(range(8))
    pos = {node: k for k, node in enumerate(order)}
    for (l, i), k in pos.items():
        if l > 0:
            assert pos[(l - 1, 2 * i)] < k and pos[(l - 1, 2 * i + 1)] < k
            # rule 3: every leaf up to the node's end was built before it
            end = (i + 1) << l
            assert all(pos[(0, j)] < k for j in range(end))
    assert (3, 0) in pos
    assert not fake.violations


def test_at_most_jobs_calls_run_at_once(make):
    import time
    fake = FakeSummarizer(delay=0.05)
    m = make(fake, jobs=2)
    # pre-build many leaves so lots of merges become ready at once
    for k in range(16):
        m.log("user", f"{k} " + "z" * 600)
    assert m.settle(timeout=20)
    deadline = time.time() + 20
    while m.store.node(4, 0) is None and time.time() < deadline:
        time.sleep(0.01)
    assert m.store.node(4, 0) is not None
    assert fake.max_active <= 2


def test_failed_node_is_retried_after_delay_and_reported_once(make, caplog):
    fake = FakeSummarizer(fail_first=3)
    m = make(fake, retry=0.05)
    with caplog.at_level("WARNING"):
        m.log("user", "q" * 1000)
        assert m.settle(timeout=5)
    assert len(fake.calls) == 4
    fails = [r for r in caplog.records if "failed" in r.message]
    assert len(fails) == 1


def test_settle_times_out_while_summarizer_keeps_failing(make):
    fake = FakeSummarizer(fail_first=10**6)
    m = make(fake, retry=0.01)
    m.log("user", "q" * 1000)
    assert m.settle(timeout=0.3) is False
    assert "(not summarized yet: zoom it)" in m.render()


def test_reopen_reuses_tree_and_refolds_view(chat_dir):
    fake = FakeSummarizer()
    m = Memory(chat_dir, summarizer=fake, budget=2000)
    m.open()
    for k in range(12):
        m.log("user", f"{k} " + "w" * 700)
    assert m.settle(timeout=10)
    import time
    deadline = time.time() + 10
    while m.busy and time.time() < deadline:
        time.sleep(0.01)
    rendered = m.render()
    m.close()
    again = FakeSummarizer()
    m2 = Memory(chat_dir, summarizer=again, budget=2000)
    m2.open()
    try:
        assert m2.render() == rendered
        assert again.calls == []
    finally:
        m2.close()


def test_without_summarizer_settle_fails_fast_on_material_needing_a_model(make):
    import time
    m = make(None)
    m.log("user", "short is fine")
    assert m.settle(timeout=5)
    m.log("user", "L" * 1000)
    t0 = time.monotonic()
    assert m.settle(timeout=5) is False
    assert time.monotonic() - t0 < 1


def test_log_dedupes_by_source_key(make):
    m = make()
    assert m.log("user", "a", src="uid:1") == 0
    assert m.log("user", "a again", src="uid:1") == 0
    assert len(m.store.messages) == 1
    assert m.has_src("uid:1")


def _complete(m, l, i, text):
    # what a finished summarizer job does (Memory._job), in a chosen completion order
    with m._lock:
        m._commit(l, i, text)
        m._pump()


def test_restart_reconstructs_live_view_after_out_of_order_completions(chat_dir):
    # node=150: 200-byte messages and 201-byte merges are never free, so only the
    # completions below build nodes, in exactly this order.
    m = Memory(chat_dir, summarizer=None, budget=350, node=150)
    m.open()
    for k in range(4):
        m.log("user", f"{k}" * 200)
    for k in range(4):
        _complete(m, 0, k, f"{k}" * 100)  # 400 bytes > 350, no parent yet
    _complete(m, 1, 1, "B" * 100)  # merges 2..3 -> 300 bytes, under budget
    _complete(m, 1, 0, "A" * 100)  # arrives late: the live view must not use it
    live, size = list(m.view.parts), m.view.size(m.store)
    assert live == [(0, 0), (0, 1), (1, 1)]
    m.close()

    m2 = Memory(chat_dir, summarizer=None, budget=350, node=150)
    m2.open()
    try:
        assert m2.view.parts == live
        assert m2.view.size(m2.store) == size
        m2.log("user", "4" * 200)
        _complete(m2, 0, 4, "4" * 100)
        live = list(m2.view.parts)
    finally:
        m2.close()
    m3 = Memory(chat_dir, summarizer=None, budget=350, node=150)
    m3.open()
    try:
        assert m3.view.parts == live
    finally:
        m3.close()
