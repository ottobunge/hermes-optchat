import time

from fakes import FakeSummarizer
from optchat.memory import Memory


def test_logging_stays_fast_as_the_chat_grows(chat_dir):
    m = Memory(chat_dir, summarizer=FakeSummarizer(), budget=20_000)
    m.open()
    try:
        t0 = time.monotonic()
        for k in range(4000):
            m.log("user" if k % 2 == 0 else "talk", f"message {k}")
        assert m.settle(10)
        elapsed = time.monotonic() - t0
        # Disk fsync latency varies by host; this guards quadratic regressions,
        # not a fixed real-time performance contract.
        assert elapsed < 30, elapsed
        assert m.view.size(m.store) <= 20_000
    finally:
        m.close()
    t0 = time.monotonic()
    m2 = Memory(chat_dir, summarizer=FakeSummarizer(), budget=20_000)
    m2.open()
    try:
        assert time.monotonic() - t0 < 10
    finally:
        m2.close()
