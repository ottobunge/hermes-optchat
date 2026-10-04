import json
import sys
import textwrap

import pytest

from optchat.summarizer import CLAUDE_CODE_ARGV, CommandSummarizer, SummarizerError, from_env

ECHO_JSON = textwrap.dedent("""
    import json, os, sys
    data = json.load(sys.stdin)
    out = {"system": data["system"], "n": len(data["messages"]),
           "child": os.environ.get("OPTCHAT_SUMMARIZER_CHILD"), "argv": sys.argv[1:]}
    print(json.dumps(out))
""")


def _script(tmp_path, body, name="s.py"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_runs_argv_with_json_stdin_and_marks_child(tmp_path):
    s = CommandSummarizer([sys.executable, _script(tmp_path, ECHO_JSON), "{system}"], timeout=30)
    out = json.loads(s("SYS PROMPT", [{"role": "user", "content": "hi"}]))
    assert out == {"system": "SYS PROMPT", "n": 1, "child": "1", "argv": ["SYS PROMPT"]}


def test_no_shell_interpretation(tmp_path):
    marker = tmp_path / "pwned"
    s = CommandSummarizer([sys.executable, _script(tmp_path, ECHO_JSON), f"$(touch {marker})", "{system}"],
                          timeout=30)
    out = json.loads(s(f"; touch {marker}; `touch {marker}`", [{"role": "user", "content": f"$(touch {marker})"}]))
    assert not marker.exists()
    assert out["argv"][0] == f"$(touch {marker})"


def test_nonzero_exit_raises(tmp_path):
    s = CommandSummarizer([sys.executable, _script(tmp_path, "import sys; sys.exit(3)")], timeout=30)
    with pytest.raises(SummarizerError):
        s("S", [{"role": "user", "content": "x"}])


def test_timeout_kills_and_raises(tmp_path):
    import time
    s = CommandSummarizer([sys.executable, _script(tmp_path, "import time; time.sleep(30)")], timeout=0.5)
    t0 = time.monotonic()
    with pytest.raises(SummarizerError):
        s("S", [{"role": "user", "content": "x"}])
    assert time.monotonic() - t0 < 5


def test_refuses_to_call_hermes_recursively():
    with pytest.raises(ValueError):
        CommandSummarizer(["/usr/bin/hermes", "chat"], timeout=5)
    with pytest.raises(ValueError):
        CommandSummarizer([], timeout=5)


def test_text_mode_renders_the_conversation(tmp_path):
    s = CommandSummarizer([sys.executable, _script(tmp_path, "import sys; print(sys.stdin.read())")],
                          timeout=30, input_format="text")
    out = s("S", [{"role": "user", "content": [{"type": "text", "text": "<chat>\n</chat>"},
                                               {"type": "text", "text": "Compress this"}]},
                  {"role": "assistant", "content": "too long line"},
                  {"role": "user", "content": "That line is 600 bytes"}])
    assert "<chat>\n</chat>\n\nCompress this" in out
    assert "too long line" in out and out.rstrip().endswith("That line is 600 bytes")


def test_close_terminates_an_active_summarizer_process(tmp_path):
    import threading
    import time
    started = tmp_path / "started"
    script = _script(tmp_path, f"from pathlib import Path\nimport time\nPath({str(started)!r}).write_text('ready')\ntime.sleep(30)\n")
    summarizer = CommandSummarizer([sys.executable, script], timeout=40)
    errors = []
    thread = threading.Thread(target=lambda: _record_summary_error(summarizer, errors))
    thread.start()
    deadline = time.monotonic() + 5
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()
    summarizer.close()
    thread.join(3)
    assert not thread.is_alive(), "a closed chat must not leave a summarizer running"
    assert errors and isinstance(errors[0], SummarizerError)


def _record_summary_error(summarizer, errors):
    try:
        summarizer("S", [{"role": "user", "content": "text"}])
    except Exception as exc:
        errors.append(exc)


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPTCHAT_SUMMARIZER", raising=False)
    monkeypatch.delenv("OPTCHAT_SUMMARIZER_CMD", raising=False)
    assert from_env() is None
    monkeypatch.setenv("OPTCHAT_SUMMARIZER", "claude-code")
    s = from_env()
    assert s.argv[0] == "claude" and "--tools" in s.argv and s.input_format == "text"
    assert s.argv == CLAUDE_CODE_ARGV
    monkeypatch.setenv("OPTCHAT_SUMMARIZER_MODEL", "haiku")
    assert from_env().argv[from_env().argv.index("--model") + 1] == "haiku"
    monkeypatch.delenv("OPTCHAT_SUMMARIZER")
    monkeypatch.setenv("OPTCHAT_SUMMARIZER_CMD", json.dumps(["python3", "x.py"]))
    monkeypatch.setenv("OPTCHAT_SUMMARIZER_TIMEOUT", "12")
    s = from_env()
    assert s.argv == ["python3", "x.py"] and s.timeout == 12 and s.input_format == "json"
    monkeypatch.setenv("OPTCHAT_SUMMARIZER_CMD", "python3 x.py; rm -rf /")
    with pytest.raises(ValueError):
        from_env()
