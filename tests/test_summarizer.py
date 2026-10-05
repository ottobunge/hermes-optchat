import json
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from optchat.summarizer import CLAUDE_CODE_ARGV, CommandSummarizer, HermesSummarizer, SummarizerError, from_env

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

def test_default_uses_host_auxiliary_backend_without_a_model_call():
    from optchat.summarizer import HermesSummarizer

    summarizer = from_env({})
    assert isinstance(summarizer, HermesSummarizer)
    assert summarizer.task == "optchat"


def test_hermes_backend_sends_all_turns_through_optchat_task():
    with (patch("hermes_cli.config.load_config_readonly", return_value={}),
          patch("agent.auxiliary_client.call_llm", return_value=SimpleNamespace(
              choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])) as call):
        messages = [{"role": "user", "content": [{"type": "text", "text": "<chat/>"}]},
                    {"role": "assistant", "content": "old attempt"},
                    {"role": "user", "content": "try again"}]
        result = from_env({})("system prompt", messages)
    assert result == "summary"
    assert call.call_args.kwargs == {
        "task": "optchat", "provider": "openai-codex", "model": "gpt-6-luna",
        "fallback_policy": "task_chain_only",
        "messages": [{"role": "system", "content": "system prompt"}, *messages],
        "timeout": 300.0,
    }


def test_auxiliary_config_overrides_provider_model_and_timeout():
    config = {"auxiliary": {"optchat": {"provider": "openrouter", "model": "vendor/fast",
                                        "timeout": 44}}}
    with (patch("hermes_cli.config.load_config_readonly", return_value=config),
          patch("agent.auxiliary_client.call_llm", return_value=SimpleNamespace(
              choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])) as call):
        assert from_env({})("S", [{"role": "user", "content": "hi"}]) == "ok"
    assert call.call_args.kwargs["provider"] == "openrouter"
    assert call.call_args.kwargs["model"] == "vendor/fast"
    assert call.call_args.kwargs["timeout"] == 44
    assert call.call_args.kwargs["fallback_policy"] == "task_chain_only"


def test_process_timeout_takes_precedence_over_auxiliary_task_timeout():
    with (patch("hermes_cli.config.load_config_readonly", return_value={
              "auxiliary": {"optchat": {"timeout": 44}}}),
          patch("agent.auxiliary_client.call_llm", return_value=SimpleNamespace(
              choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])) as call):
        summarizer = from_env({"OPTCHAT_SUMMARIZER_TIMEOUT": "12"})
        assert summarizer("S", [{"role": "user", "content": "hi"}]) == "ok"
    assert call.call_args.kwargs["timeout"] == 12


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), TimeoutError("timed out")])
def test_hermes_backend_propagates_provider_and_timeout_errors(failure):
    with (patch("hermes_cli.config.load_config_readonly", return_value={}),
          patch("agent.auxiliary_client.call_llm", side_effect=failure)):
        with pytest.raises(type(failure), match=str(failure)):
            from_env({})("S", [{"role": "user", "content": "text"}])


def test_hermes_backend_rejects_empty_model_output():
    with (patch("hermes_cli.config.load_config_readonly", return_value={}),
          patch("agent.auxiliary_client.call_llm", return_value=SimpleNamespace(
              choices=[SimpleNamespace(message=SimpleNamespace(content=None))]))):
        with pytest.raises(SummarizerError, match="empty"):
            from_env({})("S", [{"role": "user", "content": "text"}])


def test_hermes_backend_close_prevents_new_calls():
    summarizer = from_env({})
    summarizer.close()
    with patch("agent.auxiliary_client.call_llm") as call:
        with pytest.raises(SummarizerError, match="closed"):
            summarizer("S", [{"role": "user", "content": "text"}])
    call.assert_not_called()


def test_hermes_backend_close_discards_in_flight_output():
    import threading

    entered, release = threading.Event(), threading.Event()
    errors = []

    def blocked(**kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="late summary"))])

    summarizer = from_env({})
    with (patch("hermes_cli.config.load_config_readonly", return_value={}),
          patch("agent.auxiliary_client.call_llm", side_effect=blocked)):
        thread = threading.Thread(target=_record_summary_error, args=(summarizer, errors))
        thread.start()
        try:
            assert entered.wait(3)
            summarizer.close()
        finally:
            release.set()
            thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SummarizerError)


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPTCHAT_SUMMARIZER", raising=False)
    monkeypatch.delenv("OPTCHAT_SUMMARIZER_CMD", raising=False)
    assert isinstance(from_env(), HermesSummarizer)
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


@pytest.mark.parametrize("selection", ["none", "off"])
def test_explicit_none_disables_even_a_configured_command(selection):
    assert from_env({"OPTCHAT_SUMMARIZER": selection,
                     "OPTCHAT_SUMMARIZER_CMD": '["python3","worker.py"]'}) is None


def test_explicit_hermes_backend_ignores_command_override():
    assert isinstance(from_env({"OPTCHAT_SUMMARIZER": "hermes",
                                "OPTCHAT_SUMMARIZER_CMD": '["python3","worker.py"]'}), HermesSummarizer)
