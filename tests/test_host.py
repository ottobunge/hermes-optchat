"""Integration with the real Hermes host code (loader + per-request seams), no model calls."""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PKG = Path(__file__).resolve().parents[1] / "optchat"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "plugins").mkdir(parents=True)
    os.symlink(PKG, home / "plugins" / "optchat")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for k in ("OPTCHAT_SUMMARIZER_CMD", "OPTCHAT_HOME"):
        monkeypatch.delenv(k, raising=False)
    # the host imports user engines under its own namespace: start clean each test
    for name in [m for m in sys.modules if m.startswith("_hermes_user_context_engine")]:
        del sys.modules[name]
    yield home
    for name in [m for m in sys.modules if m.startswith("_hermes_user_context_engine")]:
        mod = sys.modules[name]
        if hasattr(mod, "shutdown_all"):
            mod.shutdown_all()


def test_host_discovers_and_loads_user_installed_engine(hermes_home):
    from plugins.context_engine import discover_context_engines, load_context_engine

    found = {name: (desc, ok) for name, desc, ok in discover_context_engines()}
    assert "optchat" in found
    assert found["optchat"][1] is True
    assert "opt-in" in found["optchat"][0].lower()
    assert "core" in found["optchat"][0].lower()
    engine = load_context_engine("optchat")
    assert engine is not None and engine.name == "optchat"
    from agent.context_engine import ContextEngine
    assert isinstance(engine, ContextEngine)
    assert not (hermes_home / "optchat").exists()  # discovery/probing created no state


def test_never_auto_activated(hermes_home):
    from agent.agent_init import _select_context_engine

    assert _select_context_engine({}) is None
    assert _select_context_engine({"context": {"engine": "compressor"}}) is None
    picked = _select_context_engine({"context": {"engine": "optchat"}})
    assert picked is not None and picked.name == "optchat"


def test_general_plugin_registration_exposes_optchat_auxiliary_task(hermes_home):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from optchat import register

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="optchat"), manager)
    register(ctx)
    assert manager._aux_tasks["optchat"]["plugin"] == "optchat"
    assert manager._aux_tasks["optchat"]["defaults"]["provider"] == "openai-codex"
    assert manager._aux_tasks["optchat"]["defaults"]["model"] == "gpt-6-luna"
    assert manager._context_engine.name == "optchat"


def test_enabled_plugin_task_survives_context_loader_and_clone(hermes_home):
    from hermes_cli.plugins import get_plugin_auxiliary_tasks
    from plugins.context_engine import load_context_engine

    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: [optchat]\n")
    engine = load_context_engine("optchat")
    assert engine is not None
    tasks = {entry["key"]: entry for entry in get_plugin_auxiliary_tasks()}
    assert tasks["optchat"]["plugin"] == "optchat"
    from agent.auxiliary_client import _get_auxiliary_task_config
    route = _get_auxiliary_task_config("optchat")
    assert (route["provider"], route["model"], route["timeout"]) == ("openai-codex", "gpt-6-luna", 300.0)
    clone = engine.clone_for_agent()
    assert clone is not engine
    assert clone._summarizer is engine._summarizer
    assert clone._turn is None
    assert not (hermes_home / "optchat").exists()


def test_default_backend_compacts_a_real_engine_node_without_network(hermes_home, monkeypatch):
    from plugins.context_engine import load_context_engine

    monkeypatch.setenv("OPTCHAT_SUMMARIZER", "hermes")
    engine = load_context_engine("optchat")
    assert engine is not None
    engine.on_session_start("compaction-session", hermes_home=str(hermes_home))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="short summary"))])
    with patch("agent.auxiliary_client.call_llm", return_value=response) as call:
        memory = engine._memory()
        memory.log("user", "private test message " * 80)
        assert memory.settle(5)
        assert memory.store.node(0, 0) == "short summary"
    assert call.call_count == 1
    assert call.call_args.kwargs["task"] == "optchat"
    assert call.call_args.kwargs["provider"] == "openai-codex"


class StubAgent:
    def __init__(self, engine):
        self.context_compressor = engine
        self.session_id = "stub-session"


def test_real_host_seams_route_turns_through_optchat(hermes_home):
    from agent.conversation_loop import _apply_context_engine_selection, _notify_context_engine_turn_complete
    from plugins.context_engine import load_context_engine

    engine = load_context_engine("optchat")
    engine.on_session_start("stub-session", hermes_home=str(hermes_home), platform="cli")
    agent = StubAgent(engine)
    log = logging.getLogger("test")
    history = [{"role": "user", "content": "OLD-BEFORE-OPTCHAT", "message_uid": "o1"},
               {"role": "assistant", "content": "old answer", "message_uid": "o2"},
               {"role": "user", "content": "remember: deploy on fridays is banned", "message_uid": "u1"}]
    api = [{"role": "system", "content": "SYS"}] + [{k: v for k, v in m.items() if k != "message_uid"}
                                                   for m in history]
    out = _apply_context_engine_selection(agent, api, history, history[2], logger=log)
    assert out is not api
    assert "OLD-BEFORE-OPTCHAT" not in json.dumps(out)
    assert out[1]["content"][1]["text"] == "remember: deploy on fridays is banned"
    assert all("message_uid" not in m for m in out)

    history.append({"role": "assistant", "content": "Noted.", "message_uid": "a1"})
    _notify_context_engine_turn_complete(agent, history, usage=None, logger=log, turn_id="t1")
    history.append({"role": "user", "content": "what is banned?", "message_uid": "u2"})
    api = [{"role": "system", "content": "SYS"}] + [{k: v for k, v in m.items() if k != "message_uid"}
                                                   for m in history]
    out = _apply_context_engine_selection(agent, api, history, history[-1], logger=log)
    view = out[1]["content"][0]["text"]
    assert view == ("<chat>\n0+1|user: remember: deploy on fridays is banned\n"
                    "1+1|talk: Noted.\n</chat>")
    assert len(out) == 2


def test_engine_tools_follow_host_schema_normalization(hermes_home):
    from agent.memory_manager import normalize_tool_schema
    from plugins.context_engine import load_context_engine

    engine = load_context_engine("optchat")
    names = [normalize_tool_schema(s)["name"] for s in engine.get_tool_schemas()]
    assert names == ["zoom", "date"]


def test_fresh_request_preserves_selected_view_and_uses_optchat_cache_plan(hermes_home, monkeypatch):
    from agent import turn_request_assembly as assembly
    from plugins.context_engine import load_context_engine

    engine = load_context_engine("optchat")
    engine.on_session_start("cache-session", hermes_home=str(hermes_home), platform="cli")
    history = [{"role": "user", "content": "prior preference", "message_uid": "u1"}]
    engine.select_context([{"role": "system", "content": "SYS"}, {"role": "user", "content": "prior preference"}],
                          conversation_messages=history, incoming_message=history[0])
    history.append({"role": "user", "content": "fresh question", "message_uid": "u2"})
    # A large settled view makes the three OptChat-specific cache breakpoints observable.
    engine.memory.settle(5)
    view = "<chat>\n" + "".join(f"{i}+1|user: " + "x" * 170 + "\n" for i in range(600)) + "</chat>"
    monkeypatch.setattr(engine.memory, "render", lambda: view)
    source = [{"role": "system", "content": "SYS"},
              {"role": "user", "content": "prior preference"},
              {"role": "user", "content": "fresh question"}]
    monkeypatch.setattr(assembly, "build_api_messages", lambda *args, **kwargs: (source, "SYS"))
    import agent.conversation_loop as loop
    monkeypatch.setattr(loop, "_midturn_request_pressure_tokens", lambda *args: 0)
    monkeypatch.setattr(loop, "_pressure_with_real_floor", lambda *args: 0)
    agent = SimpleNamespace(context_compressor=engine, prefill_messages=[], api_mode="anthropic_messages",
                            tools=[], _use_prompt_caching=True, provider="anthropic", _cache_ttl="5m",
                            model="claude-sonnet-4-6", base_url="https://api.anthropic.com",
                            _use_native_cache_layout=True, _direct_native_anthropic_tool_cache_capability=lambda: True,
                            _sanitize_api_messages=lambda m: m,
                            _drop_thinking_only_and_merge_users=lambda m, **kw: m)
    result = assembly.assemble_api_request(
        agent, messages=history, current_turn_user_idx=1, _ext_prefetch_cache=None,
        _plugin_user_context=None, moa_config=None, active_system_prompt=None,
        original_user_message="fresh question", pending_moa_prepared_request=None,
        request_logger=logging.getLogger("test"))
    sent = result.api_messages
    assert len(sent) == 2  # no earlier raw transcript on this fresh request
    pieces = sent[1]["content"]
    assert "".join(p["text"] for p in pieces[:-1]) == view
    assert pieces[-1]["text"] == "fresh question"
    assert sum("cache_control" in p for p in pieces[:-1]) == 3
    assert "cache_control" not in sent[0]


def test_codex_app_server_cannot_bypass_optchat_selection(hermes_home):
    from run_agent import AIAgent
    from plugins.context_engine import load_context_engine

    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent.context_compressor = load_context_engine("optchat")
    agent.api_mode = "codex_app_server"
    agent._cached_system_prompt = "You are helpful."
    agent.save_trajectories = False
    with (patch.object(agent, "_run_codex_app_server_turn") as app_server,
          patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"),
          patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation("fresh question")
    app_server.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()
    assert result["failed"] is True
    assert "codex_app_server" in result["final_response"]
    assert "No provider call was sent" in result["final_response"]
