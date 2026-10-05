"""Compactor model backends.

A summarizer is any callable ``(system: str, messages: list[dict]) -> str`` where
``messages`` is a chat in OpenAI shape (user / assistant turns; the first user turn holds
two text blocks: the ``<chat>`` context and the step). Tests use in-process fakes.

``HermesSummarizer`` is the default: one host auxiliary-client call on task ``optchat``.
Directory context-engine discovery supplies a minimal collector without ``ctx.llm``, so
this backend uses the host auxiliary API directly and never starts another Hermes agent.

``CommandSummarizer`` runs an explicitly configured local command:

* argv list only, never a shell (``shell=False``); ``"{system}"`` as a whole argv element is
  replaced by the system prompt; nothing else is interpolated;
* bounded wall-clock timeout, after which the whole process group is killed;
* the command must not be Hermes itself (no recursive agent call), and the child gets
  ``OPTCHAT_SUMMARIZER_CHILD=1`` so an OptChat engine inside it refuses to compact;
* runs in an empty temporary directory, so project files (CLAUDE.md, AGENTS.md) are not read.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading

DEFAULT_TIMEOUT = 300.0

# Claude Code, print mode, Sonnet, no tools, no settings/hooks/MCP/plugins, no saved session.
# Uses whatever login the user already has for `claude`.
CLAUDE_CODE_ARGV = [
    "claude", "-p",
    "--model", "sonnet",
    "--output-format", "text",
    "--tools", "",
    "--setting-sources", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--system-prompt", "{system}",
]

_FORBIDDEN = {"hermes", "hermes-agent", "hermes_agent"}
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_MODEL = "gpt-6-luna"
TASK = "optchat"


class SummarizerError(RuntimeError):
    pass


class HermesSummarizer:
    """Host-owned auxiliary model route; no recursive agent process."""

    task = TASK

    def __init__(self, *, timeout: float | None = None):
        self.timeout = float(timeout) if timeout is not None else None
        self._closed = threading.Event()

    def close(self) -> None:
        self._closed.set()

    def __call__(self, system: str, messages) -> str:
        if self._closed.is_set():
            raise SummarizerError("summarizer is closed")
        from agent.auxiliary_client import call_llm
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        auxiliary = config.get("auxiliary", {}) if isinstance(config, dict) else {}
        configured = auxiliary.get(TASK, {}) if isinstance(auxiliary, dict) else {}
        configured = configured if isinstance(configured, dict) else {}
        response = call_llm(task=TASK,
                            provider=configured.get("provider", DEFAULT_PROVIDER),
                            model=configured.get("model", DEFAULT_MODEL),
                            fallback_policy="task_chain_only",
                            messages=[{"role": "system", "content": system}, *messages],
                            timeout=self.timeout if self.timeout is not None else
                            float(configured.get("timeout") or DEFAULT_TIMEOUT))
        if self._closed.is_set():
            raise SummarizerError("summarizer is closed")
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise SummarizerError("summarizer returned empty model output")
        return text


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def render_text(messages) -> str:
    """One prompt for CLIs that take a single message: the turns in order."""
    out = []
    for k, m in enumerate(messages):
        if m["role"] == "assistant":
            out.append("Your previous answer:\n" + _text(m["content"]))
        else:
            out.append(_text(m["content"]))
    return "\n\n".join(out)


class CommandSummarizer:
    def __init__(self, argv, *, timeout: float = DEFAULT_TIMEOUT, input_format: str = "json"):
        argv = [str(a) for a in argv]
        if not argv:
            raise ValueError("summarizer command is empty")
        if os.path.basename(argv[0]).lower() in _FORBIDDEN:
            raise ValueError("the OptChat summarizer must not be Hermes itself (recursive agent call)")
        if input_format not in ("json", "text"):
            raise ValueError(f"unknown summarizer input format {input_format!r}")
        self.argv, self.timeout, self.input_format = argv, float(timeout), input_format
        self._lock = threading.Lock()
        self._children: set[subprocess.Popen] = set()
        self._closed = False

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for proc in self._children:
                self._kill(proc)

    def __call__(self, system: str, messages) -> str:
        argv = [system if a == "{system}" else a for a in self.argv]
        if self.input_format == "json":
            stdin = json.dumps({"system": system, "messages": messages}, ensure_ascii=False)
        else:
            stdin = render_text(messages)
        env = dict(os.environ)
        env["OPTCHAT_SUMMARIZER_CHILD"] = "1"
        env.pop("CLAUDECODE", None)  # allow `claude` even when Hermes runs inside Claude Code
        with tempfile.TemporaryDirectory(prefix="optchat-sum-") as cwd:
            with self._lock:
                if self._closed:
                    raise SummarizerError("summarizer is closed")
                try:
                    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, cwd=cwd, env=env, shell=False,
                                            start_new_session=True)
                except OSError as exc:
                    raise SummarizerError(f"cannot run summarizer {argv[0]!r}: {exc}") from exc
                self._children.add(proc)
            try:
                try:
                    out, err = proc.communicate(stdin.encode("utf-8"), timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    self._kill(proc)
                    proc.communicate()
                    raise SummarizerError(f"summarizer timed out after {self.timeout:g}s")
            finally:
                with self._lock:
                    self._children.discard(proc)
        if proc.returncode != 0:
            tail = err.decode("utf-8", "replace").strip()[-300:]
            raise SummarizerError(f"summarizer exited {proc.returncode}: {tail}")
        return out.decode("utf-8", "replace")


def from_env(env=None):
    """Build the summarizer from the environment (Hermes auxiliary route by default).

    OPTCHAT_SUMMARIZER=hermes            Hermes auxiliary task ``optchat`` (default)
    OPTCHAT_SUMMARIZER=none|off          Explicitly disable model compaction
    OPTCHAT_SUMMARIZER=claude-code       Claude Code CLI preset (model: OPTCHAT_SUMMARIZER_MODEL,
                                         default "sonnet")
    OPTCHAT_SUMMARIZER_CMD='["argv",..]' any local command, JSON argv list (never a shell string)
    OPTCHAT_SUMMARIZER_INPUT=json|text   stdin format for OPTCHAT_SUMMARIZER_CMD (default json)
    OPTCHAT_SUMMARIZER_TIMEOUT=seconds   per call (default 300; overrides auxiliary.optchat.timeout)
    """
    env = os.environ if env is None else env
    raw_timeout = env.get("OPTCHAT_SUMMARIZER_TIMEOUT")
    timeout = float(raw_timeout or DEFAULT_TIMEOUT)
    preset = (env.get("OPTCHAT_SUMMARIZER") or "").strip().lower()
    if preset in ("claude-code", "claude"):
        argv = list(CLAUDE_CODE_ARGV)
        model = env.get("OPTCHAT_SUMMARIZER_MODEL")
        if model:
            argv[argv.index("--model") + 1] = model
        return CommandSummarizer(argv, timeout=timeout, input_format="text")
    if preset in ("none", "off"):
        return None
    if preset in ("hermes", "hermes-native"):
        return HermesSummarizer(timeout=timeout if raw_timeout else None)
    if preset:
        raise ValueError(f"unknown OPTCHAT_SUMMARIZER preset {preset!r}")
    raw = env.get("OPTCHAT_SUMMARIZER_CMD")
    if not raw:
        return HermesSummarizer(timeout=timeout if raw_timeout else None)
    try:
        argv = json.loads(raw)
    except ValueError:
        raise ValueError("OPTCHAT_SUMMARIZER_CMD must be a JSON list of argv strings") from None
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError("OPTCHAT_SUMMARIZER_CMD must be a JSON list of argv strings")
    return CommandSummarizer(argv, timeout=timeout,
                             input_format=(env.get("OPTCHAT_SUMMARIZER_INPUT") or "json").strip().lower())
