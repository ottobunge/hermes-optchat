# OptChat for Hermes (experimental)

An opt-in Hermes `ContextEngine` plugin inspired by Victor Taelin's [OptChat specification](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449). Each profile has one durable chat directory. It keeps a separate append-only log and binary summary tree, constructs an incrementally coarsened view, starts each turn with that view rather than the earlier Hermes transcript, and exposes `zoom(id, n)` / `date(id)`.

**Development build, not yet a released or default Hermes feature.** The companion [Hermes core integration branch](https://github.com/ottobunge/hermes-agent/tree/feat/optchat-integration) adds a transactional event outbox, fail-closed context selection, and opt-in Anthropic/OpenAI Responses cache planning. The plugin remains disabled in the user's active profile. Do not switch important existing chats to this unmerged build; Hermes's native transcript and the OptChat directory both need secure backups. See the remaining parity limits below.

## Roadmap and status

| Phase | Status |
| --- | --- |
| Context-engine interface and request-local fresh-turn view | Implemented; isolated Hermes/Codex turns confirmed history retrieval through `zoom` |
| Private append-only daily ROOT/tree JSONL, fsync, process writer lock, torn-line recovery | Implemented; backup and multi-host sync remain the operator's responsibility |
| Binary free nodes, bounded model summaries, background workers, incremental view/restart replay and recovery tools | Implemented; fake-summarizer tests plus a live Claude Code/Sonnet summary smoke test; long-run quality/cost not characterized |
| Transactional host event outbox, turn admission, hard-stop request selection | Implemented in the **unmerged core checkout**, with targeted tests; remaining ingress paths are under audit |
| Cache-aware Anthropic/OpenAI Responses request layout | Implemented in the **unmerged core checkout** for native Anthropic and direct OpenAI GPT-5.6+ routes; mocked transport tests pass, live cache usage unmeasured |
| Full integration and operational readiness | An isolated, real-model CLI session completed multiple turns, a `zoom` tool round, and a long-message Sonnet compaction; focused core/plugin regressions pass. Desktop attached images now reach the outbox before agent build. Full release regression, @-expansion parity, migration/browser/backup UX and measured provider caching remain open |

## Install (does not activate)

Copy or symlink the **`optchat/` directory**, not the repository root, to `$HERMES_HOME/plugins/optchat`. For the default profile:

```sh
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins"
ln -s /path/to/hermes-optchat/optchat "${HERMES_HOME:-$HOME/.hermes}/plugins/optchat"
```

The source directory may be elsewhere; adjust the path. This is a context-engine plugin, not a desktop UI plugin; it does **not** need `plugins.enabled`. **The released Hermes executable does not include this checkout's core seams.** Run the integrated source checkout (or build/install a package containing it) as well as linking the plugin; otherwise the plugin falls back to best-effort legacy mode. For an isolated test, create a dedicated `HERMES_HOME`, link `plugins/optchat` there, authenticate that profile on-device, then set `context.engine` using `HERMES_HOME=<test-home> hermes config set context.engine optchat`. Run the integrated source with `PYTHONPATH=<hermes-core-checkout>:<optchat-repo> HERMES_HOME=<test-home> <core-test-venv>/bin/python -m hermes_cli.main chat ...`. **Do not flip an active gateway/profile that is serving important sessions:** configuration may be picked up by new sessions. No existing session history is imported automatically.

A configured model summarizer is necessary to summarize material above 512 UTF-8 bytes. With none set, free short nodes work but a longer history cannot settle. The optional preset uses your authenticated Claude Code CLI with Sonnet, no tools, no project settings and no persistent CLI session:

```sh
export OPTCHAT_SUMMARIZER=claude-code
# optional: OPTCHAT_SUMMARIZER_MODEL=sonnet, OPTCHAT_SUMMARIZER_TIMEOUT=300
```

Set the variable in the **Hermes host/gateway process environment**, not in a chat. Alternatively `OPTCHAT_SUMMARIZER_CMD` is a JSON array of argv strings for a local no-shell command reading one JSON request (`{system,messages}`) from stdin and writing only its summary to stdout; `OPTCHAT_SUMMARIZER_INPUT=text` passes plain text. Never point it to `hermes` itself. Long text user messages are given to the summarizer in full; tool-result echoes **and their provider-facing copies** are capped to 30,000 characters (the original Hermes transcript is unchanged; a longer original is also kept in the media sidecar, see below). The `claude` preset invokes a fresh CLI process per attempt and serializes retry history into text, so it does not preserve a literal single vendor conversation across retries. Its `--system-prompt` replaces Claude Code's normal prompt, and `--tools ''` disables its tools. This operation can consume your subscription quota; select the engine only when you intend that.

Data goes under `$HERMES_HOME/optchat/chat/{main,tree}/YYYY-MM-DD.jsonl` and `chat/assets/` (original media payloads) by default. Each record has an append sequence across the two streams so restarts replay late node completions without silently changing the view. Files contain unredacted conversation/tool content. The plugin creates private `0700` folders and `0600` files; back up the directory securely. One process owns the chat's advisory `flock`; a second Hermes process using the same profile cannot write until the first exits. Profile roots are isolated. Do not point multiple machines at the same uncoordinated directory.

## Tests

```sh
PYTHONPATH=/path/to/hermes-agent:/path/to/hermes-optchat \
  python -m pytest -q /path/to/hermes-optchat/tests
```

`tests/test_outbox.py` needs a host with the durable outbox (it skips otherwise). Tests use temporary chat directories and databases and fake summarizers; no user sessions or paid model calls. Replace the paths and Python interpreter as appropriate. The plugin itself only needs Python stdlib and the host's `agent.context_engine`.

## Durable outbox mode (hosts with `SessionDB.read_durable_events`)

When the host binds a session database that has the durable event outbox (`bind_session_state`), the outbox is the **only** source of ROOT lines; the live-history reconciliation and `on_turn_complete` logging are not used:

- `project_durable_events` runs inside the host's transcript transaction and maps each committed user/assistant/tool row to one event on stream `optchat.root` (role, exact stored content, tool calls, `message_uid`, tool-call occurrence uids, timestamp). It is pure: no files, no network, no tools. Model reasoning is never in it. A delegated child (`platform='subagent'` or a `parent_session_id`) projects nothing; its report reaches ROOT as the parent's row.
- The consumer drains the stream in global `seq` order (all sessions of the profile, including events whose commit callback was missed, crashed processes and deleted sessions), 1000 events per page, into ROOT with fsync and `src` dedupe. `chat/outbox/cursor.jsonl` advances only after every ROOT line of an event is durable, so a crash replays the event once.
- **Turn admission:** before any user event is logged, the view must settle; its parts (node ids, not text) are frozen in `chat/outbox/admit.jsonl` keyed by the event, and only then is the event logged. `select_context` renders the frozen parts of the current user event — on tool-loop requests, provider retries and after a restart alike — and drains new events without changing the view. A mid-turn steer is admitted the same way and does not change the turn's frozen view.
- If the view cannot settle, the plugin raises the host's `ContextSelectionBlocked`: no provider request is sent. The user event stays in the outbox and is logged by the next drain once the memory recovers. A host without that sentinel gets the older memory-unavailable notice instead (it cannot abort the request).
- A chat directory follows one host database only; binding it to another profile's database blocks rather than mixing cursors.
- A subagent or a detached background-review fork (`bind_session_state(None, "")`) reads the view and logs nothing. If another process holds the writer lock, a subagent gets a read-only view and `zoom`/`date` reloaded from disk; a root session in that situation is blocked.

In the integrated core checkout, `raw_content` carries JSON-safe image/audio/file parts into the transactional event; the consumer archives media payloads into private assets before writing the ROOT line. Desktop images already attached at `prompt.submit` are included in the first event, before agent construction. Older host write paths that only provide text remain flagged `content_fidelity: "host-text"`. **Remaining boundary:** file/@-reference expansion that occurs later in the agent prologue cannot change an already-admitted immutable event; the original user text is preserved but its later expanded file contents are not archived as that event's media. Hermes's `codex_app_server` route is hard-blocked for OptChat rather than letting it send ordinary history. The commit callback can wait up to the settle timeout while admitting a user event.

## Media and large tool results

The same archival path handles the integrated host's raw multimodal events and the older live-history path, provided the event contains `raw_content`; text-only host rows cannot reconstruct attachments.

The view and the compactor are text only. Media never reaches them as base64, and a fabricated `[image]` never stands in for content. Instead each ROOT line keeps two separate things:

- `text`, which is what the view, the compactor and `zoom` show. For each attachment it holds a line such as `[attachment 1 not shown: image/png, 1234 bytes, sha256 9d0c38e7aafe062c; original archived, zoom this message (n = 1) for the file]`.
- `orig` and `assets`: the message's original content, with each payload string replaced by an `{"$optchat": "asset", "sha256", "enc"}` marker, plus the files those markers refer to. `Store.original(i)` rebuilds the exact original content (for data URLs, the exact string) from them.

Payloads go to `chat/assets/<sha256[:2]>/<sha256>.<ext>`. This sidecar is content-addressed, append-only, `0700`/`0600`, owned by the user and never reached through a symlink. File names come only from the digest and a fixed extension table, never from message content. Each asset is written to `assets/.tmp`, fsynced, hard-linked into place (never overwriting an existing file), then its folder is fsynced, all **before** the ROOT line is written. Reads and dedupes verify the sha256. If an asset write fails, no ROOT line is written and the outbox cursor does not advance; a later drain retries. A crash between asset and ROOT writes leaves an unreferenced but complete file (`Store.orphan_assets()`), which the retry reuses. Interrupted temporary files are removed at the next open. Assets are never deleted, so back them up together with `main/` and `tree/`. The host captures authorized local images as data URLs *before* acknowledging a Desktop submit; arbitrary paths in message parts are never opened by the plugin.

`zoom(id, 1)` on such a message prints its text, then one `- <type>, <bytes> bytes, sha256 <hex>: <absolute path>` line per file. Pass that path to `vision_analyze` or another file tool. If a file is missing or the wrong size, zoom says so.

| Content part | Archived | Notes |
| --- | --- | --- |
| `image_url` / `input_image` / `video_url` / `audio_url` with a base64 data URL | decoded bytes | Non-canonical base64 (e.g. with line breaks) or a non-base64 data URL is kept verbatim as a UTF-8 text file. |
| `input_audio.data`, Anthropic `image`/`document`/`audio` `source.type=base64`, `file`/`input_file` `file_data` | decoded bytes | MIME type from the part or the file name. |
| `file://` URL or absolute local path in a user/model/tool part | path only; **no file read** | The message text notes that its bytes were not archived. A message part is not authorization to read an arbitrary host file. Host-attached Desktop images are converted to immutable data URLs and SHA-256 manifests before the submit row commits; those bytes are archived by the consumer even after the original file is removed. |
| `http(s)` and other URLs | URL only, verbatim | The text says only the URL is kept and that its content may change or disappear. The plugin never fetches it. |
| Provider `file_id` | id only | The text says it is held by the provider and not archived. |
| Any other part type | the whole part as exact JSON | A value that isn't JSON (e.g. a Python object) is still **rejected**: the turn gets a memory-unavailable notice and nothing is logged. |
| Model thinking parts | not logged, not archived | As specified in §2. `orig` is the content *without* thoughts. |

**Tool results (deliberate deviation from spec §7).** The spec caps an `echo` at 30,000 characters in ROOT. Here ROOT `text` and the provider-facing copy are still capped (head and tail, with a cut note; attachment lines are never cut). A longer original is also kept whole in the sidecar, and zoom points to it. Tool results at or under the cap are logged exactly as before. Screenshots inside tool results are forwarded to the provider unchanged, with only their text capped.

Limits: the archive stores message *content* only (ROOT already holds tool calls as text). Only the plugin process writes it; it is not a host-level journal. Images are forwarded to the provider only if the host sends them; the plugin adds no vision capability.

## Scope and remaining gaps

- Released Hermes installations without the new outbox still use best-effort live-history reconciliation. They cannot recover events absent from host history or hard-stop a provider request if memory cannot settle. Use the integrated checkout for durable operation.
- The integrated checkout journals raw media on ordinary agent flushes and idle Desktop images already attached at submit. Busy-queued images are captured at *dispatch*, not at their earlier queue acknowledgement; if their files disappear or the gateway restarts before dispatch, they cannot yet be recovered as OptChat media. File/@-reference expansion later in a Desktop turn may rewrite the transcript after its initial event committed, so the immutable first ROOT event retains the original user text rather than the expanded file contents.
- The host retains its own system instructions, safety layers, and tool framework rather than replacing them with the reference harness's exact `MASTER` prompt. Hermes's `codex_app_server` and MoA routes are blocked for OptChat because they bypass or precede the selected view; the generic Codex Responses and Anthropic routes are supported. Native Anthropic and direct OpenAI GPT-5.6+ caching have wire-shape tests, **not measured cache-hit data**. Other compatible Responses endpoints retain their ordinary cache behavior.
- Subagents read a settled view and their internal tool calls are excluded from ROOT; background reports and mid-turn steering follow Hermes's own lifecycle, not the reference's exact harness timing.
- The reference HTML browser, automatic history import, backup/sync service and multi-host operation are not part of the plugin. The JSONL log/tree remain directly browsable, and `zoom`/`date` navigate them. Plan backups of both `$HERMES_HOME/state.db` and `$HERMES_HOME/optchat/chat/`.
