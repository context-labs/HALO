# HALO with Sign in with ChatGPT

HALO supports an independent OAuth session in its Python CLI and
engine. HALO registers its own client and requests permission to use your ChatGPT
plan. It does not borrow the Codex application's login or read its credentials.
The existing API-key mode remains the default. The HALO desktop application's
account settings are unchanged; these commands configure the Python CLI and engine.

## Start

Run these commands from this checkout:

```sh
uv sync --locked
uv run halo-chatgpt login
uv run halo-chatgpt models
```

Complete **Continue with ChatGPT** in the system browser and authorize HALO.
Select a model slug returned by `models`; substitute it for `YOUR_MODEL` below:

```sh
uv run halo-chatgpt check --model YOUR_MODEL
uv run halo tests/fixtures/tiny_traces.jsonl \
  --auth-mode chatgpt --model YOUR_MODEL \
  --max-depth 0 --max-turns 3 \
  -p "Briefly inspect the traces and report one concrete improvement."
```

`check` makes one small real request. It reports success only after
`response.completed`. The trace example is a fixture smoke test; it does not
evaluate your own agent or prove that a harness change improves performance.
For your own traces, replace the JSONL path and optionally pass
`--repo-path /path/to/your/harness` for read-only code inspection. Trace IDs must
follow HALO's existing input format.

## Accounts and credentials

```sh
uv run halo-chatgpt status
uv run halo-chatgpt login --new-account
uv run halo-chatgpt use SAVED_CLIENT_ID
uv run halo-chatgpt login --enable-plan
uv run halo-chatgpt logout
```

`login` reuses the selected registration. `--new-account` explicitly registers a
different account/workspace. `use` selects a saved issued client ID shown by
`status`. A running analysis remains bound to the account selected at its start.
An identity-only login is retained with plan usage disabled. `--enable-plan`
explicitly requests renewed consent; ordinary returning login does not force it.

Credentials default to `$XDG_CONFIG_HOME/halo/chatgpt.json`, or
`~/.config/halo/chatgpt.json`. Override this with `HALO_CHATGPT_AUTH_FILE`, or use
`halo-chatgpt --auth-file /protected/path/chatgpt.json COMMAND` and the matching
`halo --auth-file ... --auth-mode chatgpt`. Keep credentials outside the source
checkout. Files are atomically replaced with owner-only Unix permissions; on
Windows, use a directory protected by your user account's ACL. This implementation
does not manage Windows ACLs or OS keychains.

Refresh is serialized across processes, rotates the refresh token, and respects
the server's earliest-refresh time. Logout attempts remote revocation and clears
local tokens while retaining the host/client identity. A remote revocation failure
is reported; disconnect HALO in ChatGPT Settings if needed. No token is printed by
`status`. Do not enable HTTP debug logging or share credential files.

## Request behavior

- Root analysis and subagents use the Agents SDK with local tools in a `halo`
  namespace. HALO's saved tool history is restored with that namespace on replay.
- Synthesis and compaction also use streamed Responses in ChatGPT mode.
- Every inference request uses `https://api.openai.com/v1/responses`,
  `store=false`, `stream=true`, full local context, and the HALO-owned OAuth token.
- CLI rejects API keys, alternate URLs/headers, temperature and output-token
  limits in ChatGPT mode. The transport strips unsupported SDK default fields.
  Unrelated `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and API project billing settings
  cannot redirect this mode. Agents SDK hosted tracing is disabled in this mode;
  HALO's separate `--telemetry` feature remains opt-in.
- Stream failures, incompletion, or interruption cannot produce a successful
  report. Structured quota/capability rejections are terminal. There is no
  automatic switch to paid API-key inference.

Models and availability depend on the selected account and deployment. Manage
authorized usage in [ChatGPT Settings → Usage](https://chatgpt.com/settings/usage).
Real browser consent and successful live inference are separate from offline
test results. The auth/protocol regression tests are in
[`tests/unit/test_chatgpt.py`](../tests/unit/test_chatgpt.py).

## Verify and roll back

```sh
uv run python scripts/validate_pinned_versions.py
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest tests/unit
```

API-key runs remain available by omitting `--auth-mode chatgpt`. Credentials are
independent of Git; rolling back the code does not revoke authorization. Use
`halo-chatgpt logout` separately to revoke authorization, and retain analysis
traces when changing authentication modes.

## Primary references

- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Accounts, refresh and logout](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview request limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
- [OpenAI local OAuth reference implementation](https://github.com/openai/sign-in-with-chatgpt-devkit/blob/main/packages/local/src/oauth.ts)
