# oumigo Python API — Reference

> Status: implemented. The client library lives under `oumigo.api` (the manager/worker
> *handles*, the spawn/attach functions, and the `agent` inference layer). The *services*
> those handles talk to live under `oumigo.service`. Everything in this document is
> importable from the top-level `oumigo` package.

The API has two layers:

1. **Fleet control** — spawn or attach a manager and workers from Python:
   `oumigo_get_or_create_manager`, `oumigo_create_worker`, and the `OumigoManager` /
   `OumigoWorker` handles.
2. **Inference** — run chats and tools against the fleet's OpenAI-compatible data plane:
   `OumigoManager.create_agent(...)` → `OumigoAgent` → `OumigoChat` → `OumigoResponse`,
   plus the `@tool` decorator.

```python
from oumigo import (
    oumigo_get_or_create_manager, oumigo_create_worker,   # fleet control
    OumigoManager, OumigoWorker,                          # handles
    tool, Tool, ToolDefinitionError,                      # tools
    OumigoAgent, OumigoChat, OumigoResponse,              # inference
)
```

---

## Quickstart

```python
from oumigo import oumigo_get_or_create_manager, tool

# 1. Attach to a LAN manager, or spawn one (dies with this process).
manager = oumigo_get_or_create_manager(config_file="./manager.yaml")

# 2. Declare a tool — its JSON schema is inferred from the signature + docstring.
@tool
def get_current_date(utc_offset_hours: int = 9) -> str:
    """Get the current date at a given UTC offset.

    Args:
        utc_offset_hours: Timezone offset from UTC, in hours (e.g. 9 for JST).
    """
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone(timedelta(hours=utc_offset_hours))).strftime("%Y-%m-%d")

# 3. Make an agent (tools + sampling defaults) and a stateful chat.
agent = manager.create_agent(tools=[get_current_date])
chat = agent.create_chat(system="You are concise.")

# 4. Ask. The tool loop runs automatically.
print(chat.request("What's the date?").text)

# Streaming:
for piece in chat.request("And tomorrow?", stream=True):
    print(piece, end="")
```

---

## Fleet control

### `oumigo_get_or_create_manager(...) -> OumigoManager`

Return a manager handle, **reusing** a live manager discovered on the LAN (mDNS) or
**spawning** one as a child of this process (armed with `PR_SET_PDEATHSIG`, so it exits
when this process does). Blocks until the control plane is healthy.

```python
oumigo_get_or_create_manager(
    bearer_token=None, config_file=None,
    data_host=..., data_port=..., control_host=..., control_port=...,
    provider=..., model=...,
    *, discover_timeout=3.0, startup_timeout=20.0,
) -> OumigoManager
```

Settings resolve **explicit arg > `config_file` (YAML) > built-in default**. A missing or
unparseable `config_file` is ignored (a warning is logged).

> **Discovery caveat:** if a manager is already advertising on the LAN, it is reused and
> your `config_file` is **ignored** (the returned handle has `owned=False`). To guarantee
> your config is used, stop any other manager first, or edit the config of the manager
> that is actually running.

### `oumigo_create_worker(...) -> OumigoWorker`

Spawn a worker child on this host and block until its vLLM/HF replica is `SERVING`
(includes the model load/download — potentially many minutes).

```python
oumigo_create_worker(
    bearer_token=None, hf_home="~/.hf_cache", vllm_cache_root="~/.vllm_cache",
    hf_token=None, model_name=None,
    *, manager=None, backend="vllm", manager_url=None,
    discover_timeout=10.0, serving_timeout=None, poll_interval=2.0,
) -> OumigoWorker
```

The manager is resolved as: `manager` handle > `manager_url` > the last manager created in
this process > mDNS. `serving_timeout=None` waits **indefinitely** (you own the give-up
decision); the wait still ends early on a definitive failure (child exits or node reaches
`FAILED`).

### `OumigoManager`

A handle to a running manager (spawned or discovered). Context manager; `stop()` is a
no-op for a discovered (`owned=False`) manager.

| Member | Description |
|---|---|
| `control_url` / `data_url` | Control-plane and data-plane base URLs. |
| `token` | Shared bearer token (or `None`). |
| `owned` | `True` only if this process spawned the child. |
| `is_healthy(timeout_s=2.0) -> bool` | `True` once `/healthz` answers 200. |
| `workers(timeout_s=5.0) -> list[dict]` | Current worker registry records. |
| `metrics(*, since=None, prefixes=None, timeout_s=5.0) -> list[dict]` | Latest slot per node, or raw historical points when `since` is set. |
| `create_agent(...) -> OumigoAgent` | Mint an inference agent (see below). |
| `stop()` | Terminate the spawned control-plane child (owned only). |

### `OumigoWorker`

A handle to a worker child this process spawned. Context manager.

| Member | Description |
|---|---|
| `address` / `port` / `model` / `backend` / `node_id` | Replica identity. |
| `state() -> str \| None` | Node state the manager last saw (e.g. `"serving"`). |
| `is_serving() -> bool` | `True` when the replica is `SERVING`. |
| `is_alive() -> bool` | `True` while the child process runs. |
| `stop()` | Drain and shut down the replica (35 s grace before SIGKILL). |

---

## Inference

### `OumigoManager.create_agent(...) -> OumigoAgent`

```python
manager.create_agent(
    tools=None,
    *, max_iterations=5,
    temperature=None, max_tokens=None, top_p=None, stop=None,
) -> OumigoAgent
```

An **agent** is a capability bundle bound to the manager's data plane: the tools and
sampling defaults shared by every chat it spawns.

- `tools` — a sequence of `Tool` (or plain functions, which are wrapped via the strict
  `@tool` validator). Duplicate tool names raise `ValueError`.
- `max_iterations` — cap on model round-trips per request (the runaway tool-loop guard;
  default 5). On the cap, the response's `finish_reason` is `"max_iterations"`.
- Sampling defaults apply to every chat/request; override individually later as needed.

The system prompt and tools are **server-owned** — they come from your code here, never
from a client. (See [Security & the trust boundary](#security--the-trust-boundary).)

### `OumigoAgent.create_chat(...) -> OumigoChat`

```python
agent.create_chat(system=None, max_history_turns=3, history=None) -> OumigoChat
```

Start a **stateful** conversation.

- `system` — system-role content, prepended to every request in this chat.
- `max_history_turns` — how many prior `(user, assistant)` exchanges to carry into each
  request. `0` disables memory. Default 3.
- `history` — prior conversation to seed (for stateless servers; see
  [Stateless servers](#stateless-servers-history-rehydration)). Only `user`/`assistant`
  turns are accepted.

> `OumigoChat` is **not thread-safe** — use one chat per session.

### `OumigoChat`

| Member | Description |
|---|---|
| `request(contents: str, stream=False) -> OumigoResponse` | Run one user turn to completion (executes the tool loop). |
| `history -> list[dict]` | A copy of the carried `{"role","content"}` turns — persist this to rehydrate later. Never includes system/tool/reasoning. |

`request()` assembles `[system?] + recent history + {"role":"user","content":contents}`,
posts to the data plane, and runs the **agent loop**: while the model asks for tools,
execute the matching Python callbacks, feed their results back, and repeat — until the
model returns prose or `max_iterations` is reached. The `(user, final-answer)` exchange is
appended to `history`.

### `OumigoResponse`

The result of one `request()` — the same type whether or not you streamed.

| Attribute | Description |
|---|---|
| `text: str` | The final answer (complete immediately for non-stream; complete after iteration for stream). |
| `reasoning: str` | The model's thinking (`reasoning_content`), kept **out of** `text`. `""` unless the worker runs a `--reasoning-parser`. Output-only. |
| `finish_reason: str \| None` | `"stop"`, `"length"`, or `"max_iterations"`. |
| `tool_calls_made: list[dict]` | One `{"name","arguments","result"}` per executed tool, in order. |
| `raw: dict \| None` | The last raw completion payload (escape hatch). |

**Iteration** (streaming):

```python
resp = chat.request("hi", stream=True)

for piece in resp:            # answer text deltas (str) — the default
    ...

for piece in resp.stream():   # identical to `for piece in resp`
    ...

for text, reasoning in resp.stream(get_reasoning=True):   # dual channel
    if reasoning:
        ...  # live "thinking" (reasoning_content deltas)
    if text:
        ...  # the answer (content deltas)
```

In `stream(get_reasoning=True)`, exactly one side of each pair is non-empty (the other is
`""`, never `None`). Iterating a response drives its one-shot request, so use `for piece
in resp` **or** one `stream(...)` call — not both. Either way `text` and `reasoning`
accumulate; after consumption, re-iterating yields the accumulated value once.

---

## Tools

Decorate a plain function with `@tool`; its JSON schema is inferred from the signature
(types → required/optional) and the Google-style docstring (descriptions).

```python
from typing import Literal
from oumigo import tool

@tool
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. Tokyo.
        units: Temperature unit.
    """
    ...
```

A decorated function is a `Tool` that **remains callable** (`get_weather("Tokyo")` still
works) and carries `.name`, `.description`, `.parameters`, plus `.to_openai()` and
`.invoke(**kwargs)`.

### Strict validation (fail fast at import)

`@tool` validates the declaration **at decoration time** and raises a single, aggregated
`ToolDefinitionError` (listing every problem, with `file:line`) if anything is wrong.
Hard errors include:

- a parameter with no type annotation, or an unsupported type (`Any`, bare `list`/`dict`,
  multi-type unions, arbitrary classes);
- supported types are `str`, `int`, `float`, `bool`, `list[T]`, `Literal[...]` (→ enum),
  and `Optional[X]` / `X | None`;
- `*args` / `**kwargs` or positional-only parameters (the loop calls tools by keyword);
- a default whose type doesn't match its annotation;
- a missing docstring, an **undocumented** parameter, or a docstring `Args:` that drifts
  from the signature;
- a missing (or non-serializable) return annotation.

**Escape hatches:**
- `@tool(strict=False)` demotes *documentation* checks (docstring/param descriptions) to
  warnings; structural checks stay hard errors.
- `@tool(parameters={...})` supplies an explicit JSON Schema instead of inferring — still
  validated against the signature (property names must match; each needs a type +
  description).

### Runtime tool errors

If a tool raises when the model calls it (bad args, exception, unknown tool), the loop
**catches it and feeds the error back to the model** as the tool result (`"Error: ..."`),
so the model can recover — `request()` never crashes on a tool failure.

---

## Stateless servers (history rehydration)

The XBCOM app (and any web service) is stateless per request, while `OumigoChat` is
stateful. The rule: **store the history (data), not the Chat (object).** Persist a
session's `chat.history` in your own store (dict / Redis / DB), and rehydrate a fresh,
ephemeral chat per request:

```python
SESSIONS: dict[str, list] = {}                 # your session store
AGENT = manager.create_agent(tools=[...])      # server-owned tools
SYSTEM = "You are the XBCOM recommender."

def handle(session_id: str, user_msg: str) -> str:
    chat = AGENT.create_chat(system=SYSTEM, history=SESSIONS.get(session_id))  # rehydrate
    resp = chat.request(user_msg)                                             # run one turn
    SESSIONS[session_id] = chat.history                                       # persist
    return resp.text
```

This is stateless-scalable (any process can serve any session), restart-safe, and
sidesteps `OumigoChat`'s single-thread constraint (each request gets its own chat).

> `max_history_turns` currently bounds **both** what the model sees and what `chat.history`
> retains (the most recent N exchanges). Set it to how much context you want to keep.

---

## Security & the trust boundary

Rehydrating history from a client-held or stored blob is a hardening point. `create_chat(
history=...)` enforces a **trust boundary** (`_normalize_history`):

- **Only `user`/`assistant` turns are accepted.** A `system` or `tool` role raises
  `ValueError` — so a blob cannot inject a fake system prompt or a forged tool result.
- **Smuggled keys are stripped** (`tool_calls`, `reasoning_content`, …); each turn is
  rebuilt as a clean `{"role","content"}` dict — so spoofed tool calls can't sneak in.
- **System prompt and tools stay server-owned** — supplied by your code via `create_agent`
  / `create_chat(system=...)`, never from the request payload.

Two caveats:

1. This protects the *history*, not the *current user turn*, which an attacker always
   controls. Prompt injection via the live message ("ignore your instructions…") is a
   separate concern — screen it with a guardrail layer, and never let model output
   directly authorize a privileged action (tool calls touching real systems must be
   independently authorized server-side).
2. `reasoning_content` is **output-only**: surface it for display/debugging, but it is
   never stored in history or fed back to the model.

---

## Fleet configuration notes (vLLM)

Behavior of the inference layer depends on how each worker's vLLM was launched. Pass extra
flags through the `model.extra_args` list in `manager.yaml`; they are appended verbatim to
`vllm serve`.

**Tool calling** requires vLLM to be started with an auto tool-choice parser, or any
request carrying `tools` fails with HTTP 400:

```yaml
model:
  name: google/gemma-4-12B-it-qat-w4a16-ct
  extra_args:
    - --enable-auto-tool-choice
    - --tool-call-parser
    - gemma4            # pick the parser matching your model
```

**Reasoning** must be separated by a matching parser, or the model's channel/thinking
control tokens leak into `content`. Add:

```yaml
    - --reasoning-parser
    - gemma4
```

With the reasoning parser on, `resp.text` is the clean answer and `resp.reasoning` holds
the thinking; without it, `resp.reasoning` is `""`.

> `extra_args` flows `manager.yaml` → `NodeSpec` → the worker's `vllm serve`. The worker
> fetches its spec from the manager at startup, so a change requires **restarting the
> manager and the worker**.

---

## Notes & limits

- The inference layer is synchronous/blocking; streaming is a synchronous generator.
- Every model call funnels through one internal seam (`OumigoChat._payload`), reserved for
  a future guardrail interceptor chain — not yet wired.
- Sampling defaults currently cover `temperature` / `max_tokens` / `top_p` / `stop`.
